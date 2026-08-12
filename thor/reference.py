"""Plaintext reference execution and sample preparation for THOR."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from datasets import Dataset, load_from_disk
from safetensors import safe_open
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import easyfhe as etorch
from easyfhe import fhe

from .config import RunConfig, SEQUENCE_LENGTH
from .layout import pack_attention_mask, pack_embedding


@dataclass(frozen=True)
class PreparedSample:
    index: int
    label: int
    packed_input: np.ndarray
    attention_mask_bundle: fhe.ConstantBundle
    reference_logits: np.ndarray
    reference_prediction: int
    attention_tokens: int
    sequence_length: int


@dataclass(frozen=True)
class ReferenceAssets:
    split: Dataset
    tokenizer: PreTrainedTokenizerBase
    state: dict[str, np.ndarray]


def model_schema() -> dict[str, tuple[int, ...]]:
    """Return the exact raw tensor contract for the fixed BERT-base graph."""

    schema = {
        "bert.embeddings.word_embeddings.weight": (30522, 768),
        "bert.embeddings.position_embeddings.weight": (512, 768),
        "bert.embeddings.token_type_embeddings.weight": (2, 768),
        "bert.embeddings.LayerNorm.weight": (768,),
        "bert.embeddings.LayerNorm.bias": (768,),
        "bert.pooler.dense.weight": (768, 768),
        "bert.pooler.dense.bias": (768,),
        "cls.seq_relationship.weight": (2, 768),
        "cls.seq_relationship.bias": (2,),
    }
    for layer in range(12):
        base = f"bert.encoder.layer.{layer}"
        for projection in ("query", "key", "value"):
            prefix = f"{base}.attention.self.{projection}"
            schema[f"{prefix}.weight"] = (768, 768)
            schema[f"{prefix}.bias"] = (768,)
        schema.update(
            {
                f"{base}.attention.output.dense.weight": (768, 768),
                f"{base}.attention.output.dense.bias": (768,),
                f"{base}.attention.output.LayerNorm.weight": (768,),
                f"{base}.attention.output.LayerNorm.bias": (768,),
                f"{base}.intermediate.dense.weight": (3072, 768),
                f"{base}.intermediate.dense.bias": (3072,),
                f"{base}.output.dense.weight": (768, 3072),
                f"{base}.output.dense.bias": (768,),
                f"{base}.output.LayerNorm.weight": (768,),
                f"{base}.output.LayerNorm.bias": (768,),
            }
        )
    return schema


def validate_model_state(state: dict[str, np.ndarray]) -> None:
    schema = model_schema()
    if set(state) != set(schema):
        missing = sorted(set(schema) - set(state))
        extra = sorted(set(state) - set(schema))
        raise ValueError(
            "model tensor schema mismatch: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    for name, shape in schema.items():
        value = np.asarray(state[name])
        if value.shape != shape:
            raise ValueError(
                f"model tensor {name!r} must have shape {shape}, got {value.shape}"
            )
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(
                f"model tensor {name!r} must be floating point, got {value.dtype}"
            )
        if not np.isfinite(value).all():
            raise FloatingPointError(f"model tensor {name!r} is non-finite")


def load_reference_assets(config: RunConfig) -> ReferenceAssets:
    dataset = load_from_disk(str(config.dataset_path))
    if config.split not in dataset:
        raise KeyError(
            f"dataset has no {config.split!r} split; available splits: {tuple(dataset)}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.tokenizer_path),
        model_max_length=SEQUENCE_LENGTH,
        local_files_only=True,
    )
    with safe_open(str(config.weights_path), framework="numpy") as handle:
        state = {key: np.asarray(handle.get_tensor(key)) for key in handle.keys()}
    validate_model_state(state)
    return ReferenceAssets(dataset[config.split], tokenizer, state)


def prepare_sample(
    sample_index: int,
    assets: ReferenceAssets,
) -> PreparedSample:
    if not 0 <= int(sample_index) < len(assets.split):
        raise IndexError(
            f"sample index {sample_index} is outside split of length {len(assets.split)}"
        )
    sample = assets.split[int(sample_index)]
    inputs = assets.tokenizer(
        sample["sentence1"],
        sample["sentence2"],
        max_length=SEQUENCE_LENGTH,
        padding="max_length",
        truncation=True,
    )
    input_ids = np.asarray(inputs["input_ids"], dtype=np.int64)
    token_type_ids = np.asarray(
        inputs.get("token_type_ids", np.zeros_like(input_ids)),
        dtype=np.int64,
    )
    attention_mask = np.asarray(inputs["attention_mask"], dtype=np.int64)
    input_embedding, reference_logits = reference_bert_forward(
        input_ids,
        token_type_ids,
        attention_mask,
        assets.state,
    )
    packed_attention_mask = pack_attention_mask(attention_mask)
    attention_mask_bundle = fhe.ConstantBundle(
        vectors={
            f"attention_mask.{index}": fhe.PackedRaw(
                etorch.as_tensor(packed_attention_mask[index])
            )
            for index in range(8)
        },
        cache_mode="middle",
    )
    reference_logits = np.asarray(reference_logits, dtype=np.float64)
    if reference_logits.shape != (2,):
        raise ValueError(
            f"reference classifier must return two logits, got "
            f"{reference_logits.shape}"
        )
    if not np.isfinite(reference_logits).all():
        raise FloatingPointError(
            f"sample {sample_index}: reference logits are non-finite: "
            f"{reference_logits}"
        )
    return PreparedSample(
        index=int(sample_index),
        label=int(sample["label"]),
        packed_input=pack_embedding(input_embedding.astype(np.float64)),
        attention_mask_bundle=attention_mask_bundle,
        reference_logits=reference_logits,
        reference_prediction=int(np.argmax(reference_logits)),
        attention_tokens=int(attention_mask.sum()),
        sequence_length=int(attention_mask.shape[0]),
    )


def _layernorm(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * weight + bias


def _gelu(x: np.ndarray) -> np.ndarray:
    z = x / math.sqrt(2.0)
    sign = np.sign(z)
    absolute = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * absolute)
    erf = sign * (
        1.0
        - (
            (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * np.exp(-(absolute * absolute))
        )
    )
    return 0.5 * x * (1.0 + erf)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _linear(
    x: np.ndarray,
    state: dict[str, np.ndarray],
    weight_key: str,
    bias_key: str,
) -> np.ndarray:
    return x @ state[weight_key].T + state[bias_key]


def reference_bert_forward(
    input_ids: np.ndarray,
    token_type_ids: np.ndarray,
    attention_mask: np.ndarray,
    state: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Run the exact plaintext graph used as the encrypted result reference."""

    position_ids = np.arange(input_ids.shape[0], dtype=np.int64)
    hidden = (
        state["bert.embeddings.word_embeddings.weight"][input_ids]
        + state["bert.embeddings.position_embeddings.weight"][position_ids]
        + state["bert.embeddings.token_type_embeddings.weight"][token_type_ids]
    ).astype(np.float64)
    hidden = _layernorm(
        hidden,
        state["bert.embeddings.LayerNorm.weight"],
        state["bert.embeddings.LayerNorm.bias"],
    )
    input_embedding = hidden

    sequence_length = input_ids.shape[0]
    additive_mask = np.where(attention_mask.astype(bool), 0.0, -10000.0).reshape(
        1, 1, sequence_length
    )
    for layer in range(12):
        query = _linear(
            hidden,
            state,
            f"bert.encoder.layer.{layer}.attention.self.query.weight",
            f"bert.encoder.layer.{layer}.attention.self.query.bias",
        ).reshape(sequence_length, 12, 64).transpose(1, 0, 2)
        key = _linear(
            hidden,
            state,
            f"bert.encoder.layer.{layer}.attention.self.key.weight",
            f"bert.encoder.layer.{layer}.attention.self.key.bias",
        ).reshape(sequence_length, 12, 64).transpose(1, 0, 2)
        value = _linear(
            hidden,
            state,
            f"bert.encoder.layer.{layer}.attention.self.value.weight",
            f"bert.encoder.layer.{layer}.attention.self.value.bias",
        ).reshape(sequence_length, 12, 64).transpose(1, 0, 2)
        scores = np.matmul(query, np.swapaxes(key, -1, -2)) / math.sqrt(64.0)
        probabilities = _softmax(scores + additive_mask, axis=-1)
        context = (
            np.matmul(probabilities, value)
            .transpose(1, 0, 2)
            .reshape(sequence_length, 768)
        )
        attention_dense = _linear(
            context,
            state,
            f"bert.encoder.layer.{layer}.attention.output.dense.weight",
            f"bert.encoder.layer.{layer}.attention.output.dense.bias",
        )
        attention_norm = _layernorm(
            attention_dense + hidden,
            state[f"bert.encoder.layer.{layer}.attention.output.LayerNorm.weight"],
            state[f"bert.encoder.layer.{layer}.attention.output.LayerNorm.bias"],
        )
        intermediate = _gelu(
            _linear(
                attention_norm,
                state,
                f"bert.encoder.layer.{layer}.intermediate.dense.weight",
                f"bert.encoder.layer.{layer}.intermediate.dense.bias",
            )
        )
        output_dense = _linear(
            intermediate,
            state,
            f"bert.encoder.layer.{layer}.output.dense.weight",
            f"bert.encoder.layer.{layer}.output.dense.bias",
        )
        hidden = _layernorm(
            output_dense + attention_norm,
            state[f"bert.encoder.layer.{layer}.output.LayerNorm.weight"],
            state[f"bert.encoder.layer.{layer}.output.LayerNorm.bias"],
        )

    pooled = np.tanh(
        _linear(
            hidden[0][None, :],
            state,
            "bert.pooler.dense.weight",
            "bert.pooler.dense.bias",
        )[0]
    )
    logits = (
        pooled @ state["cls.seq_relationship.weight"].T
        + state["cls.seq_relationship.bias"]
    )
    return input_embedding, logits
