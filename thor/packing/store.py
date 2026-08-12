from pathlib import Path
from dataclasses import dataclass

import numpy as np

import easyfhe as torch
from easyfhe import fhe

from ..fhe_ops import LazyPackedRaw, PackedRaw, SLOTS
from .weights import (
    build_att_vectors_from_raw,
    build_classifier_vectors_from_raw,
    build_ff_vectors_from_raw,
    build_pooler_vectors_from_raw,
)
from .masks import ThorMaskPacker


@dataclass(frozen=True)
class PackedWeightBundle:
    """Application-owned constants with public access to lazy packed sources."""

    raw_vectors: dict[str, PackedRaw]
    constants: fhe.ConstantBundle

    @classmethod
    def build(cls, vectors: dict, *, cache_mode: str = "none"):
        raw_vectors = wrap_packed_vectors(vectors)
        return cls(
            raw_vectors=raw_vectors,
            constants=fhe.ConstantBundle(vectors=raw_vectors, cache_mode=cache_mode),
        )

    def plaintext(self, *args, **kwargs):
        return self.constants.plaintext(*args, **kwargs)


@dataclass(frozen=True)
class LayerWeights:
    layer: int
    bundle: PackedWeightBundle


class WeightStore:
    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str = "cuda",
    ):
        self.model_path = self.require_file(Path(model_path), "model safetensors")
        self.device = device
        self._masks = None
        self._layers = {}
        self._pooler = None
        self._classifier = None

    @staticmethod
    def require_file(path: Path, label: str) -> Path:
        if path.exists():
            return path
        raise FileNotFoundError(f"could not find {label}: {path}")

    def masks(self):
        if self._masks is None:
            mask_vectors, _ = ThorMaskPacker(num_slots=SLOTS).pack()
            self._masks = fhe.ConstantBundle(
                vectors=wrap_packed_vectors(mask_vectors),
                cache_mode="middle",
            )
        return self._masks

    def layer(self, layer: int, *, attention_key_scale: float) -> LayerWeights:
        layer = int(layer)
        if layer not in self._layers:
            att_vectors = build_att_vectors_from_raw(
                self.model_path,
                layer,
                attention_key_scale=attention_key_scale,
                device=self.device,
            )
            ff_vectors = build_ff_vectors_from_raw(
                self.model_path,
                layer,
                device=self.device,
            )
            overlap = set(att_vectors).intersection(ff_vectors)
            if overlap:
                raise ValueError(
                    f"layer {layer} attention/FF bundle keys overlap: "
                    f"{sorted(overlap)[:5]}"
                )
            bundle = PackedWeightBundle.build(
                {**att_vectors, **ff_vectors},
                cache_mode="none",
            )
            self._layers[layer] = LayerWeights(
                layer=layer,
                bundle=bundle,
            )
        return self._layers[layer]

    def release_layer(self, layer: int):
        layer = int(layer)
        self._layers.pop(layer, None)

    def pooler(self):
        if self._pooler is None:
            vectors = build_pooler_vectors_from_raw(self.model_path)
            self._pooler = PackedWeightBundle.build(vectors, cache_mode="none")
        return self._pooler

    def classifier(self):
        if self._classifier is None:
            vectors = build_classifier_vectors_from_raw(self.model_path)
            self._classifier = PackedWeightBundle.build(vectors, cache_mode="none")
        return self._classifier


def wrap_packed_vectors(vectors: dict) -> dict:
    wrapped = {}
    for name, vector in vectors.items():
        if isinstance(vector, PackedRaw):
            wrapped[name] = vector
        elif torch.is_tensor(vector):
            wrapped[name] = PackedRaw(pad_tensor_slots(vector))
        else:
            wrapped[name] = PackedRaw(
                torch.as_tensor(pad_array_slots(np.asarray(vector)))
            )
    return wrapped


def pad_array_slots(array: np.ndarray, slots: int = SLOTS) -> np.ndarray:
    if array.shape[-1] == slots:
        return array
    if array.shape[-1] > slots:
        raise ValueError(
            f"constant vector length {array.shape[-1]} exceeds slots {slots}"
        )
    padded = np.zeros((*array.shape[:-1], slots), dtype=array.dtype)
    padded[..., : array.shape[-1]] = array
    return padded


def pad_tensor_slots(tensor, slots: int = SLOTS):
    if int(tensor.shape[-1]) == slots:
        return tensor
    if int(tensor.shape[-1]) > slots:
        raise ValueError(
            f"constant tensor length {int(tensor.shape[-1])} exceeds slots {slots}"
        )
    padded = torch.zeros(
        (*tuple(tensor.shape[:-1]), slots),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[..., : int(tensor.shape[-1])] = tensor
    return padded
