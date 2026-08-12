from pathlib import Path

import numpy as np
from safetensors import safe_open

import easyfhe as torch

from ..config import SLOTS
from ..fhe_ops import LazyPackedRaw
from .triton import (
    pack_w_att_triton_indices_gpu,
    pack_w_ff_triton_indices_gpu,
)
from .masks import format_packed_name, to_blocks


def _require_model_file(model_path: Path) -> Path:
    model_path = Path(model_path)
    if model_path.exists():
        return model_path
    raise FileNotFoundError(f"could not find raw model safetensors: {model_path}")


def load_raw_tensor(model_path: Path, key: str, *, device: str):
    with safe_open(str(_require_model_file(model_path)), framework="numpy") as handle:
        value = np.asarray(handle.get_tensor(key))
    return torch.as_tensor(value, device=device)


def load_raw_array(model_path: Path, key: str) -> np.ndarray:
    with safe_open(str(_require_model_file(model_path)), framework="numpy") as handle:
        return np.asarray(handle.get_tensor(key))


def vectors_from_packed_array(key: str, packed: np.ndarray) -> dict:
    vectors = {}
    for index in np.ndindex(packed.shape):
        value = packed[index]
        if value is not None:
            vectors[format_packed_name(key, index)] = np.asarray(value)
    return vectors


def pack_b_array(
    b: np.ndarray,
    *,
    n_blocks: int,
    n_out: int,
    pack: int = 16,
    n_slot: int = 16,
    pad_index=None,
    scale: float = 1.0,
) -> np.ndarray:
    dim = 128
    if pad_index is None:
        pad_index = [i for i in range(n_slot) if i >= n_blocks]
    elif n_blocks + len(pad_index) != n_slot:
        raise ValueError("Parameters do not match")
    if b.shape[0] % n_blocks != 0:
        raise ValueError("Block size does not match")

    blocks = np.split(np.asarray(b), n_blocks)
    packed = np.full((n_out // pack,), None, dtype=object)
    for out in range(n_out // pack):
        msg = np.zeros((SLOTS,), dtype=np.float64)
        for j in range(pack):
            temp = j * (2**11)
            r = out * pack + j
            c = 0
            for d in range(n_slot):
                if d in pad_index:
                    continue
                block = blocks[c]
                msg[temp + np.arange(dim) * n_slot + d] = (
                    scale * block[(r + np.arange(dim)) % block.shape[0]]
                ) / 2
                c += 1
        packed[out] = msg
    return packed


def pack_b_pooler_array(
    b: np.ndarray,
    *,
    n_blocks: int = 6,
    n_slot: int = 16,
    pad_index=None,
) -> np.ndarray:
    dim = 128
    if pad_index is None:
        pad_index = [i for i in range(n_slot) if i >= n_blocks]
    elif n_blocks + len(pad_index) != n_slot:
        raise ValueError("Parameters do not match")
    if b.shape[0] % n_blocks != 0:
        raise ValueError("Block size does not match")

    blocks = np.split(np.asarray(b), n_blocks)
    packed = np.full((1,), None, dtype=object)
    msg = np.zeros((2**11,), dtype=np.float64)
    for t in range(dim):
        c = 0
        for d in range(n_slot):
            if d in pad_index:
                continue
            msg[t * n_slot + d] = blocks[c][t] / 2
            c += 1
    packed[0] = np.tile(msg, 16)
    return packed


def pack_w_pooler_array(w: np.ndarray, b_shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    pack = 16
    if w.shape[0] % b_shape[0] != 0 or w.shape[1] % b_shape[1] != 0:
        raise ValueError("Dimension does not match")

    ld_blocks, (ll, _) = to_blocks(w, b_shape, diag=True)
    packed = np.full((ll, 4), None, dtype=object)
    for diag in range(ll):
        blocks = ld_blocks[diag]
        blocks_arr = np.stack([blocks[d] for d in range(6)])
        m_idx = np.arange(2**7)
        d_idx = np.arange(6)
        for n in range(4):
            msg = np.zeros((SLOTS,), dtype=np.complex128)
            for j in range(pack):
                i = n * 16 + j
                temp = j * (2**11)
                values = blocks_arr[:, m_idx, i] / 2 - 1j * (
                    blocks_arr[:, m_idx, i + 64] / 2
                )
                msg[temp + m_idx[:, None] * 16 + d_idx] = values.T
            packed[diag, n] = msg
    return packed


def pack_w_cls_array(w: np.ndarray) -> np.ndarray:
    if w.shape[1] != 768:
        raise ValueError(f"Shape of W_cls should be (cls, 768). Shape is {w.shape}")
    packed = np.full((w.shape[0],), None, dtype=object)
    for n in range(w.shape[0]):
        blocks = np.split(w[n], 6)
        msg = np.zeros((SLOTS,), dtype=np.float64)
        for t in range(128):
            for d in range(6):
                msg[t * 16 + d] = blocks[d][t]
        packed[n] = msg
    return packed


def pack_b_cls_array(b: np.ndarray) -> np.ndarray:
    packed = np.full((b.shape[0],), None, dtype=object)
    for n in range(b.shape[0]):
        msg = np.zeros((SLOTS,), dtype=np.float64)
        msg[0] = b[n]
        packed[n] = msg
    return packed


def attach_batch_metadata(
    raw: LazyPackedRaw,
    *,
    source_id: str,
    index: tuple[int, ...],
    batch_packer,
):
    raw._thor_source_id = source_id
    raw._thor_index = tuple(int(x) for x in index)
    raw._thor_batch_packer = batch_packer
    return raw


def make_lazy_packed_raw(
    raw_tensor,
    *,
    source_id: str,
    index: tuple[int, ...],
    batch_packer,
):
    def packer(tensor, slots, context, index=index):
        packed = batch_packer(tensor, np.asarray([index], dtype=np.int64), slots, context)
        return packed[0]

    return attach_batch_metadata(
        LazyPackedRaw(raw_tensor, packer),
        source_id=source_id,
        index=index,
        batch_packer=batch_packer,
    )


def make_att_weight_entries(
    key: str,
    raw_tensor,
    *,
    n_in: int,
    n_out: int,
    b_shape: tuple[int, int],
    scale: float = 1.0,
    device: str = "cuda",
) -> dict:
    weight_shape = tuple(int(x) for x in raw_tensor.shape)
    ll = min(weight_shape[0] // b_shape[0], weight_shape[1] // b_shape[1])
    n_in_c = n_in // 2

    def batch_packer(tensor, indices, slots, context):
        if int(slots) != SLOTS:
            raise ValueError(f"THOR packed weights require {SLOTS} slots, got {slots}")
        return pack_w_att_triton_indices_gpu(
            tensor,
            indices,
            n_in=n_in,
            n_out=n_out,
            b_shape=b_shape,
            scale=scale,
            device=device,
        )

    vectors = {}
    for out in range(n_out // 16):
        for diag in range(ll):
            for n in range(n_in_c):
                index = (out, diag, n)
                vectors[f"{key}__{out}x{diag}x{n}"] = make_lazy_packed_raw(
                    raw_tensor,
                    source_id=key,
                    index=index,
                    batch_packer=batch_packer,
                )
    return vectors


def make_ff_weight_entries(
    key: str,
    raw_tensor,
    *,
    vsplit: int = 0,
    hsplit: int = 0,
    scale: float = 1.0,
    device: str = "cuda",
) -> dict:
    if bool(vsplit) == bool(hsplit):
        raise ValueError("provide exactly one of vsplit or hsplit")
    weight_shape = tuple(int(x) for x in raw_tensor.shape)
    split_shape = (
        (weight_shape[0] // vsplit, weight_shape[1])
        if vsplit
        else (weight_shape[0], weight_shape[1] // hsplit)
    )
    ll = min(split_shape[0] // 128, split_shape[1] // 128)

    def batch_packer(tensor, indices, slots, context):
        if int(slots) != SLOTS:
            raise ValueError(f"THOR packed weights require {SLOTS} slots, got {slots}")
        return pack_w_ff_triton_indices_gpu(
            tensor,
            indices,
            vsplit=vsplit,
            hsplit=hsplit,
            scale=scale,
            device=device,
        )

    vectors = {}
    for rep in range(2):
        for out in range(8):
            for diag in range(ll):
                for n in range(64):
                    index = (rep, out, diag, n)
                    vectors[f"{key}__{rep}x{out}x{diag}x{n}"] = make_lazy_packed_raw(
                        raw_tensor,
                        source_id=key,
                        index=index,
                        batch_packer=batch_packer,
                    )
    return vectors


def build_att_vectors_from_raw(
    model_path: Path,
    layer: int,
    prefixes=(),
    *,
    attention_key_scale: float,
    device: str = "cuda",
) -> dict:
    if not prefixes:
        prefixes = []
        for qkv in ("query", "key", "value"):
            base = f"bert.encoder.layer.{layer}.attention.self.{qkv}"
            prefixes.extend((f"{base}.weight", f"{base}.bias"))
        prefixes.extend(
            (
                f"bert.encoder.layer.{layer}.attention.output.dense.weight",
                f"bert.encoder.layer.{layer}.attention.output.dense.bias",
                f"bert.encoder.layer.{layer}.attention.output.LayerNorm.weight",
                f"bert.encoder.layer.{layer}.attention.output.LayerNorm.bias",
            )
        )
    prefixes = tuple(prefixes)
    vectors = {}
    small_prefixes = []
    for prefix in prefixes:
        if prefix.endswith(".weight") and ".attention.self." in prefix:
            qkv = prefix.split(".")[-2]
            scale = attention_key_scale if qkv == "key" else 1.0
            raw = load_raw_tensor(model_path, prefix, device=device)
            vectors.update(
                make_att_weight_entries(
                    prefix,
                    raw,
                    n_in=128,
                    n_out=64,
                    b_shape=(64, 128),
                    scale=scale,
                    device=device,
                )
            )
        elif prefix == f"bert.encoder.layer.{layer}.attention.output.dense.weight":
            raw = load_raw_tensor(model_path, prefix, device=device)
            vectors.update(
                make_att_weight_entries(
                    prefix,
                    raw,
                    n_in=64,
                    n_out=128,
                    b_shape=(128, 64),
                    device=device,
                )
            )
        else:
            small_prefixes.append(prefix)
    if small_prefixes:
        vectors.update(
            build_att_small_vectors_from_raw(
                model_path,
                layer,
                tuple(small_prefixes),
                attention_key_scale=attention_key_scale,
            )
        )
    return vectors


def build_ff_vectors_from_raw(
    model_path: Path,
    layer: int,
    *,
    device: str = "cuda",
) -> dict:
    vectors = {}
    ff1 = f"bert.encoder.layer.{layer}.intermediate.dense.weight"
    ff2 = f"bert.encoder.layer.{layer}.output.dense.weight"
    raw = load_raw_tensor(model_path, ff1, device=device)
    vectors.update(
        make_ff_weight_entries(ff1, raw, vsplit=4, scale=1 / 64, device=device)
    )
    raw = load_raw_tensor(model_path, ff2, device=device)
    vectors.update(make_ff_weight_entries(ff2, raw, hsplit=4, device=device))
    small_prefixes = (
        f"bert.encoder.layer.{layer}.intermediate.dense.bias",
        f"bert.encoder.layer.{layer}.output.dense.bias",
        f"bert.encoder.layer.{layer}.output.LayerNorm.weight",
        f"bert.encoder.layer.{layer}.output.LayerNorm.bias",
    )
    vectors.update(build_ff_small_vectors_from_raw(model_path, layer, small_prefixes))
    return vectors


def build_att_small_vectors_from_raw(
    model_path: Path,
    layer: int,
    prefixes: tuple[str, ...],
    *,
    attention_key_scale: float,
) -> dict:
    vectors = {}
    for prefix in prefixes:
        if prefix.endswith(".bias") and ".attention.self." in prefix:
            qkv = prefix.split(".")[-2]
            scale = attention_key_scale if qkv == "key" else 1.0
            packed = pack_b_array(
                load_raw_array(model_path, prefix),
                n_blocks=12,
                n_out=64,
                scale=scale,
            )
        elif prefix == f"bert.encoder.layer.{layer}.attention.output.dense.bias":
            packed = pack_b_array(
                load_raw_array(model_path, prefix),
                n_blocks=6,
                n_out=128,
            )
        elif prefix in (
            f"bert.encoder.layer.{layer}.attention.output.LayerNorm.weight",
            f"bert.encoder.layer.{layer}.attention.output.LayerNorm.bias",
        ):
            packed = pack_b_array(
                load_raw_array(model_path, prefix),
                n_blocks=6,
                n_out=128,
            )
        else:
            raise KeyError(f"unsupported attention raw small vector prefix: {prefix}")
        vectors.update(vectors_from_packed_array(prefix, packed))
    return vectors


def build_ff_small_vectors_from_raw(
    model_path: Path,
    layer: int,
    prefixes: tuple[str, ...],
) -> dict:
    vectors = {}
    for prefix in prefixes:
        if prefix == f"bert.encoder.layer.{layer}.intermediate.dense.bias":
            bias_halves = np.split(load_raw_array(model_path, prefix), 2)
            packed = np.full((2, 8), None, dtype=object)
            for index, bias in enumerate(bias_halves):
                packed[index] = pack_b_array(
                    bias,
                    n_blocks=12,
                    n_out=128,
                    pad_index=(6, 7, 14, 15),
                    scale=1 / 64,
                )
        elif prefix == f"bert.encoder.layer.{layer}.output.dense.bias":
            packed = pack_b_array(
                load_raw_array(model_path, prefix),
                n_blocks=6,
                n_out=128,
                n_slot=16,
            )
        elif prefix in (
            f"bert.encoder.layer.{layer}.output.LayerNorm.weight",
            f"bert.encoder.layer.{layer}.output.LayerNorm.bias",
        ):
            packed = pack_b_array(
                load_raw_array(model_path, prefix),
                n_blocks=6,
                n_out=128,
                n_slot=16,
            )
        else:
            raise KeyError(f"unsupported FF raw small vector prefix: {prefix}")
        vectors.update(vectors_from_packed_array(prefix, packed))
    return vectors


def build_pooler_vectors_from_raw(model_path: Path) -> dict:
    vectors = {}
    weight_key = "bert.pooler.dense.weight"
    bias_key = "bert.pooler.dense.bias"
    vectors.update(
        vectors_from_packed_array(
            weight_key,
            pack_w_pooler_array(load_raw_array(model_path, weight_key)),
        )
    )
    vectors.update(
        vectors_from_packed_array(
            bias_key,
            pack_b_pooler_array(load_raw_array(model_path, bias_key), n_blocks=6),
        )
    )
    return vectors


def build_classifier_vectors_from_raw(model_path: Path) -> dict:
    cls_name = "cls.seq_relationship"
    vectors = {}
    weight_key = f"{cls_name}.weight"
    bias_key = f"{cls_name}.bias"
    vectors.update(
        vectors_from_packed_array(
            weight_key,
            pack_w_cls_array(load_raw_array(model_path, weight_key)),
        )
    )
    vectors.update(
        vectors_from_packed_array(
            bias_key,
            pack_b_cls_array(load_raw_array(model_path, bias_key)),
        )
    )
    return vectors
