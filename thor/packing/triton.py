from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import sys
import types

import numpy as np

import easyfhe as torch

if not hasattr(torch, "version"):
    torch.version = types.SimpleNamespace(hip=None)
elif not hasattr(torch.version, "hip"):
    torch.version.hip = None
sys.modules.setdefault("torch", torch)
import triton
import triton.language as tl

from ..config import SLOTS

PACK = 16
DIM = 128


@dataclass(frozen=True)
class PackedChunk:
    indices: np.ndarray
    values: object


def _enable_triton_easyfhe_bridge() -> None:
    """Let Triton use EasyFHE's torch-compatible CUDA runtime."""
    import triton.backends.driver as triton_driver

    if getattr(triton_driver.GPUDriver, "_easyfhe_bridge", False):
        return

    def init(self):
        self.get_device_capability = torch.cuda.get_device_capability
        self.get_current_stream = lambda idx: torch.cuda.current_stream(idx).cuda_stream
        self.get_current_device = torch.cuda.current_device
        self.set_current_device = torch.cuda.set_device

    triton_driver.GPUDriver.__init__ = init
    triton_driver.GPUDriver._easyfhe_bridge = True


def _as_weight_tensor(weight, *, device: str = "cuda", dtype=None):
    if dtype is None:
        dtype = torch.float64
    if hasattr(weight, "to") and hasattr(weight, "device"):
        return weight.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(weight), dtype=dtype, device=device)


def _as_index_tensor(values: np.ndarray, *, device: str):
    return torch.as_tensor(np.array(values, dtype=np.int64, copy=True), dtype=torch.int64, device=device)


def _normalize_indices(indices, rank: int) -> np.ndarray:
    if indices is None:
        raise ValueError("indices must be provided")
    arr = np.asarray(indices, dtype=np.int64)
    if arr.ndim == 1:
        if arr.size != rank:
            raise ValueError(f"expected index rank {rank}, got {arr.size}")
        arr = arr.reshape(1, rank)
    if arr.ndim != 2 or arr.shape[1] != rank:
        raise ValueError(f"expected indices with shape [N, {rank}], got {arr.shape}")
    return np.ascontiguousarray(arr)


def _chunk_indices(indices: np.ndarray, chunk_size: int) -> Iterator[np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(indices), chunk_size):
        yield indices[start : start + chunk_size]


def all_w_att_indices(n_in: int, n_out: int, weight_shape: tuple[int, int], b_shape: tuple[int, int]) -> np.ndarray:
    _validate_block_shape(weight_shape, b_shape, n_out)
    ll = min(weight_shape[0] // b_shape[0], weight_shape[1] // b_shape[1])
    n_in_c = n_in // 2
    return np.asarray(
        np.meshgrid(
            np.arange(n_out // PACK, dtype=np.int64),
            np.arange(ll, dtype=np.int64),
            np.arange(n_in_c, dtype=np.int64),
            indexing="ij",
        )
    ).reshape(3, -1).T.copy()


def all_w_ff_indices(n_in: int, n_out: int, weight_shape: tuple[int, int], b_shape: tuple[int, int]) -> np.ndarray:
    _validate_block_shape(weight_shape, b_shape, n_out)
    ll = min(weight_shape[0] // b_shape[0], weight_shape[1] // b_shape[1])
    n_in_c = n_in // 2
    return np.asarray(
        np.meshgrid(
            np.arange(2, dtype=np.int64),
            np.arange(n_out // PACK, dtype=np.int64),
            np.arange(ll, dtype=np.int64),
            np.arange(n_in_c, dtype=np.int64),
            indexing="ij",
        )
    ).reshape(4, -1).T.copy()


def iter_pack_w_att_gpu(
    weight,
    *,
    n_in: int,
    n_out: int,
    b_shape: tuple[int, int],
    scale: float = 1.0,
    indices=None,
    device: str = "cuda",
    chunk_size: int = 128,
    weight_dtype=None,
    out_dtype=None,
) -> Iterator[PackedChunk]:
    w = _as_weight_tensor(weight, device=device, dtype=weight_dtype)
    weight_shape = tuple(int(x) for x in w.shape)
    if indices is None:
        indices = all_w_att_indices(n_in, n_out, weight_shape, b_shape)
    else:
        indices = _normalize_indices(indices, 3)
    for chunk in _chunk_indices(indices, chunk_size):
        yield PackedChunk(chunk, _pack_w_att_chunk_gpu(w, chunk, n_in=n_in, n_out=n_out, b_shape=b_shape, scale=scale, device=device, out_dtype=out_dtype))


def pack_w_att_triton_indices_gpu(
    weight,
    indices,
    *,
    n_in: int,
    n_out: int,
    b_shape: tuple[int, int],
    scale: float = 1.0,
    device: str = "cuda",
    weight_dtype=None,
    out_dtype=None,
    block_size: int = 256,
):
    w = _as_weight_tensor(weight, device=device, dtype=weight_dtype)
    weight_shape = tuple(int(x) for x in w.shape)
    indices = _normalize_indices(indices, 3)
    ll = min(weight_shape[0] // b_shape[0], weight_shape[1] // b_shape[1])
    n_in_c = n_in // 2
    linear = ((indices[:, 0] * ll + indices[:, 1]) * n_in_c + indices[:, 2]).astype(np.int64)
    if len(linear) and np.array_equal(linear, np.arange(linear[0], linear[0] + len(linear), dtype=np.int64)):
        dd = max(weight_shape[0] // b_shape[0], weight_shape[1] // b_shape[1])
        return _pack_w_att_triton_chunk_gpu(
            w,
            int(linear[0]),
            int(len(linear)),
            weight_shape=weight_shape,
            n_in=n_in,
            n_out=n_out,
            b_shape=b_shape,
            ll=ll,
            dd=dd,
            scale=scale,
            device=device,
            out_dtype=out_dtype,
            block_size=block_size,
        )
    return pack_w_att_gpu(
        w,
        n_in=n_in,
        n_out=n_out,
        b_shape=b_shape,
        scale=scale,
        indices=indices,
        device=device,
        chunk_size=max(1, len(indices)),
        weight_dtype=weight_dtype,
        out_dtype=out_dtype,
    )


def pack_w_att_gpu(
    weight,
    *,
    n_in: int,
    n_out: int,
    b_shape: tuple[int, int],
    scale: float = 1.0,
    indices=None,
    device: str = "cuda",
    chunk_size: int = 128,
    weight_dtype=None,
    out_dtype=None,
):
    chunks = [
        chunk.values
        for chunk in iter_pack_w_att_gpu(
            weight,
            n_in=n_in,
            n_out=n_out,
            b_shape=b_shape,
            scale=scale,
            indices=indices,
            device=device,
            chunk_size=chunk_size,
            weight_dtype=weight_dtype,
            out_dtype=out_dtype,
        )
    ]
    return _cat_chunks(chunks)


def iter_pack_w_ff_gpu(
    weight,
    *,
    n_in: int = 128,
    n_out: int = 128,
    b_shape: tuple[int, int] = (128, 128),
    vsplit: int = 0,
    hsplit: int = 0,
    scale: float = 1.0,
    indices=None,
    device: str = "cuda",
    chunk_size: int = 128,
    weight_dtype=None,
    out_dtype=None,
) -> Iterator[PackedChunk]:
    if bool(vsplit) == bool(hsplit):
        raise ValueError("provide exactly one of vsplit or hsplit")
    w = _as_weight_tensor(weight, device=device, dtype=weight_dtype)
    weight_shape = tuple(int(x) for x in w.shape)
    if indices is None:
        indices = all_w_ff_indices(n_in, n_out, weight_shape, b_shape)
    else:
        indices = _normalize_indices(indices, 4)
    for chunk in _chunk_indices(indices, chunk_size):
        yield PackedChunk(
            chunk,
            _pack_w_ff_chunk_gpu(
                w,
                chunk,
                n_in=n_in,
                n_out=n_out,
                b_shape=b_shape,
                vsplit=vsplit,
                hsplit=hsplit,
                scale=scale,
                device=device,
                out_dtype=out_dtype,
            ),
        )


def pack_w_ff_triton_indices_gpu(
    weight,
    indices,
    *,
    n_in: int = 128,
    n_out: int = 128,
    b_shape: tuple[int, int] = (128, 128),
    vsplit: int = 0,
    hsplit: int = 0,
    scale: float = 1.0,
    device: str = "cuda",
    weight_dtype=None,
    out_dtype=None,
    block_size: int = 256,
):
    if bool(vsplit) == bool(hsplit):
        raise ValueError("provide exactly one of vsplit or hsplit")
    w = _as_weight_tensor(weight, device=device, dtype=weight_dtype)
    weight_shape = tuple(int(x) for x in w.shape)
    indices = _normalize_indices(indices, 4)
    split_shape = (weight_shape[0] // vsplit, weight_shape[1]) if vsplit else (weight_shape[0], weight_shape[1] // hsplit)
    split_mode = 1 if vsplit else 2
    ll = min(split_shape[0] // b_shape[0], split_shape[1] // b_shape[1])
    n_in_c = n_in // 2
    n_out_p = n_out // PACK
    linear = (((indices[:, 0] * n_out_p + indices[:, 1]) * ll + indices[:, 2]) * n_in_c + indices[:, 3]).astype(np.int64)
    if len(linear) and np.array_equal(linear, np.arange(linear[0], linear[0] + len(linear), dtype=np.int64)):
        dd = max(split_shape[0] // b_shape[0], split_shape[1] // b_shape[1])
        return _pack_w_ff_triton_chunk_gpu(
            w,
            int(linear[0]),
            int(len(linear)),
            weight_shape=weight_shape,
            split_shape=split_shape,
            split_mode=split_mode,
            n_in=n_in,
            n_out=n_out,
            b_shape=b_shape,
            ll=ll,
            dd=dd,
            scale=scale,
            device=device,
            out_dtype=out_dtype,
            block_size=block_size,
        )
    return pack_w_ff_gpu(
        w,
        n_in=n_in,
        n_out=n_out,
        b_shape=b_shape,
        vsplit=vsplit,
        hsplit=hsplit,
        scale=scale,
        indices=indices,
        device=device,
        chunk_size=max(1, len(indices)),
        weight_dtype=weight_dtype,
        out_dtype=out_dtype,
    )


def pack_w_ff_gpu(
    weight,
    *,
    n_in: int = 128,
    n_out: int = 128,
    b_shape: tuple[int, int] = (128, 128),
    vsplit: int = 0,
    hsplit: int = 0,
    scale: float = 1.0,
    indices=None,
    device: str = "cuda",
    chunk_size: int = 128,
    weight_dtype=None,
    out_dtype=None,
):
    chunks = [
        chunk.values
        for chunk in iter_pack_w_ff_gpu(
            weight,
            n_in=n_in,
            n_out=n_out,
            b_shape=b_shape,
            vsplit=vsplit,
            hsplit=hsplit,
            scale=scale,
            indices=indices,
            device=device,
            chunk_size=chunk_size,
            weight_dtype=weight_dtype,
            out_dtype=out_dtype,
        )
    ]
    return _cat_chunks(chunks)


def _cat_chunks(chunks: list):
    if not chunks:
        return torch.zeros((0, SLOTS), dtype=torch.complex128, device="cuda")
    if len(chunks) == 1:
        return chunks[0]
    if hasattr(torch, "cat"):
        return torch.cat(chunks, dim=0)
    arrays = [chunk.cpu().numpy() for chunk in chunks]
    return torch.as_tensor(np.concatenate(arrays, axis=0), device=chunks[0].device)


def _validate_block_shape(weight_shape: tuple[int, int], b_shape: tuple[int, int], n_out: int) -> None:
    if weight_shape[0] % b_shape[0] != 0 or weight_shape[1] % b_shape[1] != 0:
        raise ValueError(f"weight shape {weight_shape} is not divisible by block shape {b_shape}")
    if n_out % PACK != 0:
        raise ValueError("n_out must be divisible by PACK")


def _diagonal_block_geometry(weight_shape: tuple[int, int], b_shape: tuple[int, int], l_values: np.ndarray):
    v = weight_shape[0] // b_shape[0]
    h = weight_shape[1] // b_shape[1]
    dd = max(v, h)
    d_idx = np.arange(dd, dtype=np.int64)
    block_rows = (l_values[:, None] + d_idx[None, :]) % v
    block_cols = np.broadcast_to(d_idx[None, :] % h, block_rows.shape)
    return block_rows, block_cols, d_idx


def _pack_common_indices(index_chunk: np.ndarray, *, n_in: int, n_out: int, b_shape: tuple[int, int], weight_shape: tuple[int, int]):
    _validate_block_shape(weight_shape, b_shape, n_out)
    n_in_c = n_in // 2
    out_values = index_chunk[:, 0]
    l_values = index_chunk[:, 1]
    n_values = index_chunk[:, 2]
    block_rows, block_cols, d_idx = _diagonal_block_geometry(weight_shape, b_shape, l_values)

    j_idx = np.arange(PACK, dtype=np.int64)
    t_idx = np.arange(DIM, dtype=np.int64)

    base = ((n_values[:, None] // PACK) * PACK + out_values[:, None] * PACK + (n_values[:, None] + j_idx[None, :]) % PACK) % n_in_c
    r = out_values[:, None] * PACK + j_idx[None, :]
    i = (base - r) % n_in
    local_rows = (t_idx[None, None, :, None] + r[:, :, None, None]) % b_shape[0]
    real_local_cols = (i[:, :, None, None] + t_idx[None, None, :, None] + r[:, :, None, None]) % b_shape[1]
    imag_local_cols = ((i[:, :, None, None] + n_in_c) % n_in + t_idx[None, None, :, None] + r[:, :, None, None]) % b_shape[1]

    row_idx = block_rows[:, None, None, :] * b_shape[0] + local_rows
    real_col_idx = block_cols[:, None, None, :] * b_shape[1] + real_local_cols
    imag_col_idx = block_cols[:, None, None, :] * b_shape[1] + imag_local_cols
    positions = j_idx[None, :, None, None] * (SLOTS // PACK) + t_idx[None, None, :, None] * PACK + d_idx[None, None, None, :]

    return (
        np.ascontiguousarray(row_idx.reshape(len(index_chunk), -1)),
        np.ascontiguousarray(real_col_idx.reshape(len(index_chunk), -1)),
        np.ascontiguousarray(imag_col_idx.reshape(len(index_chunk), -1)),
        np.ascontiguousarray(np.broadcast_to(positions, row_idx.shape).reshape(len(index_chunk), -1)),
    )


def _pack_w_att_chunk_gpu(w, index_chunk: np.ndarray, *, n_in: int, n_out: int, b_shape: tuple[int, int], scale: float, device: str, out_dtype=None):
    if out_dtype is None:
        out_dtype = torch.complex128
    row_idx, real_col_idx, imag_col_idx, positions = _pack_common_indices(
        index_chunk,
        n_in=n_in,
        n_out=n_out,
        b_shape=b_shape,
        weight_shape=tuple(int(x) for x in w.shape),
    )
    packed = torch.zeros((len(index_chunk), SLOTS), dtype=out_dtype, device=device)
    real = w[_as_index_tensor(row_idx, device=device), _as_index_tensor(real_col_idx, device=device)]
    imag = w[_as_index_tensor(row_idx, device=device), _as_index_tensor(imag_col_idx, device=device)]
    values = (real * (scale / 2)).to(out_dtype) - 1j * (imag * (scale / 2)).to(out_dtype)
    packed[_row_positions(len(index_chunk), positions.shape[1], device=device), _as_index_tensor(positions, device=device)] = values
    return packed


def _split_offsets(weight_shape: tuple[int, int], *, b_shape: tuple[int, int], vsplit: int, hsplit: int, rep_values: np.ndarray):
    if vsplit:
        split_rows = weight_shape[0] // vsplit
        split_cols = weight_shape[1]
        row_offsets = (rep_values * 2)[:, None] * split_rows
        col_offsets = np.zeros((len(rep_values), 1), dtype=np.int64)
    else:
        split_rows = weight_shape[0]
        split_cols = weight_shape[1] // hsplit
        row_offsets = np.zeros((len(rep_values), 1), dtype=np.int64)
        col_offsets = (rep_values * 2)[:, None] * split_cols
    if split_rows % b_shape[0] != 0 or split_cols % b_shape[1] != 0:
        raise ValueError("split weight shape is not divisible by block shape")
    return (split_rows, split_cols), row_offsets, col_offsets


def _pack_w_ff_chunk_gpu(
    w,
    index_chunk: np.ndarray,
    *,
    n_in: int,
    n_out: int,
    b_shape: tuple[int, int],
    vsplit: int,
    hsplit: int,
    scale: float,
    device: str,
    out_dtype=None,
):
    if out_dtype is None:
        out_dtype = torch.complex128
    rep_values = index_chunk[:, 0]
    att_like_indices = index_chunk[:, 1:4]
    weight_shape = tuple(int(x) for x in w.shape)
    split_shape, row_offsets, col_offsets = _split_offsets(weight_shape, b_shape=b_shape, vsplit=vsplit, hsplit=hsplit, rep_values=rep_values)
    row_idx, real_col_idx, imag_col_idx, positions1 = _pack_common_indices(
        att_like_indices,
        n_in=n_in,
        n_out=n_out,
        b_shape=b_shape,
        weight_shape=split_shape,
    )
    positions2 = positions1 + 8

    row_idx1 = row_idx + row_offsets
    real_col_idx1 = real_col_idx + col_offsets
    imag_col_idx1 = imag_col_idx + col_offsets
    if vsplit:
        row_idx2 = row_idx1 + split_shape[0]
        real_col_idx2 = real_col_idx1
        imag_col_idx2 = imag_col_idx1
    else:
        row_idx2 = row_idx1
        real_col_idx2 = real_col_idx1 + split_shape[1]
        imag_col_idx2 = imag_col_idx1 + split_shape[1]

    packed = torch.zeros((len(index_chunk), SLOTS), dtype=out_dtype, device=device)
    row1 = _as_index_tensor(row_idx1, device=device)
    real_col1 = _as_index_tensor(real_col_idx1, device=device)
    imag_col1 = _as_index_tensor(imag_col_idx1, device=device)
    row2 = _as_index_tensor(row_idx2, device=device)
    real_col2 = _as_index_tensor(real_col_idx2, device=device)
    imag_col2 = _as_index_tensor(imag_col_idx2, device=device)

    values1 = (w[row1, real_col1] * (scale / 2)).to(out_dtype) - 1j * (w[row1, imag_col1] * (scale / 2)).to(out_dtype)
    values2 = (w[row2, real_col2] * (scale / 2)).to(out_dtype) - 1j * (w[row2, imag_col2] * (scale / 2)).to(out_dtype)
    rows = _row_positions(len(index_chunk), positions1.shape[1], device=device)
    packed[rows, _as_index_tensor(positions1, device=device)] = values1
    packed[rows, _as_index_tensor(positions2, device=device)] = values2
    return packed


def _pack_w_att_triton_chunk_gpu(
    w,
    vector_start: int,
    vector_count: int,
    *,
    weight_shape: tuple[int, int],
    n_in: int,
    n_out: int,
    b_shape: tuple[int, int],
    ll: int,
    dd: int,
    scale: float,
    device: str,
    out_dtype=None,
    block_size: int,
):
    _enable_triton_easyfhe_bridge()
    if out_dtype is None:
        out_dtype = torch.complex128
    packed = torch.zeros((vector_count, SLOTS), dtype=out_dtype, device=device)
    packed_real = packed.view(torch.float64)
    count = PACK * DIM * dd
    grid = (vector_count, triton.cdiv(count, block_size))
    _pack_w_att_triton_kernel[grid](
        w,
        packed_real,
        int(vector_start),
        int(weight_shape[1]),
        int(n_in),
        int(n_in // 2),
        int(n_out // PACK),
        int(b_shape[0]),
        int(b_shape[1]),
        int(weight_shape[0] // b_shape[0]),
        int(weight_shape[1] // b_shape[1]),
        int(ll),
        int(dd),
        float(scale * 0.5),
        COUNT=count,
        BLOCK=block_size,
    )
    return packed


@triton.jit
def _pack_w_att_triton_kernel(
    w,
    packed_real,
    VECTOR_START: tl.constexpr,
    W_COLS: tl.constexpr,
    N_IN: tl.constexpr,
    N_IN_C: tl.constexpr,
    N_OUT_P: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    V_BLOCKS: tl.constexpr,
    H_BLOCKS: tl.constexpr,
    LL: tl.constexpr,
    DD: tl.constexpr,
    HALF_SCALE: tl.constexpr,
    COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local_vec = tl.program_id(0)
    vector_id = VECTOR_START + local_vec
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < COUNT

    d = offsets % DD
    tmp = offsets // DD
    t = tmp % 128
    j = tmp // 128

    n = vector_id % N_IN_C
    l = (vector_id // N_IN_C) % LL
    out = (vector_id // (N_IN_C * LL)) % N_OUT_P

    base = ((n // 16) * 16 + out * 16 + ((n + j) % 16)) % N_IN_C
    r = out * 16 + j
    i = (base + N_IN * 2 - r) % N_IN

    block_row = (l + d) % V_BLOCKS
    block_col = d % H_BLOCKS
    row = block_row * BR + ((t + r) % BR)
    real_col = block_col * BC + ((i + t + r) % BC)
    imag_col = block_col * BC + ((((i + N_IN_C) % N_IN) + t + r) % BC)

    real = tl.load(w + row * W_COLS + real_col, mask=mask, other=0.0).to(tl.float64) * HALF_SCALE
    imag = tl.load(w + row * W_COLS + imag_col, mask=mask, other=0.0).to(tl.float64) * HALF_SCALE
    slot = j * 2048 + t * 16 + d
    out_base = (local_vec * 32768 + slot) * 2
    tl.store(packed_real + out_base, real, mask=mask)
    tl.store(packed_real + out_base + 1, -imag, mask=mask)


def _pack_w_ff_triton_chunk_gpu(
    w,
    vector_start: int,
    vector_count: int,
    *,
    weight_shape: tuple[int, int],
    split_shape: tuple[int, int],
    split_mode: int,
    n_in: int,
    n_out: int,
    b_shape: tuple[int, int],
    ll: int,
    dd: int,
    scale: float,
    device: str,
    out_dtype=None,
    block_size: int,
):
    _enable_triton_easyfhe_bridge()
    if out_dtype is None:
        out_dtype = torch.complex128
    packed = torch.zeros((vector_count, SLOTS), dtype=out_dtype, device=device)
    packed_real = packed.view(torch.float64)
    count = PACK * DIM * dd * 2
    grid = (vector_count, triton.cdiv(count, block_size))
    _pack_w_ff_triton_kernel[grid](
        w,
        packed_real,
        int(vector_start),
        int(weight_shape[1]),
        int(split_shape[0]),
        int(split_shape[1]),
        int(split_mode),
        int(n_in),
        int(n_in // 2),
        int(n_out // PACK),
        int(b_shape[0]),
        int(b_shape[1]),
        int(split_shape[0] // b_shape[0]),
        int(split_shape[1] // b_shape[1]),
        int(ll),
        int(dd),
        float(scale * 0.5),
        COUNT=count,
        BLOCK=block_size,
    )
    return packed


@triton.jit
def _pack_w_ff_triton_kernel(
    w,
    packed_real,
    VECTOR_START: tl.constexpr,
    W_COLS: tl.constexpr,
    SPLIT_ROWS: tl.constexpr,
    SPLIT_COLS: tl.constexpr,
    SPLIT_MODE: tl.constexpr,
    N_IN: tl.constexpr,
    N_IN_C: tl.constexpr,
    N_OUT_P: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    V_BLOCKS: tl.constexpr,
    H_BLOCKS: tl.constexpr,
    LL: tl.constexpr,
    DD: tl.constexpr,
    HALF_SCALE: tl.constexpr,
    COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local_vec = tl.program_id(0)
    vector_id = VECTOR_START + local_vec
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < COUNT

    one_part = 16 * 128 * DD
    part = offsets // one_part
    rem = offsets - part * one_part
    d = rem % DD
    tmp = rem // DD
    t = tmp % 128
    j = tmp // 128

    n = vector_id % N_IN_C
    l = (vector_id // N_IN_C) % LL
    out = (vector_id // (N_IN_C * LL)) % N_OUT_P
    rep = vector_id // (N_OUT_P * LL * N_IN_C)
    split_index = rep * 2 + part

    base = ((n // 16) * 16 + out * 16 + ((n + j) % 16)) % N_IN_C
    r = out * 16 + j
    i = (base + N_IN * 2 - r) % N_IN

    block_row = (l + d) % V_BLOCKS
    block_col = d % H_BLOCKS
    row = block_row * BR + ((t + r) % BR)
    real_col = block_col * BC + ((i + t + r) % BC)
    imag_col = block_col * BC + ((((i + N_IN_C) % N_IN) + t + r) % BC)

    if SPLIT_MODE == 1:
        row = row + split_index * SPLIT_ROWS
    else:
        real_col = real_col + split_index * SPLIT_COLS
        imag_col = imag_col + split_index * SPLIT_COLS

    real = tl.load(w + row * W_COLS + real_col, mask=mask, other=0.0).to(tl.float64) * HALF_SCALE
    imag = tl.load(w + row * W_COLS + imag_col, mask=mask, other=0.0).to(tl.float64) * HALF_SCALE
    slot = j * 2048 + t * 16 + d + part * 8
    out_base = (local_vec * 32768 + slot) * 2
    tl.store(packed_real + out_base, real, mask=mask)
    tl.store(packed_real + out_base + 1, -imag, mask=mask)


def _row_positions(batch: int, width: int, *, device: str):
    rows = np.broadcast_to(np.arange(batch, dtype=np.int64)[:, None], (batch, width))
    return _as_index_tensor(np.ascontiguousarray(rows), device=device)
