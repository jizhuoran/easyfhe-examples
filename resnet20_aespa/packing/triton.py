"""Triton kernels that expand compact ResNet20 constants on the GPU."""

from __future__ import annotations

import sys
import types

import easyfhe as torch

if not hasattr(torch, "version"):
    torch.version = types.SimpleNamespace(hip=None)
elif not hasattr(torch.version, "hip"):
    torch.version.hip = None
sys.modules.setdefault("torch", torch)

import triton
import triton.language as tl


@triton.jit
def _factorized_pack_kernel(
    coefficients,
    masks,
    mask_ids,
    output,
    total,
    slots: tl.constexpr,
    channels: tl.constexpr,
    spatial_size: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    active = offsets < total
    row = offsets // slots
    slot = offsets - row * slots
    channel = slot // spatial_size
    spatial = slot - channel * spatial_size
    mask_id = tl.load(mask_ids + row, mask=active, other=0)
    coefficient = tl.load(
        coefficients + row * channels + channel,
        mask=active,
        other=0.0,
    )
    mask_value = tl.load(
        masks + mask_id * spatial_size + spatial,
        mask=active,
        other=0,
    )
    tl.store(output + offsets, coefficient * mask_value, mask=active)


@triton.jit
def _dictionary_pack_kernel(
    values,
    codes,
    output,
    total,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    active = offsets < total
    code = tl.load(codes + offsets, mask=active, other=0)
    value = tl.load(values + code, mask=active, other=0.0)
    tl.store(output + offsets, value, mask=active)


def pack_factorized(coefficients, masks, mask_ids, *, slots: int):
    """Expand `[rows, channels] × row spatial masks` into slot vectors."""

    _enable_triton_easyfhe_bridge()
    rows, channels = (int(value) for value in coefficients.shape)
    spatial_size = int(masks.shape[1])
    slots = int(slots)
    if slots != channels * spatial_size:
        raise ValueError(
            f"factorized packing expected {channels * spatial_size} slots, got {slots}"
        )
    if int(mask_ids.shape[0]) != rows:
        raise ValueError(
            f"factorized packing expected {rows} mask ids, got {mask_ids.shape[0]}"
        )

    output = torch.zeros(
        (rows, slots),
        dtype=torch.float64,
        device=coefficients.device,
    )
    total = rows * slots
    block_size = 256
    _factorized_pack_kernel[(triton.cdiv(total, block_size),)](
        coefficients,
        masks,
        mask_ids,
        output,
        total,
        slots=slots,
        channels=channels,
        spatial_size=spatial_size,
        block_size=block_size,
        num_warps=4,
    )
    return output


def pack_dictionary(values, codes, *, slots: int):
    """Expand one dictionary-coded slot vector on the GPU."""

    _enable_triton_easyfhe_bridge()
    slots = int(slots)
    if int(codes.numel()) != slots:
        raise ValueError(
            f"dictionary packing expected {int(codes.numel())} slots, got {slots}"
        )
    output = torch.zeros((slots,), dtype=torch.float64, device=values.device)
    block_size = 256
    _dictionary_pack_kernel[(triton.cdiv(slots, block_size),)](
        values,
        codes,
        output,
        slots,
        block_size=block_size,
        num_warps=4,
    )
    return output


def _enable_triton_easyfhe_bridge() -> None:
    """Let Triton launch kernels against EasyFHE's torch-compatible runtime."""

    import triton.backends.driver as triton_driver

    if getattr(triton_driver.GPUDriver, "_easyfhe_bridge", False):
        return

    def init(self):
        self.get_device_capability = torch.cuda.get_device_capability
        self.get_current_stream = lambda index: torch.cuda.current_stream(
            index
        ).cuda_stream
        self.get_current_device = torch.cuda.current_device
        self.set_current_device = torch.cuda.set_device

    triton_driver.GPUDriver.__init__ = init
    triton_driver.GPUDriver._easyfhe_bridge = True
