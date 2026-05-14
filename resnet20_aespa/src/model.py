from dataclasses import dataclass

import easyfhe.fhe as fhe

from .convs import (
    aespa_add_shortcut,
    aespa_nonlinear,
    conv3x3,
    initial_conv3x3,
    pointwise_conv,
)
from .utils import (
    WeightPack,
    broadcast_slot_sum,
    downsample1024to256,
    downsample256to64,
    sum_adjacent_slots,
    sum_channel_groups,
)


@dataclass
class AespaRuntime:
    ctx: object
    weights: WeightPack
    config: object
    bootstrap_constants: dict[int, object]


@dataclass(frozen=True)
class _SameShapeBlockSpec:
    block_id: int
    img_width: int
    channels: int
    rot_offset: int
    bootstrap_l0: object = None


@dataclass(frozen=True)
class _DownsampleBlockSpec:
    block_id: int
    in_img_width: int
    in_channels: int
    out_img_width: int
    out_channels: int
    first_rot: int
    second_rot: int
    downsample_kind: str
    bootstrap_l0: object
    rescale_after_add: bool = False


def _convbn_weight_prefix(block_id, conv_id):
    return f"layer{block_id}-conv{conv_id}bn{conv_id}"


def _conv3x3_kernel_prefixes(block_id, conv_id, channels, channel_offset=0):
    prefix = _convbn_weight_prefix(block_id, conv_id)
    return [f"{prefix}-ch{channel + channel_offset}" for channel in range(channels)]


def _pointwise_kernel_keys(block_id, conv_id, channels, channel_offset=0):
    return [
        f"layer{block_id}dx-conv{conv_id}bn{conv_id}-ch{channel + channel_offset}-k1"
        for channel in range(channels)
    ]


def _pointwise_bias_key(block_id, conv_id, bias_offset):
    return f"layer{block_id}dx-conv{conv_id}bn{conv_id}-bias{bias_offset}"


def _conv_then_aespa_nonlinear(input, block_id, conv_id, img_width, channels, rot_offset, runtime, scale=1):
    res = conv3x3(
        input,
        _conv3x3_kernel_prefixes(block_id, conv_id, channels),
        img_width,
        1,
        rot_offset,
        scale,
        runtime.ctx,
        runtime.weights,
    )
    res = fhe.rescale_one_level(res, runtime.ctx)
    return aespa_nonlinear(res, _convbn_weight_prefix(block_id, conv_id), runtime.ctx, runtime.weights, scale)


def _downsample_spatial_pair(sx0, sx1, dx0, dx1, in_channels, downsample_kind, runtime):
    if downsample_kind == "1024to256":
        return (
            downsample1024to256(sx0, sx1, in_channels, runtime.ctx, runtime.weights),
            downsample1024to256(dx0, dx1, in_channels, runtime.ctx, runtime.weights),
        )
    if downsample_kind == "256to64":
        return (
            downsample256to64(sx0, sx1, in_channels, runtime.ctx, runtime.weights),
            downsample256to64(dx0, dx1, in_channels, runtime.ctx, runtime.weights),
        )
    raise ValueError(f"Unsupported downsample kind: {downsample_kind}")


def _downsample_conv_pair(input, block_id, in_img_width, in_channels, first_rot, runtime, scale):
    first_half = conv3x3(
        input,
        _conv3x3_kernel_prefixes(block_id, 1, in_channels),
        in_img_width,
        1,
        first_rot,
        scale,
        runtime.ctx,
        runtime.weights,
    )
    second_half = conv3x3(
        input,
        _conv3x3_kernel_prefixes(block_id, 1, in_channels, channel_offset=in_channels),
        in_img_width,
        1,
        first_rot,
        scale,
        runtime.ctx,
        runtime.weights,
    )
    return fhe.rescale_one_level(first_half, runtime.ctx), fhe.rescale_one_level(second_half, runtime.ctx)


def _projection_input_for_downsample(input, runtime):
    if runtime.ctx.rescaleTech != "FIXEDMANUAL":
        return input
    return fhe.align_to(input, fhe.CipherState(input.cur_limbs - 2, input.noise_deg), runtime.ctx)


def _downsample_projection_pair(input, block_id, in_channels, first_rot, runtime, scale):
    input = _projection_input_for_downsample(input, runtime)
    first_half = pointwise_conv(
        input,
        _pointwise_kernel_keys(block_id, 1, in_channels),
        _pointwise_bias_key(block_id, 1, "1"),
        first_rot,
        scale,
        runtime.ctx,
        runtime.weights,
    )
    second_half = pointwise_conv(
        input,
        _pointwise_kernel_keys(block_id, 1, in_channels, channel_offset=in_channels),
        _pointwise_bias_key(block_id, 1, "2"),
        first_rot,
        scale,
        runtime.ctx,
        runtime.weights,
    )
    return first_half, second_half


def _same_shape_residual_block(
    input,
    spec,
    runtime,
):
    scale = 1
    res = _conv_then_aespa_nonlinear(
        input,
        spec.block_id,
        1,
        spec.img_width,
        spec.channels,
        spec.rot_offset,
        runtime,
        scale,
    )
    res = conv3x3(
        res,
        _conv3x3_kernel_prefixes(spec.block_id, 2, spec.channels),
        spec.img_width,
        1,
        spec.rot_offset,
        scale,
        runtime.ctx,
        runtime.weights,
    )
    res = aespa_add_shortcut(res, input, _convbn_weight_prefix(spec.block_id, 2), runtime.ctx, runtime.weights, scale)
    res = fhe.rescale_one_level(res, runtime.ctx)

    if spec.bootstrap_l0 is not None:
        log_bs_slots = runtime.config.log_bs_slots[0]
        res = fhe.homo_bootstrap(
            res,
            runtime.ctx,
            runtime.bootstrap_constants[int(log_bs_slots)],
            L0=spec.bootstrap_l0,
        )
    return aespa_nonlinear(res, _convbn_weight_prefix(spec.block_id, 2), runtime.ctx, runtime.weights, scale)


def _downsample_residual_block(
    input,
    spec,
    runtime,
):
    scale_sx = 1
    scale_dx = 1

    sx0, sx1 = _downsample_conv_pair(
        input,
        spec.block_id,
        spec.in_img_width,
        spec.in_channels,
        spec.first_rot,
        runtime,
        scale_sx,
    )
    dx0, dx1 = _downsample_projection_pair(input, spec.block_id, spec.in_channels, spec.first_rot, runtime, scale_dx)
    sx, dx = _downsample_spatial_pair(sx0, sx1, dx0, dx1, spec.in_channels, spec.downsample_kind, runtime)

    sx = fhe.rescale_one_level(sx, runtime.ctx)
    sx = aespa_nonlinear(sx, _convbn_weight_prefix(spec.block_id, 1), runtime.ctx, runtime.weights, scale_sx)
    sx = conv3x3(
        sx,
        _conv3x3_kernel_prefixes(spec.block_id, 2, spec.out_channels),
        spec.out_img_width,
        1,
        spec.second_rot,
        scale_dx,
        runtime.ctx,
        runtime.weights,
    )
    res = fhe.homo_add(sx, dx, runtime.ctx)
    if spec.rescale_after_add:
        res = fhe.rescale_one_level(res, runtime.ctx)

    log_bs_slots = runtime.config.log_bs_slots[0]
    res = fhe.homo_bootstrap(
        res,
        runtime.ctx,
        runtime.bootstrap_constants[int(log_bs_slots)],
        L0=spec.bootstrap_l0,
    )
    return aespa_nonlinear(res, _convbn_weight_prefix(spec.block_id, 2), runtime.ctx, runtime.weights, scale_dx)


def initial_layer(input, runtime):
    res = initial_conv3x3(
        input,
        [f"conv1bn1-ch{channel}" for channel in range(16)],
        32,
        1,
        1024,
        1,
        runtime.ctx,
        runtime.weights,
    )
    res = fhe.rescale_one_level(res, runtime.ctx)
    return aespa_nonlinear(res, "conv1bn1", runtime.ctx, runtime.weights)


def layer1(input, runtime):
    res = _same_shape_residual_block(input, _SameShapeBlockSpec(1, 32, 16, -1024), runtime)
    res = _same_shape_residual_block(
        res,
        _SameShapeBlockSpec(
            2,
            32,
            16,
            -1024,
            bootstrap_l0=runtime.ctx.L - (runtime.config.max_levels_remaining - 5),
        ),
        runtime,
    )
    return _same_shape_residual_block(
        res,
        _SameShapeBlockSpec(3, 32, 16, -1024, bootstrap_l0=runtime.ctx.L),
        runtime,
    )


def layer2(input, runtime):
    res = _downsample_residual_block(
        input,
        _DownsampleBlockSpec(
            block_id=4,
            in_img_width=32,
            in_channels=16,
            out_img_width=16,
            out_channels=32,
            first_rot=-1024,
            second_rot=-256,
            downsample_kind="1024to256",
            bootstrap_l0=runtime.ctx.L - (runtime.config.max_levels_remaining - 9),
        ),
        runtime,
    )
    res = _same_shape_residual_block(res, _SameShapeBlockSpec(5, 16, 32, -256), runtime)
    return _same_shape_residual_block(res, _SameShapeBlockSpec(6, 16, 32, -256, bootstrap_l0=runtime.ctx.L - 1), runtime)


def layer3(input, runtime):
    res = _downsample_residual_block(
        input,
        _DownsampleBlockSpec(
            block_id=7,
            in_img_width=16,
            in_channels=32,
            out_img_width=8,
            out_channels=64,
            first_rot=-256,
            second_rot=-64,
            downsample_kind="256to64",
            bootstrap_l0=runtime.ctx.L,
            rescale_after_add=True,
        ),
        runtime,
    )
    res = _same_shape_residual_block(res, _SameShapeBlockSpec(8, 8, 64, -64), runtime)
    return _same_shape_residual_block(res, _SameShapeBlockSpec(9, 8, 64, -64), runtime)


def final_layer(input, runtime):
    channels = 64
    spatial_size = 64
    fc_repeat = 16

    res = sum_adjacent_slots(input, spatial_size, runtime.ctx)
    res = fhe.homo_mul_pt(
        res,
        runtime.weights.plaintext(
            f"mask_mod_{spatial_size}_{1.0 / spatial_size}_{res.slots}",
            runtime.ctx.L - res.cur_limbs,
            res.slots,
            runtime.ctx,
        ),
        runtime.ctx,
    )
    res = broadcast_slot_sum(res, fc_repeat, runtime.ctx)
    res = fhe.rescale_one_level(res, runtime.ctx)
    weight = runtime.weights.plaintext_for_cipher(f"fc_{res.slots}", res, runtime.ctx)
    res = fhe.homo_mul_pt(res, weight, runtime.ctx)
    res = fhe.rescale_one_level(res, runtime.ctx)
    res = sum_channel_groups(res, spatial_size, channels, runtime.ctx)

    bias = runtime.weights.plaintext_for_cipher(f"bias_{res.slots}", res, runtime.ctx)
    return fhe.homo_add_pt(res, bias, runtime.ctx)


def infer_one(image_vector, runtime):
    in_ct = runtime.ctx.encrypt(image_vector, runtime.ctx.device, 1, 19, 16 * 32 * 32)
    first_layer = initial_layer(in_ct, runtime)
    res_layer1 = layer1(first_layer, runtime)
    res_layer2 = layer2(res_layer1, runtime)
    res_layer3 = layer3(res_layer2, runtime)
    return final_layer(res_layer3, runtime)


__all__ = [
    "AespaRuntime",
    "final_layer",
    "infer_one",
    "initial_layer",
    "layer1",
    "layer2",
    "layer3",
]
