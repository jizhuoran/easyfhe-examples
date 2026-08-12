"""Encrypted ResNet20 graph for the canonical u64 CKKS example."""

from dataclasses import dataclass

import easyfhe.bs.openfhe as bs
import easyfhe.fhe as fhe

from .layout import (
    broadcast_slot_sum,
    downsample1024to256_pair,
    downsample256to64,
    sum_adjacent_slots,
    sum_channel_groups,
)
from .ops import (
    _add_plaintext_for_cipher,
    _mul_plaintext_for_cipher,
    aespa_add_shortcut,
    aespa_nonlinear,
    conv3x3,
    initial_conv3x3,
    pointwise_conv,
)
from ..runtime import ResNet20Runtime


@dataclass(frozen=True)
class _ResidualBlockSpec:
    block_id: int
    image_width: int
    channels: int
    rotation_offset: int
    bootstrap_output_levels: int | None = None


def _bootstrap(cipher, output_levels, runtime):
    cipher = fhe.normalize_scale(cipher, runtime.context)
    program = runtime.bootstrap_programs[int(output_levels)]
    return bs.bootstrap(cipher, runtime.context, program)


def _residual_block(input_cipher, spec, runtime):
    context = runtime.context
    conv1_prefix = f"layer{spec.block_id}-conv1bn1"
    conv2_prefix = f"layer{spec.block_id}-conv2bn2"

    result = conv3x3(
        input_cipher,
        f"{conv1_prefix}-ch0-{spec.channels - 1}-k",
        spec.image_width,
        spec.rotation_offset,
        context,
        runtime.weights,
        group_size=spec.channels,
    )
    result = aespa_nonlinear(result, conv1_prefix, context, runtime.weights)
    result = conv3x3(
        result,
        f"{conv2_prefix}-ch0-{spec.channels - 1}-k",
        spec.image_width,
        spec.rotation_offset,
        context,
        runtime.weights,
        group_size=spec.channels,
    )
    result = aespa_add_shortcut(
        result, input_cipher, conv2_prefix, context, runtime.weights
    )
    if spec.bootstrap_output_levels is not None:
        result = _bootstrap(result, spec.bootstrap_output_levels, runtime)
    return aespa_nonlinear(result, conv2_prefix, context, runtime.weights)


def _layer2_downsample_block(input_cipher, runtime):
    context = runtime.context
    group_size = 16
    sx0 = conv3x3(
        input_cipher,
        "layer4-conv1bn1-ch0-15-k",
        32,
        -1024,
        context,
        runtime.weights,
        group_size=group_size,
    )
    sx1 = conv3x3(
        input_cipher,
        "layer4-conv1bn1-ch16-31-k",
        32,
        -1024,
        context,
        runtime.weights,
        group_size=group_size,
    )
    # The projection skips two main-path multiplications, so drop two Q limbs
    # before encoding its pointwise weights.
    projection_input = fhe.align_to(
        input_cipher,
        input_cipher.state.replace(cur_limbs=input_cipher.state.cur_limbs - 2),
        context,
    )
    dx0 = pointwise_conv(
        projection_input,
        "layer4dx-conv1bn1-ch0-15-k1",
        "layer4dx-conv1bn1-bias1",
        -1024,
        context,
        runtime.weights,
        group_size=group_size,
    )
    dx1 = pointwise_conv(
        projection_input,
        "layer4dx-conv1bn1-ch16-31-k1",
        "layer4dx-conv1bn1-bias2",
        -1024,
        context,
        runtime.weights,
        group_size=group_size,
    )
    sx = downsample1024to256_pair(
        sx0, sx1, group_size, context, runtime.weights
    )
    dx = downsample1024to256_pair(
        dx0, dx1, group_size, context, runtime.weights
    )
    sx = aespa_nonlinear(
        sx, "layer4-conv1bn1", context, runtime.weights
    )
    sx = conv3x3(
        sx,
        "layer4-conv2bn2-ch0-31-k",
        16,
        -256,
        context,
        runtime.weights,
        group_size=32,
    )
    sx = fhe.normalize_scale(sx, context)
    dx = fhe.normalize_scale(dx, context)
    result = fhe.homo_add(sx, dx, context)
    result = _bootstrap(result, 9, runtime)
    return aespa_nonlinear(
        result, "layer4-conv2bn2", context, runtime.weights
    )


def _layer3_downsample_block(input_cipher, runtime):
    context = runtime.context
    group_size = 32
    sx0 = conv3x3(
        input_cipher,
        "layer7-conv1bn1-ch0-31-k",
        16,
        -256,
        context,
        runtime.weights,
        group_size=group_size,
    )
    sx1 = conv3x3(
        input_cipher,
        "layer7-conv1bn1-ch32-63-k",
        16,
        -256,
        context,
        runtime.weights,
        group_size=group_size,
    )
    # As above, the projection is aligned with the two-operation main path.
    projection_input = fhe.align_to(
        input_cipher,
        input_cipher.state.replace(cur_limbs=input_cipher.state.cur_limbs - 2),
        context,
    )
    dx0 = pointwise_conv(
        projection_input,
        "layer7dx-conv1bn1-ch0-31-k1",
        "layer7dx-conv1bn1-bias1",
        -256,
        context,
        runtime.weights,
        group_size=group_size,
    )
    dx1 = pointwise_conv(
        projection_input,
        "layer7dx-conv1bn1-ch32-63-k1",
        "layer7dx-conv1bn1-bias2",
        -256,
        context,
        runtime.weights,
        group_size=group_size,
    )
    sx = downsample256to64(sx0, sx1, group_size, context, runtime.weights)
    dx = downsample256to64(dx0, dx1, group_size, context, runtime.weights)
    sx = aespa_nonlinear(
        sx, "layer7-conv1bn1", context, runtime.weights
    )
    sx = conv3x3(
        sx,
        "layer7-conv2bn2-ch0-63-k",
        8,
        -64,
        context,
        runtime.weights,
        group_size=64,
    )
    sx = fhe.normalize_scale(sx, context)
    dx = fhe.normalize_scale(dx, context)
    result = fhe.homo_add(sx, dx, context)
    result = _bootstrap(result, 12, runtime)
    return aespa_nonlinear(
        result, "layer7-conv2bn2", context, runtime.weights
    )


def initial_layer(input_cipher, runtime):
    result = initial_conv3x3(
        input_cipher,
        "conv1bn1-ch0-15-k",
        32,
        1024,
        runtime.context,
        runtime.weights,
        group_size=16,
    )
    return aespa_nonlinear(
        result, "conv1bn1", runtime.context, runtime.weights
    )


def layer1(input_cipher, runtime):
    result = _residual_block(
        input_cipher, _ResidualBlockSpec(1, 32, 16, -1024), runtime
    )
    result = _residual_block(
        result, _ResidualBlockSpec(2, 32, 16, -1024, 5), runtime
    )
    return _residual_block(
        result, _ResidualBlockSpec(3, 32, 16, -1024, 12), runtime
    )


def layer2(input_cipher, runtime):
    result = _layer2_downsample_block(input_cipher, runtime)
    result = _residual_block(
        result, _ResidualBlockSpec(5, 16, 32, -256), runtime
    )
    return _residual_block(
        result, _ResidualBlockSpec(6, 16, 32, -256, 11), runtime
    )


def layer3(input_cipher, runtime):
    result = _layer3_downsample_block(input_cipher, runtime)
    result = _residual_block(
        result, _ResidualBlockSpec(8, 8, 64, -64), runtime
    )
    return _residual_block(
        result, _ResidualBlockSpec(9, 8, 64, -64), runtime
    )


def final_layer(input_cipher, runtime):
    context = runtime.context
    spatial_size = 64
    result = sum_adjacent_slots(input_cipher, spatial_size, context)
    result = fhe.homo_mul_pt_rescale(
        result,
        _mul_plaintext_for_cipher(
            runtime.weights,
            f"mask_mod_{spatial_size}_{1.0 / spatial_size}_{result.slots}",
            result,
            context,
        ),
        context,
    )
    result = broadcast_slot_sum(result, 16, context)
    result = fhe.homo_mul_pt_rescale(
        result,
        _mul_plaintext_for_cipher(
            runtime.weights, f"fc_{result.slots}", result, context
        ),
        context,
    )
    result = sum_channel_groups(result, spatial_size, 64, context)
    return fhe.homo_add_pt(
        result,
        _add_plaintext_for_cipher(
            runtime.weights, f"bias_{result.slots}", result, context
        ),
        context,
    )


def infer_encrypted(input_cipher, runtime):
    result = initial_layer(input_cipher, runtime)
    result = layer1(result, runtime)
    result = layer2(result, runtime)
    result = layer3(result, runtime)
    return final_layer(result, runtime)


__all__ = [
    "final_layer",
    "infer_encrypted",
    "initial_layer",
    "layer1",
    "layer2",
    "layer3",
]
