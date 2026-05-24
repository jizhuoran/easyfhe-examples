from dataclasses import dataclass

import easyfhe.bs.openfhe as bs
import easyfhe.fhe as fhe

from .layout import (
    broadcast_slot_sum,
    downsample1024to256,
    downsample256to64_pair,
    sum_adjacent_slots,
    sum_channel_groups,
)
from .ops import (
    aespa_add_shortcut,
    aespa_nonlinear,
    conv3x3,
    initial_conv3x3,
    pointwise_conv,
)


@dataclass
class AespaRuntime:
    ctx: object
    client: object
    weights: object
    config: object
    bootstrap_material: dict[int, tuple[object, object]]


@dataclass(frozen=True)
class _SameShapeBlockSpec:
    block_id: int
    img_width: int
    channels: int
    rot_offset: int
    bootstrap_l0: object = None


def _residual_block(
    input,
    spec,
    rt,
):
    scale = "scale.one"
    conv1_prefix = f"layer{spec.block_id}-conv1bn1"
    conv2_prefix = f"layer{spec.block_id}-conv2bn2"
    res = conv3x3(
        input,
        f"{conv1_prefix}-ch0-{spec.channels - 1}-k",
        spec.img_width,
        1,
        spec.rot_offset,
        scale,
        rt.ctx,
        rt.weights,
        group_size=spec.channels,
    )
    res = fhe.rescale_one_level(res, rt.ctx)
    res = aespa_nonlinear(res, conv1_prefix, rt.ctx, rt.weights, scale)
    res = conv3x3(
        res,
        f"{conv2_prefix}-ch0-{spec.channels - 1}-k",
        spec.img_width,
        1,
        spec.rot_offset,
        scale,
        rt.ctx,
        rt.weights,
        group_size=spec.channels,
    )
    res = aespa_add_shortcut(res, input, conv2_prefix, rt.ctx, rt.weights, scale)
    res = fhe.rescale_one_level(res, rt.ctx)

    if spec.bootstrap_l0 is not None:
        constants, plan = rt.bootstrap_material[int(rt.config.log_bs_slots[0])]
        res = bs.bootstrap(
            res,
            rt.ctx,
            constants,
            plan,
            L0=spec.bootstrap_l0,
            bootstrap_mode=rt.config.bootstrap_mode,
        )
    return aespa_nonlinear(res, conv2_prefix, rt.ctx, rt.weights, scale)


def _layer2_downsample_block(input, rt):
    scale = "scale.one"
    group_size = 32 // 2
    sx = conv3x3(
        input,
        "layer4-conv1bn1-sx",
        32,
        1,
        -1024,
        scale,
        rt.ctx,
        rt.weights,
        group_size=group_size,
        final_rotate=group_size * -1024,
    )
    sx = fhe.rescale_one_level(sx, rt.ctx)
    projection_input = fhe.align_to(input, fhe.CipherState(input.state.cur_limbs - 2, input.state.noise_deg), rt.ctx)
    dx = pointwise_conv(
        projection_input,
        "layer4dx-conv1bn1-sx",
        "layer4dx-conv1bn1-bias-sx",
        -1024,
        scale,
        rt.ctx,
        rt.weights,
        group_size=group_size,
        final_rotate=group_size * -1024,
    )
    sx = downsample1024to256(sx, 16, rt.ctx, rt.weights)
    dx = downsample1024to256(dx, 16, rt.ctx, rt.weights)
    sx = fhe.rescale_one_level(sx, rt.ctx)
    sx = aespa_nonlinear(sx, "layer4-conv1bn1", rt.ctx, rt.weights, scale)
    sx = conv3x3(
        sx,
        "layer4-conv2bn2-ch0-31-k",
        16,
        1,
        -256,
        scale,
        rt.ctx,
        rt.weights,
        group_size=32,
    )
    res = fhe.homo_add(sx, dx, rt.ctx)
    constants, plan = rt.bootstrap_material[int(rt.config.log_bs_slots[0])]
    res = bs.bootstrap(
        res,
        rt.ctx,
        constants,
        plan,
        L0=rt.ctx.L,
        bootstrap_mode=rt.config.bootstrap_mode,
    )
    return aespa_nonlinear(res, "layer4-conv2bn2", rt.ctx, rt.weights, scale)


def _layer3_downsample_block(input, rt):
    scale = "scale.one"
    group_size = 64 // 2
    sx = conv3x3(
        input,
        "layer7-conv1bn1-sx",
        16,
        1,
        -256,
        scale,
        rt.ctx,
        rt.weights,
        group_size=group_size,
        final_rotate=group_size * -256,
    )
    sx = fhe.rescale_one_level(sx, rt.ctx)
    dx = pointwise_conv(
        input,
        "layer7dx-conv1bn1-sx",
        "layer7dx-conv1bn1-bias-sx",
        -256,
        scale,
        rt.ctx,
        rt.weights,
        group_size=group_size,
        final_rotate=group_size * -256,
    )
    sx, dx = downsample256to64_pair(sx, dx, 32, rt.ctx, rt.weights)
    sx = fhe.rescale_one_level(sx, rt.ctx)
    sx = aespa_nonlinear(sx, "layer7-conv1bn1", rt.ctx, rt.weights, scale)
    sx = conv3x3(
        sx,
        "layer7-conv2bn2-ch0-63-k",
        8,
        1,
        -64,
        scale,
        rt.ctx,
        rt.weights,
        group_size=64,
    )
    res = fhe.homo_add(sx, dx, rt.ctx)
    res = fhe.rescale_one_level(res, rt.ctx)
    res = aespa_nonlinear(res, "layer7-conv2bn2", rt.ctx, rt.weights, scale)
    constants, plan = rt.bootstrap_material[int(rt.config.log_bs_slots[0])]
    return bs.bootstrap(
        res,
        rt.ctx,
        constants,
        plan,
        L0=rt.ctx.L,
        bootstrap_mode=rt.config.bootstrap_mode,
    )


def initial_layer(input, rt):
    res = initial_conv3x3(
        input,
        "conv1bn1-ch0-15-k",
        32,
        1,
        1024,
        "scale.one",
        rt.ctx,
        rt.weights,
        group_size=16,
    )
    res = fhe.rescale_one_level(res, rt.ctx)
    return aespa_nonlinear(res, "conv1bn1", rt.ctx, rt.weights)


def layer1(input, rt):
    res = _residual_block(input, _SameShapeBlockSpec(1, 32, 16, -1024), rt)
    res = _residual_block(res, _SameShapeBlockSpec(2, 32, 16, -1024), rt)
    return _residual_block(res, _SameShapeBlockSpec(3, 32, 16, -1024), rt)


def layer2(input, rt):
    constants, plan = rt.bootstrap_material[int(rt.config.log_bs_slots[0])]
    input = bs.bootstrap(
        input,
        rt.ctx,
        constants,
        plan,
        L0=rt.ctx.L,
        bootstrap_mode=rt.config.bootstrap_mode,
    )
    input = fhe.expand_slots(input, input.slots << 1, rt.ctx)
    input = fhe.align_to(input, fhe.CipherState(input.state.cur_limbs - 1, input.state.noise_deg), rt.ctx)
    res = _layer2_downsample_block(input, rt)
    res = fhe.align_to(res, fhe.CipherState(res.state.cur_limbs - 1, res.state.noise_deg), rt.ctx)
    res = _residual_block(res, _SameShapeBlockSpec(5, 16, 32, -256), rt)
    return _residual_block(res, _SameShapeBlockSpec(6, 16, 32, -256), rt)


def layer3(input, rt):
    constants, plan = rt.bootstrap_material[int(rt.config.log_bs_slots[0])]
    input = bs.bootstrap(
        input,
        rt.ctx,
        constants,
        plan,
        L0=rt.ctx.L,
        bootstrap_mode=rt.config.bootstrap_mode,
    )
    input = fhe.expand_slots(input, input.slots << 1, rt.ctx)
    res = _layer3_downsample_block(input, rt)
    res = _residual_block(res, _SameShapeBlockSpec(8, 8, 64, -64), rt)
    return _residual_block(res, _SameShapeBlockSpec(9, 8, 64, -64), rt)


def final_layer(input, rt):
    channels = 64
    spatial_size = 64
    fc_repeat = 16

    res = sum_adjacent_slots(input, spatial_size, rt.ctx)
    res = fhe.homo_mul_pt(
        res,
        rt.weights.plaintext(
            f"mask_mod_{spatial_size}_{1.0 / spatial_size}_{res.slots}",
            rt.ctx.L - res.state.cur_limbs,
            res.slots,
            rt.ctx,
        ),
        rt.ctx,
    )
    res = broadcast_slot_sum(res, fc_repeat, rt.ctx)
    res = fhe.rescale_one_level(res, rt.ctx)
    weight = rt.weights.plaintext(
        f"fc_{res.slots}",
        rt.ctx.L - res.state.cur_limbs,
        res.slots,
        rt.ctx,
    )
    res = fhe.homo_mul_pt(res, weight, rt.ctx)
    res = fhe.rescale_one_level(res, rt.ctx)
    res = sum_channel_groups(res, spatial_size, channels, rt.ctx)

    bias = rt.weights.plaintext(
        f"bias_{res.slots}",
        rt.ctx.L - res.state.cur_limbs,
        res.slots,
        rt.ctx,
    )
    return fhe.homo_add_pt(res, bias, rt.ctx)


def encrypt_input(image_vector, rt):
    return rt.client.encrypt(
        image_vector,
        device=rt.ctx.device,
        scale_deg=1,
        level=rt.config.input_level,
        slots=16 * 32 * 32,
    )


def infer_encrypted(input_cipher, rt):
    first_layer = initial_layer(input_cipher, rt)
    res_layer1 = layer1(first_layer, rt)
    res_layer2 = layer2(res_layer1, rt)
    res_layer3 = layer3(res_layer2, rt)
    return final_layer(res_layer3, rt)


def infer_one(image_vector, rt):
    return infer_encrypted(encrypt_input(image_vector, rt), rt)


__all__ = [
    "AespaRuntime",
    "encrypt_input",
    "final_layer",
    "infer_encrypted",
    "infer_one",
    "initial_layer",
    "layer1",
    "layer2",
    "layer3",
]
