"""Ciphertext slot-layout transformations for the u64 ResNet20 graph."""

import math

import easyfhe.fhe as fhe

from .ops import _mul_plaintext_for_cipher

__all__ = [
    "broadcast_slot_sum",
    "downsample1024to256_pair",
    "downsample256to64",
    "sum_adjacent_slots",
    "sum_channel_groups",
]


def downsample1024to256_pair(c1, c2, channel_count, context, weights):
    return _downsample_spatial_pair(
        c1,
        c2,
        channel_count,
        context,
        weights,
        row_mask_prefix="mask_first_n_mod",
        row_width=16,
        spatial_size=1024,
        output_spatial_size=256,
        row_count=16,
        row_rotation=48,
        include_gen8=True,
    )


def downsample256to64(c1, c2, channel_count, context, weights):
    return _downsample_spatial_pair(
        c1,
        c2,
        channel_count,
        context,
        weights,
        row_mask_prefix="mask_first_n_mod2",
        row_width=8,
        spatial_size=256,
        output_spatial_size=64,
        row_count=32,
        row_rotation=24,
        include_gen8=False,
    )


def sum_adjacent_slots(input_cipher, slots, context):
    _require_power_of_two(slots, "slots")
    result = input_cipher
    for level in range(int(math.log2(slots))):
        result = fhe.homo_rotate_add(
            result, 1 << level, context, addend=result
        )
    return result


def sum_channel_groups(input_cipher, group_size, group_count, context):
    _require_power_of_two(group_count, "group_count")
    result = input_cipher
    for level in range(int(math.log2(group_count))):
        result = fhe.homo_rotate_add(
            result, group_size * (1 << level), context, addend=result
        )
    return result


def broadcast_slot_sum(input_cipher, slots, context):
    return fhe.homo_rotate(
        sum_adjacent_slots(input_cipher, slots, context),
        -slots + 1,
        context,
    )


def _merge_fullpack_pair(c1, c2, context, weights):
    c1 = fhe.normalize_scale(c1, context)
    c2 = fhe.normalize_scale(c2, context)
    source_slots = int(c1.slots)
    target_slots = source_slots * 2
    c1 = fhe.expand_slots(c1, target_slots, context)
    c2 = fhe.expand_slots(c2, target_slots, context)
    merged = fhe.homo_add(
        fhe.homo_mul_pt(
            c1,
            _mul_plaintext_for_cipher(
                weights,
                f"mask_first_n_{source_slots}_{target_slots}",
                c1,
                context,
            ),
            context,
        ),
        fhe.homo_mul_pt(
            c2,
            _mul_plaintext_for_cipher(
                weights,
                f"mask_second_n_{source_slots}_{target_slots}",
                c2,
                context,
            ),
            context,
        ),
        context,
    )
    return fhe.normalize_scale(merged, context)


def _downsample_spatial_pair(
    c1,
    c2,
    channel_count,
    context,
    weights,
    *,
    row_mask_prefix,
    row_width,
    spatial_size,
    output_spatial_size,
    row_count,
    row_rotation,
    include_gen8,
):
    packed = _merge_fullpack_pair(c1, c2, context, weights)
    packed = _masked_reduce(packed, 2, 1, context, weights)
    packed = _masked_reduce(packed, 4, 2, context, weights)
    if include_gen8:
        packed = _masked_reduce(packed, 8, 4, context, weights)
        packed = fhe.homo_rotate_add(packed, 8, context, addend=packed)
    else:
        packed = fhe.homo_rotate_add(packed, 4, context, addend=packed)

    rows = _pack_rows(
        packed,
        row_mask_prefix,
        row_width,
        spatial_size,
        row_count,
        row_rotation,
        context,
        weights,
    )
    channels = _pack_channels(
        rows,
        channel_count,
        spatial_size,
        output_spatial_size,
        context,
        weights,
    )
    channels = _fold_quarters(channels, context)
    return _fold_downsample_slots(channels, context, weights)


def _masked_reduce(cipher, mask_width, rotation_offset, context, weights):
    summed = fhe.homo_rotate_add(
        cipher, rotation_offset, context, addend=cipher
    )
    mask = _mul_plaintext_for_cipher(
        weights, f"gen_mask_{mask_width}_{cipher.slots}", summed, context
    )
    return fhe.homo_mul_pt_rescale(summed, mask, context)


def _pack_rows(
    packed,
    row_mask_prefix,
    row_width,
    spatial_size,
    row_count,
    row_rotation,
    context,
    weights,
):
    # Use one repeated row-rotation key instead of generating one key for every
    # absolute row offset. This keeps the canonical key set compact.
    rows = None
    for row in range(row_count):
        masked = fhe.homo_mul_pt(
            packed,
            _mul_plaintext_for_cipher(
                weights,
                f"{row_mask_prefix}_{row_width}_{spatial_size}_{row}_{packed.slots}",
                packed,
                context,
            ),
            context,
        )
        rows = masked if rows is None else fhe.homo_add(rows, masked, context)
        if row < row_count - 1:
            packed = fhe.homo_rotate(packed, row_rotation, context)
    return fhe.normalize_scale(rows, context)


def _pack_channels(
    rows,
    channel_count,
    spatial_size,
    output_spatial_size,
    context,
    weights,
):
    # This is likewise a sequential one-key schedule, intentionally trading a
    # few rotations for much smaller key residency.
    channels = None
    rotation = -(spatial_size - output_spatial_size)
    for channel in range(channel_count * 2):
        masked = fhe.homo_mul_pt(
            rows,
            _mul_plaintext_for_cipher(
                weights,
                f"mask_channel_{channel}_{channel_count}_{spatial_size}",
                rows,
                context,
            ),
            context,
        )
        channels = (
            masked if channels is None else fhe.homo_add(channels, masked, context)
        )
        channels = fhe.homo_rotate(channels, rotation, context)
    channels = fhe.homo_rotate(
        channels,
        channel_count * 2 * (spatial_size - output_spatial_size),
        context,
    )
    return fhe.normalize_scale(channels, context)


def _fold_quarters(cipher, context):
    quarter = cipher.slots // 4
    cipher = fhe.homo_rotate_add(cipher, -quarter, context, addend=cipher)
    rotated = fhe.homo_rotate(cipher, -quarter, context)
    return fhe.homo_rotate_add(rotated, -quarter, context, addend=cipher)


def _fold_downsample_slots(channels, context, weights):
    target_slots = channels.slots // 4
    mask = _mul_plaintext_for_cipher(
        weights,
        f"fold_slots_mask_{int(channels.slots)}to{int(target_slots)}",
        channels,
        context,
    )
    folded = fhe.fold_slots(channels, target_slots, context, mask=mask)
    return fhe.normalize_scale(folded, context)


def _require_power_of_two(value, name):
    if value <= 0 or value & (value - 1) != 0:
        raise ValueError(f"{name} must be a positive power of two, got {value}")
