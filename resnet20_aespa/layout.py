import math

import easyfhe.fhe as fhe

from .weight_pack import fold_slots_mask_name

__all__ = [
    "broadcast_slot_sum",
    "downsample1024to256",
    "downsample256to64_pair",
    "sum_adjacent_slots",
    "sum_channel_groups",
]


# Downsample layouts


def downsample1024to256(fullpack, num_channel, cryptoContext, weights):
    fullpack = fhe.reduce_noise_to_one(fullpack, cryptoContext)
    fullpack = _masked_reduce(fullpack, 2, 1, cryptoContext, weights)
    fullpack = _masked_reduce(fullpack, 4, 2, cryptoContext, weights)
    fullpack = _masked_reduce(fullpack, 8, 4, cryptoContext, weights)
    fullpack = fhe.homo_rotate_add(fullpack, 8, cryptoContext, addend=fullpack)
    rows = _pack_rows(
        fullpack,
        "mask_first_n_mod",
        16,
        1024,
        16,
        48,
        cryptoContext,
        weights,
    )
    channels = _pack_channels(rows, num_channel, 1024, 256, cryptoContext, weights)
    channels = fhe.rescale_one_level(channels, cryptoContext)
    channels = _fold_quarters(channels, cryptoContext)
    return _fold_downsample_slots(channels, cryptoContext, weights)


def downsample256to64_pair(c1, c2, num_channel, cryptoContext, weights):
    merged = _append_second_half(c1, c2, cryptoContext, weights)
    channels = _downsample256to64_channels(merged, num_channel, cryptoContext, weights)
    channels = _fold_quarters(channels, cryptoContext, quarter_slots=channels.slots // 8)
    channels = fhe.rescale_one_level(channels, cryptoContext)
    return _fold_merged_downsample_halves(channels, cryptoContext, weights)


# Final-layer reductions


def sum_adjacent_slots(input, slots, cryptoContext):
    _require_power_of_two(slots, "slots")
    result = input.deep_copy()
    for i in range(int(math.log2(slots))):
        result = fhe.homo_rotate_add(result, 2 ** i, cryptoContext, addend=result)
    return result


def sum_channel_groups(input, group_size, num_groups, cryptoContext):
    _require_power_of_two(num_groups, "num_groups")
    result = input.deep_copy()
    for i in range(int(math.log2(num_groups))):
        result = fhe.homo_rotate_add(result, group_size * (2 ** i), cryptoContext, addend=result)
    return result


def broadcast_slot_sum(input, slots, cryptoContext):
    return fhe.homo_rotate(sum_adjacent_slots(input, slots, cryptoContext), -slots + 1, cryptoContext)


# Downsample helpers


def _downsample256to64_channels(fullpack, num_channel, cryptoContext, weights):
    fullpack = fhe.reduce_noise_to_one(fullpack, cryptoContext)
    fullpack = _masked_reduce(fullpack, 2, 1, cryptoContext, weights)
    fullpack = _masked_reduce(fullpack, 4, 2, cryptoContext, weights)
    fullpack = fhe.homo_rotate_add(fullpack, 4, cryptoContext, addend=fullpack)
    rows = _pack_rows(
        fullpack,
        "mask_first_n_mod2",
        8,
        256,
        32,
        24,
        cryptoContext,
        weights,
    )
    return _pack_channels(rows, num_channel, 256, 64, cryptoContext, weights)


def _append_second_half(c1, c2, cryptoContext, weights):
    source_slots = int(c1.slots)
    target_slots = source_slots * 2
    c1 = fhe.expand_slots(c1, target_slots, cryptoContext)
    c2 = fhe.expand_slots(c2, target_slots, cryptoContext)
    c2 = fhe.homo_rotate(c2, -source_slots, cryptoContext)
    return fhe.homo_add(
        fhe.homo_mul_pt(
            c1,
            weights.plaintext(
                f"mask_first_n_{source_slots}_{target_slots}",
                cryptoContext.L - c1.state.cur_limbs,
                c1.slots,
                cryptoContext,
            ),
            cryptoContext,
        ),
        fhe.homo_mul_pt(
            c2,
            weights.plaintext(
                f"mask_second_n_{source_slots}_{target_slots}",
                cryptoContext.L - c2.state.cur_limbs,
                c2.slots,
                cryptoContext,
            ),
            cryptoContext,
        ),
        cryptoContext,
    )


def _fold_merged_downsample_halves(cipher, cryptoContext, weights):
    half_slots = cipher.slots // 2
    target_slots = half_slots // 4
    mask = weights.plaintext(
        fold_slots_mask_name(cipher.slots, target_slots),
        cryptoContext.L - cipher.state.cur_limbs,
        cipher.slots,
        cryptoContext,
    )
    first = fhe.fold_slots(cipher, target_slots, cryptoContext, mask=mask)
    second = fhe.homo_rotate(cipher, half_slots, cryptoContext)
    second = fhe.fold_slots(second, target_slots, cryptoContext, mask=mask)
    return first, second


def _masked_reduce(cipher, mask_n, rotate_offset, cryptoContext, weights):
    summed = fhe.homo_rotate_add(cipher, rotate_offset, cryptoContext, addend=cipher)
    cipher = fhe.homo_mul_pt(
        summed,
        weights.plaintext(
            f"gen_mask_{mask_n}_{cipher.slots}",
            cryptoContext.L - cipher.state.cur_limbs,
            cipher.slots,
            cryptoContext,
        ),
        cryptoContext,
    )
    return fhe.rescale_one_level(cipher, cryptoContext)


def _pack_rows(fullpack, row_mask_prefix, row_width, spatial_size, row_count, row_rotate, cryptoContext, weights):
    rows = None
    for i in range(row_count):
        masked = fhe.homo_mul_pt(
            fullpack,
            weights.plaintext(
                f"{row_mask_prefix}_{row_width}_{spatial_size}_{i}_{fullpack.slots}",
                cryptoContext.L - fullpack.state.cur_limbs,
                fullpack.slots,
                cryptoContext,
            ),
            cryptoContext,
        )
        rows = masked if rows is None else fhe.homo_add(rows, masked, cryptoContext)
        if i < row_count - 1:
            fullpack = fhe.homo_rotate(fullpack, row_rotate, cryptoContext)
    return fhe.rescale_one_level(rows, cryptoContext)


def _pack_channels(rows, num_channel, spatial_size, out_spatial_size, cryptoContext, weights):
    channels = None
    for i in range(num_channel * 2):
        masked = fhe.homo_mul_pt(
            rows,
            weights.plaintext(
                f"mask_channel_{i}_{num_channel}_{spatial_size}",
                cryptoContext.L - rows.state.cur_limbs,
                rows.slots,
                cryptoContext,
            ),
            cryptoContext,
        )
        channels = masked if channels is None else fhe.homo_add(channels, masked, cryptoContext)
        channels = fhe.homo_rotate(channels, -(spatial_size - out_spatial_size), cryptoContext)
    return fhe.homo_rotate(channels, num_channel * 2 * (spatial_size - out_spatial_size), cryptoContext)


def _fold_quarters(cipher, cryptoContext, *, quarter_slots=None):
    quarter = cipher.slots // 4 if quarter_slots is None else int(quarter_slots)
    cipher = fhe.homo_rotate_add(cipher, -quarter, cryptoContext, addend=cipher)
    rotated = fhe.homo_rotate(cipher, -quarter, cryptoContext)
    cipher = fhe.homo_rotate_add(rotated, -quarter, cryptoContext, addend=cipher)
    return cipher


def _fold_downsample_slots(channels, cryptoContext, weights):
    target_slots = channels.slots // 4
    mask = weights.plaintext(
        fold_slots_mask_name(channels.slots, target_slots),
        cryptoContext.L - channels.state.cur_limbs,
        channels.slots,
        cryptoContext,
    )
    return fhe.fold_slots(channels, target_slots, cryptoContext, mask=mask)


# Validation


def _require_power_of_two(value, name):
    if value <= 0 or value & (value - 1) != 0:
        raise ValueError(f"{name} must be a positive power of two, got {value}")
