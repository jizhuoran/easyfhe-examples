import easyfhe.fhe as fhe

__all__ = [
    "aespa_add_shortcut",
    "aespa_nonlinear",
    "conv3x3",
    "initial_conv3x3",
    "pointwise_conv",
]


def _conv3x3_offsets(img_width, padding):
    return [
        -img_width - padding,
        -img_width,
        -img_width + padding,
        -padding,
        0,
        padding,
        img_width - padding,
        img_width,
        img_width + padding,
    ]


def conv3x3(
    input,
    kernel_group,
    img_width,
    padding,
    rot_offset,
    scale,
    cryptoContext,
    weights,
    *,
    group_size,
    final_rotate=None,
):
    input = fhe.reduce_noise_to_one(input, cryptoContext)
    plaintexts = weights.plaintext(
        kernel_group,
        cryptoContext.L - input.state.cur_limbs,
        input.slots,
        cryptoContext,
        weights.scalar_value(scale) if isinstance(scale, str) else scale,
    )
    group_size = int(group_size)
    final_rotate = rot_offset if final_rotate is None else int(final_rotate)
    result = fhe.hoisted_mac_sum(
        input,
        _conv3x3_offsets(img_width, padding),
        plaintexts,
        rot_offset,
        group_size,
        cryptoContext,
        strategy=fhe.HOIST_NORMAL,
    )
    return fhe.homo_rotate(result, final_rotate, cryptoContext)


def initial_conv3x3(
    input,
    kernel_group,
    img_width,
    padding,
    rot_offset,
    scale,
    cryptoContext,
    weights,
    *,
    group_size,
):
    input = fhe.reduce_noise_to_one(input, cryptoContext)
    rotations = fhe.fast_rotate(input, _conv3x3_offsets(img_width, padding), cryptoContext)
    plaintexts = weights.plaintext(
        kernel_group,
        cryptoContext.L - rotations.state.cur_limbs,
        rotations.slots,
        cryptoContext,
        weights.scalar_value(scale) if isinstance(scale, str) else scale,
    )
    partial_sums = fhe.grouped_pairwise_mac(
        rotations,
        plaintexts,
        int(group_size),
        cryptoContext,
    )
    partial_sums = [
        _initial_conv_postprocess(partial_sum, cryptoContext, weights)
        for partial_sum in fhe.unpack_cipher_batch(partial_sums)
    ]
    result = fhe.giant_rotate_sum(fhe.pack_cipher_batch(partial_sums), rot_offset, cryptoContext, strategy=fhe.HOIST_NORMAL)
    return fhe.homo_rotate(result, rot_offset, cryptoContext)


def _initial_conv_postprocess(partial_sum, cryptoContext, weights):
    partial_sum = fhe.rescale_one_level(partial_sum, cryptoContext)
    base = partial_sum
    sum_rot = fhe.homo_rotate(partial_sum, 1024, cryptoContext)
    partial_sum = fhe.homo_rotate_add(sum_rot, 1024, cryptoContext, addend=sum_rot)
    partial_sum = fhe.homo_add(base, partial_sum, cryptoContext)
    return fhe.homo_mul_pt(
        partial_sum,
        weights.plaintext(
            f"mask_from_to_0_1024_{partial_sum.slots}",
            cryptoContext.L - partial_sum.state.cur_limbs,
            partial_sum.slots,
            cryptoContext,
        ),
        cryptoContext,
    )


def pointwise_conv(
    input,
    kernel_group,
    bias_key,
    rot_offset,
    scale,
    cryptoContext,
    weights,
    *,
    group_size=None,
    final_rotate=None,
):
    input = fhe.reduce_noise_to_one(input, cryptoContext)
    plaintexts = weights.plaintext(
        kernel_group,
        cryptoContext.L - input.state.cur_limbs,
        input.slots,
        cryptoContext,
        weights.scalar_value(scale) if isinstance(scale, str) else scale,
    )
    group_size = plaintexts.batch_size if group_size is None else int(group_size)
    final_rotate = rot_offset if final_rotate is None else int(final_rotate)
    input_batch = input.cipher_like(input.cv, batch_size=1)
    partial_sums = fhe.grouped_pairwise_mac(input_batch, plaintexts, group_size, cryptoContext)
    finalsum = fhe.giant_rotate_sum(partial_sums, rot_offset, cryptoContext, strategy=fhe.HOIST_NORMAL)
    finalsum = fhe.homo_rotate(finalsum, final_rotate, cryptoContext)
    finalsum = fhe.rescale_one_level(finalsum, cryptoContext)
    bias = weights.plaintext(
        bias_key,
        cryptoContext.L - finalsum.state.cur_limbs,
        finalsum.slots,
        cryptoContext,
        weights.scalar_value(scale) if isinstance(scale, str) else scale,
    )
    return fhe.homo_add_pt(finalsum, bias, cryptoContext)


def aespa_nonlinear(x, prefix, cryptoContext, weights, scale=1):
    x = fhe.reduce_noise_to_one(x, cryptoContext)
    n1 = weights.plaintext(
        f"{prefix}-n1",
        cryptoContext.L - x.state.cur_limbs,
        x.slots,
        cryptoContext,
        weights.scalar_value(scale) if isinstance(scale, str) else scale,
    )
    shifted = fhe.homo_add_pt(x, n1, cryptoContext)
    out_cur_limbs = shifted.state.cur_limbs - 1
    n2 = weights.plaintext(
        f"{prefix}-n2",
        cryptoContext.L - out_cur_limbs,
        shifted.slots,
        cryptoContext,
        weights.scalar_value(scale) if isinstance(scale, str) else scale,
    )
    return fhe.homo_mul_relin_rescale_add_pt(shifted, shifted, n2, cryptoContext)


def aespa_add_shortcut(conv_out, shortcut, prefix, cryptoContext, weights, scale=1):
    # if cryptoContext.rescale_policy == "manual":
    shortcut = fhe.align_to(
            shortcut,
            fhe.CipherState(shortcut.state.cur_limbs - (shortcut.state.cur_limbs - conv_out.state.cur_limbs), shortcut.state.noise_deg),
            cryptoContext,
        )
    a2 = weights.plaintext(
        f"{prefix}-A2",
        cryptoContext.L - shortcut.state.cur_limbs,
        shortcut.slots,
        cryptoContext,
        weights.scalar_value(scale) if isinstance(scale, str) else scale,
    )
    return fhe.homo_add(conv_out, fhe.homo_mul_pt(shortcut, a2, cryptoContext), cryptoContext)
