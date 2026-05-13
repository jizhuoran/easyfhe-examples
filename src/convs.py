import easyfhe.fhe as fhe

__all__ = [
    "aespa_add_shortcut",
    "aespa_nonlinear",
    "conv3x3",
    "initial_conv3x3",
    "pointwise_conv",
]


def _pairwise_mac(ctxs, ptxs, crypto_context):
    if len(ctxs) != len(ptxs) or len(ctxs) == 0:
        raise ValueError(f"ctxs and ptxs must have the same non-zero length, but got {len(ctxs)} and {len(ptxs)}")

    partial_sum = fhe.homo_mul_pt(ctxs[0], ptxs[0], crypto_context)
    for ctx, ptx in zip(ctxs[1:], ptxs[1:]):
        partial_sum = fhe.homo_add(partial_sum, fhe.homo_mul_pt(ctx, ptx, crypto_context), crypto_context)
    return partial_sum


def _read_kernel_rows(prefix, cipher, scale, crypto_context, weights):
    level = crypto_context.L - cipher.cur_limbs
    return [
        weights.plaintext(f"{prefix}-k{k + 1}", level, cipher.slots, crypto_context, scale)
        for k in range(9)
    ]


def _rot_input(input, img_width, padding, crypto_context):
    digits = fhe.modup_to_ext(input.cipher_like([input.cv[1]]), crypto_context)
    digits_neg_padding = fhe.eval_fast_rotate(digits, input, -padding, True, True, crypto_context)
    digits_padding = fhe.eval_fast_rotate(digits, input, padding, True, True, crypto_context)
    digits_neg_img_width = fhe.eval_fast_rotate(digits, input, -img_width, True, True, crypto_context)
    digits_img_width = fhe.eval_fast_rotate(digits, input, img_width, True, True, crypto_context)

    return [
        fhe.homo_rotate(digits_neg_padding, -img_width, crypto_context),
        digits_neg_img_width,
        fhe.homo_rotate(digits_padding, -img_width, crypto_context),
        digits_neg_padding,
        input,
        digits_padding,
        fhe.homo_rotate(digits_neg_padding, img_width, crypto_context),
        digits_img_width,
        fhe.homo_rotate(digits_padding, img_width, crypto_context),
    ]


def _conv3x3(input, prefixes, img_width, padding, rot_offset, scale, crypto_context, weights, postprocess=None):
    if not prefixes:
        raise ValueError("conv3x3 requires at least one kernel prefix")

    input = fhe.reduce_noise_to_one(input, crypto_context)
    rotations = _rot_input(input, img_width, padding, crypto_context)

    for idx, prefix in enumerate(prefixes):
        partial_sum = _pairwise_mac(rotations, _read_kernel_rows(prefix, input, scale, crypto_context, weights), crypto_context)
        if postprocess is not None:
            partial_sum = postprocess(partial_sum)
        finalsum = partial_sum.deep_copy() if idx == 0 else fhe.homo_add(finalsum, partial_sum, crypto_context)
        finalsum = fhe.homo_rotate(finalsum, rot_offset, crypto_context)
    return finalsum


def _initial_conv_postprocess(partial_sum, crypto_context, weights):
    partial_sum = fhe.rescale_one_level(partial_sum, crypto_context)
    sum_rot = fhe.homo_rotate(partial_sum, 1024, crypto_context)
    partial_sum = fhe.homo_add(partial_sum, sum_rot, crypto_context)
    partial_sum = fhe.homo_add(partial_sum, fhe.homo_rotate(sum_rot, 1024, crypto_context), crypto_context)
    return fhe.homo_mul_pt(
        partial_sum,
        weights.plaintext(
            f"mask_from_to_0_1024_{partial_sum.slots}",
            crypto_context.L - partial_sum.cur_limbs,
            partial_sum.slots,
            crypto_context,
        ),
        crypto_context,
    )


def initial_conv3x3(input, kernel_prefixes, img_width, padding, rot_offset, scale, crypto_context, weights):
    return _conv3x3(
        input,
        kernel_prefixes,
        img_width,
        padding,
        rot_offset,
        scale,
        crypto_context,
        weights,
        lambda partial_sum: _initial_conv_postprocess(partial_sum, crypto_context, weights),
    )


def conv3x3(input, kernel_prefixes, img_width, padding, rot_offset, scale, crypto_context, weights):
    return _conv3x3(
        input,
        kernel_prefixes,
        img_width,
        padding,
        rot_offset,
        scale,
        crypto_context,
        weights,
    )


def pointwise_conv(input, kernel_keys, bias_key, rot_offset, scale, crypto_context, weights):
    if not kernel_keys:
        raise ValueError("pointwise_conv requires at least one kernel key")

    input = fhe.reduce_noise_to_one(input, crypto_context)

    for idx, kernel_key in enumerate(kernel_keys):
        encoded = weights.plaintext_for_cipher(kernel_key, input, crypto_context, scale)
        partial_sum = fhe.homo_mul_pt(input, encoded, crypto_context)
        finalsum = partial_sum.deep_copy() if idx == 0 else fhe.homo_add(finalsum, partial_sum, crypto_context)
        finalsum = fhe.homo_rotate(finalsum, rot_offset, crypto_context)

    finalsum = fhe.rescale_one_level(finalsum, crypto_context)
    bias = weights.plaintext_for_cipher(bias_key, finalsum, crypto_context, scale)
    return fhe.homo_add_pt(finalsum, bias, crypto_context)


def aespa_nonlinear(x, prefix, crypto_context, weights, scale=1):
    x = fhe.reduce_noise_to_one(x, crypto_context)
    n1 = weights.plaintext_for_cipher(f"{prefix}-n1", x, crypto_context, scale)
    shifted = fhe.homo_add_pt(x, n1, crypto_context)
    squared = fhe.homo_square(shifted, crypto_context)
    squared = fhe.rescale_one_level(squared, crypto_context)
    n2 = weights.plaintext_for_cipher(f"{prefix}-n2", squared, crypto_context, scale)
    return fhe.homo_add_pt(squared, n2, crypto_context)


def aespa_add_shortcut(conv_out, shortcut, prefix, crypto_context, weights, scale=1):
    if crypto_context.rescaleTech == "FIXEDMANUAL":
        shortcut = fhe.align_to(
            shortcut,
            fhe.CipherState(shortcut.cur_limbs - (shortcut.cur_limbs - conv_out.cur_limbs), shortcut.noise_deg),
            crypto_context,
        )
    a2 = weights.plaintext_for_cipher(f"{prefix}-A2", shortcut, crypto_context, scale)
    return fhe.homo_add(conv_out, fhe.homo_mul_pt(shortcut, a2, crypto_context), crypto_context)
