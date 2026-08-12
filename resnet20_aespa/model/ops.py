"""u64 CKKS operators used by the encrypted ResNet20 graph."""

import easyfhe.fhe as fhe

__all__ = [
    "aespa_add_shortcut",
    "aespa_nonlinear",
    "conv3x3",
    "initial_conv3x3",
    "pointwise_conv",
]


def _conv3x3_offsets(image_width):
    return [
        -image_width - 1,
        -image_width,
        -image_width + 1,
        -1,
        0,
        1,
        image_width - 1,
        image_width,
        image_width + 1,
    ]


def _mul_plaintext_for_cipher(weights, name, cipher, context, *, slots=None):
    if int(cipher.state.scale_degree) != 1:
        raise ValueError(
            "plaintext multiplication requires normalized ciphertext input, "
            f"got {cipher.state}"
        )
    return weights.plaintext(
        name,
        state=cipher.state,
        slots=cipher.slots if slots is None else slots,
        context=context,
    )


def _add_plaintext_for_cipher(weights, name, cipher, context, *, slots=None):
    return weights.plaintext(
        name,
        state=cipher.state,
        slots=cipher.slots if slots is None else slots,
        context=context,
    )


def _state_after_square_rescale(cipher, context):
    output_limbs = int(cipher.state.cur_limbs) - 1
    return fhe.CipherState(
        cur_limbs=output_limbs,
        scale_degree=1,
        scaling_factor=(
            float(cipher.state.scaling_factor) ** 2
            / float(context.rescale_divisor_at(output_limbs))
        ),
    )


def conv3x3(
    input_cipher,
    kernel_group,
    image_width,
    rotation_offset,
    context,
    weights,
    *,
    group_size,
):
    input_cipher = fhe.normalize_scale(input_cipher, context)
    plaintexts = _mul_plaintext_for_cipher(
        weights, kernel_group, input_cipher, context
    )
    result = fhe.hoisted_mac_sum(
        input_cipher,
        _conv3x3_offsets(image_width),
        plaintexts,
        rotation_offset,
        int(group_size),
        context,
        strategy="normal",
    )
    return fhe.homo_rotate(result, rotation_offset, context)


def initial_conv3x3(
    input_cipher,
    kernel_group,
    image_width,
    rotation_offset,
    context,
    weights,
    *,
    group_size,
):
    input_cipher = fhe.normalize_scale(input_cipher, context)
    rotations = fhe.fast_rotate(
        input_cipher, _conv3x3_offsets(image_width), context
    )
    plaintexts = _mul_plaintext_for_cipher(
        weights, kernel_group, rotations, context
    )
    partial_sums = fhe.grouped_pairwise_mac_rescale(
        rotations, plaintexts, int(group_size), context
    )
    partial_sums = [
        _initial_conv_postprocess(partial_sum, context, weights)
        for partial_sum in fhe.unpack_cipher_batch(partial_sums)
    ]
    result = fhe.giant_rotate_sum(
        fhe.pack_cipher_batch(partial_sums),
        rotation_offset,
        context,
        strategy="normal",
    )
    return fhe.homo_rotate(result, rotation_offset, context)


def _initial_conv_postprocess(partial_sum, context, weights):
    base = partial_sum
    sum_rotation = fhe.homo_rotate(partial_sum, 1024, context)
    partial_sum = fhe.homo_rotate_add(
        sum_rotation, 1024, context, addend=sum_rotation
    )
    partial_sum = fhe.homo_add(base, partial_sum, context)
    return fhe.homo_mul_pt(
        partial_sum,
        _mul_plaintext_for_cipher(
            weights,
            f"mask_from_to_0_1024_{partial_sum.slots}",
            partial_sum,
            context,
        ),
        context,
    )


def pointwise_conv(
    input_cipher,
    kernel_group,
    bias_key,
    rotation_offset,
    context,
    weights,
    *,
    group_size,
):
    input_cipher = fhe.normalize_scale(input_cipher, context)
    # The group is stored in reverse channel order. giant_rotate_sum followed
    # by one final rotation is algebraically identical to the old per-channel
    # multiply/add/rotate recurrence, while executing it as one public batch.
    plaintexts = _mul_plaintext_for_cipher(
        weights, kernel_group, input_cipher, context
    )
    partial_sums = fhe.grouped_pairwise_mac(
        input_cipher, plaintexts, int(group_size), context
    )
    result = fhe.giant_rotate_sum(
        partial_sums, rotation_offset, context, strategy="normal"
    )
    result = fhe.homo_rotate(result, rotation_offset, context)
    result = fhe.normalize_scale(result, context)
    return fhe.homo_add_pt(
        result,
        _add_plaintext_for_cipher(weights, bias_key, result, context),
        context,
    )


def aespa_nonlinear(cipher, prefix, context, weights):
    cipher = fhe.normalize_scale(cipher, context)
    shifted = fhe.homo_add_pt(
        cipher,
        _add_plaintext_for_cipher(weights, f"{prefix}-n1", cipher, context),
        context,
    )
    output_state = _state_after_square_rescale(shifted, context)
    post_add = weights.plaintext(
        f"{prefix}-n2",
        state=output_state,
        slots=shifted.slots,
        context=context,
    )
    return fhe.homo_mul_relin_rescale_add_pt(
        shifted, shifted, post_add, context
    )


def aespa_add_shortcut(conv_out, shortcut, prefix, context, weights):
    shortcut = fhe.normalize_scale(shortcut, context)
    shortcut = fhe.align_to(
        shortcut,
        shortcut.state.replace(cur_limbs=conv_out.state.cur_limbs),
        context,
    )
    scaled_shortcut = fhe.homo_mul_pt(
        shortcut,
        _mul_plaintext_for_cipher(
            weights, f"{prefix}-A2", shortcut, context
        ),
        context,
    )
    return fhe.homo_add(conv_out, scaled_shortcut, context)
