"""Encrypted feed-forward, GELU, and second layer-normalization graph."""

import numpy as np

from easyfhe import fhe

from .approximations import (
    evaluate_inverse_sqrt,
    evaluate_polynomial_stockmeyer,
)
from ..fhe_ops import (
    CONJUGATE_ROTATION,
    SLOTS,
    add_aligned,
    align_encrypted_constant,
    bootstrap_cipher,
    add_double_scalar,
    complex_real_twice,
    grouped_mac_sum,
    imult,
    linear_weight_rescale,
    mult_double_scalar,
    mult_int_scalar_any,
    mul_relin_rescale,
    multiply_plain_vector,
    normalize_cipher_scale,
    plaintext_for_add,
    plaintext_for_mul,
    pt_ct_mult_any,
    rotate_internal,
    square_rescale,
)


def make_ff_dense1_input_copies(layernorm_out, ctx):
    """Merge LN real/imag pairs into the 64 cyclic input columns used by FF dense1."""
    mask = np.ones((SLOTS,), dtype=np.float64)
    mask[np.arange(SLOTS) % 16 >= 6] = 0
    copies = [None for _ in range(64)]
    for index in range(4):
        merged = fhe.homo_add(layernorm_out[index], imult(layernorm_out[index + 4], ctx), ctx)
        masked = multiply_plain_vector(merged, mask, f"ff.input_mask.{index}", ctx)
        copies[16 * index] = fhe.homo_add(masked, fhe.homo_rotate(masked, -8, ctx), ctx)
        for offset in range(1, 16):
            copies[16 * index + offset] = fhe.homo_rotate(copies[16 * index + offset - 1], 2**11, ctx)
    return copies


def ff_pt_ct_matmul(weights, masks, base: str, rep: int, input_copies, ctx):
    """Apply one FF packed diagonal matrix to already prepared cyclic input columns."""
    outputs = []
    for out in range(8):
        ct_sub = []
        for diag in range(6):
            ciphers = [input_copies[(16 * out + n) % 64] for n in range(64)]
            pt_names = [f"{base}__{rep}x{out}x{diag}x{n}" for n in range(64)]
            summed = grouped_mac_sum(
                ciphers,
                weights,
                pt_names,
                ctx,
                batch_name=f"{base}.grouped.{rep}.{out}.{diag}",
            )
            ct_sub.append(linear_weight_rescale(summed, ctx))

        current = ct_sub[0]
        for diag in range(1, 6):
            rotated = rotate_internal(ct_sub[diag], 6 - diag, "block_diag_2", masks, ctx)
            current = fhe.homo_add(fhe.align_to(current, rotated.state, ctx), rotated, ctx)
        outputs.append(current)
    return outputs


def ff_dense1(weights, masks, layernorm_out, ctx, *, layer: int = 0):
    weight_base = f"bert.encoder.layer.{layer}.intermediate.dense.weight"
    bias_base = f"bert.encoder.layer.{layer}.intermediate.dense.bias"
    input_copies = make_ff_dense1_input_copies(layernorm_out, ctx)
    reps = []
    for rep in range(2):
        wx = ff_pt_ct_matmul(weights, masks, weight_base, rep, input_copies, ctx)
        out = []
        for index, cipher in enumerate(wx):
            biased = fhe.homo_add_pt(
                cipher,
                plaintext_for_add(weights, f"{bias_base}__{rep}x{index}", cipher, ctx),
                ctx,
            )
            out.append(complex_real_twice(biased, ctx))
        reps.append(out)
    return reps


def make_ff_dense2_input_copies(gelu_rep, ctx):
    """Merge one GELU replica into the 64 cyclic input columns used by FF dense2."""
    copies = [None for _ in range(64)]
    for index in range(4):
        copies[16 * index] = fhe.homo_add(gelu_rep[index], imult(gelu_rep[index + 4], ctx), ctx)
        for offset in range(1, 16):
            copies[16 * index + offset] = fhe.homo_rotate(copies[16 * index + offset - 1], 2**11, ctx)
    return copies


def ff_dense2(weights, masks, gelu, ctx, *, layer: int = 0):
    weight_base = f"bert.encoder.layer.{layer}.output.dense.weight"
    bias_base = f"bert.encoder.layer.{layer}.output.dense.bias"
    wx = []
    for rep in range(2):
        input_copies = make_ff_dense2_input_copies(gelu[rep], ctx)
        wx.append(ff_pt_ct_matmul(weights, masks, weight_base, rep, input_copies, ctx))

    outputs = []
    for index in range(8):
        combined = add_aligned(wx[0][index], wx[1][index], ctx)
        combined = add_aligned(combined, fhe.homo_rotate(combined, 8, ctx), ctx)
        combined = fhe.homo_add_pt(
            combined,
            plaintext_for_add(weights, f"{bias_base}__{index}", combined, ctx),
            ctx,
        )
        outputs.append(complex_real_twice(combined, ctx))
    return outputs


def ff_residual_add_and_bootstrap(attention_ln, dense2, ctx, bootstrap_program):
    residual = []
    for index, cipher in enumerate(dense2):
        residual.append(add_aligned(attention_ln[index], cipher, ctx))

    output = [None for _ in range(8)]
    for index in range(4):
        merged = fhe.homo_add(residual[index], imult(residual[index + 4], ctx), ctx)
        merged = bootstrap_cipher(merged, ctx, bootstrap_program)
        conj = fhe.homo_rotate(merged, CONJUGATE_ROTATION, ctx)
        output[index] = fhe.homo_add(merged, conj, ctx)
        output[index + 4] = imult(fhe.homo_sub(conj, merged, ctx), ctx)
    return residual, output


def feed_forward_layernorm(
    encrypted_constants,
    residual,
    weights,
    ctx,
    bootstrap_program,
    *,
    layer: int = 0,
    min_var: float = 0.2,
    max_var: float = 150.0,
):
    var_e = 1e-5
    n = 768
    max_for_denominator = (max_var * 1.05 + var_e) * n**2
    norm_mask = np.asarray(([1 / np.sqrt(max_for_denominator)] * 6 + [0.0] * 10) * 2**11, dtype=np.float64) / 2
    first_slot_mask = np.asarray(([1.0] + [0.0] * 15) * 2**11, dtype=np.float64)

    enc_l = [multiply_plain_vector(cipher, norm_mask, f"ln2.norm_mask.{index}", ctx) for index, cipher in enumerate(residual)]

    sum_x = enc_l[0]
    for cipher in enc_l[1:]:
        sum_x = fhe.homo_add(sum_x, cipher, ctx)
    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, 2**11, ctx), ctx)
    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, 2**12, ctx), ctx)
    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, 2**13, ctx), ctx)
    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, 2**14, ctx), ctx)
    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, 1, ctx), ctx)
    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, 2, ctx), ctx)
    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, 4, ctx), ctx)
    sum_x = multiply_plain_vector(sum_x, first_slot_mask, "ln2.sum_first_slot", ctx)
    sq_sum_x = square_rescale(sum_x, ctx)
    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, -1, ctx), ctx)
    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, -2, ctx), ctx)
    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, -4, ctx), ctx)

    numerator = []
    for cipher in enc_l:
        nx = mult_int_scalar_any(cipher, n, ctx)
        numerator.append(fhe.homo_sub(fhe.align_to(nx, sum_x.state, ctx), sum_x, ctx))

    sigma_x2 = square_rescale(enc_l[0], ctx)
    for cipher in enc_l[1:]:
        sigma_x2 = fhe.homo_add(sigma_x2, square_rescale(cipher, ctx), ctx)
    sigma_x2 = fhe.homo_add(sigma_x2, fhe.homo_rotate(sigma_x2, 2**11, ctx), ctx)
    sigma_x2 = fhe.homo_add(sigma_x2, fhe.homo_rotate(sigma_x2, 2**12, ctx), ctx)
    sigma_x2 = fhe.homo_add(sigma_x2, fhe.homo_rotate(sigma_x2, 2**13, ctx), ctx)
    sigma_x2 = fhe.homo_add(sigma_x2, fhe.homo_rotate(sigma_x2, 2**14, ctx), ctx)
    sigma_x2 = fhe.homo_add(sigma_x2, fhe.homo_rotate(sigma_x2, 1, ctx), ctx)
    sigma_x2 = fhe.homo_add(sigma_x2, fhe.homo_rotate(sigma_x2, 2, ctx), ctx)
    sigma_x2 = fhe.homo_add(sigma_x2, fhe.homo_rotate(sigma_x2, 4, ctx), ctx)
    sigma_x2 = multiply_plain_vector(sigma_x2, first_slot_mask, "ln2.sigma_first_slot", ctx)

    variance = fhe.homo_sub(mult_int_scalar_any(sigma_x2, n, ctx), fhe.align_to(sq_sum_x, sigma_x2.state, ctx), ctx)
    variance = add_double_scalar(variance, var_e / max_for_denominator, ctx)

    enc_one = align_encrypted_constant(
        encrypted_constants.layernorm_one,
        variance,
        ctx,
    )
    denominator = evaluate_inverse_sqrt(
        enc_one,
        variance,
        min_var / max_var,
        0.001,
        first_slot_mask,
        ctx,
        bootstrap_program,
    )
    if denominator.state.cur_limbs < 8:
        denominator = bootstrap_cipher(denominator, ctx, bootstrap_program)

    denominator = fhe.homo_add(denominator, fhe.homo_rotate(denominator, -1, ctx), ctx)
    denominator = fhe.homo_add(denominator, fhe.homo_rotate(denominator, -2, ctx), ctx)
    denominator = fhe.homo_add(denominator, fhe.homo_rotate(denominator, -4, ctx), ctx)

    gamma_base = f"bert.encoder.layer.{layer}.output.LayerNorm.weight"
    beta_base = f"bert.encoder.layer.{layer}.output.LayerNorm.bias"
    output = []
    for index, cipher in enumerate(numerator):
        gamma_den = fhe.rescale(
            pt_ct_mult_any(plaintext_for_mul(weights, f"{gamma_base}__{index}", denominator, ctx), denominator, ctx),
            ctx,
        )
        ln = mul_relin_rescale(cipher, gamma_den, ctx)
        ln = fhe.homo_add_pt(ln, plaintext_for_add(weights, f"{beta_base}__{index}", ln, ctx), ctx)
        output.append(fhe.homo_add(ln, ln, ctx))
    return output


def bootstrap_dense1_pairs(dense1, ctx, bootstrap_program):
    output = [[None for _ in range(8)] for _ in range(2)]
    for index in range(8):
        merged = fhe.homo_add(dense1[0][index], imult(dense1[1][index], ctx), ctx)
        if int(merged.state.cur_limbs) <= 1:
            merged = bootstrap_cipher(merged, ctx, bootstrap_program)
            merged = mult_double_scalar(merged, 0.5, ctx)
        else:
            merged = mult_double_scalar(merged, 0.5, ctx)
            merged = bootstrap_cipher(merged, ctx, bootstrap_program)
        conj = fhe.homo_rotate(merged, CONJUGATE_ROTATION, ctx)
        output[0][index] = fhe.homo_add(merged, conj, ctx)
        output[1][index] = imult(fhe.homo_sub(conj, merged, ctx), ctx)
    return output


def _evaluate_tanh(cipher, ctx):
    p1 = [
        -1.06240033e-05, 1.64454894e-04, -5.83533517e-04, -3.80912692e-04,
        2.24431193e-03, 8.92295204e-03, -1.05277477e-02, -1.91827040e-02,
        -2.04634786e-01, 4.54014410e-01, -5.40759203e-01, 5.67745523e00,
        -1.36433727e01, 1.82574621e01, -8.48849601e01, 1.28686741e02,
        3.66720281e02, -1.01400159e03, -1.26278856e02, 2.21728878e03,
        -9.95421415e02, -2.31059465e03, 1.73583957e03, 1.27394360e03,
        -1.27836230e03, -3.66781716e02, 4.79663919e02, 4.94610178e01,
        -9.06754761e01, -2.36515790e00, 8.74311855e00, 1.62838703e-02,
    ]
    p2 = [
        -1.70270667e02, 6.81076279e01, 1.79197364e03, -6.81621043e02,
        -8.49256169e03, 3.05629446e03, 2.39579397e04, -8.10435126e03,
        -4.48145152e04, 1.41297616e04, 5.86197512e04, -1.70371505e04,
        -5.51326382e04, 1.45532495e04, 3.77866438e04, -8.87673890e03,
        -1.89514802e04, 3.84972853e03, 6.94169727e03, -1.16901058e03,
        -1.84658407e03, 2.41693754e02, 3.54452276e02, -3.24499570e01,
        -4.91918227e01, 2.58122977e00, 5.78392852e00, -9.45171527e-02,
    ]
    p1.reverse()
    p2.reverse()
    p1_x = evaluate_polynomial_stockmeyer(np.asarray(p1), cipher, ctx)
    return evaluate_polynomial_stockmeyer(np.asarray(p2) * 0.5, p1_x, ctx)


def gelu(dense1_bs, ctx):
    gelu = [[None for _ in range(8)] for _ in range(2)]
    for rep in range(2):
        for index in range(4):
            x0 = dense1_bs[rep][index]
            x1 = dense1_bs[rep][index + 4]
            tanh_x0 = _evaluate_tanh(x0, ctx)
            tanh_x1 = _evaluate_tanh(x1, ctx)
            one_plus_tanh_x0 = add_double_scalar(tanh_x0, 0.5, ctx)
            one_plus_tanh_x1 = add_double_scalar(tanh_x1, 0.5, ctx)
            x0_scaled = mult_int_scalar_any(x0, 64, ctx)
            x1_scaled = mult_int_scalar_any(x1, 64, ctx)
            gelu[rep][index] = normalize_cipher_scale(
                mul_relin_rescale(x0_scaled, one_plus_tanh_x0, ctx),
                ctx,
            )
            gelu[rep][index + 4] = normalize_cipher_scale(
                mul_relin_rescale(x1_scaled, one_plus_tanh_x1, ctx),
                ctx,
            )
    return gelu
