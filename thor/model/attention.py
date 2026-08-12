import numpy as np

from easyfhe import fhe

from .approximations import (
    evaluate_inverse_sqrt,
    evaluate_polynomial_stockmeyer,
)
from ..fhe_ops import (
    CONJUGATE_ROTATION,
    SLOTS,
    add_double_scalar,
    align_encrypted_constant,
    batched_plaintext_from_names,
    bootstrap_cipher,
    complex_real_twice,
    conjugate,
    ct_ct_mult_triplet,
    _encode_int_for_scalar_op,
    _plaintext_scale,
    grouped_mac_sum,
    imult,
    linear_weight_rescale,
    make_rotated_copies,
    mult_double_scalar,
    mul_relin_rescale,
    mult_int_scalar_any,
    multiply_plain_vector,
    negate,
    normalize_cipher_scale,
    packed_name,
    plaintext_for_add,
    plaintext_for_mul,
    pt_ct_mult_any,
    relinearize_triplet,
    rotate_internal,
    rotsum,
    square_rescale,
    transpose_upper_to_lower_rotations,
)


INVERSE_BOOTSTRAP_MIN_LIMBS = 3
INVERSE_BOOTSTRAP_IMAG_BOOST = 1024
INVERSE_BOOTSTRAP_OUTPUT_SCALE = 1.0 / 1.2


def make_qkv_input_copies(x_cplx, ctx):
    """Expose token-block positions needed by diagonal Q/K/V weights."""

    return make_rotated_copies(x_cplx, ctx)


def combine_qkv_diagonals(ct_sub, masks, ctx):
    outputs = []
    for out in range(4):
        current = ct_sub[out][0]
        for diag in range(1, 6):
            rotated = rotate_internal(
                ct_sub[out][diag],
                12 - diag,
                "block_diag_1",
                masks,
                ctx,
            )
            current = fhe.homo_add(
                fhe.align_to(current, rotated.state, ctx),
                rotated,
                ctx,
            )
        outputs.append(current)
    return outputs


def pt_ct_matmul_qkv_grouped(
    weights,
    masks,
    qkvs,
    input_copies,
    ctx,
    *,
    layer: int = 0,
):
    """Run grouped Q/K/V MACs while packing each cipher batch once."""

    qkvs = tuple(qkvs)
    ct_sub = {
        qkv: [[None for _ in range(6)] for _ in range(4)] for qkv in qkvs
    }
    for out in range(4):
        ciphers = [input_copies[(16 * out + n) % 64] for n in range(64)]
        cipher_batch = fhe.pack_cipher_batch(ciphers)
        for qkv in qkvs:
            base = f"bert.encoder.layer.{layer}.attention.self.{qkv}.weight"
            for diag in range(6):
                names = [
                    packed_name(base, (out, diag, n)) for n in range(64)
                ]
                plaintext_batch = batched_plaintext_from_names(
                    weights,
                    names,
                    ciphers[0],
                    ctx,
                    f"{base}.grouped.{out}.{diag}",
                )
                summed_batch = fhe.grouped_pairwise_mac_rescale(
                    cipher_batch,
                    plaintext_batch,
                    1,
                    ctx,
                )
                ct_sub[qkv][out][diag] = fhe.unpack_cipher_batch(
                    summed_batch
                )[0]

    return {
        qkv: combine_qkv_diagonals(ct_sub[qkv], masks, ctx) for qkv in qkvs
    }


def run_qkv_projections(qkvs, weights, masks, x_cplx, ctx, *, layer: int = 0):
    input_copies = make_qkv_input_copies(x_cplx, ctx)
    qkvs = tuple(qkvs)
    raw = pt_ct_matmul_qkv_grouped(
        weights,
        masks,
        qkvs,
        input_copies,
        ctx,
        layer=layer,
    )
    return {
        qkv: add_qkv_bias(weights, qkv, raw[qkv], ctx, layer=layer)
        for qkv in qkvs
    }


def add_qkv_bias(weights, qkv: str, outputs, ctx, *, layer: int = 0):
    base = f"bert.encoder.layer.{layer}.attention.self.{qkv}.bias"
    biased = []
    for out, cipher in enumerate(outputs):
        name = packed_name(base, (out,))
        biased.append(
            fhe.homo_add_pt(
                cipher,
                plaintext_for_add(weights, name, cipher, ctx),
                ctx,
            )
        )
    return biased


def attention_rotations():
    """Full non-bootstrap rotation set for the canonical THOR attention graph."""
    rotations = [
        CONJUGATE_ROTATION,
        2**11,
        -(2**11),
        2**12,
        2**13,
        2**14,
        6,
        3,
        5,
        8,
        -8,
        2**10,
        -(2**10),
        *range(7, 12),
        *range(-6, 0),
        1,
        2,
        4,
        -1,
        -2,
        -4,
    ]
    rotations.extend(transpose_upper_to_lower_rotations())
    rotations.extend(256 * giant for giant in range(1, 8))
    rotations.append(-2032)

    return tuple(dict.fromkeys(rotations))


def _evaluate_exp_default(cipher, ctx):
    mid_x = (-27.2493 + 21.72692) / 2
    coeffs = np.asarray(
        list(
            reversed(
                [
                    0.032855468333339584,
                    0.05948672763856172,
                    0.03881607331549499,
                    0.0670090353368128,
                    0.15202099984697098,
                    0.20618261949210986,
                    0.23721029007596767,
                    0.26787311936472025,
                    0.27220647178765545,
                    0.2379982262906916,
                    0.1780344447042791,
                    0.11128698173597897,
                    0.05566510463488879,
                    0.020873931555133732,
                    0.005218196900295354,
                    0.0006522770224130905,
                ]
            )
        ),
        dtype=np.float64,
    )
    shifted = add_double_scalar(cipher, -mid_x / 32, ctx)
    exp_x = evaluate_polynomial_stockmeyer(coeffs, shifted, ctx)
    return square_rescale(exp_x, ctx)


def _evaluate_exp_layer2(cipher, ctx):
    coeffs = np.asarray(
        list(
            reversed(
                [
                    0.008201736399899691,
                    0.014226972463907047,
                    -0.008386712802267769,
                    -0.009262268572236316,
                    0.0397324053296174,
                    0.04817928878801878,
                    0.016604320800445653,
                    0.02336452059478656,
                    0.04217318306400685,
                    0.03517495704921328,
                    0.022268203231858744,
                    0.014216636671807894,
                    0.00749909544008294,
                    0.0027930565779849003,
                    0.0006877176070101981,
                    8.615994668877663e-05,
                ]
            )
        ),
        dtype=np.float64,
    )
    exp_x = evaluate_polynomial_stockmeyer(coeffs, cipher, ctx)
    return square_rescale(exp_x, ctx)


def evaluate_exp(cipher, ctx, *, variant: str):
    if variant == "softmax1":
        return _evaluate_exp_default(cipher, ctx)
    if variant == "softmax2":
        return _evaluate_exp_layer2(cipher, ctx)
    raise ValueError(f"unsupported softmax exponential variant: {variant!r}")


def apply_attention_mask(ciphers, mask_bundle, ctx):
    return [
        fhe.rescale(
            pt_ct_mult_any(plaintext_for_mul(mask_bundle, f"attention_mask.{index}", cipher, ctx), cipher, ctx),
            ctx,
        )
        for index, cipher in enumerate(ciphers)
    ]


def final_inv_mask_vector():
    mask = np.zeros((SLOTS,), dtype=np.float64)
    mask[: 2**11] = np.asarray(([1.0] * 12 + [0.0] * 4) * 2**7, dtype=np.float64)
    return mask


def calculate_sigma_exp(exp_masked, ctx):
    sigma = exp_masked[0]
    for cipher in exp_masked[1:]:
        sigma = fhe.homo_add(sigma, cipher, ctx)
    return rotsum(sigma, interval=2**11, ctx=ctx)


class ScaledCipher:
    def __init__(self, cipher, delta: float):
        self.cipher = cipher
        self.delta = float(delta)


def maybe_bootstrap_inv_pair(an: ScaledCipher, bn: ScaledCipher, ctx, bootstrap_program):
    if an.cipher.state.cur_limbs >= INVERSE_BOOTSTRAP_MIN_LIMBS:
        return an, bn

    scale_adjust = int(1 / (bn.delta * 4) / 2)
    if scale_adjust > 1:
        an_conj = conjugate(an.cipher, ctx)
        bn_conj = conjugate(bn.cipher, ctx)
        an = ScaledCipher(fhe.homo_add(an.cipher, an_conj, ctx), an.delta * 2)
        bn = ScaledCipher(fhe.homo_add(bn.cipher, bn_conj, ctx), bn.delta * 2)
        scale_adjust = int(1 / (bn.delta * 4) / 2)
    else:
        scale_adjust = 1

    an_cipher = fhe.align_to(an.cipher, bn.cipher.state, ctx)
    bn_cipher = bn.cipher
    bn_scaled = ScaledCipher(mult_int_scalar_any(bn_cipher, scale_adjust, ctx), bn.delta * scale_adjust)
    an_scaled = ScaledCipher(mult_int_scalar_any(an_cipher, 1, ctx), an.delta)

    bn_for_bootstrap = ScaledCipher(
        mult_int_scalar_any(bn_scaled.cipher, INVERSE_BOOTSTRAP_IMAG_BOOST, ctx),
        bn_scaled.delta * INVERSE_BOOTSTRAP_IMAG_BOOST,
    )

    temp = fhe.homo_add(an_scaled.cipher, imult(bn_for_bootstrap.cipher, ctx), ctx)
    temp = bootstrap_cipher(temp, ctx, bootstrap_program)
    temp = mult_double_scalar(temp, INVERSE_BOOTSTRAP_OUTPUT_SCALE, ctx)
    conj = conjugate(temp, ctx)
    real_part = fhe.homo_add(temp, conj, ctx)
    imag_part = imult(fhe.homo_sub(conj, temp, ctx), ctx)
    imag_part = mult_double_scalar(imag_part, 1.0 / INVERSE_BOOTSTRAP_IMAG_BOOST, ctx)
    real_part = fhe.align_to(real_part, imag_part.state, ctx)
    return (
        ScaledCipher(real_part, an_scaled.delta),
        ScaledCipher(imag_part, bn_scaled.delta),
    )


def evaluate_inverse(
    encrypted_constants,
    denominator,
    ctx,
    bootstrap_program,
    *,
    epsilon: float,
    alpha: float,
):
    numerator = align_encrypted_constant(
        encrypted_constants.inverse_numerator,
        denominator,
        ctx,
    )
    an = ScaledCipher(numerator, 1.0)
    bn = ScaledCipher(denominator, 1.0)
    en = float(epsilon)

    while en < 1 - alpha:
        an, bn = maybe_bootstrap_inv_pair(an, bn, ctx, bootstrap_program)
        kn = 2 / (en + 1)

        an_temp = ScaledCipher(
            negate(add_double_scalar(bn.cipher, -2 / kn * bn.delta, ctx), ctx),
            bn.delta,
        )
        an = ScaledCipher(
            mul_relin_rescale(an.cipher, an_temp.cipher, ctx),
            an.delta * an_temp.delta / kn**2,
        )

        bn_temp = ScaledCipher(
            negate(add_double_scalar(bn.cipher, -2 / kn * bn.delta, ctx), ctx),
            bn.delta,
        )
        bn = ScaledCipher(
            mul_relin_rescale(bn.cipher, bn_temp.cipher, ctx),
            bn.delta * bn_temp.delta / kn**2,
        )

        en = kn * en * (2 - kn * en)
        scale_adjust = int(1 / bn.delta / 2**8)
        if scale_adjust > 1:
            an_conj = conjugate(an.cipher, ctx)
            bn_conj = conjugate(bn.cipher, ctx)
            an = ScaledCipher(fhe.homo_add(an.cipher, an_conj, ctx), an.delta * 2)
            bn = ScaledCipher(fhe.homo_add(bn.cipher, bn_conj, ctx), bn.delta * 2)
            scale_adjust = int(1 / bn.delta / 2**8)
        else:
            scale_adjust = 1

        an = ScaledCipher(mult_int_scalar_any(an.cipher, scale_adjust, ctx), an.delta * scale_adjust)
        bn = ScaledCipher(mult_int_scalar_any(bn.cipher, scale_adjust, ctx), bn.delta * scale_adjust)

    return an.cipher, an.delta, en


def refine_inverse(
    encrypted_constants,
    exp_u,
    attention_mask_bundle,
    inv_d,
    d_delta: float,
    output_precision: float,
    ctx,
    bootstrap_program,
    *,
    alpha: float,
    final_inv: bool = False,
):
    inv_d_list = apply_attention_mask(
        [inv_d for _ in range(8)],
        attention_mask_bundle,
        ctx,
    )
    exp_2u = [None for _ in range(8)]
    k = max(int(1 / d_delta / 2), 1)

    for index in range(4):
        left0 = fhe.homo_add(exp_u[index], exp_u[index], ctx)
        part0 = mul_relin_rescale(left0, inv_d_list[index], ctx)
        part0 = mult_int_scalar_any(part0, k, ctx)
        exp_2u[index] = square_rescale(part0, ctx)

        left1 = fhe.homo_add(exp_u[index + 4], exp_u[index + 4], ctx)
        part1 = mul_relin_rescale(left1, inv_d_list[index + 4], ctx)
        part1 = mult_int_scalar_any(part1, k, ctx)
        exp_2u[index + 4] = square_rescale(part1, ctx)

    summation = exp_2u[0]
    for cipher in exp_2u[1:]:
        summation = fhe.homo_add(summation, cipher, ctx)

    summation = mult_int_scalar_any(summation, 2**6, ctx)
    summation = bootstrap_cipher(summation, ctx, bootstrap_program)
    summation = mult_double_scalar(summation, 1 / 2**6, ctx)
    summation = rotsum(summation, interval=2**11, ctx=ctx)

    epsilon2 = output_precision / 128 / 2
    inv_d, d_delta, output_precision = evaluate_inverse(
        encrypted_constants,
        summation,
        ctx,
        bootstrap_program,
        epsilon=epsilon2,
        alpha=alpha / 10,
    )

    if final_inv:
        inv_d = multiply_plain_vector(inv_d, final_inv_mask_vector(), "softmax.final_inv_mask", ctx)

    return exp_2u, inv_d, d_delta, output_precision, summation


def attention_layernorm(
    encrypted_constants,
    residual,
    weights,
    ctx,
    bootstrap_program,
    *,
    layer: int = 0,
):
    var_e = 1e-5
    min_var = 0.15
    max_var = 10.0
    n = 768
    max_for_denominator = (max_var * 1.05 + var_e) * n**2

    norm_mask = np.asarray(([1 / np.sqrt(max_for_denominator)] * 6 + [0.0] * 10) * 2**11, dtype=np.float64)
    first_slot_mask = np.asarray(([1.0] + [0.0] * 15) * 2**11, dtype=np.float64)

    enc_l = [multiply_plain_vector(cipher, norm_mask, f"ln1.norm_mask.{index}", ctx) for index, cipher in enumerate(residual)]

    sum_x = enc_l[0]
    for cipher in enc_l[1:]:
        sum_x = fhe.homo_add(sum_x, cipher, ctx)
    sum_x = rotsum(sum_x, 2**11, ctx)
    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, 1, ctx), ctx)
    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, 2, ctx), ctx)
    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, 4, ctx), ctx)
    sum_x = multiply_plain_vector(sum_x, first_slot_mask, "ln1.sum_first_slot", ctx)
    sq_sum_x = square_rescale(sum_x, ctx)

    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, -1, ctx), ctx)
    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, -2, ctx), ctx)
    sum_x = fhe.homo_add(sum_x, fhe.homo_rotate(sum_x, -4, ctx), ctx)

    numerator = []
    for cipher in enc_l:
        nx = mult_int_scalar_any(cipher, n, ctx)
        aligned_nx = fhe.align_to(nx, sum_x.state, ctx)
        numerator.append(fhe.homo_sub(aligned_nx, sum_x, ctx))

    sigma_x2 = square_rescale(enc_l[0], ctx)
    for cipher in enc_l[1:]:
        sigma_x2 = fhe.homo_add(sigma_x2, square_rescale(cipher, ctx), ctx)
    sigma_x2 = rotsum(sigma_x2, 2**11, ctx)
    sigma_x2 = fhe.homo_add(sigma_x2, fhe.homo_rotate(sigma_x2, 1, ctx), ctx)
    sigma_x2 = fhe.homo_add(sigma_x2, fhe.homo_rotate(sigma_x2, 2, ctx), ctx)
    sigma_x2 = fhe.homo_add(sigma_x2, fhe.homo_rotate(sigma_x2, 4, ctx), ctx)
    sigma_x2 = multiply_plain_vector(sigma_x2, first_slot_mask, "ln1.sigma_first_slot", ctx)

    n_sigma_x2 = mult_int_scalar_any(sigma_x2, n, ctx)
    aligned_sq_sum = fhe.align_to(sq_sum_x, n_sigma_x2.state, ctx)
    variance = fhe.homo_sub(n_sigma_x2, aligned_sq_sum, ctx)
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

    gamma_base = f"bert.encoder.layer.{layer}.attention.output.LayerNorm.weight"
    beta_base = f"bert.encoder.layer.{layer}.attention.output.LayerNorm.bias"
    outputs = []
    for index, cipher in enumerate(numerator):
        gamma_den = fhe.rescale(
            pt_ct_mult_any(plaintext_for_mul(weights, f"{gamma_base}__{index}", denominator, ctx), denominator, ctx),
            ctx,
        )
        ln = mul_relin_rescale(cipher, gamma_den, ctx)
        ln = fhe.homo_add_pt(ln, plaintext_for_add(weights, f"{beta_base}__{index}", ln, ctx), ctx)
        outputs.append(fhe.homo_add(ln, ln, ctx))
    return outputs


def calculate_softmax_prob(
    exp_u,
    inv_d,
    d_delta: float,
    ctx,
    bootstrap_program,
    *,
    force_bootstrap_exp: bool = False,
):
    rotated_inv_d = [inv_d]
    for _ in range(15):
        rotated_inv_d.append(fhe.homo_rotate(rotated_inv_d[-1], -2**11, ctx))

    cplx_softmax1 = []
    cplx_softmax2 = []
    k = int(1 / (2 * d_delta)) + 1
    for index in range(4):
        exp_cplx = fhe.homo_add(exp_u[index], imult(exp_u[index + 4], ctx), ctx)
        exp_cplx = mult_int_scalar_any(exp_cplx, k, ctx)
        if force_bootstrap_exp or exp_cplx.state.cur_limbs < 4:
            exp_cplx = normalize_cipher_scale(exp_cplx, ctx)
            exp_cplx = bootstrap_cipher(exp_cplx, ctx, bootstrap_program)
        for shift in range(16):
            masked_softmax = mul_relin_rescale(exp_cplx, rotated_inv_d[shift], ctx)
            softmax_copied = rotsum(masked_softmax, interval=2**11, ctx=ctx)
            conj = conjugate(softmax_copied, ctx)
            cplx_softmax1.append(fhe.homo_add(softmax_copied, conj, ctx))
            cplx_softmax2.append(imult(fhe.homo_sub(conj, softmax_copied, ctx), ctx))
    return cplx_softmax1 + cplx_softmax2


def split_complex_input(x_cplx, ctx):
    split = [None for _ in range(8)]
    for index in range(4):
        conj = conjugate(x_cplx[index], ctx)
        split[index] = mult_double_scalar(fhe.homo_add(x_cplx[index], conj, ctx), 0.5, ctx)
        split[index + 4] = mult_double_scalar(imult(fhe.homo_sub(conj, x_cplx[index], ctx), ctx), 0.5, ctx)
    return split


def split_context_temp(rotated_value, att_copy, n: int, masks, ctx):
    triplet = ct_ct_mult_triplet(rotated_value, att_copy, ctx)
    if n % 16 == 0:
        part0 = masked_triplet(0, n, triplet, masks, ctx)
        return [part0, fhe.homo_sub(delta_scaled_triplet(triplet, ctx), part0, ctx), None, None]

    parts = [masked_triplet(i, n, triplet, masks, ctx) for i in range(3)]
    return [*parts, fhe.homo_sub(delta_scaled_triplet(triplet, ctx), add_triplet_terms(parts, ctx), ctx)]


def apply_source_moves(ttemp, source_temp, source_index: int, moves, ctx):
    for dst_out, dst_part, src_out, src_part in moves:
        if src_out != source_index:
            continue
        ttemp[dst_out][dst_part] = add_optional(ttemp[dst_out][dst_part], source_temp[src_part], ctx)


def cached_baby_rotations(cipher, count: int, step: int, ctx):
    babies = [cipher]
    for _ in range(1, int(count)):
        babies.append(fhe.homo_rotate(babies[-1], step, ctx))
    return babies


def rotate_cached_score_input(babies, n: int, ctx):
    rotated = babies[n % 16]
    giant = n // 16
    if giant:
        rotated = fhe.homo_rotate(rotated, 256 * giant, ctx)
    return rotated


def context_moves(q_block: int, j_zero: bool):
    if j_zero:
        if q_block == 0:
            return [(0, 0, 0, 0), (0, 1, 0, 1), (1, 0, 1, 0), (1, 1, 1, 1)]
        if q_block == 1:
            return [(0, 2, 1, 0), (0, 3, 1, 1), (1, 0, 0, 0), (1, 1, 0, 1)]
        if q_block == 2:
            return [(0, 2, 0, 0), (0, 3, 0, 1), (1, 2, 1, 0), (1, 3, 1, 1)]
        if q_block == 3:
            return [(0, 0, 1, 0), (0, 1, 1, 1), (1, 2, 0, 0), (1, 3, 0, 1)]
        raise AssertionError(q_block)

    if q_block == 0:
        return [(0, 2, 1, 2), (0, 3, 1, 3), (0, 0, 0, 0), (0, 1, 0, 1),
                (1, 0, 0, 2), (1, 1, 0, 3), (1, 0, 1, 0), (1, 1, 1, 1)]
    if q_block == 1:
        return [(0, 2, 0, 2), (0, 3, 0, 3), (0, 2, 1, 0), (0, 3, 1, 1),
                (1, 2, 1, 2), (1, 3, 1, 3), (1, 0, 0, 0), (1, 1, 0, 1)]
    if q_block == 2:
        return [(0, 0, 1, 2), (0, 1, 1, 3), (0, 2, 0, 0), (0, 3, 0, 1),
                (1, 2, 0, 2), (1, 3, 0, 3), (1, 2, 1, 0), (1, 3, 1, 1)]
    if q_block == 3:
        return [(0, 0, 0, 2), (0, 1, 0, 3), (0, 0, 1, 0), (0, 1, 1, 1),
                (1, 0, 1, 2), (1, 1, 1, 3), (1, 2, 0, 0), (1, 3, 0, 1)]
    raise AssertionError(q_block)


def calculate_attention_context(v_cplx, att_prob, masks, ctx):
    if len(v_cplx) != 2 or len(att_prob) != 128:
        raise ValueError(f"expected v_cplx len 2 and att_prob len 128, got {len(v_cplx)} {len(att_prob)}")

    n_out = 2
    ttemp = [[None for _ in range(4)] for _ in range(n_out)]
    for out in range(n_out):
        triplet = ct_ct_mult_triplet(v_cplx[out], att_prob[0], ctx)
        ttemp[out][0] = pt_ct_mult_any(
            plaintext_for_mul(masks, "scale.attention_context", triplet, ctx),
            triplet,
            ctx,
        )

    for source_index, cipher in enumerate(v_cplx):
        babies = cached_baby_rotations(cipher, 16, -2032, ctx)
        for n in range(1, 128):
            source_temp = split_context_temp(
                rotate_cached_score_input(babies, n, ctx),
                att_prob[n],
                n,
                masks,
                ctx,
            )
            apply_source_moves(
                ttemp,
                source_temp,
                source_index,
                context_moves((n // 16) % 4, n % 16 == 0),
                ctx,
            )

    output = [None for _ in range(n_out)]
    for out in range(n_out):
        relin = [relinearize_triplet(ttemp[out][part], ctx) for part in range(4)]
        relin[1] = fhe.homo_rotate(relin[1], -2**11, ctx)
        relin[3] = fhe.homo_rotate(relin[3], -2**11, ctx)
        relin[2] = imult(conjugate(fhe.homo_add(relin[2], relin[3], ctx), ctx), ctx)
        cplx = fhe.homo_add(fhe.homo_add(relin[0], relin[1], ctx), relin[2], ctx)
        output[out] = fhe.rescale(cplx, ctx)
    return output


def attention_dense_matmul(
    weights,
    masks,
    base: str,
    input_copies,
    n_out_packed: int,
    n_diag: int,
    n_in_c: int,
    ctx,
):
    """Apply the canonical packed diagonal attention-output matrix."""
    ct_sub = [[None for _ in range(n_diag)] for _ in range(n_out_packed)]
    for out in range(n_out_packed):
        for diag in range(n_diag):
            ciphers = [input_copies[(16 * out + n) % n_in_c] for n in range(n_in_c)]
            pt_names = [f"{base}__{out}x{diag}x{n}" for n in range(n_in_c)]
            summed = grouped_mac_sum(
                ciphers,
                weights,
                pt_names,
                ctx,
                batch_name=f"{base}.grouped.{out}.{diag}",
            )
            ct_sub[out][diag] = linear_weight_rescale(summed, ctx)

    outputs = []
    for out in range(n_out_packed):
        current = ct_sub[out][0]
        for diag in range(1, n_diag):
            rotated = rotate_internal(
                ct_sub[out][diag],
                12 - diag,
                "block_diag_1",
                masks,
                ctx,
            )
            current = fhe.homo_add(fhe.align_to(current, rotated.state, ctx), rotated, ctx)
        outputs.append(current)
    return outputs


def attention_dense(weights, masks, context, ctx, *, layer: int = 0):
    """Run attention output dense, including the input copies its diagonal layout consumes."""
    weight_base = f"bert.encoder.layer.{layer}.attention.output.dense.weight"
    bias_base = f"bert.encoder.layer.{layer}.attention.output.dense.bias"
    context_copies = make_rotated_copies(context, ctx)
    wx = attention_dense_matmul(
        weights,
        masks,
        weight_base,
        context_copies,
        n_out_packed=8,
        n_diag=6,
        n_in_c=32,
        ctx=ctx,
    )
    outputs = []
    for out, cipher in enumerate(wx):
        masked = fhe.rescale(
            pt_ct_mult_any(plaintext_for_mul(masks, "attention_dense.mask1", cipher, ctx), cipher, ctx),
            ctx,
        )
        rotated = fhe.homo_rotate(masked, 6, ctx)
        current = fhe.homo_add(fhe.align_to(cipher, rotated.state, ctx), rotated, ctx)
        current = fhe.homo_add_pt(
            current,
            plaintext_for_add(weights, f"{bias_base}__{out}", current, ctx),
            ctx,
        )
        current = complex_real_twice(current, ctx)
        outputs.append(current)
    return outputs


def add_optional(current, term, ctx):
    if current is None:
        return term
    return fhe.homo_add(current, term, ctx)


def add_triplet_terms(terms, ctx):
    total = terms[0]
    for term in terms[1:]:
        total = fhe.homo_add(total, term, ctx)
    return total


def delta_scaled_triplet(cipher, ctx):
    scale = _plaintext_scale(ctx, cipher.state.cur_limbs)
    scalar = _encode_int_for_scalar_op(round(scale), cipher.state.cur_limbs, ctx)
    scaled = fhe.homo_mul_scalar(cipher, scalar, ctx)
    return scaled.cipher_like(
        scaled.cv,
        state=fhe.CipherState(
            scaled.state.cur_limbs,
            scaled.state.scale_degree + 1,
            scaled.state.scaling_factor * scale,
        ),
    )


def ct_ct_mask_name(mask_index: int, n: int) -> str:
    return f"ct_ct_matmul.{mask_index}.{n}"


def masked_triplet(mask_index: int, n: int, triplet, masks, ctx):
    return pt_ct_mult_any(plaintext_for_mul(masks, ct_ct_mask_name(mask_index, n), triplet, ctx), triplet, ctx)


def split_score_temp(rotated_k, q_copy, n: int, masks, ctx):
    triplet = ct_ct_mult_triplet(rotated_k, q_copy, ctx)
    if n % 16 == 0:
        part0 = masked_triplet(0, n, triplet, masks, ctx)
        return [part0, fhe.homo_sub(delta_scaled_triplet(triplet, ctx), part0, ctx)]

    parts = [masked_triplet(i, n, triplet, masks, ctx) for i in range(3)]
    return [*parts, fhe.homo_sub(delta_scaled_triplet(triplet, ctx), add_triplet_terms(parts, ctx), ctx)]


def score_moves(q_block: int, j_zero: bool):
    if j_zero:
        if q_block == 1:
            return [(0, 2, 3, 0), (0, 3, 3, 1), (1, 0, 0, 0), (1, 1, 0, 1),
                    (2, 0, 1, 0), (2, 1, 1, 1), (3, 0, 2, 0), (3, 1, 2, 1)]
        if q_block == 2:
            return [(0, 2, 2, 0), (0, 3, 2, 1), (1, 2, 3, 0), (1, 3, 3, 1),
                    (2, 0, 0, 0), (2, 1, 0, 1), (3, 0, 1, 0), (3, 1, 1, 1)]
        if q_block == 3:
            return [(0, 2, 1, 0), (0, 3, 1, 1), (1, 2, 2, 0), (1, 3, 2, 1),
                    (2, 2, 3, 0), (2, 3, 3, 1), (3, 0, 0, 0), (3, 1, 0, 1)]
        if q_block == 0:
            return []
        raise AssertionError(q_block)
    if q_block == 0:
        return [(0, 2, 3, 2), (0, 3, 3, 3), (0, 0, 0, 0), (0, 1, 0, 1),
                (1, 0, 0, 2), (1, 1, 0, 3), (1, 0, 1, 0), (1, 1, 1, 1),
                (2, 0, 1, 2), (2, 1, 1, 3), (2, 0, 2, 0), (2, 1, 2, 1),
                (3, 0, 2, 2), (3, 1, 2, 3), (3, 0, 3, 0), (3, 1, 3, 1)]
    if q_block == 1:
        return [(0, 2, 2, 2), (0, 3, 2, 3), (0, 2, 3, 0), (0, 3, 3, 1),
                (1, 2, 3, 2), (1, 3, 3, 3), (1, 0, 0, 0), (1, 1, 0, 1),
                (2, 0, 0, 2), (2, 1, 0, 3), (2, 0, 1, 0), (2, 1, 1, 1),
                (3, 0, 1, 2), (3, 1, 1, 3), (3, 0, 2, 0), (3, 1, 2, 1)]
    if q_block == 2:
        return [(0, 2, 1, 2), (0, 3, 1, 3), (0, 2, 2, 0), (0, 3, 2, 1),
                (1, 2, 2, 2), (1, 3, 2, 3), (1, 2, 3, 0), (1, 3, 3, 1),
                (2, 2, 3, 2), (2, 3, 3, 3), (2, 0, 0, 0), (2, 1, 0, 1),
                (3, 0, 0, 2), (3, 1, 0, 3), (3, 0, 1, 0), (3, 1, 1, 1)]
    if q_block == 3:
        return [(0, 2, 0, 2), (0, 3, 0, 3), (0, 2, 1, 0), (0, 3, 1, 1),
                (1, 2, 1, 2), (1, 3, 1, 3), (1, 2, 2, 0), (1, 3, 2, 1),
                (2, 2, 2, 2), (2, 3, 2, 3), (2, 2, 3, 0), (2, 3, 3, 1),
                (3, 2, 3, 2), (3, 3, 3, 3), (3, 0, 0, 0), (3, 1, 0, 1)]
    raise AssertionError(q_block)


def calculate_attention_score(k_cplx, q_copies, masks, ctx):
    if len(k_cplx) != 4 or len(q_copies) != 64:
        raise ValueError(f"expected k_cplx len 4 and q_copies len 64, got {len(k_cplx)} {len(q_copies)}")

    n_in = 64
    n_out = 4
    ttemp = [[None for _ in range(4)] for _ in range(n_out)]

    for out in range(n_out):
        triplet = ct_ct_mult_triplet(k_cplx[out], q_copies[0], ctx)
        ttemp[out][0] = pt_ct_mult_any(
            plaintext_for_mul(masks, "scale.attention_score", triplet, ctx),
            triplet,
            ctx,
        )

    for source_index, cipher in enumerate(k_cplx):
        babies = cached_baby_rotations(cipher, 16, -2032, ctx)
        for n in range(1, n_in):
            source_temp = split_score_temp(
                rotate_cached_score_input(babies, n, ctx),
                q_copies[n],
                n,
                masks,
                ctx,
            )
            apply_source_moves(
                ttemp,
                source_temp,
                source_index,
                score_moves(n // 16, n % 16 == 0),
                ctx,
            )

    output = [None for _ in range(8)]
    for out in range(n_out):
        relin = [relinearize_triplet(ttemp[out][part], ctx) for part in range(4)]
        relin[1] = fhe.homo_rotate(relin[1], -2**11, ctx)
        relin[3] = fhe.homo_rotate(relin[3], -2**11, ctx)
        relin[2] = imult(conjugate(fhe.homo_add(relin[2], relin[3], ctx), ctx), ctx)
        cplx = fhe.homo_add(fhe.homo_add(relin[0], relin[1], ctx), relin[2], ctx)
        cplx = fhe.rescale(cplx, ctx)
        conj = conjugate(cplx, ctx)
        output[out] = fhe.homo_add(cplx, conj, ctx)
        output[out + n_out] = imult(fhe.homo_sub(conj, cplx, ctx), ctx)
        output[out] = fhe.homo_add(output[out], output[out], ctx)
        output[out + n_out] = fhe.homo_add(output[out + n_out], output[out + n_out], ctx)
    return output
