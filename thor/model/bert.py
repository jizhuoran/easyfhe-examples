"""The complete fixed twelve-layer encrypted THOR BERT graph."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import numpy as np
from easyfhe import fhe

from ..config import LayerPlan, NUM_LAYERS, SLOTS, layer_plans
from ..fhe_ops import (
    CONJUGATE_ROTATION,
    add_aligned,
    bootstrap_cipher,
    bootstrap_complex_pair,
    complex_real_twice,
    imult,
    linear_weight_rescale,
    make_copies,
    multiply_plain_vector,
    normalize_cipher_scale,
    plaintext_for_add,
    plaintext_for_mul,
    pt_ct_mult_any,
    rotate_internal,
    rotsum,
    transpose_upper_to_lower,
)
from .approximations import evaluate_polynomial_stockmeyer
from .attention import (
    apply_attention_mask,
    attention_dense,
    attention_layernorm,
    calculate_attention_context,
    calculate_attention_score,
    calculate_sigma_exp,
    calculate_softmax_prob,
    evaluate_exp,
    evaluate_inverse,
    refine_inverse,
    run_qkv_projections,
    split_complex_input,
)
from .feed_forward import (
    bootstrap_dense1_pairs,
    ff_dense1,
    ff_dense2,
    ff_residual_add_and_bootstrap,
    feed_forward_layernorm,
    gelu,
)

if TYPE_CHECKING:
    from ..runtime import ThorRuntime


def boundary_real8_to_complex4(real8, ctx):
    """Recombine a real eight-cipher layer result for the next QKV input."""

    combined = []
    for index in range(4):
        merged = fhe.homo_add(real8[index], imult(real8[index + 4], ctx), ctx)
        combined.append(
            fhe.homo_add(merged, fhe.homo_rotate(merged, -6, ctx), ctx)
        )
    return combined


def bootstrap_real8_pairs(real8, ctx, bootstrap_program):
    """Bootstrap four real pairs while preserving the eight-cipher layout."""

    output = [None] * 8
    for index in range(4):
        output[index], output[index + 4] = bootstrap_complex_pair(
            real8[index],
            real8[index + 4],
            ctx,
            bootstrap_program,
            split_scale=0.5,
        )
    return output


def run_encoder_layer(
    plan: LayerPlan,
    *,
    encrypted_constants,
    ctx,
    bootstrap_program,
    masks,
    attention_mask_bundle,
    weights,
    x_cplx,
    input_real8=None,
):
    """Run one layer using the choices recorded in ``LayerPlan``."""

    raw_qkv = run_qkv_projections(
        ("query", "key", "value"),
        weights,
        masks,
        x_cplx,
        ctx,
        layer=plan.index,
    )
    real_qkv = {
        name: [complex_real_twice(cipher, ctx) for cipher in raw_qkv[name]]
        for name in ("query", "key", "value")
    }

    lower_key = transpose_upper_to_lower(real_qkv["key"], masks, ctx)
    complex_key = []
    for index in range(4):
        rotated_imag = imult(
            rotate_internal(lower_key[index], 64, "att", masks, ctx),
            ctx,
        )
        complex_key.append(
            fhe.homo_add(
                fhe.align_to(lower_key[index], rotated_imag.state, ctx),
                rotated_imag,
                ctx,
            )
        )
    query_copies = make_copies(real_qkv["query"], masks, ctx)
    attention_score = calculate_attention_score(
        complex_key,
        query_copies,
        masks,
        ctx,
    )

    bootstrapped_score = [None] * 8
    for index in range(4):
        bootstrapped_score[index], bootstrapped_score[index + 4] = (
            bootstrap_complex_pair(
                attention_score[index],
                attention_score[index + 4],
                ctx,
                bootstrap_program,
                split_scale=0.5,
            )
        )

    exp_score = [
        evaluate_exp(cipher, ctx, variant=plan.softmax_variant)
        for cipher in bootstrapped_score
    ]
    masked_exp = apply_attention_mask(exp_score, attention_mask_bundle, ctx)
    sigma_exp = calculate_sigma_exp(masked_exp, ctx)
    inverse, inverse_scale, precision = evaluate_inverse(
        encrypted_constants,
        sigma_exp,
        ctx,
        bootstrap_program,
        epsilon=plan.softmax_epsilon,
        alpha=0.01,
    )
    if plan.refine_softmax_inverse:
        exp_score, inverse, inverse_scale, precision, _ = refine_inverse(
            encrypted_constants,
            exp_score,
            attention_mask_bundle,
            inverse,
            inverse_scale,
            precision,
            ctx,
            bootstrap_program,
            alpha=0.1,
        )
    exp_score, inverse, inverse_scale, _, _ = refine_inverse(
        encrypted_constants,
        exp_score,
        attention_mask_bundle,
        inverse,
        inverse_scale,
        precision,
        ctx,
        bootstrap_program,
        alpha=0.01,
        final_inv=True,
    )
    attention_probability = calculate_softmax_prob(
        exp_score,
        inverse,
        inverse_scale,
        ctx,
        bootstrap_program,
        force_bootstrap_exp=plan.force_softmax_bootstrap,
    )

    complex_value = [
        fhe.homo_add(
            real_qkv["value"][0], imult(real_qkv["value"][2], ctx), ctx
        ),
        fhe.homo_add(
            real_qkv["value"][1], imult(real_qkv["value"][3], ctx), ctx
        ),
    ]
    attention_context = calculate_attention_context(
        complex_value,
        attention_probability,
        masks,
        ctx,
    )
    attention_context = [
        bootstrap_cipher(cipher, ctx, bootstrap_program)
        for cipher in attention_context
    ]
    dense_output = attention_dense(
        weights,
        masks,
        attention_context,
        ctx,
        layer=plan.index,
    )
    residual_left = (
        input_real8
        if input_real8 is not None
        else split_complex_input(x_cplx, ctx)
    )
    attention_residual = [
        fhe.homo_add(residual_left[index], dense_output[index], ctx)
        for index in range(8)
    ]
    attention_norm = attention_layernorm(
        encrypted_constants,
        attention_residual,
        weights,
        ctx,
        bootstrap_program,
        layer=plan.index,
    )

    dense1 = ff_dense1(
        weights,
        masks,
        attention_norm,
        ctx,
        layer=plan.index,
    )
    dense1 = bootstrap_dense1_pairs(dense1, ctx, bootstrap_program)
    activated = gelu(dense1, ctx)
    dense2 = ff_dense2(
        weights,
        masks,
        activated,
        ctx,
        layer=plan.index,
    )
    _, residual = ff_residual_add_and_bootstrap(
        attention_norm,
        dense2,
        ctx,
        bootstrap_program,
    )
    return feed_forward_layernorm(
        encrypted_constants,
        residual,
        weights,
        ctx,
        bootstrap_program,
        layer=plan.index,
        min_var=plan.ff_layernorm_min_var,
        max_var=plan.ff_layernorm_max_var,
    )


def pooler_classifier_rotations():
    """Application rotations used by the fixed pooler and classifier."""

    return tuple(
        dict.fromkeys(
            [
                CONJUGATE_ROTATION,
                *[-16 * (2**index) for index in range(7)],
                *range(1, 6),
                *[2**index for index in range(4, 11)],
                *range(-5, 0),
            ]
        )
    )


def pooler_dense(weights, masks, x8, ctx):
    if len(x8) != 8:
        raise ValueError(
            f"pooler_dense expects 8 real packed ciphers, got {len(x8)}"
        )
    x = [
        fhe.homo_add(x8[index], imult(x8[index + 4], ctx), ctx)
        for index in range(4)
    ]

    mask = np.zeros((SLOTS,), dtype=np.float64)
    mask[np.arange(SLOTS) % (2**11) < 6] = 1.0
    for index, cipher in enumerate(x):
        masked = multiply_plain_vector(
            cipher,
            mask,
            f"pooler.input_mask.{index}",
            ctx,
        )
        for rotation_index in range(7):
            masked = fhe.homo_add(
                masked,
                fhe.homo_rotate(masked, -16 * (2**rotation_index), ctx),
                ctx,
            )
        x[index] = masked

    products = []
    for diagonal in range(6):
        names = [
            f"bert.pooler.dense.weight__{diagonal}x{index}"
            for index in range(4)
        ]
        terms = [
            fhe.homo_mul_pt(
                cipher,
                plaintext_for_mul(weights, name, cipher, ctx),
                ctx,
            )
            for cipher, name in zip(x, names)
        ]
        total = terms[0]
        for term in terms[1:]:
            total = fhe.homo_add(total, term, ctx)
        products.append(linear_weight_rescale(total, ctx))

    current = products[0]
    for diagonal in range(1, 6):
        rotated = rotate_internal(
            products[diagonal],
            6 - diagonal,
            "block_diag_2",
            masks,
            ctx,
        )
        current = add_aligned(current, rotated, ctx)

    current = rotsum(current, 2**11, ctx)
    current = fhe.homo_add_pt(
        current,
        plaintext_for_add(
            weights,
            "bert.pooler.dense.bias__0",
            current,
            ctx,
        ),
        ctx,
    )
    return [complex_real_twice(current, ctx)]


def pooler_tanh(cipher, ctx, bootstrap_program):
    first_coefficients = np.asarray(
        list(
            reversed(
                [
                    -7.14529052e03,
                    -7.76519925e01,
                    2.74279201e04,
                    2.45150249e02,
                    -4.25793697e04,
                    -3.01953016e02,
                    3.42189880e04,
                    1.82989351e02,
                    -1.51158283e04,
                    -5.64098990e01,
                    3.58757327e03,
                    8.17596753e00,
                    -4.13341496e02,
                    -4.29024545e-01,
                    1.95056729e01,
                    2.06201784e-03,
                ]
            )
        ),
        dtype=np.float64,
    )
    second_coefficients = np.asarray(
        list(
            reversed(
                [
                    -9.02573450e-03,
                    -1.12320034e-04,
                    1.08762008e-01,
                    7.96793166e-04,
                    -5.41327356e-01,
                    -1.42873183e-03,
                    1.46476749e00,
                    -2.22416152e-03,
                    -2.43259032e00,
                    1.17381072e-02,
                    2.74974898e00,
                    -1.77631073e-02,
                    -2.38934873e00,
                    1.30194294e-02,
                    2.02874846e00,
                    -4.08442578e-03,
                ]
            )
        ),
        dtype=np.float64,
    )
    cipher = bootstrap_cipher(cipher, ctx, bootstrap_program)
    tanh_x = evaluate_polynomial_stockmeyer(first_coefficients, cipher, ctx)
    tanh_x = evaluate_polynomial_stockmeyer(
        second_coefficients,
        tanh_x,
        ctx,
    )
    tanh_x = normalize_cipher_scale(tanh_x, ctx)
    return bootstrap_cipher(tanh_x, ctx, bootstrap_program)


def classifier_dense(weights, pooled, ctx):
    outputs = []
    prefix = "cls.seq_relationship"
    for index in range(2):
        name = f"{prefix}.weight__{index}"
        product = pt_ct_mult_any(
            plaintext_for_mul(weights, name, pooled[0], ctx),
            pooled[0],
            ctx,
        )
        total = fhe.homo_add(product, fhe.homo_rotate(product, 1, ctx), ctx)
        doubled = fhe.homo_add(total, fhe.homo_rotate(total, 2, ctx), ctx)
        total = fhe.homo_add(doubled, fhe.homo_rotate(doubled, 4, ctx), ctx)
        for power in range(4, 11):
            total = fhe.homo_add(
                total,
                fhe.homo_rotate(total, 2**power, ctx),
                ctx,
            )
        total = linear_weight_rescale(total, ctx)
        total = fhe.homo_add_pt(
            total,
            plaintext_for_add(
                weights,
                f"{prefix}.bias__{index}",
                total,
                ctx,
            ),
            ctx,
        )
        outputs.append(total)
    return outputs


def infer_encrypted(runtime: "ThorRuntime", input_ciphers, attention_mask_bundle):
    """Run all encoder layers, followed by the pooler and classifier."""

    x_cplx = list(input_ciphers)
    input_real8 = None
    final_layer = None
    for plan in layer_plans():
        layer_weights = runtime.weights.layer(
            plan.index,
            attention_key_scale=plan.attention_key_scale,
        )
        try:
            final_layer = run_encoder_layer(
                plan,
                encrypted_constants=runtime.encrypted_constants,
                ctx=runtime.context,
                bootstrap_program=runtime.bootstrap_program,
                masks=runtime.masks,
                attention_mask_bundle=attention_mask_bundle,
                weights=layer_weights.bundle,
                x_cplx=x_cplx,
                input_real8=input_real8,
            )
        finally:
            runtime.weights.release_layer(plan.index)
        del layer_weights
        gc.collect()

        if plan.index + 1 < NUM_LAYERS:
            input_real8 = bootstrap_real8_pairs(
                final_layer,
                runtime.context,
                runtime.bootstrap_program,
            )
            x_cplx = boundary_real8_to_complex4(input_real8, runtime.context)

    if min(int(cipher.state.cur_limbs) for cipher in final_layer) < 8:
        final_layer = bootstrap_real8_pairs(
            final_layer,
            runtime.context,
            runtime.bootstrap_program,
        )
    dense = pooler_dense(
        runtime.weights.pooler(),
        runtime.masks,
        final_layer,
        runtime.context,
    )
    if int(dense[0].state.cur_limbs) <= 1:
        dense[0] = bootstrap_cipher(
            dense[0],
            runtime.context,
            runtime.bootstrap_program,
        )
    dense[0] = multiply_plain_vector(
        dense[0],
        np.full((SLOTS,), 1 / 40, dtype=np.float64),
        "pooler.tanh_scale",
        runtime.context,
    )
    pooled = [
        pooler_tanh(
            dense[0],
            runtime.context,
            runtime.bootstrap_program,
        )
    ]
    return classifier_dense(
        runtime.weights.classifier(),
        pooled,
        runtime.context,
    )


__all__ = [
    "infer_encrypted",
    "pooler_classifier_rotations",
    "run_encoder_layer",
]
