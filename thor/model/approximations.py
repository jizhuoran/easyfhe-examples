"""Shared encrypted polynomial and inverse-square-root approximations."""

from __future__ import annotations

import numpy as np

from ..fhe_ops import (
    add_aligned,
    add_double_scalar,
    bootstrap_cipher,
    mult_double_scalar,
    mul_relin_rescale,
    mult_int_scalar_any,
    multiply_plain_vector,
    normalize_cipher_scale,
    square_rescale,
    sub_from_plain_vector,
)


def _evaluate_baby_polynomial(coefficients, powers, context):
    result = mult_double_scalar(powers[1], float(coefficients[1]), context)
    if len(coefficients) >= 3:
        result = add_aligned(
            result,
            mult_double_scalar(powers[2], float(coefficients[2]), context),
            context,
        )
    if len(coefficients) >= 4:
        cubic = mul_relin_rescale(
            powers[2],
            mult_double_scalar(powers[1], float(coefficients[3]), context),
            context,
        )
        result = add_aligned(result, cubic, context)
    return add_double_scalar(result, float(coefficients[0]), context)


def evaluate_polynomial_stockmeyer(coefficients, cipher, context):
    """Evaluate the degree-15/27/31 polynomials used by the fixed graph."""

    coefficients = np.asarray(coefficients, dtype=np.float64)
    if len(coefficients) < 16 or len(coefficients) % 4:
        raise ValueError(f"unsupported polynomial length {len(coefficients)}")

    x2 = square_rescale(cipher, context)
    x3 = mul_relin_rescale(cipher, x2, context)
    powers = [None, cipher, x2, x3]
    x4 = square_rescale(x2, context)
    x8 = square_rescale(x4, context)
    x16 = square_rescale(x8, context) if len(coefficients) > 16 else None

    subpolynomials = np.split(coefficients, len(coefficients) // 4)
    baby_results = [
        _evaluate_baby_polynomial(values, powers, context)
        for values in subpolynomials
    ]
    first_level = []
    for index in range(0, len(baby_results), 2):
        if index + 1 == len(baby_results):
            first_level.append(baby_results[index])
        else:
            first_level.append(
                add_aligned(
                    baby_results[index],
                    mul_relin_rescale(
                        baby_results[index + 1], x4, context
                    ),
                    context,
                )
            )

    second_level = []
    for index in range(0, len(first_level), 2):
        if index + 1 == len(first_level):
            second_level.append(first_level[index])
        else:
            second_level.append(
                add_aligned(
                    first_level[index],
                    mul_relin_rescale(first_level[index + 1], x8, context),
                    context,
                )
            )

    if len(second_level) == 1:
        return second_level[0]
    if len(second_level) == 2 and x16 is not None:
        return add_aligned(
            second_level[0],
            mul_relin_rescale(second_level[1], x16, context),
            context,
        )
    raise ValueError(f"unsupported polynomial length {len(coefficients)}")


def evaluate_inverse_sqrt(
    numerator,
    denominator,
    epsilon: float,
    alpha: float,
    mask: np.ndarray,
    context,
    bootstrap_program,
):
    """Evaluate the fixed iterative inverse-square-root approximation."""

    an = denominator
    bn = numerator
    en = float(epsilon)
    iteration = 0

    while en < 1 - alpha:
        iteration += 1
        kn = np.roots([1 - en**3, 6 * en**2 - 6, 9 - 9 * en])[1].real
        bn1 = multiply_plain_vector(
            bn,
            (kn**1.5 / 2) * mask,
            f"ln.invsqrt.bn1.{iteration}",
            context,
        )

        if an.state.cur_limbs < 6 or bn1.state.cur_limbs < 6:
            an = mult_int_scalar_any(an, 2**6, context)
            an = bootstrap_cipher(an, context, bootstrap_program)
            an = mult_double_scalar(an, 1 / 2**6, context)
            bn1 = bootstrap_cipher(bn1, context, bootstrap_program)

        bn2 = sub_from_plain_vector(
            (3 / kn) * mask,
            an,
            f"ln.invsqrt.bn2.{iteration}",
            context,
        )
        bn = mul_relin_rescale(bn1, bn2, context)

        an1 = multiply_plain_vector(
            an,
            (kn**3 / 4) * mask,
            f"ln.invsqrt.an1.{iteration}",
            context,
        )
        an2_base = sub_from_plain_vector(
            (3 / kn) * mask,
            an,
            f"ln.invsqrt.an2_base.{iteration}",
            context,
        )
        an2 = square_rescale(an2_base, context)
        an = mul_relin_rescale(an1, an2, context)
        en = kn * en * (3 - kn * en) ** 2 / 4
        an = normalize_cipher_scale(an, context)
        bn = normalize_cipher_scale(bn, context)

    return bn


__all__ = ["evaluate_inverse_sqrt", "evaluate_polynomial_stockmeyer"]
