"""Low-level u64 CKKS operations used by the canonical THOR graph."""

from __future__ import annotations

import numpy as np

import easyfhe as torch
from easyfhe import fhe
import easyfhe.bs.openfhe as bs

from .config import SLOTS

CONJUGATE_ROTATION = (2 * SLOTS * 2) - 1


PackedRaw = fhe.PackedRaw


class LazyPackedRaw(PackedRaw):
    """Application-owned lazy packing for large THOR weight matrices."""

    def __init__(self, tensor, packer):
        super().__init__(tensor)
        self._packer = packer

    def packed_tensor(self, slots, context=None):
        return self._packer(self.tensor, slots, context)


def validate_cipher(cipher, ctx, label: str = "cipher"):
    if int(cipher.state.cur_limbs) <= 0:
        raise AssertionError(f"{label}: invalid limb count: {cipher.state}")
    if int(cipher.state.scale_degree) <= 0:
        raise AssertionError(f"{label}: invalid scale degree: {cipher.state}")
    if float(cipher.state.scaling_factor) <= 0.0:
        raise AssertionError(f"{label}: invalid scaling factor: {cipher.state}")
    return cipher


def packed_name(key: str, index: tuple[int, ...]) -> str:
    return f"{key}__{'x'.join(str(i) for i in index)}"


def plaintext_for_add(bundle, name: str, cipher, ctx, *, slots: int = SLOTS, cache: bool = True):
    return bundle.plaintext(
        name,
        state=cipher.state,
        slots=slots,
        context=ctx,
        cache=cache,
    )


def plaintext_for_mul(bundle, name: str, cipher, ctx, *, slots: int = SLOTS, cache: bool = True):
    return bundle.plaintext(
        name,
        state=_multiplication_plaintext_state(cipher, ctx),
        slots=slots,
        context=ctx,
        cache=cache,
    )


def multiply_plain_vector(cipher, vector: np.ndarray, name: str, ctx):
    """Multiply by one ad-hoc slot vector and consume one u64 level."""

    vector = np.asarray(vector, dtype=np.float64)
    bundle = fhe.ConstantBundle(
        vectors={name: PackedRaw(torch.as_tensor(vector))},
        cache_mode="none",
    )
    product = pt_ct_mult_any(
        plaintext_for_mul(bundle, name, cipher, ctx),
        cipher,
        ctx,
    )
    return fhe.rescale(product, ctx)


def add_plain_vector(cipher, vector: np.ndarray, name: str, ctx):
    """Add one ad-hoc slot vector without retaining a constant cache."""

    vector = np.asarray(vector, dtype=np.float64)
    bundle = fhe.ConstantBundle(
        vectors={name: PackedRaw(torch.as_tensor(vector))},
        cache_mode="none",
    )
    return fhe.homo_add_pt(
        cipher,
        plaintext_for_add(bundle, name, cipher, ctx),
        ctx,
    )


def sub_from_plain_vector(vector: np.ndarray, cipher, name: str, ctx):
    return add_plain_vector(negate(cipher, ctx), vector, name, ctx)


def add_aligned(left, right, ctx):
    """Align two ciphertext limb counts before addition."""

    if left.state.cur_limbs > right.state.cur_limbs:
        left = fhe.align_to(left, right.state, ctx)
    elif right.state.cur_limbs > left.state.cur_limbs:
        right = fhe.align_to(right, left.state, ctx)
    return fhe.homo_add(left, right, ctx)


def align_encrypted_constant(constant, reference, ctx):
    """Drop a setup-time encrypted constant to a graph value's limb count."""

    target = constant.state.replace(cur_limbs=int(reference.state.cur_limbs))
    return fhe.align_to(constant, target, ctx)


def _multiplication_plaintext_state(cipher, ctx):
    cur_limbs = int(cipher.state.cur_limbs)
    return fhe.CipherState(cur_limbs, 1, float(ctx.scale_at(cur_limbs)))


def batched_plaintext_from_names(bundle, names, cipher, ctx, batch_name: str, *, slots: int = SLOTS):
    vectors = [bundle.raw_vectors[name] for name in names]
    batch_tensor = _batch_tensor_from_vectors(vectors, slots, ctx)
    batch_bundle = fhe.ConstantBundle(vectors={batch_name: PackedRaw(batch_tensor)}, cache_mode="none")
    return batch_bundle.plaintext(
        batch_name,
        state=_multiplication_plaintext_state(cipher, ctx),
        slots=slots,
        context=ctx,
        cache=False,
    )


def _plaintext_scale(ctx, cur_limbs):
    return float(ctx.scale_at(int(cur_limbs)))


def linear_weight_rescale(cipher, ctx):
    return fhe.rescale(cipher, ctx)


def _scalar_for_add(cipher, scalar: float, ctx):
    return fhe.encode_scalar(
        float(scalar),
        cur_limbs=cipher.state.cur_limbs,
        scale_degree=cipher.state.scale_degree,
        context=ctx,
        mode="scaled",
        scaling_factor=float(cipher.state.scaling_factor),
    )


def _scalar_for_mul_rescale(cipher, scalar: float, ctx):
    return fhe.encode_scalar(
        float(scalar),
        cur_limbs=cipher.state.cur_limbs,
        scale_degree=1,
        context=ctx,
        mode="scaled",
        scaling_factor=float(ctx.scale_at(cipher.state.cur_limbs)),
    )


def normalize_cipher_scale(cipher, ctx):
    if int(cipher.state.scale_degree) == 1:
        return cipher
    return fhe.normalize_scale(cipher, ctx)


def _batch_tensor_from_vectors(vectors, slots: int, ctx):
    if not vectors:
        raise ValueError("batch plaintext requires at least one vector")
    first = vectors[0]
    source_id = getattr(first, "_thor_source_id", None)
    batch_packer = getattr(first, "_thor_batch_packer", None)
    if source_id is not None and callable(batch_packer):
        indices = []
        for vector in vectors:
            if getattr(vector, "_thor_source_id", None) != source_id:
                break
            indices.append(getattr(vector, "_thor_index"))
        else:
            return batch_packer(first.tensor, np.asarray(indices, dtype=np.int64), slots, ctx)

    packed = []
    for vector in vectors:
        if hasattr(vector, "packed_tensor"):
            packed.append(vector.packed_tensor(slots, ctx))
        else:
            packed.append(vector)
    if all(hasattr(vector, "dim") for vector in packed):
        return torch.stack(packed, dim=0)
    return torch.as_tensor(np.stack(packed, axis=0), device=ctx.device)


def grouped_mac_batch(
    ciphers,
    bundle,
    names,
    ctx,
    *,
    batch_name: str,
    slots: int = SLOTS,
    groups: int = 1,
):
    cipher_batch = fhe.pack_cipher_batch(ciphers)
    plaintext_batch = batched_plaintext_from_names(
        bundle,
        names,
        ciphers[0],
        ctx,
        batch_name,
        slots=slots,
    )
    return fhe.grouped_pairwise_mac(cipher_batch, plaintext_batch, groups, ctx)


def grouped_mac_sum(
    ciphers,
    bundle,
    names,
    ctx,
    *,
    batch_name: str,
    slots: int = SLOTS,
):
    return fhe.unpack_cipher_batch(
        grouped_mac_batch(
            ciphers,
            bundle,
            names,
            ctx,
            batch_name=batch_name,
            slots=slots,
            groups=1,
        )
    )[0]


def make_rotated_copies(ciphers, ctx):
    """Build the 16 cyclic input copies with successive 2048-slot rotations."""
    rots = [None] * (16 * len(ciphers))
    for index, cipher in enumerate(ciphers):
        rots[16 * index] = cipher
        for offset in range(1, 16):
            rots[16 * index + offset] = fhe.homo_rotate(rots[16 * index + offset - 1], 2**11, ctx)
    return rots


def rotate_internal(cipher, delta: int, mode: str, masks, ctx):
    if delta == 0:
        return cipher
    if mode == "att":
        left_delta = 16 * int(delta)
        right_delta = 2**11 - left_delta
    elif mode == "block_diag_1":
        left_delta = int(delta)
        right_delta = 12 - left_delta
    elif mode == "block_diag_2":
        left_delta = int(delta)
        right_delta = 6 - left_delta
    else:
        raise ValueError(f"unsupported internal rotation mode: {mode}")

    masked = fhe.rescale(pt_ct_mult_any(plaintext_for_mul(masks, f"rot_internal.{mode}.{delta}", cipher, ctx), cipher, ctx), ctx)
    unmasked = fhe.homo_sub(fhe.align_to(cipher, masked.state, ctx), masked, ctx)
    return fhe.homo_add(
        fhe.homo_rotate(unmasked, left_delta, ctx),
        fhe.homo_rotate(masked, -right_delta, ctx),
        ctx,
    )


def rotsum(cipher, interval: int, ctx):
    reps = int(np.log2(SLOTS / interval))
    total = cipher
    for index in range(reps):
        total = fhe.homo_add(total, fhe.homo_rotate(total, interval * 2**index, ctx), ctx)
    return total


def transpose_upper_to_lower_rotations():
    """Rotation keys for the canonical cached baby/giant transpose schedule."""
    rotations = []
    for index in range(4):
        n_diag = 16 * index
        delta = (((64 - n_diag) % 64) * 16) % SLOTS
        if delta:
            rotations.append(delta)

    rotations.append(4080)
    for index in range(4):
        for n_diag_u in range(16 * index + 1, 16 * (index + 1)):
            rotations.append(1024 - 128 * (n_diag_u // 8))

    return tuple(dict.fromkeys(rotations))


def transpose_upper_to_lower(ciphers, masks, ctx):
    """Transpose with the canonical cached baby/giant rotation schedule."""
    if len(ciphers) != 4:
        raise ValueError(f"transpose_upper_to_lower expects 4 ciphers, got {len(ciphers)}")
    temp = [[None, None] for _ in range(4)]
    for index in range(4):
        n_diag = 16 * index
        delta = (((64 - n_diag) % 64) * 16) % SLOTS
        rotated = fhe.homo_rotate(ciphers[index], delta, ctx) if delta else ciphers[index]
        part0 = fhe.rescale(
            pt_ct_mult_any(plaintext_for_mul(masks, f"transpose.mask0.{index}", rotated, ctx), rotated, ctx),
            ctx,
        )
        part1 = fhe.rescale(
            pt_ct_mult_any(plaintext_for_mul(masks, f"transpose.mask1.{index}", rotated, ctx), rotated, ctx),
            ctx,
        )
        temp[(4 - index) % 4][0] = part0
        temp[(4 - index) % 4][1] = part1

    for index in range(4):
        cached_babies = [ciphers[index]]
        for _ in range(1, 8):
            cached_babies.append(fhe.homo_rotate(cached_babies[-1], 4080, ctx))
        for n_diag_u in range(16 * index + 1, 16 * (index + 1)):
            rotated = cached_babies[n_diag_u % 8]
            rotated = fhe.homo_rotate(rotated, 1024 - 128 * (n_diag_u // 8), ctx)
            part2 = fhe.rescale(
                pt_ct_mult_any(plaintext_for_mul(masks, f"transpose.mask2.{n_diag_u}", rotated, ctx), rotated, ctx),
                ctx,
            )
            part3 = fhe.rescale(
                pt_ct_mult_any(plaintext_for_mul(masks, f"transpose.mask3.{n_diag_u}", rotated, ctx), rotated, ctx),
                ctx,
            )
            target = (3 - index) % 4
            temp[target][0] = fhe.homo_add(temp[target][0], part2, ctx)
            temp[target][1] = fhe.homo_add(temp[target][1], part3, ctx)

    return [
        fhe.homo_add(temp[index][0], fhe.homo_rotate(temp[index][1], -2**11, ctx), ctx)
        for index in range(4)
    ]


def make_copies(ciphers, masks, ctx):
    if len(ciphers) % 2:
        raise ValueError(f"make_copies expects an even cipher count, got {len(ciphers)}")
    copies = [None] * (16 * len(ciphers))
    half = len(ciphers) // 2
    for index in range(half):
        merged = fhe.homo_add(ciphers[index], imult(ciphers[index + half], ctx), ctx)
        for sub in range(8):
            masked = fhe.rescale(
                pt_ct_mult_any(plaintext_for_mul(masks, f"make_copies_2.{sub}", merged, ctx), merged, ctx),
                ctx,
            )
            copied = rotsum(masked, 2**12, ctx)
            copied0 = fhe.rescale(
                pt_ct_mult_any(plaintext_for_mul(masks, "make_copies.local0", copied, ctx), copied, ctx),
                ctx,
            )
            copied1 = fhe.rescale(
                pt_ct_mult_any(plaintext_for_mul(masks, "make_copies.local1", copied, ctx), copied, ctx),
                ctx,
            )
            copied0 = fhe.homo_add(copied0, fhe.homo_rotate(copied0, -2**11, ctx), ctx)
            copied1 = fhe.homo_add(copied1, fhe.homo_rotate(copied1, 2**11, ctx), ctx)
            conj0 = conjugate(copied0, ctx)
            conj1 = conjugate(copied1, ctx)
            copies[index * 16 + 2 * sub] = fhe.homo_add(copied0, conj0, ctx)
            copies[(index + half) * 16 + 2 * sub] = imult(fhe.homo_sub(conj0, copied0, ctx), ctx)
            copies[index * 16 + 2 * sub + 1] = fhe.homo_add(copied1, conj1, ctx)
            copies[(index + half) * 16 + 2 * sub + 1] = imult(fhe.homo_sub(conj1, copied1, ctx), ctx)
    return copies


def ct_ct_mult_triplet(left, right, ctx):
    """Return an unrelinearized ct-ct product, matching THOR relin=False."""
    return fhe.homo_mul_no_relin(left, right, ctx)


def pt_ct_mult_any(plain, cipher, ctx):
    """Multiply a plaintext with a 2-component cipher or a 3-component triplet."""
    return fhe.homo_mul_pt(cipher, plain, ctx)


def relinearize_triplet(cipher, ctx):
    return fhe.homo_relinearize(cipher, ctx)


def mult_int_scalar_any(cipher, scalar: int, ctx):
    encoded = _encode_int_for_scalar_op(int(scalar), cipher.state.cur_limbs, ctx)
    return fhe.homo_mul_scalar(cipher, encoded, ctx)


def negate(cipher, ctx):
    return mult_int_scalar_any(cipher, -1, ctx)


def mult_double_scalar(cipher, scalar: float, ctx):
    validate_cipher(cipher, ctx, "mult_double_scalar.input")
    encoded = _scalar_for_mul_rescale(cipher, float(scalar), ctx)
    return validate_cipher(
        fhe.homo_mul_scalar_rescale(cipher, encoded, ctx),
        ctx,
        "mult_double_scalar",
    )


def add_double_scalar(cipher, scalar: float, ctx):
    encoded = _scalar_for_add(cipher, float(scalar), ctx)
    return validate_cipher(fhe.homo_add_scalar(cipher, encoded, ctx), ctx, "add_double_scalar")


def mul_relin_rescale(left, right, ctx):
    validate_cipher(left, ctx, "mul_relin_rescale.left")
    validate_cipher(right, ctx, "mul_relin_rescale.right")
    left, right = _align_for_mul(left, right, ctx)
    return validate_cipher(
        fhe.rescale(fhe.homo_mul_relin(left, right, ctx), ctx),
        ctx,
        "mul_relin_rescale",
    )


def square_rescale(cipher, ctx):
    return mul_relin_rescale(cipher, cipher, ctx)


def imult(cipher, ctx, *, neg: bool = False):
    """Multiply by i without consuming a level, matching THOR's mult_imag."""
    return fhe.homo_mul_i(cipher, ctx, negative=neg)


def conjugate(cipher, ctx):
    return fhe.homo_rotate(cipher, CONJUGATE_ROTATION, ctx)


def complex_real_twice(cipher, ctx):
    return fhe.homo_add(cipher, conjugate(cipher, ctx), ctx)


def bootstrap_cipher(cipher, ctx, program):
    validate_cipher(cipher, ctx, "bootstrap_cipher.input")
    return validate_cipher(
        bs.bootstrap(cipher, ctx, program),
        ctx,
        "bootstrap_cipher",
    )


def bootstrap_complex_pair(real, imag, ctx, program, *, split_scale: float | None = None):
    temp = fhe.homo_add(real, imult(imag, ctx), ctx)
    temp = bootstrap_cipher(temp, ctx, program)
    conj = conjugate(temp, ctx)
    real_out = fhe.homo_add(temp, conj, ctx)
    imag_out = imult(fhe.homo_sub(conj, temp, ctx), ctx)
    if split_scale is not None:
        real_out = mult_double_scalar(real_out, split_scale, ctx)
        imag_out = mult_double_scalar(imag_out, split_scale, ctx)
    return real_out, imag_out


def _align_for_mul(left, right, ctx):
    target_limbs = min(int(left.state.cur_limbs), int(right.state.cur_limbs))
    left_target = left.state.replace(cur_limbs=target_limbs)
    right_target = right.state.replace(cur_limbs=target_limbs)
    left = fhe.align_to(left, left_target, ctx)
    right = fhe.align_to(right, right_target, ctx)
    validate_cipher(left, ctx, "_align_for_mul.left")
    validate_cipher(right, ctx, "_align_for_mul.right")
    return left, right


def _encode_int_for_scalar_op(scalar: int, cur_limbs: int, ctx):
    return fhe.encode_scalar(
        int(scalar),
        cur_limbs=cur_limbs,
        scale_degree=0,
        context=ctx,
        mode="integer",
    )
