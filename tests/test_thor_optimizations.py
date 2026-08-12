from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from thor import reference
from thor.fhe_ops import SLOTS, transpose_upper_to_lower_rotations
from thor.model import approximations, attention
from thor.packing.masks import ThorMaskPacker


@pytest.mark.parametrize("coefficient_count", (16, 28, 32))
def test_shared_stockmeyer_tree_matches_plain_polynomial(
    monkeypatch,
    coefficient_count,
):
    monkeypatch.setattr(approximations, "square_rescale", lambda value, ctx: value**2)
    monkeypatch.setattr(
        approximations,
        "mul_relin_rescale",
        lambda left, right, ctx: left * right,
    )
    monkeypatch.setattr(
        approximations,
        "mult_double_scalar",
        lambda value, scalar, ctx: value * scalar,
    )
    monkeypatch.setattr(
        approximations,
        "add_aligned",
        lambda left, right, ctx: left + right,
    )
    monkeypatch.setattr(
        approximations,
        "add_double_scalar",
        lambda value, scalar, ctx: value + scalar,
    )

    coefficients = np.linspace(-0.25, 0.5, coefficient_count)
    value = 0.125
    actual = approximations.evaluate_polynomial_stockmeyer(
        coefficients,
        value,
        object(),
    )
    expected = np.polynomial.polynomial.polyval(value, coefficients)
    assert actual == pytest.approx(expected, abs=1e-15)


def test_cached_attention_offsets_match_the_cyclic_diagonal_oracle():
    for diagonal in range(1, 128):
        baby = diagonal % 16
        giant = diagonal // 16
        cyclic_diagonal_offset = -(
            ((2**11) * baby - 16 * diagonal) % SLOTS
        )
        cached_offset = -2032 * baby + 256 * giant
        assert cyclic_diagonal_offset % SLOTS == cached_offset % SLOTS

    rotations = attention.attention_rotations()
    assert -2032 in rotations
    assert -4064 not in rotations
    assert all(256 * giant in rotations for giant in range(1, 8))


def test_cached_transpose_offsets_match_the_layout_oracle():
    for index in range(4):
        for diagonal in range(16 * index + 1, 16 * (index + 1)):
            layout_offset = (
                (64 - diagonal) * 2**4
                + (((diagonal % 48) * 2) % 16) * 2**11
            ) % SLOTS
            cached_offset = 4080 * (diagonal % 8) + 1024 - 128 * (diagonal // 8)
            assert layout_offset == cached_offset % SLOTS

    rotations = transpose_upper_to_lower_rotations()
    assert 4080 in rotations
    assert 8160 not in rotations


def test_source_wise_move_accumulation_preserves_the_oracle(monkeypatch):
    monkeypatch.setattr(
        attention,
        "add_optional",
        lambda current, term, ctx: (current or frozenset()) | {term},
    )
    schedules = (
        *(attention.score_moves(block, zero) for block in range(4) for zero in (False, True)),
        *(attention.context_moves(block, zero) for block in range(4) for zero in (False, True)),
    )
    for moves in schedules:
        width = max((source_part for _, _, _, source_part in moves), default=0) + 1
        temporary = [[(source, part) for part in range(width)] for source in range(4)]
        oracle = [[None for _ in range(4)] for _ in range(4)]
        source_wise = [[None for _ in range(4)] for _ in range(4)]
        for destination, part, source, source_part in moves:
            oracle[destination][part] = attention.add_optional(
                oracle[destination][part],
                temporary[source][source_part],
                object(),
            )
        for source in range(4):
            attention.apply_source_moves(
                source_wise,
                temporary[source],
                source,
                moves,
                object(),
            )
        assert source_wise == oracle


def test_shared_mask_bundle_contains_all_static_runtime_masks():
    vectors, _ = ThorMaskPacker(num_slots=SLOTS).pack()
    local0 = vectors["make_copies.local0"]
    local1 = vectors["make_copies.local1"]
    dense = vectors["attention_dense.mask1"]

    np.testing.assert_array_equal(local0 + local1, np.ones((SLOTS,)))
    assert np.all(local0[: 2**11] == 1)
    assert np.all(local0[2**11 : 2**12] == 0)
    assert np.all(dense[:6] == 0)
    assert np.all(dense[6:16] == 1)
    assert "transpose.mask2.1" in vectors
    assert "ct_ct_matmul.0.1" in vectors


def test_grouped_qkv_packs_each_output_cipher_batch_once(monkeypatch):
    counts = {"pack": 0, "mac": 0, "plaintext": 0}

    def pack_cipher_batch(ciphers):
        counts["pack"] += 1
        assert len(ciphers) == 64
        return ("cipher_batch", counts["pack"])

    def batched_plaintext(*args, **kwargs):
        counts["plaintext"] += 1
        return ("plaintext_batch", counts["plaintext"])

    def grouped_mac(*args, **kwargs):
        counts["mac"] += 1
        return ("summed_batch", counts["mac"])

    monkeypatch.setattr(attention.fhe, "pack_cipher_batch", pack_cipher_batch)
    monkeypatch.setattr(
        attention,
        "batched_plaintext_from_names",
        batched_plaintext,
    )
    monkeypatch.setattr(
        attention.fhe,
        "grouped_pairwise_mac_rescale",
        grouped_mac,
    )
    monkeypatch.setattr(
        attention.fhe,
        "unpack_cipher_batch",
        lambda batch: (batch,),
    )
    monkeypatch.setattr(
        attention,
        "combine_qkv_diagonals",
        lambda values, masks, ctx: values,
    )

    outputs = attention.pt_ct_matmul_qkv_grouped(
        object(),
        object(),
        ("query", "key", "value"),
        [object() for _ in range(64)],
        object(),
        layer=0,
    )

    assert set(outputs) == {"query", "key", "value"}
    assert counts == {"pack": 4, "mac": 72, "plaintext": 72}


def test_prepared_sample_has_one_shared_attention_mask_bundle(monkeypatch):
    tokenizer = lambda *args, **kwargs: {
        "input_ids": np.zeros((128,), dtype=np.int64),
        "token_type_ids": np.zeros((128,), dtype=np.int64),
        "attention_mask": np.ones((128,), dtype=np.int64),
    }
    assets = reference.ReferenceAssets(
        split=[{"sentence1": "a", "sentence2": "b", "label": 1}],
        tokenizer=tokenizer,
        state={},
    )
    monkeypatch.setattr(
        reference,
        "reference_bert_forward",
        lambda *args: (
            np.zeros((128, 768), dtype=np.float64),
            np.asarray([0.25, 0.75], dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        reference,
        "pack_attention_mask",
        lambda mask: np.ones((8, 4), dtype=np.float64),
    )
    monkeypatch.setattr(
        reference,
        "pack_embedding",
        lambda embedding: np.zeros((4, 4), dtype=np.complex128),
    )

    prepared = reference.prepare_sample(0, assets)

    assert isinstance(prepared, reference.PreparedSample)
    assert len(prepared.attention_mask_bundle) == 8
    assert prepared.attention_mask_bundle.cache_mode == "middle"
    assert prepared.reference_prediction == 1


@dataclass(frozen=True)
class _State:
    cur_limbs: int
    scale_degree: int = 1
    scaling_factor: float = 1.0

    def replace(self, **changes):
        values = {
            "cur_limbs": self.cur_limbs,
            "scale_degree": self.scale_degree,
            "scaling_factor": self.scaling_factor,
        }
        values.update(changes)
        return _State(**values)


def test_setup_encrypted_constants_are_only_aligned_downward(monkeypatch):
    constant = SimpleNamespace(state=_State(cur_limbs=31))
    reference_cipher = SimpleNamespace(state=_State(cur_limbs=7, scale_degree=2))
    captured = {}

    def align_to(value, target, context):
        captured.update(value=value, target=target, context=context)
        return "aligned"

    monkeypatch.setattr(attention.fhe, "align_to", align_to)
    context = object()
    result = attention.align_encrypted_constant(constant, reference_cipher, context)

    assert result == "aligned"
    assert captured["value"] is constant
    assert captured["target"] == _State(cur_limbs=7, scale_degree=1)
    assert captured["context"] is context
