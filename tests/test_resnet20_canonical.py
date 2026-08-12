from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

import easyfhe
import easyfhe.bs.openfhe as bs

from resnet20_aespa import benchmark
from resnet20_aespa import main as resnet_main
from resnet20_aespa.assets import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    DATASET_SHA256,
    WEIGHTS_REPO_ID,
    WEIGHTS_REVISION,
    WEIGHTS_SHA256,
)
from resnet20_aespa.config import (
    BOOTSTRAP_MODE,
    BOOTSTRAP_OUTPUT_LEVELS,
    BOOTSTRAP_STRATEGY,
    DEFAULT_ASSET_DIR,
    DEFAULT_DATASET_PATH,
    DEFAULT_WEIGHTS_PATH,
    INPUT_LIMBS,
    LOG_N,
    NETWORK_ROTATIONS,
    SECRET_KEY_DIST,
    RunConfig,
    bootstrap_specs,
    parse_args,
)
from resnet20_aespa.packing import load_weights
from resnet20_aespa.packing.triton import pack_dictionary, pack_factorized
from resnet20_aespa.runtime import ResNet20Runtime


def test_fixed_bootstrap_requirements_cover_the_canonical_context():
    specs = bootstrap_specs()
    assert tuple(spec.output_levels for spec in specs) == BOOTSTRAP_OUTPUT_LEVELS
    assert {spec.strategy for spec in specs} == {BOOTSTRAP_STRATEGY}
    assert {spec.mode for spec in specs} == {BOOTSTRAP_MODE}

    requirements = bs.requirements(
        specs,
        log_n=LOG_N,
        secret_key_dist=SECRET_KEY_DIST,
    )
    rotations = tuple(
        dict.fromkeys((*NETWORK_ROTATIONS, *requirements.rotations))
    )
    assert requirements.context_depth == 30
    assert INPUT_LIMBS == 18
    assert len(NETWORK_ROTATIONS) == 39
    assert len(rotations) == 133


def test_run_config_and_server_runtime_have_one_responsibility():
    assert [field.name for field in fields(RunConfig)] == [
        "runs",
        "warmup",
        "dataset_path",
        "weights_path",
    ]
    assert [field.name for field in fields(ResNet20Runtime)] == [
        "context",
        "weights",
        "bootstrap_programs",
    ]


def test_cli_requires_positive_runs_and_allows_zero_warmup():
    config = parse_args(["--runs", "2", "--warmup", "0"])
    assert config.runs == 2
    assert config.warmup == 0

    with pytest.raises(SystemExit):
        parse_args(["--runs", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--warmup", "-1"])


def test_default_assets_are_external_and_content_addressed():
    assert DEFAULT_DATASET_PATH.parent == DEFAULT_ASSET_DIR
    assert DEFAULT_WEIGHTS_PATH.parent == DEFAULT_ASSET_DIR
    assert DATASET_REPO_ID == "jizhuoran/easyfhe-resnet20-cifar10"
    assert WEIGHTS_REPO_ID == "jizhuoran/easyfhe-resnet20-aespa"
    assert all(
        len(value) == 40 for value in (DATASET_REVISION, WEIGHTS_REVISION)
    )
    assert all(len(value) == 64 for value in (DATASET_SHA256, WEIGHTS_SHA256))


def test_missing_assets_fail_before_context_setup(tmp_path):
    with pytest.raises(FileNotFoundError, match="resnet20_aespa.assets"):
        resnet_main._require_assets(
            tmp_path / "missing-dataset.npz",
            tmp_path / "missing-weights.npz",
        )


def test_numeric_dataset_contract_uses_a_synthetic_fixture(tmp_path):
    path = tmp_path / "dataset.npz"
    images = np.arange(2 * 3072, dtype=np.float64).reshape(2, 3072)
    labels = np.asarray([3, 7], dtype=np.int64)
    np.savez_compressed(path, images=images, labels=labels)

    actual_images, actual_labels = benchmark._load_dataset(path)
    np.testing.assert_array_equal(actual_images, images)
    np.testing.assert_array_equal(actual_labels, labels)


def test_benchmark_fails_loudly_on_nonfinite_logits(monkeypatch):
    class Decrypted:
        def cpu(self):
            return self

        def numpy(self):
            return np.array([np.nan, *range(1, 10)], dtype=np.float64)

    class Client:
        def encrypt(self, *args, **kwargs):
            return object()

        def decrypt(self, cipher):
            return Decrypted()

    monkeypatch.setattr(benchmark, "_sync", lambda: None)
    monkeypatch.setattr(
        benchmark,
        "infer_encrypted",
        lambda cipher, runtime: cipher,
    )
    with pytest.raises(FloatingPointError):
        benchmark._run_sample(
            Client(),
            object(),
            np.zeros((1, 3072), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            0,
        )


def _write_synthetic_weight_archive(path: Path):
    arrays = {
        "__schema_version__": np.asarray([1], dtype=np.int64),
        "__vector_count__": np.asarray([233], dtype=np.int64),
    }
    for index in range(233):
        name = f"synthetic.{index}"
        arrays[f"{name}::values"] = np.asarray(
            [float(index), float(index + 1)],
            dtype=np.float64,
        )
        arrays[f"{name}::codes"] = np.asarray([0, 1, 0, 1], dtype=np.int32)
    np.savez_compressed(path, **arrays)


def test_weight_loader_uses_compact_sources_and_middle_cache(tmp_path):
    path = tmp_path / "weights.npz"
    _write_synthetic_weight_archive(path)

    bundle = load_weights(path, device="cpu")

    assert len(bundle) == 233
    assert bundle.constants.cache_mode == "middle"
    assert bundle.raw_vectors["synthetic.0"].slots == 4


@pytest.mark.skipif(
    not easyfhe.cuda.is_available(),
    reason="runtime packing requires CUDA",
)
def test_triton_packers_match_small_numpy_fixtures():
    values = easyfhe.as_tensor(
        np.asarray([1.5, -2.0], dtype=np.float64),
        dtype=easyfhe.float64,
        device="cuda",
    )
    codes = easyfhe.as_tensor(
        np.asarray([0, 1, 1, 0], dtype=np.int32),
        dtype=easyfhe.int32,
        device="cuda",
    )
    dictionary = pack_dictionary(values, codes, slots=4).cpu().numpy()
    np.testing.assert_array_equal(dictionary, [1.5, -2.0, -2.0, 1.5])

    coefficients = easyfhe.as_tensor(
        np.asarray([[2.0, 3.0], [5.0, 7.0]], dtype=np.float64),
        dtype=easyfhe.float64,
        device="cuda",
    )
    masks = easyfhe.as_tensor(
        np.asarray([[1, 0], [0, 1]], dtype=np.int32),
        dtype=easyfhe.int32,
        device="cuda",
    )
    mask_ids = easyfhe.as_tensor(
        np.asarray([0, 1], dtype=np.int32),
        dtype=easyfhe.int32,
        device="cuda",
    )
    factorized = pack_factorized(
        coefficients,
        masks,
        mask_ids,
        slots=4,
    ).cpu().numpy()
    np.testing.assert_array_equal(
        factorized,
        np.asarray([[2.0, 0.0, 3.0, 0.0], [0.0, 5.0, 0.0, 7.0]]),
    )


@pytest.mark.parametrize(
    ("channels", "slots", "rotation"),
    ((16, 64, -8), (32, 128, -16)),
)
def test_grouped_pointwise_order_matches_the_scalar_recurrence(
    channels,
    slots,
    rotation,
):
    rng = np.random.default_rng(1234)
    packed = rng.normal(size=(channels, slots))
    individual = packed[::-1]
    input_slots = np.linspace(-1.0, 1.0, slots)

    legacy = None
    for weight in individual:
        term = input_slots * weight
        legacy = term if legacy is None else legacy + term
        legacy = np.roll(legacy, rotation)

    terms = input_slots * packed
    grouped = terms[-1]
    for index in range(channels - 2, -1, -1):
        grouped = np.roll(grouped, rotation) + terms[index]
    grouped = np.roll(grouped, rotation)

    np.testing.assert_array_equal(grouped, legacy)
