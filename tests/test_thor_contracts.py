from dataclasses import FrozenInstanceError, fields
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from thor import benchmark, reference, runtime
from thor.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_TOKENIZER_PATH,
    DEFAULT_WEIGHTS_PATH,
    DEVICE,
    INPUT_LIMBS,
    LayerPlan,
    RunConfig,
    parse_args,
)
from thor.assets import (
    DATASET_REVISION,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_SHA256,
)
from thor.model import infer_encrypted
from thor.model import run_encoder_layer


def test_layer_plan_records_only_the_two_arithmetic_exceptions():
    ordinary = LayerPlan.for_layer(0)
    softmax2 = LayerPlan.for_layer(2)
    late_norm = LayerPlan.for_layer(9)

    assert ordinary.softmax_variant == "softmax1"
    assert ordinary.attention_key_scale == 1 / 512
    assert softmax2.softmax_variant == "softmax2"
    assert softmax2.attention_key_scale == 1 / 1024
    assert softmax2.refine_softmax_inverse
    assert softmax2.force_softmax_bootstrap
    assert (late_norm.ff_layernorm_min_var, late_norm.ff_layernorm_max_var) == (
        0.75,
        2500.0,
    )
    with pytest.raises(FrozenInstanceError):
        ordinary.index = 4
    with pytest.raises(ValueError):
        LayerPlan.for_layer(12)


def test_cli_has_only_asset_run_and_dataset_options(tmp_path):
    dataset = tmp_path / "dataset"
    tokenizer = tmp_path / "tokenizer"
    weights = tmp_path / "model.safetensors"
    dataset.mkdir()
    tokenizer.mkdir()
    weights.touch()

    config = parse_args(
        [
            "--dataset",
            str(dataset),
            "--weights",
            str(weights),
            "--tokenizer",
            str(tokenizer),
            "--split",
            "validation",
            "--warmup",
            "2",
            "--runs",
            "3",
            "--start-index",
            "4",
        ]
    )
    assert config == RunConfig(
        dataset_path=dataset.resolve(),
        weights_path=weights.resolve(),
        tokenizer_path=tokenizer.resolve(),
        split="validation",
        warmup=2,
        runs=3,
        start_index=4,
    )
    assert DEVICE == "cuda"
    assert INPUT_LIMBS == 10


def test_default_assets_match_the_pinned_hugging_face_layout():
    assert DEFAULT_WEIGHTS_PATH.parent == DEFAULT_TOKENIZER_PATH
    assert DEFAULT_DATASET_PATH.name == "mrpc"
    assert MODEL_REPO_ID == "jizhuoran/easyfhe-thor-mrpc"
    assert len(MODEL_REVISION) == 40
    assert len(MODEL_SHA256) == 64
    assert DATASET_REVISION == "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"


def test_run_config_and_server_runtime_have_separate_responsibilities():
    assert [field.name for field in fields(RunConfig)] == [
        "dataset_path",
        "weights_path",
        "tokenizer_path",
        "split",
        "warmup",
        "runs",
        "start_index",
    ]
    assert [field.name for field in fields(runtime.ThorRuntime)] == [
        "plan",
        "context",
        "bootstrap_program",
        "weights",
        "masks",
        "encrypted_constants",
    ]
    assert "weights_path" in inspect.signature(runtime.create_runtime).parameters
    assert "config" not in inspect.signature(runtime.create_runtime).parameters
    assert "runtime" in inspect.signature(infer_encrypted).parameters


def test_runtime_plan_merges_application_and_bootstrap_rotations(monkeypatch, tmp_path):
    requirements = SimpleNamespace(context_depth=30, rotations=(7, 8, 9))
    monkeypatch.setattr(
        runtime.bs,
        "BootstrapSpec",
        lambda **values: SimpleNamespace(**values),
        raising=False,
    )
    monkeypatch.setattr(
        runtime.bs,
        "requirements",
        lambda *args, **kwargs: requirements,
        raising=False,
    )
    monkeypatch.setattr(runtime, "attention_rotations", lambda: (1, 7, 2))
    monkeypatch.setattr(runtime, "pooler_classifier_rotations", lambda: (2, 3))
    plan = runtime.plan_runtime()

    assert plan.bootstrap_requirements is requirements
    assert plan.rotations == (1, 7, 2, 3, 8, 9)
    assert plan.bootstrap_spec.level_budget == (3, 3)
    assert plan.bootstrap_spec.output_levels == 14
    assert plan.bootstrap_spec.mode == "modraise_first"


def test_runtime_plan_canonicalizes_equivalent_rotation_keys():
    plan = runtime.plan_runtime()

    assert len(plan.rotations) == 88
    assert 32752 in plan.rotations
    assert -16 not in plan.rotations
    assert 32256 in plan.rotations
    assert -512 not in plan.rotations


def test_server_graph_has_no_client_parameter():
    assert "client" not in run_encoder_layer.__annotations__
    assert "client" not in runtime.ThorRuntime.__dataclass_fields__
    assert "config" not in runtime.ThorRuntime.__dataclass_fields__
    assert "client" not in inspect.signature(run_encoder_layer).parameters


def test_warmup_and_measured_samples_have_separate_released_mask_caches(
    monkeypatch,
):
    bundles = []

    class Bundle:
        def __init__(self):
            self.clear_count = 0

        def clear_cache(self):
            self.clear_count += 1

    def prepare(index, assets):
        bundle = Bundle()
        bundles.append(bundle)
        return SimpleNamespace(index=index, attention_mask_bundle=bundle)

    def run_sample(client, runtime_value, sample):
        return benchmark.SampleResult(
            index=sample.index,
            label=0,
            reference_prediction=0,
            prediction=0,
            reference_logits=np.asarray([1.0, 0.0]),
            logits=np.asarray([1.0, 0.0]),
            relative_l2_error=0.0,
            encrypt_seconds=0.0,
            inference_seconds=0.0,
            decrypt_seconds=0.0,
        )

    monkeypatch.setattr(benchmark, "prepare_sample", prepare)
    monkeypatch.setattr(benchmark, "_run_sample", run_sample)
    config = SimpleNamespace(
        start_index=0,
        runs=2,
        warmup=1,
        split="train",
        depth=30,
        input_limbs=10,
        bootstrap_level_budget=(3, 3),
        bootstrap_output_levels=14,
    )
    runtime_value = SimpleNamespace(
        context=SimpleNamespace(max_limbs=31),
        plan=SimpleNamespace(
            rotations=(),
            bootstrap_requirements=SimpleNamespace(context_depth=30),
        ),
    )

    benchmark.run_benchmark(
        object(),
        runtime_value,
        SimpleNamespace(split=[object(), object()]),
        config,
    )

    assert len(bundles) == 3
    assert len({id(bundle) for bundle in bundles}) == 3
    assert [bundle.clear_count for bundle in bundles] == [1, 1, 1]


def test_result_validation_rejects_nonfinite_reference():
    sample = SimpleNamespace(
        index=0,
        reference_logits=np.asarray([np.nan, 0.0]),
        reference_prediction=0,
    )
    with pytest.raises(FloatingPointError, match="reference logits"):
        benchmark._validate_result(
            sample,
            np.asarray([1.0, 0.0]),
            0,
            np.nan,
            tolerance=5e-2,
        )


def test_benchmark_request_is_validated_before_runtime_setup():
    config = SimpleNamespace(start_index=2, runs=1, split="train")
    with pytest.raises(ValueError, match="exceed"):
        benchmark.validate_benchmark_request(
            config, SimpleNamespace(split=[object(), object()])
        )


def test_fixed_model_schema_has_all_graph_tensors():
    schema = reference.model_schema()

    assert len(schema) == 201
    assert schema["bert.embeddings.word_embeddings.weight"] == (30522, 768)
    assert schema["bert.encoder.layer.11.output.dense.weight"] == (768, 3072)
    assert schema["cls.seq_relationship.weight"] == (2, 768)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (np.zeros((3,), dtype=np.float64), ValueError),
        (np.zeros((2,), dtype=np.int64), TypeError),
        (np.asarray([np.nan, 0.0]), FloatingPointError),
    ],
)
def test_model_schema_rejects_bad_tensor_shape_dtype_and_values(
    monkeypatch, value, error
):
    monkeypatch.setattr(reference, "model_schema", lambda: {"weight": (2,)})
    with pytest.raises(error):
        reference.validate_model_state({"weight": value})


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--warmup", "-1", "--warmup must be non-negative"),
        ("--runs", "0", "--runs must be positive"),
        ("--start-index", "-1", "--start-index must be non-negative"),
    ],
)
def test_cli_rejects_invalid_run_counts(option, value, message, capsys):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset",
                str(Path("missing-dataset")),
                "--weights",
                str(Path("missing-weights")),
                "--tokenizer",
                str(Path("missing-tokenizer")),
                option,
                value,
            ]
        )
    assert message in capsys.readouterr().err
