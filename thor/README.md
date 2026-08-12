# THOR: canonical u64 BERT inference

This directory is the canonical EasyFHE u64 THOR example. It exposes one
validated CUDA design instead of an algorithm-selection surface: twelve BERT
encoder layers, encrypted pooler and classifier, and a plaintext NumPy
reference used only for correctness checks.

## Install

Install EasyFHE separately, either as a package or by placing its repository on
`PYTHONPATH`. Then install the example's data/model dependencies:

```bash
python -m pip install -r thor/requirements.txt
```

No EasyFHE version is bundled or pinned by this example.

## Assets

Only application source is tracked in GitHub. The validated 418 MB BERT-base
checkpoint and matching tokenizer are version-pinned in
[`jizhuoran/easyfhe-thor-mrpc`](https://huggingface.co/jizhuoran/easyfhe-thor-mrpc).
MRPC comes directly from the official Hugging Face `glue/mrpc` dataset at a
fixed commit; the old 268 MB local directory was almost entirely disposable
`datasets` cache files and is not redistributed.

Download and verify all assets into the ignored `thor/assets` directory:

```bash
python -m thor.assets
```

The downloader verifies the model SHA-256, saves a clean MRPC `DatasetDict`,
and never depends on a moving Hub `main` branch. `--output` can select another
asset root. On a proxy where Xet connection shutdown is unreliable, force the
ordinary resumable HTTP path with:

```bash
HF_HUB_DISABLE_XET=1 python -m thor.assets
```

Before creating any FHE keys, the loader validates the fixed 201-tensor
BERT-base schema, every tensor shape, and finite floating-point values. Sample
ranges are likewise checked before the CUDA context is generated.

## Run

```bash
python -m thor.main --split train --warmup 1 --runs 1 --start-index 1
```

The default paths are those created by `thor.assets`. The
`--dataset`, `--weights`, and `--tokenizer` overrides remain available for an
alternate local asset root.

Warmup executions are synchronized and excluded from the reported inference
times. Every warmup and measured run must produce finite logits, the same
prediction as the NumPy reference, and a relative-L2 logit error within the
configured tolerance. A mismatch stops the run immediately.

`python -m thor.main --help` lists the complete CLI. Cryptographic and graph choices
are deliberately not command-line options; they are documented in
[ALGORITHM.md](ALGORITHM.md).

## Code map

- `main.py`: parse assets -> load/validate -> create runtime -> benchmark.
- `assets.py`: pinned Hugging Face download and integrity checks.
- `config.py`: fixed circuit constants, per-layer plans, and the small run CLI.
- `reference.py`: tokenizer/sample preparation and NumPy BERT.
- `runtime.py`: bootstrap requirements, context/program assembly, and setup-time
  encrypted constants. It contains no model execution.
- `benchmark.py`: synchronized warmup, measurement, decryption,
  correctness checks, and summary output.
- `fhe_ops.py`: reusable low-level u64 ciphertext and plaintext operations.
- `model/bert.py`: encoder-layer orchestration, twelve-layer graph, pooler, and
  classifier.
- `model/attention.py`: grouped Q/K/V, score/context products, softmax, and the
  first layer normalization.
- `model/feed_forward.py`: dense/GELU/dense and second layer normalization.
- `model/approximations.py`: shared polynomial and inverse-square-root circuits.
- `packing/store.py`: per-layer weight lifetime, caches, and release policy.
- `packing/weights.py`: safetensors-to-slot recipes and lazy packed sources.
- `packing/masks.py`: static mask and block-layout recipes.
- `packing/triton.py`: batched CUDA packing kernels.

The server-side `ThorRuntime` contains no secret-bearing client. Two fixed
ciphertexts needed by inverse and layer-normalization routines are encrypted
once at setup at `context.max_limbs`, then aligned downward inside the graph.

Large BERT matrices remain in their ordinary safetensors form. A layer is
loaded onto CUDA, packed lazily in batches by Triton when its constants are
requested, and released before the next layer is loaded. Runtime packing and
plaintext caching are separate choices: static and per-sample masks use a
middle cache, while layer weights use cache mode `none` to bound GPU memory.
