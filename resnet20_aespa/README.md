# Canonical u64 ResNet20 AESPA

This example is one complete, fixed CUDA circuit intended to be copied when
building an EasyFHE application. It contains no backend selector, u32 branch,
H/L rail, or bootstrap-planner logic inside the model.

The cryptographic design is deliberately not exposed as command-line tuning:

- native u64 ciphertexts and plaintexts;
- fixed-scale CKKS with manual rescaling;
- `logN=16`, 59-bit rescale primes, and sparse ternary secret keys;
- `normal_giant` / `modraise_first` bootstrapping with level budget `(4, 4)`;
- four bootstrap contracts returning 5, 9, 11, and 12 usable levels;
- 18 active input limbs.

## Public API path

[`runtime.py`](runtime.py) shows the complete application setup in five steps;
[`main.py`](main.py) is intentionally only the CLI assembly point:

1. Declare the four `BootstrapSpec` values used by the graph.
2. Call `bs.requirements(...)` for context depth and bootstrap rotations.
3. Union those rotations with the model rotations and create one
   `CKKSContextSpec`.
4. Generate one context-bound `BootstrapProgram` per spec.
5. Load compact constants, pack them on CUDA, and evaluate the encrypted model.

The model runtime contains only server-side evaluation objects: context,
constants, and bootstrap programs. The secret-key client remains in the
benchmark driver for encryption and decryption.

## Code layout

```text
main.py             parse -> create runtime -> benchmark entry point
config.py           fixed circuit constants and the four-option run CLI
assets.py           pinned Hugging Face downloads and SHA-256 checks
runtime.py          Specs -> requirements -> context -> Programs -> weights
benchmark.py        warmup, timing, decryption, and validation
model/
  graph.py          ResNet20 graph and explicit bootstrap placement
  ops.py            convolution, grouped projection, AESPA activation
  layout.py         slot transforms, downsampling, pooling, key schedules
packing/
  weights.py        compact source schema, lazy constants, middle cache
  triton.py         batched expansion into slot-ready CUDA tensors
```

`_ResidualBlockSpec` captures the repeated same-shape block without hiding the
two downsampling blocks. The row and channel packing paths intentionally reuse
one rotation key sequentially instead of generating a key for every absolute
offset. Projection convolutions use reverse-packed grouped plaintexts and the
public batch MAC/giant-rotation pipeline.

## Run

Download the pinned weights and complete 10,000-example test archive once,
then run from this repository:

```bash
python -m resnet20_aespa.assets
PYTHONPATH=/path/to/EasyFHE-u64:. \
  python -m resnet20_aespa.main --warmup 1 --runs 1
```

The CLI contains only:

```text
--runs N       positive number of measured inferences
--warmup N     non-negative number of unmeasured inferences
--dataset PATH numeric images/labels NPZ
--weights PATH canonical compact weights NPZ
```

Every phase is CUDA-synchronized. Warmup executes encryption, all 20 layers,
decryption, and prediction, but is excluded from reported timings. Decryption
errors and non-finite logits fail immediately rather than being counted as an
incorrect prediction.

No model or dataset binary is tracked in Git. `assets.py` locks both repositories
to full commit hashes and verifies SHA-256 before placing files under the
ignored `resnet20_aespa/assets/` directory:

- complete test inputs:
  [`jizhuoran/easyfhe-resnet20-cifar10`](https://huggingface.co/datasets/jizhuoran/easyfhe-resnet20-cifar10);
- compact model:
  [`jizhuoran/easyfhe-resnet20-aespa`](https://huggingface.co/jizhuoran/easyfhe-resnet20-aespa).

The downloaded dataset is pickle-free: `images` is float64 with shape
`[10000, 3072]`, and `labels` is int64 with shape `[10000]`. The 2.1 MB weight
archive contains only graph-reachable compact parameters and layout
descriptors; the previous slot-expanded archive materialized 416 MiB.

On first use, batched Triton kernels expand a constant directly into its
slot-ready CUDA tensor, after which EasyFHE performs CKKS stage-1 pre-encoding.
`ConstantBundle(cache_mode="middle")` retains that reusable middle
representation, so on-the-fly packing and caching are independent: packing is
paid once, while level-specific plaintexts are still created for the state
requested by each operation. No slot-expanded model weight is stored on disk
or materialized on the CPU.

## Lightweight checks

```bash
PYTHONPATH=/path/to/EasyFHE-u64:. python -m pytest -q tests
```

The tests use synthetic data and compact-weight fixtures, so a fresh source
checkout can validate the fixed bootstrap requirements, CLI contract,
runtime/client separation, and packing kernels without downloading assets. A
download followed by `--warmup 0 --runs 1` is the end-to-end CUDA smoke test.
