# EasyFHE ResNet-20 AESPA Demo

This repository is a standalone EasyFHE example for encrypted CIFAR-10
ResNet-20 inference. It keeps the AESPA encrypted inference path intact while
packaging the code as a small, reproducible demo that can be run outside the
EasyFHE source tree.

## What Is Included

```text
README.md
requirements.txt
scripts/
  run_resnet20.sh
src/
  resnet20_aespa.py   # CLI, EasyFHE context setup, dataset loop, reporting
  model.py            # ResNet-20 AESPA encrypted inference graph
  convs.py            # encrypted convolution, pointwise conv, AESPA nonlinear ops
  utils.py            # CIFAR-10 reader, packing/layout helpers, NPZ weight cache
assets/
  sample_output.txt
data/
  cifar10/test_batch.bin
resnet20_aespa_weights.npz
```

The default paths expect the CIFAR-10 binary test batch at
`data/cifar10/test_batch.bin` and the packed AESPA weights at
`resnet20_aespa_weights.npz`.

## Setup

Use Python 3.10 or newer. A virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`easyfhe` must be importable in the environment. If you are developing against a
local EasyFHE checkout, install that package first, then install this demo's
requirements.

## Data And Weights

This demo reads CIFAR-10 in the original binary format. The only required data
file is:

```text
data/cifar10/test_batch.bin
```

The model weights are packed into:

```text
resnet20_aespa_weights.npz
```

You can override either path without editing code:

```bash
EASYFHE_CIFAR10_TEST_BATCH=/path/to/test_batch.bin \
  ./scripts/run_resnet20.sh

EASYFHE_RESNET20_AESPA_WEIGHTS=/path/to/resnet20_aespa_weights.npz \
  ./scripts/run_resnet20.sh
```

If you prefer to point at a data directory, set
`EASYFHE_RESNET20_AESPA_DATA_DIR`; the script will look for
`$EASYFHE_RESNET20_AESPA_DATA_DIR/cifar10/test_batch.bin`.

## Run

Run one encrypted inference, which is the default:

```bash
./scripts/run_resnet20.sh
```

Run more CIFAR-10 test images:

```bash
./scripts/run_resnet20.sh 5
```

The Python entrypoint can also be called through module mode:

```bash
python -m src.resnet20_aespa --total 1
```

EasyFHE runtime arguments are passed through. For example:

```bash
EASYFHE_DEVICE=cuda ./scripts/run_resnet20.sh 1
```

## Runtime Behavior

At startup the demo prints the CKKS/AESPA parameters, input paths, selected
device, generated crypto context, and number of packed weight arrays. Each image
then runs through:

1. CIFAR-10 normalization and packing.
2. EasyFHE CKKS encryption.
3. Encrypted ResNet-20 AESPA inference.
4. Decryption of the final logits.
5. Label/prediction comparison and timing summary.

Fully homomorphic inference is compute-heavy. CPU runs can take a long time for
a single image; GPU runs depend on the EasyFHE installation and available CUDA
hardware.

See `assets/sample_output.txt` for the shape of the expected output.

## Design Notes

This demo is meant to show a realistic EasyFHE inference pipeline without
pulling in the EasyFHE development tree. The main design choices are:

`resnet20_aespa.py` is the runner

The entrypoint owns the parts an application author usually controls:
configuration, input paths, EasyFHE context construction, bootstrap constant
generation, runtime assembly, timing, decryption, and result checking.

`model.py` is the encrypted network

The model file keeps the AESPA ResNet-20 graph readable: initial convolution,
three residual stages, downsampling blocks, bootstraps at the selected points,
and the final average-pool/linear head. It receives an `AespaRuntime` containing
the crypto context, weights, config, and bootstrap constants, then performs only
encrypted operations.

`convs.py` is the FHE operator layer

Convolution, pointwise convolution, shortcut addition, and AESPA nonlinear
evaluation are isolated from the high-level network graph. This makes it easier
to inspect where rotations, plaintext multiplications, rescaling, and slot
movement happen.

`utils.py` holds data, packing, and weights

The CIFAR-10 reader normalizes images with the ResNet training mean/std and
returns the 3,072-value vector expected by the encrypted model. The same file
also contains layout helpers and the NPZ-backed weight loader.

### Context Construction

EasyFHE contexts need to know the encrypted program shape before execution. This
demo makes those choices explicit in `AespaConfig`:

```text
rotations          all rotation keys needed by convolutions, packing, pooling, and skips
maxLevelsRemaining bootstrap depth target for the AESPA circuit
logBsSlots         bootstrap slot count
logN/dnum/moduli   CKKS ring and modulus-chain parameters
rescaleTech        FIXEDMANUAL, matching the original AESPA path
```

`build_runtime()` first creates `BootstrapSpec` values, then calls
`fhe.generate_context(...)`. After the context exists, it creates bootstrap
constants with `fhe.generate_bootstrap_constants(...)` and passes them into the
model runtime. The model later uses them at the residual-block bootstrap sites
with `fhe.homo_bootstrap(...)`.

This separation is intentional: context setup is an application concern, while
the model graph only consumes the prepared context and constants.

### Weight Cache

Encrypted inference repeatedly multiplies ciphertexts by encoded plaintext
weights, masks, and biases. Encoding those plaintexts can become a noticeable
part of runtime, especially after the first image has warmed the main EasyFHE
objects.

`WeightPack` supports four cache modes through `EASYFHE_WEIGHT_CACHE_MODE`:

```text
none    never cache prepared or encoded plaintexts
middle  cache EasyFHE prepared plaintext intermediates
plain   cache final plaintext objects; default
both    cache both layers
```

The cache key includes the weight name, level, slot count, scale, extension
state, and crypto-context identity. That is important because the same logical
weight can require different plaintext encodings at different CKKS levels or
slot counts.

At the end of a run the demo prints cache entries, approximate cache memory,
hits, and misses. For multi-image runs, this helps explain why the first image
often behaves differently from later images.

### Result Checking

The final encrypted logits are decrypted only after the encrypted ResNet-20
pipeline completes. The demo prints the predicted class, the CIFAR-10 label, a
running accuracy, and the decrypted logits. This keeps correctness checking
simple while still making it clear that the model path itself is encrypted.

### Small Practical Choices

The shell script always runs module mode, `python -m src.resnet20_aespa`, so the
source files can use clean package-relative imports. The project avoids
framework-specific training code, legacy EasyFHE example readers, and PyTorch
model definitions; the only model artifact used at runtime is the packed NPZ.

## Configuration

Useful environment variables:

```text
EASYFHE_TOTAL                     default number of images when no script arg is passed
EASYFHE_DEVICE                    EasyFHE device, usually cpu or cuda
EASYFHE_CIFAR10_TEST_BATCH        exact CIFAR-10 test_batch.bin path
EASYFHE_RESNET20_AESPA_DATA_DIR   directory containing cifar10/test_batch.bin
EASYFHE_RESNET20_AESPA_WEIGHTS    exact resnet20_aespa_weights.npz path
EASYFHE_DNUM                      CKKS decomposition number, default 3
EASYFHE_WEIGHT_CACHE_MODE         none, middle, plain, or both; default plain
```

## Troubleshooting

`ModuleNotFoundError: easyfhe`

Install EasyFHE into the active Python environment. This demo intentionally
treats EasyFHE as a dependency instead of importing from an EasyFHE source tree.

`CIFAR-10 test batch ... does not exist`

Check that `data/cifar10/test_batch.bin` exists, or set
`EASYFHE_CIFAR10_TEST_BATCH` to the exact file path.

`Weight npz ... does not exist`

Check that `resnet20_aespa_weights.npz` exists, or set
`EASYFHE_RESNET20_AESPA_WEIGHTS`.

Very slow first run

The first image includes context generation, bootstrap constant generation, and
weight plaintext preparation. Use `--total` or `./scripts/run_resnet20.sh N` to
compare later images after the cache is warm.

CUDA import or runtime warnings

Use `EASYFHE_DEVICE=cpu` for a CPU-only run, or verify that your EasyFHE
installation matches the CUDA runtime on the machine.
