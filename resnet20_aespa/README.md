# EasyFHE ResNet20 AESPA Demo

This standalone example mirrors the NPZ-backed `examples/resnet20_aespa`
implementation from EasyFHE. It runs encrypted CIFAR-10 ResNet20 inference with
the packed AESPA weights and preprocessed dataset included in this directory.

## Layout

```text
resnet20_aespa/
  __init__.py
  main.py                  # CLI, EasyFHE context setup, dataset loop, reporting
  model.py                 # encrypted ResNet20 AESPA graph
  ops.py                   # encrypted convolution, pointwise conv, AESPA ops
  layout.py                # packing and slot-layout helpers
  data/cifar10/test_batch.npz
  resnet20_aespa_weights.npz
requirements.txt
```

## Setup

Use the CUDA 12.9 wheel environment from the repository root:

```bash
source .venv-cu129/bin/activate
```

Or create a fresh Python 3.10 environment and install:

```bash
python -m pip install -r resnet20_aespa/requirements.txt
```

## Run

From the repository root:

```bash
python -m resnet20_aespa.main --total 1
```

The Python entrypoint defaults to `EASYFHE_BOOTSTRAP_STRATEGY=normal_giant`,
`EASYFHE_WEIGHT_CACHE_MODE=mix`, `EASYFHE_WEIGHT_PLAIN_CACHE_POLICY=small_first`,
and sets `EASYFHE_WEIGHT_PLAIN_CACHE_GB` to the selected CUDA GPU's total memory
minus 24 GiB when the variable is not already set.

Run more CIFAR-10 test images:

```bash
python -m resnet20_aespa.main --total 5
```

The Python entrypoint always performs one fixed warmup iteration before the
measured loop. `--total` only controls the number of measured iterations.

The default device is CUDA. To run on CPU:

```bash
python -m resnet20_aespa.main --device cpu
```

## Data And Weights

The default input data and packed weights live inside this directory:

```text
resnet20_aespa/data/cifar10/test_batch.npz
resnet20_aespa/resnet20_aespa_weights.npz
```

The dataset NPZ stores a single `samples` object array of `(image, label)` tuples.

Override the preprocessed CIFAR-10 dataset with:

```bash
EASYFHE_RESNET20_AESPA_DATASET=/path/to/test_batch.npz \
python -m resnet20_aespa.main
```

Override the weight artifact with:

```bash
EASYFHE_RESNET20_AESPA_WEIGHTS=/path/to/resnet20_aespa_weights.npz \
python -m resnet20_aespa.main
```

## Configuration

Useful environment variables:

```text
EASYFHE_TOTAL                       default number of images when no script arg is passed
EASYFHE_DEVICE                      EasyFHE device, usually cpu or cuda
EASYFHE_RESNET20_AESPA_DATASET      exact preprocessed CIFAR-10 NPZ path
EASYFHE_RESNET20_AESPA_WEIGHTS      exact resnet20_aespa_weights.npz path
EASYFHE_DNUM                        CKKS decomposition number, default 3
EASYFHE_DCRT_BITS                   CKKS dcrt bits, default 59
EASYFHE_FIRST_MOD                   CKKS first modulus bits, default 60
EASYFHE_INPUT_LEVEL                 encrypted input level, default 13
EASYFHE_POST_BOOTSTRAP_LEVELS       post-bootstrap circuit depth, default 11
EASYFHE_BOOTSTRAP_STRATEGY          double_hoist, normal_giant, or normal_bsgs; default normal_giant
EASYFHE_BOOTSTRAP_MODE              classic, modraise_first, slots_first, or stc_first
EASYFHE_SECRET_KEY_DIST             SPARSE_TERNARY or UNIFORM_TERNARY
EASYFHE_WEIGHT_CACHE_MODE           none, middle, plain, both, mix; default mix
EASYFHE_WEIGHT_PLAIN_CACHE_GB       optional plain cache limit in GiB; CUDA default is selected GPU memory minus 24 GiB
EASYFHE_WEIGHT_PLAIN_CACHE_POLICY   first_fit, lru, small_first; default small_first
```
