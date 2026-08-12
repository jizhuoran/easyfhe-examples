# Canonical EasyFHE u64 examples

These examples are reference implementations for new EasyFHE applications.
Each one has a single tested u64 design, uses only public APIs, and keeps
benchmarking separate from context and model code. There are no backend,
plaintext-width, H/L rail, or experimental schedule switches in the main path.

All four examples follow the same setup sequence:

```text
BootstrapSpec -> requirements -> CKKSContextSpec -> BootstrapProgram -> evaluate
```

They use fixed-scale CKKS with manual rescaling. Multiplicative stages consume
u64 Q primes explicitly, with no mixed-width scale or rail conversion path.

## Examples

- [`benchmark`](benchmark/README.md): correctness-checked latency for seven
  representative public operations.
- [`bootstrap`](bootstrap/README.md): one full-slot complex bootstrap.
- [`resnet20_aespa`](resnet20_aespa/README.md): encrypted ResNet20 inference
  with compact weights, Triton runtime packing, and a shared middle cache.
  Model and test assets are downloaded from Hugging Face.
- [`thor`](thor/README.md): encrypted twelve-layer BERT inference with pinned
  model, tokenizer, and MRPC assets downloaded from Hugging Face.

## Run

Install EasyFHE, or point `PYTHONPATH` at a u64 EasyFHE checkout and this
repository:

```bash
PYTHONPATH=/path/to/EasyFHE-u64:. python -m benchmark.main --warmup 2 --runs 10
PYTHONPATH=/path/to/EasyFHE-u64:. python -m bootstrap.main --warmup 1 --runs 1
PYTHONPATH=/path/to/EasyFHE-u64:. python -m resnet20_aespa.assets
PYTHONPATH=/path/to/EasyFHE-u64:. python -m resnet20_aespa.main --warmup 1 --runs 1
PYTHONPATH=/path/to/EasyFHE-u64:. python -m thor.assets
PYTHONPATH=/path/to/EasyFHE-u64:. python -m thor.main --help
```

Warmup is synchronized and excluded from reported execution time. Correctness
checks decrypt the final measured bootstrap output and every reported model
result, reject non-finite values, and fail loudly instead of converting runtime
errors into incorrect predictions.
