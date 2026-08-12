# Canonical u64 operation benchmark

This example measures a small, fixed set of representative public EasyFHE
operations on CUDA:

- encryption and decryption;
- ciphertext addition;
- plaintext and ciphertext multiplication followed by one u64 rescale;
- rotation; and
- full-slot complex bootstrapping.

It deliberately is not a parameter sweep or simulator-data pipeline. One
fixed-scale u64 context is planned through `BootstrapSpec` and
`bs.requirements`, every operation receives independent immutable inputs, and
setup/key generation is outside the reported timings.

Run from the repository root:

```bash
PYTHONPATH=/path/to/EasyFHE-u64:. \
python -m benchmark.main --warmup 2 --runs 10
```

Warmup is synchronized and excluded independently for every operation. Each
reported result is decrypted after measurement and checked against its clear
NumPy result; non-finite values or excessive error stop the benchmark.

The plaintext-multiplication case also demonstrates the canonical on-GPU
constant path: a slot-ready CUDA tensor is wrapped in `PackedRaw`, transformed
to the CKKS middle representation on first use, and retained by
`ConstantBundle(cache_mode="middle")`. Level-specific plaintext material is not
kept resident.
