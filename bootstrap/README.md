# Full-slot complex bootstrap — u64 CKKS

This example bootstraps all `2^15` complex slots of an `N=2^16` CKKS
ciphertext. Every input slot contains nonzero real and imaginary components.

It presents the public application flow in order:

```text
BootstrapSpec -> requirements -> CKKSContextSpec -> BootstrapProgram -> bootstrap
```

The canonical path uses CUDA, fixed-scale u64 CKKS with manual rescaling,
`normal_giant`, `modraise_first`, and EasyFHE's default CUDA rotation-key
residency. The application supplies only the bootstrap specification and its
derived context/key requirements; the generated program owns the constants,
execution schedule, raise target, and promised output state.

Run from the examples repository root:

```bash
python -m bootstrap.main --warmup 1 --runs 5
```

When using an EasyFHE source checkout instead of an installed package:

```bash
PYTHONPATH=/path/to/EasyFHE-u64:. python -m bootstrap.main --warmup 1 --runs 5
```

The reported time contains only synchronized `bs.bootstrap()` execution.
Context and key generation, bootstrap-program generation, encryption,
decryption, warmup runs, and correctness checks are excluded.

After timing, the example decrypts all slots and verifies that every output is
finite, the ciphertext state equals `BootstrapProgram.output_state`, and the
maximum complex absolute error stays within the configured tolerance.
