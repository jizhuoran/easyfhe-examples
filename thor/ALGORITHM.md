# Canonical THOR u64 design

## Fixed cryptographic plan

THOR uses a CUDA u64 CKKS context with these fixed choices:

- `log_n=16`, `2^15` slots;
- depth 30 with a 60-bit first prime and 59-bit rescale primes;
- `dnum=3`, sparse ternary secret;
- fixed scaling and manual rescaling;
- bootstrap level budget `(3, 3)`, 14 output levels,
  `normal_giant`, `modraise_first`;
- input ciphertexts at 10 limbs.

`runtime.plan_runtime` asks the bootstrap planner for its depth and rotations,
then merges those rotations with the exact attention, transpose, pooler, and
classifier schedule. Equivalent left/right offsets are canonicalized before
key generation. Context creation consumes that single requirements result; the
application does not independently re-plan bootstrap keys.

## Sample and client/server boundary

`reference.prepare_sample` tokenizes one sentence pair to 128 tokens, runs the
NumPy BERT reference, packs the `(128, 768)` embedding into four complex CKKS
messages, and creates one eight-vector attention-mask bundle. The same mask
bundle is reused by all twelve encrypted layers.

The client encrypts inputs and decrypts the two output logits only in the
benchmark layer. The encrypted graph does not receive a client. During setup,
the client also encrypts two public fixed vectors at `context.max_limbs`:

1. the inverse numerator mask (12 active lanes per 16-lane block);
2. the layer-normalization first-slot mask.

The server aligns these ciphertexts to the required lower limb count. This
keeps secret-bearing client state out of `ThorRuntime` and the model graph.

## Encoder layer

Each layer follows one path:

1. grouped Q/K/V packed plaintext-ciphertext matrix products;
2. real extraction and cached baby/giant key transpose;
3. cached baby/giant attention-score products;
4. paired score bootstrap;
5. polynomial exponential, attention mask, inverse refinement, and softmax;
6. cached attention-context products;
7. grouped attention dense, residual, and attention layer normalization;
8. grouped feed-forward dense, GELU, dense, residual bootstrap, and feed-forward
   layer normalization;
9. paired bootstrap and real-to-complex boundary before the next layer.

`LayerPlan` makes the only validated per-layer arithmetic exceptions explicit:

- layer 2 uses the second exponential polynomial, `1/1024` attention-key
  scaling, tighter inverse epsilon, an extra inverse refinement, and a forced
  softmax bootstrap;
- layers 9 and 10 use the wider feed-forward layer-normalization variance range;
- all other layers share the ordinary plan.

The graph entry is `model.bert.infer_encrypted`. Runtime setup and encrypted
model execution are deliberately separate: `runtime.py` builds immutable
server objects, while `model/` consumes them without parsing benchmark or asset
options.

## Rotation and packing strategy

Q/K/V reuses one packed batch for each of four outputs rather than packing once
per projection and diagonal. Attention score/context and key transpose reuse
cached baby rotations and apply their giant rotations afterward. Input copies
are generated successively.

Static graph masks—including copy, transpose, attention score/context, and
attention-dense masks—live in one shared `ThorMaskPacker` bundle with middle
cache enabled. The per-sample attention mask is a separate single shared bundle,
also middle-cached and cleared after that sample. Layer weight bundles use cache
mode `none` and are released before the next layer is loaded; large matrices
remain lazily packed through the application-owned
`PackedWeightBundle.raw_vectors` interface rather than EasyFHE internals.
The weight store owns lifetime only; `packing/weights.py` owns the raw tensor to
slot-layout recipes, and `packing/triton.py` owns the Triton implementation.

## Pooler and classifier

The final eight real ciphertexts feed the loop pooler dense operation, a fixed
`1/40` scale, the polynomial tanh with bootstrap, and the two-output loop
classifier. The benchmark decrypts slot zero from each classifier ciphertext
and compares those logits with the NumPy reference.
