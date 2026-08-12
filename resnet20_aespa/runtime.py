"""Build the server runtime for the canonical u64 ResNet20 circuit."""

from dataclasses import dataclass

import easyfhe.bs.openfhe as bs
import easyfhe.fhe as fhe

from .config import (
    DNUM,
    FIRST_PRIME_BITS,
    INPUT_LIMBS,
    LOG_N,
    NETWORK_ROTATIONS,
    RESCALE_PRIME_BITS,
    SECRET_KEY_DIST,
    bootstrap_specs,
)
from .packing import PackedWeightBundle, load_weights


@dataclass(frozen=True)
class ResNet20Runtime:
    """Server-side objects needed to evaluate one encrypted input."""

    context: fhe.Context
    weights: PackedWeightBundle
    bootstrap_programs: dict[int, bs.BootstrapProgram]


def create_runtime(weights_path) -> tuple[fhe.Client, ResNet20Runtime]:
    """Plan and build the fixed context, programs, and packed constants."""

    # 1. Describe every bootstrap call made by the circuit.
    specs = bootstrap_specs()

    # 2. Ask the planner for the context depth and bootstrap rotation keys.
    requirements = bs.requirements(
        specs,
        log_n=LOG_N,
        secret_key_dist=SECRET_KEY_DIST,
    )
    rotations = tuple(
        dict.fromkeys((*NETWORK_ROTATIONS, *requirements.rotations))
    )

    # 3. Generate one fixed-scale u64 context satisfying those requirements.
    client, context = fhe.generate_client_context(
        fhe.CKKSContextSpec(
            depth=requirements.context_depth,
            log_n=LOG_N,
            dnum=DNUM,
            dcrt_bits=RESCALE_PRIME_BITS,
            first_mod=FIRST_PRIME_BITS,
            secret_key_dist=SECRET_KEY_DIST,
            scale_mode="fixed",
            rescale_policy="manual",
            rotations=rotations,
        ),
        device="cuda",
    )
    if INPUT_LIMBS > context.max_limbs:
        raise ValueError(
            f"INPUT_LIMBS={INPUT_LIMBS} exceeds context.max_limbs={context.max_limbs}"
        )

    # 4. Bind every bootstrap contract to this context.
    programs = {
        spec.output_levels: bs.generate(context, spec) for spec in specs
    }

    # 5. Load compact sources that the graph packs on first use.
    weights = load_weights(weights_path, device=context.device)

    print(
        "u64 context:",
        f"depth={requirements.context_depth}",
        f"limbs={context.max_limbs}",
        f"input_limbs={INPUT_LIMBS}",
        f"planned_rotations={len(rotations)}",
    )
    print(
        "bootstrap programs:",
        ", ".join(
            f"{levels} levels/{levels + 1} limbs" for levels in sorted(programs)
        ),
    )
    print(f"compact runtime-packed constants: {len(weights)}")
    return client, ResNet20Runtime(context, weights, programs)


__all__ = ["ResNet20Runtime", "create_runtime"]
