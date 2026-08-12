"""Build the server runtime for the canonical u64 THOR circuit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import easyfhe.bs.openfhe as bs
from easyfhe import fhe

from .config import (
    BOOTSTRAP_LEVEL_BUDGET,
    BOOTSTRAP_MODE,
    BOOTSTRAP_OUTPUT_LEVELS,
    BOOTSTRAP_STRATEGY,
    DEPTH,
    DEVICE,
    DNUM,
    FIRST_PRIME_BITS,
    INPUT_LIMBS,
    LOG_N,
    LOG_SLOTS,
    RESCALE_POLICY,
    RESCALE_PRIME_BITS,
    SCALE_MODE,
    SECRET_KEY_DIST,
    SLOTS,
)
from .fhe_ops import CONJUGATE_ROTATION
from .model import pooler_classifier_rotations
from .model.attention import attention_rotations
from .packing import WeightStore


@dataclass(frozen=True)
class RuntimePlan:
    bootstrap_spec: bs.BootstrapSpec
    bootstrap_requirements: bs.BootstrapRequirements
    rotations: tuple[int, ...]


@dataclass(frozen=True)
class EncryptedConstants:
    """Setup-time encryptions aligned downward by the server graph."""

    inverse_numerator: fhe.Cipher
    layernorm_one: fhe.Cipher


@dataclass(frozen=True)
class ThorRuntime:
    """Secret-free objects needed to evaluate encrypted THOR inputs."""

    plan: RuntimePlan
    context: fhe.Context
    bootstrap_program: bs.BootstrapProgram
    weights: WeightStore
    masks: fhe.ConstantBundle
    encrypted_constants: EncryptedConstants


def plan_runtime() -> RuntimePlan:
    """Plan bootstrap depth and all application rotation keys once."""

    bootstrap_spec = bs.BootstrapSpec(
        log_slots=LOG_SLOTS,
        level_budget=BOOTSTRAP_LEVEL_BUDGET,
        output_levels=BOOTSTRAP_OUTPUT_LEVELS,
        strategy=BOOTSTRAP_STRATEGY,
        mode=BOOTSTRAP_MODE,
    )
    requirements = bs.requirements(
        bootstrap_spec,
        log_n=LOG_N,
        secret_key_dist=SECRET_KEY_DIST,
    )
    rotations = _canonical_rotations(
        (
            *attention_rotations(),
            *pooler_classifier_rotations(),
            *requirements.rotations,
        ),
        slots=SLOTS,
    )
    return RuntimePlan(bootstrap_spec, requirements, rotations)


def _canonical_rotations(rotations, *, slots: int) -> tuple[int, ...]:
    """Deduplicate equivalent left/right slot rotations before key generation."""

    canonical = []
    for rotation in rotations:
        rotation = int(rotation)
        key = rotation if rotation == CONJUGATE_ROTATION else rotation % int(slots)
        if key and key not in canonical:
            canonical.append(key)
    return tuple(canonical)


def _encrypt_graph_constants(client, context) -> EncryptedConstants:
    inverse_numerator = np.asarray(
        ([1.0] * 12 + [0.0] * 4) * (SLOTS // 16),
        dtype=np.float64,
    )
    layernorm_one = np.asarray(
        ([1.0] + [0.0] * 15) * (SLOTS // 16),
        dtype=np.float64,
    )

    def encrypt(values):
        return client.encrypt(
            values,
            slots=SLOTS,
            device=DEVICE,
            cur_limbs=int(context.max_limbs),
        )

    return EncryptedConstants(
        inverse_numerator=encrypt(inverse_numerator),
        layernorm_one=encrypt(layernorm_one),
    )


def create_runtime(weights_path: str | Path) -> tuple[fhe.Client, ThorRuntime]:
    """Create the client and secret-free runtime for the fixed design."""

    plan = plan_runtime()
    if int(plan.bootstrap_requirements.context_depth) > DEPTH:
        raise ValueError(
            "canonical depth is smaller than the bootstrap requirement: "
            f"{DEPTH} < {plan.bootstrap_requirements.context_depth}"
        )
    client, context = fhe.generate_client_context(
        fhe.CKKSContextSpec(
            depth=DEPTH,
            log_n=LOG_N,
            dnum=DNUM,
            dcrt_bits=RESCALE_PRIME_BITS,
            first_mod=FIRST_PRIME_BITS,
            secret_key_dist=SECRET_KEY_DIST,
            scale_mode=SCALE_MODE,
            rescale_policy=RESCALE_POLICY,
            rotations=plan.rotations,
        ),
        device=DEVICE,
    )
    if INPUT_LIMBS > int(context.max_limbs):
        raise ValueError(
            f"input_limbs={INPUT_LIMBS} exceeds "
            f"context.max_limbs={context.max_limbs}"
        )
    bootstrap_program = bs.generate(context, plan.bootstrap_spec)
    encrypted_constants = _encrypt_graph_constants(client, context)
    weights = WeightStore(model_path=weights_path, device=DEVICE)
    runtime = ThorRuntime(
        plan=plan,
        context=context,
        bootstrap_program=bootstrap_program,
        weights=weights,
        masks=weights.masks(),
        encrypted_constants=encrypted_constants,
    )
    return client, runtime


__all__ = [
    "EncryptedConstants",
    "RuntimePlan",
    "ThorRuntime",
    "create_runtime",
    "plan_runtime",
]
