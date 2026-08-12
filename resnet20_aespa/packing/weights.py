"""Load compact ResNet20 parameters for cached on-GPU packing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import easyfhe
import easyfhe.fhe as fhe

from .triton import pack_dictionary, pack_factorized


SCHEMA_VERSION = 1
EXPECTED_VECTOR_COUNT = 233
_METADATA_KEYS = {"__schema_version__", "__vector_count__"}


class FactorizedPackedRaw(fhe.PackedRaw):
    """A matrix of channel coefficients expanded by Triton on first use."""

    def __init__(self, coefficients, masks, mask_ids, *, slots: int):
        super().__init__(coefficients)
        self.coefficients = coefficients
        self.masks = masks
        self.mask_ids = mask_ids
        self.slots = int(slots)

    def packed_tensor(self, slots, context=None):
        del context
        if int(slots) != self.slots:
            raise ValueError(
                f"factorized constant requires {self.slots} slots, got {slots}"
            )
        return pack_factorized(
            self.coefficients,
            self.masks,
            self.mask_ids,
            slots=self.slots,
        )


class DictionaryPackedRaw(fhe.PackedRaw):
    """A dictionary-coded vector expanded by Triton on first use."""

    def __init__(self, values, codes):
        super().__init__(values)
        self.values = values
        self.codes = codes
        self.slots = int(codes.numel())

    def packed_tensor(self, slots, context=None):
        del context
        if int(slots) != self.slots:
            raise ValueError(
                f"dictionary constant requires {self.slots} slots, got {slots}"
            )
        return pack_dictionary(self.values, self.codes, slots=self.slots)


@dataclass(frozen=True)
class PackedWeightBundle:
    """Application constants with compact sources and a shared middle cache."""

    raw_vectors: dict[str, fhe.PackedRaw]
    constants: fhe.ConstantBundle

    def __len__(self):
        return len(self.raw_vectors)

    def plaintext(self, *args, **kwargs):
        return self.constants.plaintext(*args, **kwargs)

    def cache_info(self):
        return self.constants.cache_info()

    def clear_cache(self):
        self.constants.clear_cache()


def load_weights(path, *, device="cuda") -> PackedWeightBundle:
    weight_path = Path(path)
    if not weight_path.is_file():
        raise FileNotFoundError(f"weight archive does not exist: {weight_path}")

    with np.load(weight_path, allow_pickle=False) as archive:
        _validate_archive_metadata(archive, weight_path)
        groups = _group_arrays(archive.files)
        if len(groups) != EXPECTED_VECTOR_COUNT:
            raise ValueError(
                f"expected {EXPECTED_VECTOR_COUNT} compact constants, "
                f"got {len(groups)} in {weight_path}"
            )
        vectors = {
            name: _load_vector(name, fields, archive, device=str(device))
            for name, fields in groups.items()
        }

    constants = fhe.ConstantBundle(vectors=vectors, cache_mode="middle")
    return PackedWeightBundle(vectors, constants)


def _validate_archive_metadata(archive, path: Path):
    if set(_METADATA_KEYS) - set(archive.files):
        raise ValueError(f"compact weight metadata is missing in {path}")
    version = int(np.asarray(archive["__schema_version__"]).reshape(-1)[0])
    count = int(np.asarray(archive["__vector_count__"]).reshape(-1)[0])
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported compact weight schema {version}; expected {SCHEMA_VERSION}"
        )
    if count != EXPECTED_VECTOR_COUNT:
        raise ValueError(
            f"compact weight metadata declares {count} vectors; "
            f"expected {EXPECTED_VECTOR_COUNT}"
        )


def _group_arrays(keys) -> dict[str, set[str]]:
    groups = {}
    for key in keys:
        if key in _METADATA_KEYS:
            continue
        if "::" not in key:
            raise ValueError(f"unexpected compact weight array {key!r}")
        name, field = key.rsplit("::", 1)
        groups.setdefault(name, set()).add(field)
    return groups


def _load_vector(name, fields, archive, *, device: str):
    if fields == {"coefficients", "masks", "mask_ids"}:
        coefficients = _finite_float64(archive[f"{name}::coefficients"], name)
        masks = np.asarray(archive[f"{name}::masks"])
        mask_ids = np.asarray(archive[f"{name}::mask_ids"])
        _validate_factorized(name, coefficients, masks, mask_ids)
        slots = int(coefficients.shape[1] * masks.shape[1])
        return FactorizedPackedRaw(
            easyfhe.as_tensor(coefficients, dtype=easyfhe.float64, device=device),
            easyfhe.as_tensor(
                np.asarray(masks, dtype=np.int32),
                dtype=easyfhe.int32,
                device=device,
            ),
            easyfhe.as_tensor(
                np.asarray(mask_ids, dtype=np.int32),
                dtype=easyfhe.int32,
                device=device,
            ),
            slots=slots,
        )
    if fields == {"values", "codes"}:
        values = _finite_float64(archive[f"{name}::values"], name)
        codes = np.asarray(archive[f"{name}::codes"])
        _validate_dictionary(name, values, codes)
        return DictionaryPackedRaw(
            easyfhe.as_tensor(values, dtype=easyfhe.float64, device=device),
            easyfhe.as_tensor(
                np.asarray(codes, dtype=np.int32),
                dtype=easyfhe.int32,
                device=device,
            ),
        )
    raise ValueError(f"compact constant {name!r} has unexpected fields {fields}")


def _finite_float64(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise FloatingPointError(f"compact constant {name!r} is non-finite")
    return np.ascontiguousarray(value)


def _validate_factorized(name, coefficients, masks, mask_ids):
    if coefficients.ndim != 2 or masks.ndim != 2 or mask_ids.ndim != 1:
        raise ValueError(f"factorized constant {name!r} has invalid ranks")
    if len(mask_ids) != coefficients.shape[0]:
        raise ValueError(f"factorized constant {name!r} has invalid mask ids")
    if not np.all((masks == 0) | (masks == 1)):
        raise ValueError(f"factorized constant {name!r} has non-binary masks")
    if mask_ids.size and (
        int(mask_ids.min()) < 0 or int(mask_ids.max()) >= masks.shape[0]
    ):
        raise ValueError(f"factorized constant {name!r} has out-of-range mask ids")


def _validate_dictionary(name, values, codes):
    if values.ndim != 1 or codes.ndim != 1 or not len(values):
        raise ValueError(f"dictionary constant {name!r} has invalid arrays")
    if codes.size and (
        int(codes.min()) < 0 or int(codes.max()) >= len(values)
    ):
        raise ValueError(f"dictionary constant {name!r} has out-of-range codes")
