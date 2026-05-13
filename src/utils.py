import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import easyfhe.fhe as fhe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_TEST_BATCH = DEFAULT_DATA_DIR / "cifar10" / "test_batch.bin"
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "resnet20_aespa_weights.npz"

IMAGE_SIZE = 3072
LABEL_SIZE = 1
RECORD_SIZE = LABEL_SIZE + IMAGE_SIZE
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

_CACHE_MODES = {"none", "middle", "plain", "both"}


def resolve_test_batch_path(data_dir=None, test_batch_path=None):
    if test_batch_path is None:
        test_batch_path = os.environ.get("EASYFHE_CIFAR10_TEST_BATCH")
    if test_batch_path is not None:
        return Path(test_batch_path)

    if data_dir is None:
        data_dir = os.environ.get("EASYFHE_RESNET20_AESPA_DATA_DIR", DEFAULT_DATA_DIR)
    return Path(data_dir) / "cifar10" / "test_batch.bin"


def read_image(index, data_dir=None, test_batch_path=None):
    file_path = resolve_test_batch_path(data_dir, test_batch_path)
    with open(file_path, "rb") as file:
        file.seek(index * RECORD_SIZE)
        label_data = file.read(LABEL_SIZE)
        if not label_data:
            raise ValueError(f"Failed to read CIFAR-10 label at index {index} from {file_path}")
        label = int.from_bytes(label_data, byteorder="big")

        image_data = file.read(IMAGE_SIZE)
        if len(image_data) != IMAGE_SIZE:
            raise ValueError(f"Failed to read CIFAR-10 image at index {index} from {file_path}")

    image_vector = []
    for channel, (mean, std) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
        channel_offset = channel * 1024
        for pixel_index in range(1024):
            pixel = float(image_data[channel_offset + pixel_index]) / 255.0
            image_vector.append((pixel - mean) / std)

    return image_vector, label, index


class WeightPack:
    """NPZ-backed weight reader with optional encoded plaintext caching.

    The same numeric weight may need different plaintext encodings depending on
    CKKS level, slot count, scale, and context. Cache keys include those fields
    so reuse is fast without crossing incompatible encodings.
    """

    def __init__(self, arrays, cache_mode="plain"):
        self.arrays = arrays
        self.cache_mode = self._validate_cache_mode(cache_mode)
        self._middle_cache = {}
        self._plain_cache = {}
        self._cache_stats = {
            "middle_hits": 0,
            "middle_misses": 0,
            "plain_hits": 0,
            "plain_misses": 0,
        }

    @classmethod
    def from_npz(cls, path, cache_mode="plain"):
        weight_path = Path(path)
        if not weight_path.exists():
            raise ValueError(f"Weight npz {weight_path} does not exist!")
        with np.load(weight_path) as weights:
            arrays = {name: np.asarray(weights[name], dtype=np.double) for name in weights.files}
        return cls(arrays, cache_mode=cache_mode)

    def __len__(self):
        return len(self.arrays)

    def has(self, name):
        return name in self.arrays

    def _validate_cache_mode(self, cache_mode):
        if cache_mode not in _CACHE_MODES:
            raise ValueError(f"cache_mode must be one of {sorted(_CACHE_MODES)}, got {cache_mode!r}")
        return cache_mode

    def set_cache_mode(self, cache_mode, clear=True):
        self.cache_mode = self._validate_cache_mode(cache_mode)
        if clear:
            self.clear_cache()

    def clear_cache(self):
        self._middle_cache.clear()
        self._plain_cache.clear()

    def cache_info(self):
        return {
            "mode": self.cache_mode,
            "middle_entries": len(self._middle_cache),
            "plain_entries": len(self._plain_cache),
            "middle_bytes": self._cache_nbytes(self._middle_cache),
            "plain_bytes": self._cache_nbytes(self._plain_cache),
            **self._cache_stats,
        }

    def _cache_nbytes(self, cache):
        return sum(self._object_nbytes(value) for value in cache.values())

    def _object_nbytes(self, value):
        total = 0
        for attr in ("values", "encoded_values"):
            array = getattr(value, attr, None)
            if array is None:
                continue
            if hasattr(array, "numel") and hasattr(array, "element_size"):
                total += array.numel() * array.element_size()
            elif hasattr(array, "nbytes"):
                total += array.nbytes
        for cv in getattr(value, "cv", ()):
            if hasattr(cv, "numel") and hasattr(cv, "element_size"):
                total += cv.numel() * cv.element_size()
            elif hasattr(cv, "nbytes"):
                total += cv.nbytes
        return total

    def _cache_middle(self):
        return self.cache_mode in {"middle", "both"}

    def _cache_plain(self):
        return self.cache_mode in {"plain", "both"}

    def _context_key(self, crypto_context):
        return (
            id(crypto_context),
            getattr(crypto_context, "device", None),
            getattr(crypto_context, "N", None),
            getattr(crypto_context, "L", None),
            getattr(crypto_context, "rescaleTech", None),
        )

    def _scale_key(self, scale):
        return float(scale)

    def _middle_key(self, name, slots, crypto_context, scale):
        return (name, slots, self._scale_key(scale), crypto_context.N)

    def _plain_key(self, name, level, slots, crypto_context, scale, is_ext):
        return (
            name,
            level,
            slots,
            self._scale_key(scale),
            is_ext,
            self._context_key(crypto_context),
        )

    def values(self, name, slots, scale=1.0):
        if name not in self.arrays:
            raise KeyError(f"weight {name!r} is missing")

        values = np.asarray(self.arrays[name], dtype=np.double).reshape(-1)
        if values.size < slots:
            values = np.pad(values, (0, slots - values.size))
        elif values.size > slots:
            values = values[:slots]
        if scale != 1.0:
            values = values * scale
        return values

    def prepared_plaintext(self, name, slots, crypto_context, scale=1.0):
        key = self._middle_key(name, slots, crypto_context, scale)
        if self.cache_mode != "none" and key in self._middle_cache:
            self._cache_stats["middle_hits"] += 1
            return self._middle_cache[key]

        self._cache_stats["middle_misses"] += 1
        middle = fhe.prepare_plaintext(self.values(name, slots, scale), slots, crypto_context.N)
        if self._cache_middle():
            self._middle_cache[key] = middle
        return middle

    def plaintext(self, name, level, slots, crypto_context, scale=1.0, is_ext=False):
        plain_key = self._plain_key(name, level, slots, crypto_context, scale, is_ext)
        if self.cache_mode != "none" and plain_key in self._plain_cache:
            self._cache_stats["plain_hits"] += 1
            return self._plain_cache[plain_key]

        self._cache_stats["plain_misses"] += 1
        middle = self.prepared_plaintext(name, slots, crypto_context, scale)
        plaintext = fhe.make_plaintext(middle, level, slots, is_ext, crypto_context)
        if self._cache_plain():
            self._plain_cache[plain_key] = plaintext
        return plaintext

    def plaintext_for_cipher(self, name, cipher, crypto_context, scale=1.0, is_ext=False):
        return self.plaintext(
            name,
            crypto_context.L - cipher.cur_limbs,
            cipher.slots,
            crypto_context,
            scale,
            is_ext,
        )

    def encode(self, name, level, slots, crypto_context, scale=1.0):
        return self.plaintext(name, level, slots, crypto_context, scale)

    def encode_for_cipher(self, name, cipher, crypto_context, scale=1.0):
        return self.plaintext_for_cipher(name, cipher, crypto_context, scale)


def downsample1024to256(c1, c2, num_channel, crypto_context, weights):
    return _downsample_spatial(
        c1,
        c2,
        num_channel,
        crypto_context,
        weights,
        _DownsampleSpec(
            spatial_size=1024,
            out_spatial_size=256,
            row_mask_prefix="mask_first_n_mod",
            row_width=16,
            row_count=16,
            row_rotate=48,
            include_gen8=True,
            initial_rescale="if_needed",
            rescale_before_fold=True,
            rescale_after_fold=False,
        ),
    )


def downsample256to64(c1, c2, num_channel, crypto_context, weights):
    return _downsample_spatial(
        c1,
        c2,
        num_channel,
        crypto_context,
        weights,
        _DownsampleSpec(
            spatial_size=256,
            out_spatial_size=64,
            row_mask_prefix="mask_first_n_mod2",
            row_width=8,
            row_count=32,
            row_rotate=24,
            include_gen8=False,
            initial_rescale="always",
            rescale_before_fold=False,
            rescale_after_fold=True,
        ),
    )


def sum_adjacent_slots(input, slots, crypto_context):
    _require_power_of_two(slots, "slots")
    result = input.deep_copy()
    for i in range(int(math.log2(slots))):
        result = fhe.homo_add(result, fhe.homo_rotate(result, 2**i, crypto_context), crypto_context)
    return result


def sum_channel_groups(input, group_size, num_groups, crypto_context):
    _require_power_of_two(num_groups, "num_groups")
    result = input.deep_copy()
    for i in range(int(math.log2(num_groups))):
        result = fhe.homo_add(
            result,
            fhe.homo_rotate(result, group_size * (2**i), crypto_context),
            crypto_context,
        )
    return result


def broadcast_slot_sum(input, slots, crypto_context):
    return fhe.homo_rotate(sum_adjacent_slots(input, slots, crypto_context), -slots + 1, crypto_context)


@dataclass(frozen=True)
class _DownsampleSpec:
    spatial_size: int
    out_spatial_size: int
    row_mask_prefix: str
    row_width: int
    row_count: int
    row_rotate: int
    include_gen8: bool
    initial_rescale: str
    rescale_before_fold: bool
    rescale_after_fold: bool


def _merge_fullpack(c1, c2, crypto_context, weights):
    old_slots = c1.slots
    c1 = fhe.slot_resize(c1, c1.slots * 2, crypto_context)
    c2 = fhe.slot_resize(c2, c2.slots * 2, crypto_context)
    second_mask_key = f"mask_second_n_{old_slots}_{c2.slots}"
    if not weights.has(second_mask_key):
        second_mask_key = f"mask_scecond_n_{old_slots}_{c2.slots}"
    return fhe.homo_add(
        fhe.homo_mul_pt(
            c1,
            weights.plaintext(
                f"mask_first_n_{old_slots}_{c1.slots}",
                crypto_context.L - c1.cur_limbs,
                c1.slots,
                crypto_context,
            ),
            crypto_context,
        ),
        fhe.homo_mul_pt(
            c2,
            weights.plaintext(
                second_mask_key,
                crypto_context.L - c2.cur_limbs,
                c2.slots,
                crypto_context,
            ),
            crypto_context,
        ),
        crypto_context,
    )


def _double_rotate(cipher, crypto_context):
    return fhe.homo_rotate(fhe.homo_rotate(cipher, 1, crypto_context), 1, crypto_context)


def _masked_reduce(cipher, mask_n, rotated, crypto_context, weights):
    cipher = fhe.homo_mul_pt(
        fhe.homo_add(cipher, rotated, crypto_context),
        weights.plaintext(
            f"gen_mask_{mask_n}_{cipher.slots}",
            crypto_context.L - cipher.cur_limbs,
            cipher.slots,
            crypto_context,
        ),
        crypto_context,
    )
    return fhe.rescale_one_level(cipher, crypto_context)


def _spatial_reduce(fullpack, crypto_context, weights, include_gen8, initial_rescale):
    if initial_rescale == "always":
        fullpack = fhe.rescale_one_level(fullpack, crypto_context)
    else:
        fullpack = fhe.reduce_noise_to_one(fullpack, crypto_context)
    fullpack = _masked_reduce(fullpack, 2, fhe.homo_rotate(fullpack, 1, crypto_context), crypto_context, weights)
    fullpack = _masked_reduce(fullpack, 4, _double_rotate(fullpack, crypto_context), crypto_context, weights)
    if include_gen8:
        fullpack = _masked_reduce(fullpack, 8, fhe.homo_rotate(fullpack, 4, crypto_context), crypto_context, weights)
        return fhe.homo_add(fullpack, fhe.homo_rotate(fullpack, 8, crypto_context), crypto_context)
    return fhe.homo_add(fullpack, fhe.homo_rotate(fullpack, 4, crypto_context), crypto_context)


def _pack_rows(fullpack, row_mask_prefix, row_width, spatial_size, row_count, row_rotate, crypto_context, weights):
    rows = None
    for i in range(row_count):
        masked = fhe.homo_mul_pt(
            fullpack,
            weights.plaintext(
                f"{row_mask_prefix}_{row_width}_{spatial_size}_{i}_{fullpack.slots}",
                crypto_context.L - fullpack.cur_limbs,
                fullpack.slots,
                crypto_context,
            ),
            crypto_context,
        )
        rows = masked if rows is None else fhe.homo_add(rows, masked, crypto_context)
        if i < row_count - 1:
            fullpack = fhe.homo_rotate(fullpack, row_rotate, crypto_context)
    return fhe.rescale_one_level(rows, crypto_context)


def _pack_channels(rows, num_channel, spatial_size, out_spatial_size, crypto_context, weights):
    channels = None
    for i in range(num_channel * 2):
        masked = fhe.homo_mul_pt(
            rows,
            weights.plaintext(
                f"mask_channel_{i}_{num_channel}_{spatial_size}",
                crypto_context.L - rows.cur_limbs,
                rows.slots,
                crypto_context,
            ),
            crypto_context,
        )
        channels = masked if channels is None else fhe.homo_add(channels, masked, crypto_context)
        channels = fhe.homo_rotate(channels, -(spatial_size - out_spatial_size), crypto_context)
    return fhe.homo_rotate(channels, num_channel * 2 * (spatial_size - out_spatial_size), crypto_context)


def _fold_quarters(cipher, crypto_context):
    quarter = cipher.slots // 4
    cipher = fhe.homo_add(cipher, fhe.homo_rotate(cipher, -quarter, crypto_context), crypto_context)
    cipher = fhe.homo_add(
        cipher,
        fhe.homo_rotate(fhe.homo_rotate(cipher, -quarter, crypto_context), -quarter, crypto_context),
        crypto_context,
    )
    return cipher


def _downsample_spatial(c1, c2, num_channel, crypto_context, weights, spec):
    fullpack = _merge_fullpack(c1, c2, crypto_context, weights)
    fullpack = _spatial_reduce(
        fullpack,
        crypto_context,
        weights,
        include_gen8=spec.include_gen8,
        initial_rescale=spec.initial_rescale,
    )
    rows = _pack_rows(
        fullpack,
        spec.row_mask_prefix,
        spec.row_width,
        spec.spatial_size,
        spec.row_count,
        spec.row_rotate,
        crypto_context,
        weights,
    )
    channels = _pack_channels(rows, num_channel, spec.spatial_size, spec.out_spatial_size, crypto_context, weights)
    if spec.rescale_before_fold:
        channels = fhe.rescale_one_level(channels, crypto_context)
    channels = _fold_quarters(channels, crypto_context)
    if spec.rescale_after_fold:
        channels = fhe.rescale_one_level(channels, crypto_context)
    return fhe.slot_resize(channels, channels.slots // 4, crypto_context)


def _require_power_of_two(value, name):
    if value <= 0 or value & (value - 1) != 0:
        raise ValueError(f"{name} must be a positive power of two, got {value}")


__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_TEST_BATCH",
    "DEFAULT_WEIGHTS_PATH",
    "WeightPack",
    "broadcast_slot_sum",
    "downsample1024to256",
    "downsample256to64",
    "read_image",
    "resolve_test_batch_path",
    "sum_adjacent_slots",
    "sum_channel_groups",
]
