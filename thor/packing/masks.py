"""THOR static-mask and block-layout packing recipes."""

import numpy as np


def format_packed_name(key: str, index: tuple[int, ...]) -> str:
    index_text = "x".join(str(i) for i in index)
    return f"{key}__{index_text}"


def to_blocks(matrix: np.ndarray, block_shape: tuple[int, int], diag: bool = True):
    """Split a matrix into block matrices, optionally in THOR diagonal order."""
    rows, cols = matrix.shape
    if rows % block_shape[0] != 0 or cols % block_shape[1] != 0:
        raise ValueError("Matrix shape should be divisible by block shape")
    v = rows // block_shape[0]
    h = cols // block_shape[1]
    blocks = np.empty((v, h), dtype=object)
    row_blocks = np.vsplit(matrix, v)
    for index, row_block in enumerate(row_blocks):
        blocks[index] = np.hsplit(row_block, h)
    if not diag:
        return blocks, (v, h)

    l = min(v, h)
    d = max(v, h)
    diag_blocks = np.empty((l, d), dtype=object)
    for t in range(d):
        for index in range(l):
            diag_blocks[index, t] = blocks[(index + t) % v, t % h]
    return diag_blocks, (l, d)


class ThorMaskPacker:
    """Pack THOR plaintext masks without CKKS encoding."""

    def __init__(self, num_slots: int = 2**15, level: int = 15):
        self.num_slots = int(num_slots)
        self.level = int(level)

    def pack(self):
        vectors = {}
        metadata = {}

        def add(name, array):
            vector = np.asarray(array)
            vectors[name] = vector
            metadata[name] = {
                "level": self.level,
                "slots": self.num_slots,
                "dtype": str(vector.dtype),
                "shape": list(vector.shape),
            }

        slots = self.num_slots
        idx = np.arange(slots)

        for index in range(1, 128):
            array = np.ones((slots,), dtype=int)
            array[idx % (2**11) >= (16 * index)] = 0
            add(f"rot_internal.att.{index}", array)

        for index in range(1, 16):
            array = np.ones((slots,), dtype=int)
            array[idx % 16 >= index] = 0
            add(f"rot_internal.block_diag_1.{index}", array)

        for index in range(1, 8):
            array = np.ones((slots,), dtype=int)
            array[idx % 8 >= index] = 0
            add(f"rot_internal.block_diag_2.{index}", array)

        for index in range(16):
            array = np.zeros((slots,), dtype=int)
            array[2**11 * index : 2**11 * (index + 1)] = 1
            add(f"make_copies_merge.{index}", array * (1 / 4))

        for index in range(8):
            array = np.zeros((slots,), dtype=int)
            array[2**12 * index : 2**12 * (index + 1)] = 1
            add(f"make_copies_2.{index}", array * (1 / 4))

        local0 = np.ones((slots,), dtype=float)
        local0[idx % (2**12) >= 2**11] = 0
        add("make_copies.local0", local0)

        local1 = np.ones((slots,), dtype=float)
        local1[idx % (2**12) < 2**11] = 0
        add("make_copies.local1", local1)

        attention_dense_mask = np.ones((slots,), dtype=float)
        attention_dense_mask[idx % 16 < 6] = 0
        add("attention_dense.mask1", attention_dense_mask)

        for index in range(4):
            n_diag = 16 * index
            arr0 = np.array([1] * 16 * (64 + (n_diag - 16) % 64 + 16))
            arr1 = np.array(
                [0] * (slots - 16 * (64 - ((n_diag - 16) % 64 + 16)))
                + [1] * 16 * (64 - ((n_diag - 16) % 64 + 16))
            )
            add(f"transpose.mask0.{index}", arr0)
            add(f"transpose.mask1.{index}", arr1)
            for j in range(16 * index + 1, 16 * (index + 1)):
                l_value = 64 - j
                arr2 = np.array([0] * 2**11 * (16 - j % 16) + [1] * (128 - l_value) * 2**4)
                arr3 = np.array(
                    [0] * 2**11 * (16 - j % 16 - 1)
                    + [0] * (128 - l_value) * 2**4
                    + [1] * 16 * l_value
                )
                add(f"transpose.mask2.{j}", arr2)
                add(f"transpose.mask3.{j}", arr3)

        for n in range(1, 128):
            rot = n
            j = n % 16
            arr0 = np.full((slots,), 1, dtype=float)
            arr0[idx % (2**11) >= (2**11 - 16 * rot)] = 0

            arr1 = np.full((slots,), 0, dtype=float)
            arr1[idx % (2**11) >= (2**11 - 16 * rot)] = 1

            if j == 0:
                add(f"ct_ct_matmul.0.{n}", arr0)
                add(f"ct_ct_matmul.1.{n}", arr1)
            else:
                arr0[: (2**11) * j] = 0
                add(f"ct_ct_matmul.0.{n}", arr0)

                arr1[-(2**11) :] = 0
                if j > 1:
                    arr1[: (2**11) * (j - 1)] = 0
                add(f"ct_ct_matmul.1.{n}", arr1)

                arr2 = np.full((slots,), 1, dtype=float)
                arr2[idx % (2**11) >= (2**11 - 16 * rot)] = 0
                arr2[(2**11) * j :] = 0
                add(f"ct_ct_matmul.2.{n}", arr2)

                arr3 = np.full((slots,), 1, dtype=float)
                arr3 = arr3 - arr0 - arr1 - arr2
                add(f"ct_ct_matmul.3.{n}", arr3)

        add("scale.attention_score", np.ones((slots,), dtype=float))
        add("scale.attention_context", np.ones((slots,), dtype=float))
        return vectors, metadata
