"""Canonical THOR slot layouts for inputs and attention masks."""

from __future__ import annotations

import numpy as np

from .config import HIDDEN_SIZE, SEQUENCE_LENGTH, SLOTS


def _diagonal_entry(matrix: np.ndarray, diagonal: int, column: int):
    rows, columns = matrix.shape
    return matrix[(diagonal + column) % rows, column % columns]


def pack_embedding(embedding: np.ndarray) -> np.ndarray:
    """Pack a ``(128, 768)`` embedding into THOR's four complex messages."""

    embedding = np.asarray(embedding)
    if embedding.shape != (SEQUENCE_LENGTH, HIDDEN_SIZE):
        raise ValueError(
            f"embedding shape must be ({SEQUENCE_LENGTH}, {HIDDEN_SIZE}), "
            f"got {embedding.shape}"
        )
    blocks = np.vsplit(embedding.T, 6)
    packed = np.empty((4, SLOTS), dtype=np.complex128)
    for packed_index in range(4):
        message = np.zeros((SLOTS,), dtype=np.complex128)
        for lane_group in range(16):
            offset = lane_group * (2**11)
            diagonal = packed_index * 16 + lane_group
            for token in range(SEQUENCE_LENGTH):
                for block_index in range(12):
                    block = blocks[block_index % 6]
                    message[offset + token * 16 + block_index] = complex(
                        _diagonal_entry(block, diagonal, token),
                        _diagonal_entry(block, diagonal + 64, token),
                    )
        packed[packed_index] = message
    return packed


def pack_attention_mask(attention_mask: np.ndarray) -> np.ndarray:
    """Pack a token-validity mask into the eight diagonal mask messages."""

    attention_mask = np.asarray(attention_mask)
    if attention_mask.shape != (SEQUENCE_LENGTH,):
        raise ValueError(
            f"attention_mask shape must be ({SEQUENCE_LENGTH},), "
            f"got {attention_mask.shape}"
        )
    packed = np.empty((8, SLOTS), dtype=np.float64)
    for packed_index in range(8):
        message = np.zeros((SLOTS,), dtype=np.float64)
        for lane_group in range(16):
            offset = lane_group * (2**11)
            diagonal = packed_index * 16 + lane_group
            for token in range(SEQUENCE_LENGTH):
                column = (diagonal + token) % SEQUENCE_LENGTH
                value = float(bool(attention_mask[column]))
                for head in range(12):
                    message[offset + token * 16 + head] = value
        packed[packed_index] = message
    return packed
