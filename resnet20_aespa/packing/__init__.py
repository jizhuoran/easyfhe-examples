"""Compact ResNet20 weight sources and GPU packing kernels."""

from .weights import PackedWeightBundle, load_weights


__all__ = ["PackedWeightBundle", "load_weights"]
