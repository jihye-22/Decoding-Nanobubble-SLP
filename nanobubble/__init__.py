"""Physics-constrained decoding of multi-nanobubble configurations in graphene.

A target density-of-states (DOS) spectrum is modelled as a non-negative mixture of a
library of single-nanobubble spectra. A single dense layer without bias holds the
mixing ratios; its weights are read off directly as the configuration.
"""
from .data import (BASIS_LABELS, ENERGY_GRID, basis_index, build_library, default_data_dir,
                   label_str, make_target, true_weights)
from .evaluate import (DETECTION_THRESHOLD, parse_checkpoint_name, read_weights, score_weights)

__all__ = [
    "BASIS_LABELS", "ENERGY_GRID", "basis_index", "build_library", "default_data_dir", "label_str",
    "make_target", "true_weights",
    "DETECTION_THRESHOLD", "parse_checkpoint_name", "read_weights", "score_weights",
]
