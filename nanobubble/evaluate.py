"""Reading checkpoints and scoring recovered mixing ratios."""
import re

import h5py
import numpy as np

from .data import BASIS_LABELS, basis_index, true_weights

#: a basis spectrum counts as "detected" if its weight is at least this value (as in the paper)
DETECTION_THRESHOLD = 0.01

_CKPT_RE = re.compile(r"N(\d+)__([^_]+(?:\+[^_]+)*)(?:__snr([0-9.]+))?\.hdf5$")
_LAB_RE = re.compile(r"s(\d+)h(\d+)")


def read_weights(path):
    """Kernel of the ``basis_head`` layer, read with h5py (no TensorFlow needed)."""
    with h5py.File(path, "r") as h:
        return np.array(h["model_weights/basis_head/basis_head/kernel:0"]).ravel()


def parse_checkpoint_name(path):
    """``N3__s30h20+s30h20+s30h11__snr0.03.hdf5`` -> dict(N, labels, snr).

    The ground truth of every shipped checkpoint is encoded in its file name.
    """
    name = path.replace("\\", "/").split("/")[-1]
    m = _CKPT_RE.search(name)
    if not m:
        raise ValueError(f"cannot parse checkpoint name {name}")
    labels = [(int(s), int(h)) for s, h in _LAB_RE.findall(m.group(2))]
    assert len(labels) == int(m.group(1)), name
    return {"N": int(m.group(1)), "labels": labels, "snr": float(m.group(3)) if m.group(3) else 0.0}


def score_weights(w, indices, threshold=DETECTION_THRESHOLD, n_basis=None):
    """Compare recovered weights with the ground truth of configuration ``indices``.

    Returns a dict with
      exact_support : detected set == true set of distinct constituents
      n_true / n_pred : number of distinct true / detected constituents
      rel_l2_error : ||w - w_true||_2 / ||w_true||_2   (in %, as ``rel_l2_error_pct``)
      max_abs_error, weight_sum
    """
    w = np.asarray(w, float).ravel()
    w_true = true_weights(indices, n_basis or len(w))
    det = set(np.flatnonzero(w >= threshold).tolist())
    true = set(np.flatnonzero(w_true > 0).tolist())
    err = np.linalg.norm(w - w_true) / np.linalg.norm(w_true)
    return {
        "exact_support": det == true,
        "n_true": len(true),
        "n_pred": len(det),
        "false_pos": len(det - true),
        "false_neg": len(true - det),
        "rel_l2_error_pct": 100.0 * float(err),
        "max_abs_error": float(np.max(np.abs(w - w_true))),
        "weight_sum": float(w.sum()),
    }


def indices_of(labels, library=BASIS_LABELS):
    return [basis_index(l, library) for l in labels]
