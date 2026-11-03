"""Single-nanobubble DOS library and synthetic multi-bubble targets.

Raw files (not shipped, see data/README.md):

    <data_dir>/sigma{20,25,30}/condAll[sm{sigma}]h{hh}.dat     hh = 00 ... 25

(the sigma = 30 files are named condAllh{hh}.dat; sigma is always taken from the folder name).

Each file has two whitespace-separated columns, energy E and DOS, on a uniform grid
(19 980 - 19 996 points, E in [-0.1, 0.1]). sigma is the bubble width parameter and h its height;
h = 0 is flat (pristine) graphene.

Pre-processing follows the original notebooks exactly:

1. linear interpolation of every spectrum onto ``ENERGY_GRID``
   (50 000 points in [-0.099, 0.099]),
2. stacking all 3 x 26 = 78 spectra and sorting them by (h, sigma),
3. dropping the first two rows, i.e. the flat-graphene (h = 0) spectra of sigma = 20 and 25,
   which leaves 76 basis spectra with flat graphene kept once, as (30, 0).
"""
import os
import re

import numpy as np

SIGMAS = (20, 25, 30)
HEIGHTS = tuple(range(26))
ENERGY_GRID = np.linspace(-0.099, 0.099, 50000)

#: (sigma, h) of the 76 library columns, in the order used by every checkpoint.
BASIS_LABELS = [(30, 0)] + [(s, h) for h in HEIGHTS[1:] for s in SIGMAS]

_NAME_RE = re.compile(r"h(\d+)\.dat$")
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")


def default_data_dir():
    """``$NANOBUBBLE_DATA_DIR`` if set, else ``<repo>/data/full`` (may not exist)."""
    env = os.environ.get("NANOBUBBLE_DATA_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "full")


def parse_dat_name(path):
    """``.../sigma25/condAllsm25h07.dat`` -> (25, 7); sigma from the folder, h from the file name."""
    folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    m_s = re.fullmatch(r"sigma(\d+)", folder)
    m_h = _NAME_RE.search(os.path.basename(path))
    if not (m_s and m_h):
        raise ValueError(f"not a DOS file: {path}")
    return int(m_s.group(1)), int(m_h.group(1))


def basis_index(label, labels=BASIS_LABELS):
    """Column index of (sigma, h) in the library. Pristine (any sigma, h=0) maps to (30, 0)."""
    s, h = int(round(label[0])), int(round(label[1]))
    if h == 0:
        s = 30
    return list(labels).index((s, h))


def label_str(label):
    return f"s{int(label[0])}h{int(label[1]):02d}"


def load_raw(data_dir):
    """Return {(sigma, h): (E, DOS)} for every ``.dat`` file below ``data_dir``."""
    out = {}
    for s in SIGMAS:
        folder = os.path.join(data_dir, f"sigma{s}")
        for fname in sorted(os.listdir(folder)):
            if fname.startswith(".") or not fname.endswith(".dat"):
                continue
            arr = np.loadtxt(os.path.join(folder, fname))
            out[parse_dat_name(os.path.join(folder, fname))] = (arr[:, 0], arr[:, 1])
    return out


def build_library(data_dir=None, grid=ENERGY_GRID, cache=True):
    """Return ``(E, X, labels)`` with ``X`` of shape (len(grid), 76) holding the DOS (linear scale).

    The interpolated library is cached in ``<repo>/.cache/`` (the raw data directory is never written to).
    """
    data_dir = data_dir or default_data_dir()
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DOS library not found in {data_dir!r} (see data/README.md)")
    tag = re.sub(r"[^A-Za-z0-9]+", "_", os.path.abspath(data_dir)).strip("_")
    cache_path = os.path.join(_CACHE_DIR, f"library_{tag}_{len(grid)}.npz")
    if cache and os.path.exists(cache_path):
        z = np.load(cache_path)
        return z["E"], z["X"], [tuple(l) for l in z["labels"].tolist()]

    raw = load_raw(data_dir)
    keys = list(raw)
    lab = np.array(keys, dtype=float)
    dos = np.stack([np.interp(grid, *raw[k]) for k in keys])
    order = np.lexsort((lab[:, 0], lab[:, 1]))          # sort by h, then sigma
    dos, lab = dos[order][2:], lab[order][2:]            # drop duplicate pristine (20,0), (25,0)
    labels = [(int(s), int(h)) for s, h in lab]
    assert labels == BASIS_LABELS, "unexpected library layout"
    X = dos.T.copy()
    if cache:
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            np.savez_compressed(cache_path, E=grid, X=X, labels=np.array(labels))
        except OSError:
            pass
    return grid, X, labels


def true_weights(indices, n_basis=len(BASIS_LABELS)):
    """Ground-truth mixing ratios: 1/N per constituent (k/N for a basis repeated k times)."""
    w = np.zeros(n_basis)
    for i in indices:
        w[i] += 1.0 / len(indices)
    return w


def make_target(X, indices, snr=0.0, rng=None):
    """Log-DOS of the configuration ``indices``, optionally with measurement noise.

    y = log( (1/N) sum_{i in S} DOS_i ) + eps,   eps ~ Normal(0, (snr * |max y|)^2)

    This is the noise model of the paper (Gaussian noise on log-DOS whose standard deviation
    is a fraction ``snr`` of the maximum of the clean log-DOS).
    """
    y = np.log(X[:, list(indices)].sum(axis=1) / len(indices))
    if snr > 0:
        rng = rng if rng is not None else np.random.default_rng()
        y = y + rng.normal(0.0, snr * abs(y.max()), size=y.shape)
    return y
