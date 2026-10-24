"""Fit the mixing ratios of one synthetic multi-bubble target.

Examples
    # three given constituents (sigma, h), noise-free, original Keras procedure
    python scripts/train.py --labels 20,4 25,7 30,10
    # five random constituents, 5 % noise, fast L-BFGS-B solver of the same objective
    python scripts/train.py --n 5 --seed 3 --snr 0.05 --solver lbfgs
    # incomplete library: only the first 70 basis spectra are offered to the model
    python scripts/train.py --labels 20,4 25,7 30,25 --n-basis 70

The DOS library is read from --data-dir (default: $NANOBUBBLE_DATA_DIR or data/full).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nanobubble import BASIS_LABELS, basis_index, build_library, label_str, make_target, score_weights  # noqa: E402
from nanobubble.fit import fit_target, solve_bounded  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--labels", nargs="+", help="constituents as sigma,h (repeat a label for a repeated bubble)")
    g.add_argument("--indices", nargs="+", type=int, help="constituents as library column indices 0..75")
    g.add_argument("--n", type=int, help="number of distinct random constituents")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random configuration, noise and init")
    ap.add_argument("--snr", type=float, default=0.0, help="noise std as a fraction of max(log DOS)")
    ap.add_argument("--n-basis", type=int, default=76, help="offer only the first n basis spectra (incomplete library)")
    ap.add_argument("--solver", choices=["keras", "lbfgs"], default="keras")
    ap.add_argument("--epochs", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs"), help="folder for checkpoint + json")
    ap.add_argument("--verbose", type=int, default=0)
    a = ap.parse_args()

    _, X, _ = build_library(a.data_dir)
    rng = np.random.default_rng(a.seed)
    if a.labels:
        idx = [basis_index(tuple(float(v) for v in s.split(","))) for s in a.labels]
    elif a.indices:
        idx = a.indices
    else:
        idx = sorted(rng.choice(len(BASIS_LABELS), size=a.n, replace=False).tolist())
    y = make_target(X, idx, a.snr, rng)
    Xs = X[:, :a.n_basis]
    name = "N%d__%s" % (len(idx), "+".join(label_str(BASIS_LABELS[i]) for i in idx))
    if a.snr:
        name += f"__snr{a.snr:.2f}"
    if a.n_basis != X.shape[1]:
        name += f"__basis{a.n_basis}"
    os.makedirs(a.out, exist_ok=True)
    print("target:", [BASIS_LABELS[i] for i in idx], f"snr={a.snr}", f"library={a.n_basis} spectra")

    t0 = time.time()
    if a.solver == "keras":
        w, hist = fit_target(Xs, y, checkpoint=os.path.join(a.out, name + ".hdf5"), learning_rate=a.lr,
                             epochs=a.epochs, seed=a.seed, verbose=a.verbose)
        extra = {"epochs_run": len(hist["loss"]), "final_loss": float(min(hist["loss"]))}
    else:
        w, loss = solve_bounded(Xs, y)
        extra = {"final_loss": loss}
    extra["seconds"] = round(time.time() - t0, 1)

    w_full = np.zeros(X.shape[1])
    w_full[:a.n_basis] = w
    res = {"target": [BASIS_LABELS[i] for i in idx], "indices": idx, "snr": a.snr, "n_basis": a.n_basis,
           "solver": a.solver, **extra, "weights": np.round(w_full, 6).tolist()}
    if a.n_basis == X.shape[1]:
        res["score"] = score_weights(w_full, idx)
    with open(os.path.join(a.out, name + f"__{a.solver}.json"), "w") as f:
        json.dump(res, f, indent=1)
    top = np.argsort(w_full)[::-1][:max(len(set(idx)) + 3, 5)]
    print("largest weights:", ", ".join(f"{BASIS_LABELS[i]}: {w_full[i]:.4f}" for i in top))
    if "score" in res:
        print("score:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["score"].items()})
    print({k: v for k, v in extra.items()})


if __name__ == "__main__":
    main()
