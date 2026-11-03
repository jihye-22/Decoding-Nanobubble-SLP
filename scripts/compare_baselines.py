"""Comparison of the trained single-layer network with conventional linear regression.

For every N in --n-list and every target t < --targets, N distinct library spectra are drawn
(seed = (--seed, N, t)); the same configuration is then corrupted with noise at every SNR in
--snr-list (noise seed = (--seed, N, t, SNR)) and decomposed by

  nnls      non-negative least squares on the linear DOS
  lasso_cv  LassoCV(positive=True) on the linear DOS

The row "this work" is not refitted here: it is the score of the Keras checkpoints in
pretrained/noise/ with the same N and SNR. Their noise realisations were not stored, so these are
different targets and noise draws from the same distribution.

All methods use the same detection threshold (w >= 0.01). Outputs:
results/baseline_runs.csv, results/baseline_summary.csv, assets/baseline_comparison.png.

    python scripts/compare_baselines.py --data-dir /path/to/DOS            # ~15 min on 8 CPU cores
    python scripts/compare_baselines.py --from-runs                        # no DOS library needed
"""
import os

# one BLAS thread per worker process (many shells export OMP_NUM_THREADS = #cores)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = os.environ.get("NANOBUBBLE_THREADS", "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nanobubble import (BASIS_LABELS, basis_index, build_library, make_target,  # noqa: E402
                        parse_checkpoint_name, read_weights, score_weights)
from nanobubble.baselines import METHODS  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
ASSETS = os.path.join(ROOT, "assets")
NOISE_CKPTS = os.path.join(ROOT, "pretrained", "noise")
LABELS = {"ours": "this work (Keras checkpoints)", "nnls": "NNLS (linear DOS)",
          "lasso_cv": "LassoCV, positive (linear DOS)"}
ORDER = ["ours", "nnls", "lasso_cv"]
X = None


def configuration(seed, n, t):
    rng = np.random.default_rng([seed, n, t])
    return sorted(rng.choice(len(BASIS_LABELS), size=n, replace=False).tolist())


def target(seed, n, t, snr):
    idx = configuration(seed, n, t)
    return idx, make_target(X, idx, snr, np.random.default_rng([seed, n, t, int(round(snr * 1000))]))


def run_one(job):
    seed, n, t, snr = job
    idx, y = target(seed, n, t, snr)
    rows = []
    for name, fit in METHODS.items():
        t0 = time.time()
        w = fit(X, y)
        s = score_weights(w, idx)
        rows.append({"method": name, "N": n, "target": t, "snr": snr,
                     "config": " ".join(f"{BASIS_LABELS[i][0]},{BASIS_LABELS[i][1]}" for i in idx),
                     **s, "seconds": round(time.time() - t0, 2)})
    return rows


def checkpoint_rows(n_list, snr_list):
    """Scores of the Keras checkpoints in pretrained/noise/ for the requested N and SNR."""
    rows = []
    for p in sorted(glob.glob(os.path.join(NOISE_CKPTS, "*.hdf5"))):
        m = parse_checkpoint_name(p)
        if m["N"] not in n_list or not np.isclose(m["snr"], snr_list).any():
            continue
        idx = [basis_index(l) for l in m["labels"]]
        rows.append({"method": "ours", "N": m["N"], "target": os.path.basename(p), "snr": m["snr"],
                     "config": " ".join(f"{s},{h}" for s, h in m["labels"]), **score_weights(read_weights(p), idx)})
    return pd.DataFrame(rows)


def init(data_dir):
    global X
    _, X, _ = build_library(data_dir)


def summarise(df):
    g = df.groupby(["method", "N", "snr"])
    out = g.agg(runs=("exact_support", "size"), exact_support_rate=("exact_support", "mean"),
                median_rel_l2_error_pct=("rel_l2_error_pct", "median"),
                mean_rel_l2_error_pct=("rel_l2_error_pct", "mean"),
                mean_n_pred=("n_pred", "mean"), mean_false_pos=("false_pos", "mean"),
                mean_false_neg=("false_neg", "mean")).reset_index()
    out["method"] = pd.Categorical(out["method"], ORDER)
    return out.sort_values(["method", "N", "snr"])


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted(summary["N"].unique())
    colors = {"ours": "#2a78d6", "nnls": "#1baf7a", "lasso_cv": "#eda100"}
    marks = {"ours": "o", "nnls": "s", "lasso_cv": "^"}
    fig, axes = plt.subplots(2, len(ns), figsize=(3.2 * len(ns), 6.2), sharex=True, sharey="row", squeeze=False)
    for j, n in enumerate(ns):
        for m in ORDER:
            d = summary[(summary["method"] == m) & (summary["N"] == n)]
            kw = dict(color=colors[m], marker=marks[m], ms=5, lw=1.6 if m == "ours" else 1.1, label=LABELS[m])
            axes[0, j].plot(d["snr"], 100 * d["exact_support_rate"], **kw)
            e = d[d["snr"] > 0]  # at SNR = 0 the errors are ~0 except for LassoCV (see table)
            axes[1, j].plot(e["snr"], e["median_rel_l2_error_pct"], **kw)
        axes[0, j].set_title(f"N = {n}")
        axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel("noise level SNR")
        for ax in axes[:, j]:
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("exact support recovery [%]")
    axes[0, 0].set_ylim(-5, 105)
    axes[1, 0].set_ylabel("median relative weight error [%]\n(SNR > 0)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-list", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--snr-list", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.05, 0.09])
    ap.add_argument("--targets", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--from-runs", action="store_true",
                    help="reuse the baseline fits in results/baseline_runs.csv (no DOS library needed)")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(ASSETS, exist_ok=True)

    t0 = time.time()
    if args.from_runs:
        base = pd.read_csv(os.path.join(RESULTS, "baseline_runs.csv"))
        base = base[base["method"].isin(METHODS)]
    else:
        init(args.data_dir)  # builds the cache once before forking
        jobs = [(args.seed, n, t, s) for n in args.n_list for t in range(args.targets) for s in args.snr_list]
        with Pool(args.jobs, initializer=init, initargs=(args.data_dir,)) as pool:
            base = pd.DataFrame([r for rs in pool.imap_unordered(run_one, jobs) for r in rs])
        print(f"{len(jobs)} targets x {len(METHODS)} methods in {time.time() - t0:.0f} s")
    base = base.sort_values(["method", "N", "target", "snr"])
    base.to_csv(os.path.join(RESULTS, "baseline_runs.csv"), index=False, float_format="%.6g")

    df = pd.concat([checkpoint_rows(args.n_list, args.snr_list), base], ignore_index=True)
    summary = summarise(df)
    summary.to_csv(os.path.join(RESULTS, "baseline_summary.csv"), index=False, float_format="%.4g")
    plot(summary, os.path.join(ASSETS, "baseline_comparison.png"))

    pivot = summary.pivot_table(index="method", columns="snr", values="exact_support_rate", aggfunc="mean",
                                observed=True)
    print("exact support recovery [%] (mean over N):\n", (100 * pivot).round(0).to_string())
    print("median relative weight error [%] (all targets):\n",
          df.pivot_table(index="method", columns="snr", values="rel_l2_error_pct", aggfunc="median").round(2)
          .to_string())
    print("targets per (method, SNR):\n", df.pivot_table(index="method", columns="snr", values="N", aggfunc="size")
          .to_string())


if __name__ == "__main__":
    main()
