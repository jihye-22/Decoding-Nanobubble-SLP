"""Seeded comparison of the log-domain basis-mixing fit with conventional linear regression.

For every N in --n-list and every target t < --targets, N distinct library spectra are drawn
(seed = (--seed, N, t)); the same configuration is then corrupted with noise at every SNR in
--snr-list (noise seed = (--seed, N, t, SNR)) and decomposed by

  ours           log-domain fit of the single-layer network (same objective and [0, 1] box
                 constraint as the Keras model, minimised with L-BFGS-B, see nanobubble/fit.py)
  nnls           non-negative least squares on the linear DOS
  nnls_relative  NNLS on relative residuals (linearised log loss)
  lasso_cv       LassoCV(positive=True) on the linear DOS

All methods use the same detection threshold (w >= 0.01). Outputs:
results/baseline_runs.csv, results/baseline_summary.csv, assets/baseline_comparison.png.

    python scripts/compare_baselines.py --data-dir /path/to/DOS            # ~15 min on 10 CPU cores
    python scripts/compare_baselines.py --crosscheck                       # Keras vs L-BFGS-B (~1 h)
"""
import os

# one BLAS thread per worker process (many shells export OMP_NUM_THREADS = #cores)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = os.environ.get("NANOBUBBLE_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nanobubble import BASIS_LABELS, build_library, make_target, score_weights  # noqa: E402
from nanobubble.baselines import METHODS  # noqa: E402
from nanobubble.fit import solve_bounded  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
ASSETS = os.path.join(ROOT, "assets")
LABELS = {"ours": "this work (log-domain fit)", "nnls": "NNLS (linear DOS)",
          "nnls_relative": "NNLS, relative residuals", "lasso_cv": "LassoCV, positive (linear DOS)"}
ORDER = ["ours", "nnls_relative", "nnls", "lasso_cv"]
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
    fits = {"ours": lambda: solve_bounded(X, y)[0], **{k: (lambda f=f: f(X, y)) for k, f in METHODS.items()}}
    for name, fit in fits.items():
        t0 = time.time()
        w = fit()
        s = score_weights(w, idx)
        rows.append({"method": name, "N": n, "target": t, "snr": snr,
                     "config": " ".join(f"{BASIS_LABELS[i][0]},{BASIS_LABELS[i][1]}" for i in idx),
                     **s, "seconds": round(time.time() - t0, 2)})
    return rows


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
    colors = {"ours": "#2a78d6", "nnls_relative": "#eb6834", "nnls": "#1baf7a", "lasso_cv": "#eda100"}
    marks = {"ours": "o", "nnls_relative": "D", "nnls": "s", "lasso_cv": "^"}
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
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def crosscheck(args):
    """Train the Keras model (original settings) on a few of the same seeded targets and compare."""
    from nanobubble.fit import fit_target

    init(args.data_dir)
    cases = [(3, 0, 0.0), (3, 0, 0.01), (3, 0, 0.05), (5, 0, 0.03), (5, 0, 0.09), (1, 0, 0.09)]
    out = []
    for n, t, snr in cases[:args.crosscheck_cases]:
        idx, y = target(args.seed, n, t, snr)
        t0 = time.time()
        wk, hist = fit_target(X, y, seed=0)
        tk = time.time() - t0
        wl, ll = solve_bounded(X, y)
        rec = {"N": n, "target": t, "snr": snr, "config": [BASIS_LABELS[i] for i in idx],
               "keras": {"epochs": len(hist["loss"]), "best_loss": float(min(hist["loss"])),
                         "minutes": round(tk / 60, 1), **score_weights(wk, idx)},
               "lbfgs": {"loss": ll, **score_weights(wl, idx)},
               "max_abs_weight_difference": float(np.max(np.abs(wk - wl)))}
        print(json.dumps(rec), flush=True)
        out.append(rec)
        with open(os.path.join(RESULTS, "solver_crosscheck.json"), "w") as f:
            json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-list", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--snr-list", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.05, 0.09])
    ap.add_argument("--targets", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--crosscheck", action="store_true", help="run the Keras-vs-L-BFGS-B check instead")
    ap.add_argument("--crosscheck-cases", type=int, default=3)
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from results/baseline_summary.csv")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(ASSETS, exist_ok=True)
    if args.plot_only:
        return plot(pd.read_csv(os.path.join(RESULTS, "baseline_summary.csv")), os.path.join(ASSETS, "baseline_comparison.png"))
    if args.crosscheck:
        return crosscheck(args)

    init(args.data_dir)  # builds the cache once before forking
    jobs = [(args.seed, n, t, s) for n in args.n_list for t in range(args.targets) for s in args.snr_list]
    t0 = time.time()
    with Pool(args.jobs, initializer=init, initargs=(args.data_dir,)) as pool:
        rows = [r for rs in pool.imap_unordered(run_one, jobs) for r in rs]
    df = pd.DataFrame(rows).sort_values(["method", "N", "target", "snr"])
    df.to_csv(os.path.join(RESULTS, "baseline_runs.csv"), index=False, float_format="%.6g")
    summary = summarise(df)
    summary.to_csv(os.path.join(RESULTS, "baseline_summary.csv"), index=False, float_format="%.4g")
    plot(summary, os.path.join(ASSETS, "baseline_comparison.png"))

    pivot = summary.pivot_table(index="method", columns="snr", values="exact_support_rate", aggfunc="mean",
                                observed=True)
    print(f"{len(jobs)} targets x {len(ORDER)} methods in {time.time() - t0:.0f} s")
    print("exact support recovery (mean over N):\n", (100 * pivot).round(0).to_string())
    print("median relative weight error [%]:\n",
          df.pivot_table(index="method", columns="snr", values="rel_l2_error_pct", aggfunc="median").round(2)
          .to_string())


if __name__ == "__main__":
    main()
