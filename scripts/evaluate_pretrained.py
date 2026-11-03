"""Reproduce the checkpoint-based numbers and figures in the README.

No training happens here. Every checkpoint in pretrained/ is read with h5py; the ground-truth
configuration is encoded in its file name, so all weight metrics need only the checkpoints.
Figures that show spectra additionally need the DOS library (--data-dir, $NANOBUBBLE_DATA_DIR
or data/full); without it they are skipped.

    python scripts/evaluate_pretrained.py [--data-dir /path/to/DOS]

Writes results/pretrained_scores.csv, results/pretrained_summary.json and assets/*.png.
"""
import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nanobubble import (BASIS_LABELS, DETECTION_THRESHOLD, basis_index, build_library,  # noqa: E402
                        parse_checkpoint_name, read_weights, score_weights, true_weights)

PRE = os.path.join(ROOT, "pretrained")
ASSETS = os.path.join(ROOT, "assets")
RESULTS = os.path.join(ROOT, "results")
# categorical slots (fixed order) of the reference palette, validated for CVD separation
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, MUTED = "#222222", "#8a8a85"

# checkpoints shown in the paper's figures
FIG2 = ["random_mixtures/N1__s30h07.hdf5",
        "random_mixtures/N5__s20h24+s20h02+s30h18+s30h20+s30h00.hdf5",
        "random_mixtures/N10__s20h24+s20h22+s20h11+s30h05+s20h16+s25h03+s30h01+s20h03+s25h20+s20h21.hdf5"]
REPEATED = ["repeated/N3__s30h20+s30h20+s30h11.hdf5",
            "repeated/N5__s30h00+s20h05+s20h05+s20h05+s20h19.hdf5"]
INCOMPLETE = "incomplete_library/N3__s20h04+s25h07+s30h25.hdf5"
NOISE_EXAMPLE = ("N1__s20h24", (0.03, 0.06, 0.09))

plt.rcParams.update({"figure.dpi": 120, "font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "legend.frameon": False})


def save(fig, name):
    os.makedirs(ASSETS, exist_ok=True)
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved", os.path.relpath(path, ROOT))


def idx_of(labels):
    return [basis_index(l) for l in labels]


def short(l):
    return f"({l[0]},{l[1]})"


# --------------------------------------------------------------------------------------
def score_all():
    rows = []
    for p in sorted(glob.glob(os.path.join(PRE, "*", "*.hdf5"))):
        group = os.path.basename(os.path.dirname(p))
        if group == "incomplete_library":
            continue
        m = parse_checkpoint_name(p)
        s = score_weights(read_weights(p), idx_of(m["labels"]))
        rows.append({"group": group, "file": os.path.relpath(p, PRE), "N": m["N"], "snr": m["snr"], **s})
    return pd.DataFrame(rows)


def summarise(df):
    out = {"detection_threshold": DETECTION_THRESHOLD}

    def agg(d):
        e = d["rel_l2_error_pct"]
        return {"checkpoints": int(len(d)), "exact_support_rate": round(float(d["exact_support"].mean()), 4),
                "mean_rel_l2_error_pct": float(f"{e.mean():.4g}"), "std_rel_l2_error_pct": float(f"{e.std(ddof=0):.4g}"),
                "max_rel_l2_error_pct": float(f"{e.max():.4g}"),
                "mean_false_pos": round(float(d["false_pos"].mean()), 3),
                "mean_false_neg": round(float(d["false_neg"].mean()), 3)}

    r = df[df.group == "random_mixtures"]
    out["random_mixtures"] = {"all": agg(r), "by_N": {int(n): agg(d) for n, d in r.groupby("N")}}
    out["repeated"] = agg(df[df.group == "repeated"])
    examples = {}
    for f in REPEATED:
        m = parse_checkpoint_name(f)
        w = read_weights(os.path.join(PRE, f))
        examples[f] = {short(BASIS_LABELS[i]): round(float(w[i]), 6) for i in sorted(set(idx_of(m["labels"])))}
    out["repeated"]["examples"] = examples
    nz = df[df.group == "noise"]
    out["noise"] = {f"N={int(n)}": {f"{s:.2f}": agg(d2) for s, d2 in d.groupby("snr")} for n, d in nz.groupby("N")}
    out["noise_all_N"] = {f"{s:.2f}": agg(d) for s, d in nz.groupby("snr")}
    return out


def incomplete_summary():
    out = {}
    for p in sorted(glob.glob(os.path.join(PRE, "incomplete_library", "*.hdf5"))):
        m = parse_checkpoint_name(p)
        w = read_weights(p)
        lib = BASIS_LABELS[:len(w)]
        present = [l for l in m["labels"] if l in lib]
        missing = [l for l in m["labels"] if l not in lib]
        order = np.argsort(w)[::-1]
        out[os.path.basename(p)] = {
            "library_size": len(w), "missing_constituents": [short(l) for l in missing],
            "true_weight_each": round(1 / m["N"], 4),
            "weights_on_present_constituents": {short(l): round(float(w[lib.index(l)]), 4) for l in present},
            "n_detected": int((w >= DETECTION_THRESHOLD).sum()),
            "largest_other_weights": {short(lib[i]): round(float(w[i]), 4) for i in order
                                      if lib[i] not in present and w[i] >= DETECTION_THRESHOLD},
            "weight_sum": round(float(w.sum()), 4)}
    return out


# --------------------------------------------------------------------------------------
def stem(ax, w, idx, color, label_true=True):
    wt = true_weights(idx, len(w))
    x = np.arange(len(w))
    ax.vlines(x, 0, w, color=color, lw=1.4)
    ax.plot(x, w, "o", color=color, ms=3.2, label="learned weight")
    t = wt > 0
    ax.plot(x[t], wt[t], "_", color=INK, ms=11, mew=1.6, label="ground truth" if label_true else None)
    ax.axhline(DETECTION_THRESHOLD, color=MUTED, ls=":", lw=0.9)
    ax.set_xlim(-1.5, 76.5)
    ax.set_xlabel("basis index i")


def fig_recovery(lib):
    ncol = len(FIG2)
    rows = 2 if lib else 1
    fig, axes = plt.subplots(rows, ncol, figsize=(4.0 * ncol, 2.9 * rows), squeeze=False)
    for j, f in enumerate(FIG2):
        m = parse_checkpoint_name(f)
        idx = idx_of(m["labels"])
        w = read_weights(os.path.join(PRE, f))
        s = score_weights(w, idx)
        ax = axes[0, j]
        stem(ax, w, idx, C[0])
        ax.set_ylim(0, 1.08 if m["N"] == 1 else 1.6 / m["N"] + 0.02)
        ax.set_title(f"N = {m['N']}: {s['n_pred']} detected, weight error {s['rel_l2_error_pct']:.1e} %", fontsize=9)
        if j == 0:
            ax.set_ylabel("mixing ratio $w_i$")
            ax.legend(loc="upper right", fontsize=8)
        if lib:
            E, X, _ = lib
            y = np.log(X[:, idx].mean(axis=1))
            ax = axes[1, j]
            ax.plot(E, y, color=INK, lw=2.4, label="target")
            rec = np.log(X @ w)
            ax.plot(E, rec, color=C[1], lw=1.1, ls="--", label="reconstruction  $\\log\\sum_i w_i\\,\\mathrm{DOS}_i$")
            ax.set_title(f"max |log-DOS residual| = {np.max(np.abs(rec - y)):.1e}", fontsize=9)
            ax.set_xlabel("energy E")
            ax.set_ylim(top=y.max() + 3.2)
            if j == 0:
                ax.set_ylabel("log DOS")
                ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    fig.tight_layout()
    save(fig, "recovery_examples.png")


def fig_library(lib):
    E, X, labels = lib
    heights = (0, 5, 15, 25)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, s in zip(axes, (20, 25, 30)):
        for k, h in enumerate(heights):
            col = X[:, 0] if h == 0 else X[:, labels.index((s, h))]
            ax.plot(E, np.log(col) + 6 * k, color=C[0] if h else MUTED, lw=0.6)
            ax.text(0.104, 6 * k + 7.2, f"h = {h}", fontsize=8, color=INK, va="center")
        ax.set_title(f"σ = {s}")
        ax.set_xlabel("energy E")
        ax.set_xlim(-0.102, 0.122)
        ax.set_yticks([])
    axes[0].set_ylabel("log DOS (curves offset by 6)")
    fig.suptitle("single-bubble basis spectra (grey: flat graphene, h = 0, a single library entry)", fontsize=10)
    fig.tight_layout()
    save(fig, "basis_library.png")


def fig_repeated_incomplete(lib):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.0), gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax, f, title in zip(axes[:2], REPEATED, ["(a) N = 3", "(b) N = 5"]):
        m = parse_checkpoint_name(f)
        idx = idx_of(m["labels"])
        w = read_weights(os.path.join(PRE, f))
        stem(ax, w, idx, C[0])
        for i in sorted(set(idx)):
            ax.annotate(f"{short(BASIS_LABELS[i])}\n{w[i]:.3f}", (i, w[i]), xytext=(4, -2),
                        textcoords="offset points", fontsize=7.5, va="top", color=INK)
        ax.set_ylim(0, 0.8)
        ax.set_title(title + " with a repeated constituent:\n" + " + ".join(short(l) for l in m["labels"]), fontsize=9)
    axes[0].set_ylabel("mixing ratio $w_i$")
    axes[0].legend(loc="upper left", fontsize=8)

    ax = axes[2]
    m = parse_checkpoint_name(INCOMPLETE)
    w = read_weights(os.path.join(PRE, INCOMPLETE))
    present = [basis_index(l) for l in m["labels"] if basis_index(l) < len(w)]
    x = np.arange(len(w))
    ax.axvspan(len(w) - 0.5, 75.5, color=MUTED, alpha=0.15, lw=0)
    ax.text(73, 0.55, "not in\nlibrary", ha="center", fontsize=7.5, color=INK)
    ax.vlines(x, 0, w, color=C[0], lw=1.4)
    ax.plot(x, w, "o", color=C[0], ms=3.2)
    ax.plot(present, [1 / 3] * len(present), "_", color=INK, ms=11, mew=1.6)
    miss = basis_index((30, 25))
    ax.plot([miss], [1 / 3], "_", color=C[1], ms=11, mew=1.6)
    ax.annotate("(30,25), missing", (miss, 1 / 3), xytext=(-4, 8), textcoords="offset points", ha="right",
                fontsize=7.5, color=INK)
    ax.axhline(DETECTION_THRESHOLD, color=MUTED, ls=":", lw=0.9)
    ax.set_xlim(-1.5, 76.5)
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("basis index i")
    ax.set_title("(c) incomplete library (first 70 of 76 spectra):\n" + " + ".join(short(l) for l in m["labels"]), fontsize=9)
    for i in present:
        ax.annotate(f"{short(BASIS_LABELS[i])}\n{w[i]:.3f}", (i, w[i]), xytext=(4, -2), textcoords="offset points",
                    fontsize=7.5, va="top", color=INK)
    fig.tight_layout()
    save(fig, "repeated_and_incomplete.png")


def fig_noise(df):
    nz = df[df.group == "noise"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for k, (n, d) in enumerate(nz.groupby("N")):
        g = d.groupby("snr")
        e = g["rel_l2_error_pct"]
        x = np.array(list(g.groups))
        axes[0].errorbar(x + (k - 1.5) * 0.0012, e.mean(), yerr=e.std(ddof=0), color=C[k], marker="o", ms=4,
                         lw=1.4, capsize=2.5, label=f"N = {n}")
        axes[1].plot(x, 100 * g["exact_support"].mean(), color=C[k], marker="o", ms=4, lw=1.4, label=f"N = {n}")
    axes[0].set_ylabel("relative weight error [%]\n(mean ± std over targets)")
    axes[1].set_ylabel("exact support recovery [%]")
    axes[1].set_ylim(0, 105)
    for ax in axes:
        ax.set_xlabel("noise level SNR  (σ_noise = SNR · max log DOS)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "noise_robustness.png")


def fig_noise_examples(lib):
    E, X, _ = lib
    stemname, snrs = NOISE_EXAMPLE
    fig, axes = plt.subplots(2, len(snrs), figsize=(4.0 * len(snrs), 5.4), squeeze=False)
    rng = np.random.default_rng(0)
    for j, snr in enumerate(snrs):
        f = os.path.join(PRE, "noise", f"{stemname}__snr{snr:.2f}.hdf5")
        m = parse_checkpoint_name(f)
        idx = idx_of(m["labels"])
        w = read_weights(f)
        y = np.log(X[:, idx].mean(axis=1))
        # the original noise realisation was not stored; an equivalent draw is shown for illustration
        y_noisy = y + rng.normal(0, snr * abs(y.max()), size=y.shape)
        ax = axes[0, j]
        ax.plot(E, y_noisy, color=MUTED, lw=0.3, alpha=0.6, label="noisy target (illustrative draw)")
        ax.plot(E, y, color=INK, lw=1.6, label="clean target")
        ax.plot(E, np.log(X @ w), color=C[1], lw=1.1, ls="--", label="reconstruction")
        ax.set_title(f"SNR = {snr}", fontsize=9)
        ax.set_xlabel("energy E")
        stem(axes[1, j], w, idx, C[0])
        s = score_weights(w, idx)
        axes[1, j].set_ylim(0, 1.08)
        axes[1, j].set_title(f"weight on {short(m['labels'][0])}: {w[idx[0]]:.3f}, error {s['rel_l2_error_pct']:.1f} %",
                             fontsize=9)
    axes[0, 0].set_ylabel("log DOS")
    axes[1, 0].set_ylabel("mixing ratio $w_i$")
    fig.tight_layout()
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0))
    save(fig, "noise_examples.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    print("[1] scoring checkpoints")
    df = score_all()
    df.to_csv(os.path.join(RESULTS, "pretrained_scores.csv"), index=False, float_format="%.6g")
    summary = summarise(df)
    summary["incomplete_library"] = incomplete_summary()
    with open(os.path.join(RESULTS, "pretrained_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for n, d in summary["random_mixtures"]["by_N"].items():
        print(f"  random N={n:2d}: {d['checkpoints']:3d} ckpts, exact {100 * d['exact_support_rate']:.0f} %, "
              f"rel. error mean {d['mean_rel_l2_error_pct']:.1e} % max {d['max_rel_l2_error_pct']:.1e} %")
    print("  repeated:", {k: v for k, v in summary["repeated"].items() if k != "examples"})
    for s, d in summary["noise_all_N"].items():
        print(f"  noise SNR={s}: exact {100 * d['exact_support_rate']:.0f} %, rel. error {d['mean_rel_l2_error_pct']:.2f} %")

    try:
        lib = build_library(a.data_dir)
    except FileNotFoundError as e:
        print("  [skip spectra figures]", e)
        lib = None
    print("[2] figures")
    if lib:
        fig_library(lib)
        fig_recovery(lib)
    else:  # keep the shipped figures that show spectra instead of overwriting them with weight-only panels
        print("  [skip] basis_library.png, recovery_examples.png, noise_examples.png (need the DOS library)")
    fig_repeated_incomplete(lib)
    fig_noise(df)
    if lib:
        fig_noise_examples(lib)


if __name__ == "__main__":
    main()
