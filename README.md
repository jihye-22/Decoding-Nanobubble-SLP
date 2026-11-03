# Decoding multi-nanobubble configurations in graphene

**A physics-constrained single-layer network whose trained weights are the mixing ratios of single-nanobubble spectra, so they show directly which bubbles are present.**

Nanobubbles form readily in atomically thin graphene. Each bubble has a Gaussian height profile, so its width $\sigma$ and height $h$ are the fingerprint of the bubble. The strain shows up as sharp, geometry-dependent features in the electronic density of states (DOS). When several bubbles are present, their features overlap, so it is hard to read off from one spectrum *which* bubbles, and how many, produced it.

For spatially separated nanobubbles, KWANT quantum-transport simulation by our collaborators showed that the multi-bubble DOS is numerically equivalent to the **normalised sum of the constituent single-bubble spectra**. This repository builds that additive relation into a compact model. A target spectrum is decomposed over a library of single-bubble spectra by a single dense layer with non-negative, bounded weights. After fitting, each weight is the mixing ratio of one library geometry $(\sigma, h)$. 

This is the code, the trained checkpoints and the evaluation for
J. Kim, T. Song, and N. Myoung, *Physics-constrained machine learning for decoding multi-nanobubble configurations in graphene*, [arXiv:2607.17866](https://arxiv.org/abs/2607.17866) (2026).

---

## Model

Library: 76 single-bubble DOS spectra $\mathrm{D}_i(E)$ (bubble width $\sigma \in \{20, 25, 30\}$, height $h = 1 \dots 25$, plus flat graphene) on 50 000 energy points. Target: the log-DOS of a configuration $S$ of $N$ bubbles,

$$y(E) = \log \Big( \frac{1}{N} \sum_{i \in S} \mathrm{D}_i(E) \Big).$$

The model is one `Dense(1, use_bias=False)` unit whose inputs at energy $E$ are the 76 library values, trained with

$$\mathcal{L}(\mathbf{w}) = \Big\langle \Big( y(E) - \log \sum_{i=1}^{76} w_i\\mathrm{D}_i(E) \Big)^2 \Big\rangle_E , \qquad 0 \le w_i \le 1 .$$

| element | physical role |
| :-- | :-- |
| sum inside the log | the validated additive relation (linear superposition) for separated bubbles |
| weights clipping ([0,1]) | mixing ratios are fractions. The true solution is $w_i = 1/N$ for each constituent ($k/N$ if a geometry occurs $k$ times) |
| loss on $\log$ DOS | the DOS spans more than four orders of magnitude; the log compares spectra in relative terms, so the sharp resonances do not dominate the fit |
| readout | constituents = $\{i : w_i \ge 0.01\}$, $N_\text{pred}$ = their number, proportions = $w_i$ |

Each target is fitted independently: one network, 76 parameters per spectrum. Training (paper settings): Adam with initial learning rate $10^{-3}$, mini-batches of 128 energy points, at most $10^4$ epochs, early stopping on the loss (patience 100) and `ReduceLROnPlateau` (factor 0.5, patience 50); Glorot-uniform initialisation, no fixed seed (see [`nanobubble/`](nanobubble) and [`pretrained/README.md`](pretrained/README.md)).

![basis library](assets/basis_library.png)

---

## Results

Unless noted otherwise, all numbers come from `python scripts/evaluate_pretrained.py`. The script reads the 623 checkpoints in [`pretrained/`](pretrained). The ground truth of every checkpoint is encoded in its file name, and the outputs go to [`results/`](results) and [`assets/`](assets). The *weight error* is the normalised L1 deviation of the paper (Eq. 10), $\delta w = 100 \frac {\sum_i \lvert w_i - w_i^\text{true} \rvert} {\sum_i \lvert w_i^\text{true} \rvert}$ in %. *Exact support recovery* means that the set of detected bases ($w_i \ge 0.01$) is exactly the true set.

### 1. Noise-free configurations: exact recovery up to N = 10

![recovery examples](assets/recovery_examples.png)

| N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| targets | 76 | 26 | 17 | 16 | 17 | 10 | 14 | 20 | 25 | 17 |
| exact support recovery | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| max. weight error δw | 9e-4 % | 3e-4 % | 9e-4 % | 2e-4 % | 2e-4 % | 9e-5 % | 2e-4 % | 2e-4 % | 2e-4 % | 1e-4 % |

All 238 random configurations (N = 1: every library spectrum once) are identified exactly. The weights match $1/N$ to better than $10^{-3}$ %, and the error does not grow with the number of bubbles.

### 2. Repeated constituents and an incomplete library

![repeated and incomplete](assets/repeated_and_incomplete.png)

**Repeated constituents (a, b).** If the same geometry occurs twice, the model does not need two basis functions. It puts weight $2/3$ on that geometry, and $1/3$ on the other one. All 78 such targets are recovered exactly: 76 of the form $[i, i, j]$ and two hand-picked cases, e.g. (30,0) + 3×(20,5) + (20,19) → 0.200 / 0.600 / 0.200. The maximum weight error is $\delta w = 2\times10^{-4}$ %.

**Incomplete library (c).** Here the model sees only the first 70 spectra, and every target contains the missing geometry (30, 25):

| target (truth 1/N each) | weights on the constituents that are in the library | other detected bases | $\sum_i w_i$ |
| :-- | :-- | :-: | :-: |
| (30,25) alone | – | 11, largest: flat graphene 0.64, (30,21) 0.14 | 1.13 |
| (20,4) + (25,7) + (30,25) | 0.413, 0.395 (truth 0.333) | 7, largest: flat graphene 0.08 | 1.08 |
| 6 geometries + (30,25) | 0.150 – 0.183 (truth 0.143) | 2, largest: flat graphene 0.02 | 1.04 |

The constituents that are in the library are still found, with the largest weights. The missing spectrum is approximated by a spread of other spectra, mostly flat graphene and bubbles of similar height. The weights then no longer sum to 1, and $N_\text{pred}$ is too large. Spurious detections and a weight sum $\neq 1$ are therefore a usable sign that the library is incomplete.

### 3. Measurement noise
![noise examples](assets/noise_examples.png)

Noise model (paper, Eq. 9): Gaussian white noise added to the log-DOS, $y_\eta(E) = y(E) + \eta f(E)$ with $f \sim \mathcal{N}(0, 1)$. In the code and the checkpoint names the amplitude is given as a noise level SNR, $\eta = \text{SNR}\cdot\lvert\max_Ey\rvert$. SNR = 0.01, 0.05 and 0.09 correspond to the amplitudes η = 0.165, 0.82 and 1.49 of the paper's Fig. 4; η = 0.165 is multiplicative noise of about 18 % on the DOS.

![noise robustness](assets/noise_robustness.png)

The figure follows Fig. 4 of the paper (N = 1, 3, 5; the paper uses 7, 6 and 8 realisations per noise level). The shipped `pretrained/noise/` folder holds all trained runs, 304 noisy targets with 6–10 per cell and N = 2 in addition, so its means differ slightly from the paper's figure. Mean weight error δw / exact support recovery:

| η (SNR) | 0.165 (0.01) | 0.50 (0.03) | 0.82 (0.05) | 1.16 (0.07) | 1.49 (0.09) |
| :-- | :-: | :-: | :-: | :-: | :-: |
| N = 1 | 0.63 % / 100 % | 1.7 % / 100 % | 6.5 % / 75 % | 5.5 % / 100 % | 7.2 % / 86 % |
| N = 2 | 1.2 % / 100 % | 3.2 % / 100 % | 4.8 % / 100 % | 6.7 % / 83 % | 8.6 % / 67 % |
| N = 3 | 1.1 % / 100 % | 3.0 % / 100 % | 5.8 % / 86 % | 8.2 % / 67 % | 13.6 % / 33 % |
| N = 5 | 2.0 % / 100 % | 5.4 % / 90 % | 8.7 % / 80 % | 12.8 % / 44 % | 14.7 % / 56 % |

The weight error grows with the noise amplitude and with N. It stays at the few-percent level even when the noise visibly buries the spectrum (below). Support recovery is exact at η = 0.165 for every N. At larger η, the failures are always **false positives**: an extra basis gets a small weight just above the 0.01 threshold, and no true constituent is ever missed (`results/pretrained_scores.csv`).


### 4. Comparison with conventional regression

`python scripts/compare_baselines.py` builds 10 random configurations for each N ∈ {1, 2, 3, 5} (seeded). It adds noise at SNR ∈ {0, 0.01, 0.03, 0.05, 0.09}, giving 200 targets, and decomposes every target with two conventional regressions, as in the original comparison notebook:

* **NNLS** and **LassoCV (positive, no intercept)** on the linear DOS, $e^{y} \approx X\mathbf{w}$.

**This work** is the trained network of §3: the Keras checkpoints in `pretrained/noise/` with the same N and SNR (28–32 targets per SNR). Their noise realisations were not stored, so these are different configurations and noise draws from the same noise model, not the same 200 targets. All methods use the same detection threshold ($w_i \ge 0.01$).

![baseline comparison](assets/baseline_comparison.png)

Averaged over N (baselines: 40 targets per column; per-N values in [`results/baseline_summary.csv`](results/baseline_summary.csv)):

| exact support recovery | SNR 0 | 0.01 | 0.03 | 0.05 | 0.09 |
| :-- | :-: | :-: | :-: | :-: | :-: |
| **this work** (Keras checkpoints) | **100 %** | **100 %** | **98 %** | **85 %** | **60 %** |
| NNLS (linear DOS) | 100 % | 8 % | 8 % | 0 % | 2 % |
| LassoCV, positive (linear DOS) | 15 % | 2 % | 5 % | 0 % | 2 % |


* **Linear-space regression breaks down as soon as there is noise.** NNLS is exact on clean data, but at SNR = 0.01 it already fails to recover the support in 92 % of the targets. The fit is dominated by the few energies where the DOS is largest; the log-DOS loss compares spectra in relative terms instead.
* **LassoCV does not recover the support even without noise:** the L1 penalty shrinks the true weights, and the cross-validated penalty admits extra bases.

Runtime per target (one CPU core): NNLS 0.2 s, LassoCV 10 s. The Keras training takes 4 min (noise-free) to 20–30 min (noisy) per target on CPU.

### Summary

| question | answer |
| :-- | :-- |
| Do the trained weights identify the configuration? | **Yes.** Exactly for all 316 noise-free targets (N = 1 … 10, repeated constituents) |
| Are the weights quantitatively meaningful? | **Yes.** They equal the mixing ratios $k/N$ to $\delta w < 10^{-3}$ % |
| Robust to noise? | δw ≈ 1–2 % at η = 0.165 and ≈ 7–15 % at η = 1.49. Support recovery degrades above η ≈ 0.5 through false positives |
| Better than conventional regression? | Yes. Much better support recovery than NNLS and LassoCV on the linear DOS at every noise level > 0 |

## Limitations

* **Synthetic, in-library targets.** All targets are built from the same library that is used for decoding. The additive model is therefore exact by construction, apart from the incomplete-library test. Real spectra (e.g. STM dI/dV) add broadening, a background, tip effects and geometries that are not in the library. The additive relation itself holds only for spatially separated bubbles.
* **Incomplete library.** A missing geometry is absorbed by several other spectra (§2). The weights then cannot be read as a configuration without further checks (weight sum, residual).
* **Noise.** The fixed threshold of 0.01 produces false positives above η ≈ 0.5 (SNR ≈ 0.03), and the method gives no uncertainty estimate for the weights. The noise model is Gaussian on log-DOS, scaled to the maximum of the log-DOS. "SNR" is the name of this noise level in the code, not a power ratio.
* **Per-spectrum optimisation.** There is no amortised network; every new spectrum needs its own fit (minutes with Keras on CPU).
* **Reproducibility of the original runs.** Seeds were not fixed, and the noise realisations of the noisy checkpoints were not stored. The noise numbers in §3 are therefore statistics of those runs; only the baseline fits in §4 are seeded.

---

## Repository structure

```text
Decoding-Nanobubble-SLP/
├── nanobubble/                 # small Python package
│   ├── data.py                 #   read .dat files, build the 76-spectrum library, make targets (+ noise)
│   ├── model.py                #   ClipConstraint, log-sum MSE loss, build_model()  (TensorFlow/Keras)
│   ├── fit.py                  #   fit_target(): training with the paper settings (Keras)
│   ├── evaluate.py             #   read checkpoint weights (h5py), parse ground truth, score weights
│   └── baselines.py            #   NNLS, LassoCV(positive)
├── scripts/
│   ├── evaluate_pretrained.py  # §1–3: all checkpoint numbers and figures
│   ├── compare_baselines.py    # §4: comparison with conventional regression
│   └── train.py                # fit one target (given, random, noisy, or with an incomplete library)
├── notebooks/demo.ipynb        # short executed walkthrough
├── pretrained/                 # 623 checkpoints (17 KB each) + manifest.csv (see pretrained/README.md)
├── data/                       # format description and three sample spectra (see data/README.md)
├── results/                    # CSV / JSON outputs of the scripts
└── assets/                     # figures
```

## Reproducing

```bash
pip install -r requirements.txt

# §1–3 from the shipped checkpoints (weights only; spectrum panels need the DOS library), < 1 min
python scripts/evaluate_pretrained.py --data-dir /path/to/DOS

# §4: refit the baselines (needs the DOS library, ≈ 15 min on 8 CPU cores) ...
python scripts/compare_baselines.py --data-dir /path/to/DOS
# ... or rebuild the tables and figure from the shipped baseline fits and the checkpoints, < 1 min
python scripts/compare_baselines.py --from-runs

# fit a single target
python scripts/train.py --data-dir /path/to/DOS --labels 20,4 25,7 30,10              # given constituents
python scripts/train.py --data-dir /path/to/DOS --n 5 --seed 3 --snr 0.05              # random constituents + noise
python scripts/train.py --data-dir /path/to/DOS --labels 20,4 25,7 30,25 --n-basis 70  # incomplete library
```

```python
from nanobubble import read_weights, parse_checkpoint_name, BASIS_LABELS

w = read_weights("pretrained/repeated/N3__s30h20+s30h20+s30h11.hdf5")
print({BASIS_LABELS[i]: round(float(w[i]), 4) for i in (w >= 0.01).nonzero()[0]})
# {(30, 11): 0.3333, (30, 20): 0.6667}
```

The single-bubble DOS library comes from the collaborating group's calculations and is not redistributed (available on request; format in [`data/README.md`](data/README.md)). Without it, `evaluate_pretrained.py` still reproduces every weight-based number and the figures `repeated_and_incomplete.png` and `noise_robustness.png`; the figures that show spectra are kept as shipped. Tested with Python 3.8 and 3.10, TensorFlow 2.10, NumPy 1.24, SciPy 1.10, scikit-learn 1.3.

## Publication and presentations

* **J. Kim**, T. Song, and N. Myoung, "Physics-constrained machine learning for decoding multi-nanobubble configurations in graphene," [arXiv:2607.17866](https://arxiv.org/abs/2607.17866) (2026).
* *Interpretable Single-Layer Neural Networks for Multi-Nanobubble Characterization in Graphene*: oral, AI4Sci Korea 2026.
* *Interpretable Physics-Constrained Machine Learning for Identifying Multi-Nanobubble Configurations in Graphene*: poster, ICPS 2026, Japan.

```bibtex
@article{kim2026nanobubble,
  title   = {Physics-constrained machine learning for decoding multi-nanobubble configurations in graphene},
  author  = {Kim, Jihye and Song, Taegeun and Myoung, Nojoon},
  journal = {arXiv preprint arXiv:2607.17866},
  year    = {2026}
}
```

A related project applies the same idea, an interpretable single-layer network whose weights carry physical meaning, to signal decomposition with sinusoidal units: [Interpretable-Sine-SLP](https://github.com/jihye-22/interpretable-sine-SLP).

## Acknowledgements

This is a collaborative project, supervised by Prof. Taegeun Song (Department of Data Information and Physics, Kongju National University) and Prof. Nojoon Myoung (Chosun University). The single-nanobubble DOS spectra were provided by Prof. Myoung's group.

Author: **Jihye Kim** · [github.com/jihye-22](https://github.com/jihye-22)
