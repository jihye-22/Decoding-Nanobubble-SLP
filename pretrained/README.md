# Pretrained checkpoints

Keras `.hdf5` files (TensorFlow 2.10, Keras 2.10) from the original experiments, Sep 2025 – Apr 2026.
Each file holds one trained single-layer network, i.e. the mixing ratios of **one** target spectrum
(kernel `model_weights/basis_head/basis_head/kernel:0`, shape `(76, 1)` or `(70, 1)`).

Read the weights without TensorFlow:

```python
from nanobubble import read_weights, parse_checkpoint_name
w = read_weights("pretrained/repeated/N3__s30h20+s30h20+s30h11.hdf5")   # (76,)
parse_checkpoint_name("N3__s30h20+s30h20+s30h11.hdf5")   # {'N': 3, 'labels': [(30, 20), (30, 20), (30, 11)], 'snr': 0.0}
```

or load the model with `tf.keras.models.load_model(path, custom_objects=nanobubble.model.custom_objects())`.

## File names

`N{N}__s{σ}h{hh}+s{σ}h{hh}+…[__snr{SNR}].hdf5` — the ground-truth configuration in the order in which the
constituents were drawn (a repeated label means a repeated constituent, true weight k/N), and the noise
level for the noise runs. The original file names (e.g. `INR(N=3,[[30.0, 20.0], [30.0, 20.0], [30.0, 11.0]]).hdf5`;
INR was the internal project name) are listed in [`manifest.csv`](manifest.csv) together with the learning rate
and Adam step count stored in each file.

## Folders

| folder | files | experiment | original folder |
| :-- | :-: | :-- | :-- |
| `random_mixtures/` | 238 | noise-free targets, N = 1 … 10 distinct random constituents (N = 1: every basis once) | `MODEL_SAVE(weight-log)/INR(…)` |
| `repeated/` | 78 | noise-free targets with a repeated constituent: 76 × N = 3 (`[i, i, j]`, one per basis i) and 2 hand-picked cases | `MODEL_SAVE(weight-log)/INR_same(…)`, `MODEL_SAVE(same)/` |
| `noise/` | 304 | targets with Gaussian noise on log-DOS, N ∈ {1, 2, 3, 5}, SNR = 0.00 … 0.09 (6–10 targets per cell) | `MODEL_SAVE(noise)/INR_weightsum(…)` |
| `incomplete_library/` | 3 | the model sees only the first 70 basis spectra; every target contains the missing (30, 25) | `MODEL_SAVE(70)/INR(…)` |

## Training settings

Common to all folders (see `nanobubble/model.py`, `nanobubble/fit.py`):
one `Dense(1, use_bias=False)` layer with weights clipped to [0, 1] after every step; loss
`mean((y − log(1e-12 + relu(Xw)))²)` over the 50 000 energy points; Adam; batch 128; up to 10⁴ epochs (early stopping ended every run before epoch ≈ 2100);
`EarlyStopping(loss, patience 100)`; `ReduceLROnPlateau(loss, factor 0.5, patience 50, min_lr 1e-8, min_delta 1e-8)`;
`ModelCheckpoint(save_best_only)` on the training loss. There is no validation split: each network is fitted to a
single spectrum, the "training set" being its energy points.

| folder | initial lr | relu in loss | restore best weights | notes |
| :-- | :-: | :-: | :-: | :-- |
| `random_mixtures/` | 1e-3 ¹ | see ² | no ³ | best epoch: median ≈ 100, max ≈ 1000 |
| `repeated/` | 1e-3 | yes | yes | the 76 `[i, i, j]` runs: notebook `INR_weight-log(same)` (project git history); the 2 hand-picked runs (Mar 2026): notebook not preserved |
| `noise/` | 1e-3 | yes | yes | notebook `INR_noise_train`; best epoch median ≈ 1600 |
| `incomplete_library/` | – | – | – | trained Jan 2026; the producing notebook was not preserved. Same file layout and loss name; best epoch ≈ 1500 |

¹ The surviving notebook version of this sweep has `learning_rate=1e-4`, but 82 of the 238 files still store
lr = 1e-3 at their best epoch (the learning rate can only decrease during training), so the run that produced the
files must have started from 1e-3.
² One notebook version omits the `relu`. For non-negative weights and positive spectra `Xw > 0`, so both variants
define the same loss on the feasible set; the difference is only relevant directly after a Glorot initialisation
with negative entries.
³ `ModelCheckpoint(save_best_only=True)` writes the best-loss weights regardless.

Seeds were not fixed in the original runs, and the noise realisations of the `noise/` runs were not stored
(only the SNR). Exploratory checkpoints not used in the paper (`MODEL_SAVE(weight-log,noise)`, the
`INR_nonsum` runs in `MODEL_SAVE(70)`, three noise-free `INR(…)` files in `MODEL_SAVE(noise)`) are not included.
