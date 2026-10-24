# Data

## Single-nanobubble DOS library (not included)

The model needs a library of 78 single-nanobubble density-of-states spectra:

```text
<data_dir>/sigma20/condAllsm20h00.dat … condAllsm20h25.dat
<data_dir>/sigma25/condAllsm25h00.dat … condAllsm25h25.dat
<data_dir>/sigma30/condAllh00.dat     … condAllh25.dat        # note: no "sm30" in these names
```

* `sigma` (σ ∈ {20, 25, 30}) is the width parameter and `h` (0 … 25) the height of the bubble;
  `h = 0` is flat graphene. **TODO (author): state the units of σ and h and the bubble height profile (Gaussian?).**
* Each file has two whitespace-separated columns, energy `E` and `DOS`, on a uniform grid with
  E ∈ [−0.1, 0.1] (19 980 – 19 996 rows). **TODO (author): units of E (eV or hopping energy t?) and DOS normalisation.**
* The spectra were pre-computed with full quantum-transport simulations by the collaborating group
  (Prof. Nojoon Myoung, Chosun University). **TODO (author): method / code / reference for these calculations.**

The full library is not redistributed here; it is available from the authors on request. To run the
spectrum-level scripts, place it in `data/full/` or point to it:

```bash
export NANOBUBBLE_DATA_DIR=/path/to/DOS        # or pass --data-dir
```

`nanobubble.build_library()` interpolates every spectrum onto 50 000 points in [−0.099, 0.099], sorts the 78
spectra by (h, σ) and keeps the flat-graphene spectrum once (as σ = 30, h = 0), giving the 76 basis spectra.
The interpolated library is cached in `.cache/` (git-ignored).

## Samples (`samples/`)

Three spectra in the original format, **downsampled to every 10th energy point** to show the format
(the sharp resonances are not fully resolved at this sampling; do not use them for fitting):

| file | σ | h |
| :-- | :-: | :-: |
| `samples/sigma30/condAllh00.dat` | – | 0 (flat graphene) |
| `samples/sigma20/condAllsm20h10.dat` | 20 | 10 |
| `samples/sigma30/condAllh25.dat` | 30 | 25 |

```python
import numpy as np
E, dos = np.loadtxt("data/samples/sigma20/condAllsm20h10.dat", unpack=True)
```
