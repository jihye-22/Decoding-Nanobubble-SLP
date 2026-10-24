"""Fitting the mixing ratios of one target spectrum.

``fit_target``   – the original training procedure (Keras, Adam, mini-batches of energy points).
``solve_bounded`` – the same objective and box constraint minimised with L-BFGS-B (numpy/scipy).
                   Deterministic and ~1000x faster; used for the large baseline sweep and
                   cross-checked against the Keras checkpoints (see README).
"""
import numpy as np
from scipy.optimize import minimize

EPS = 1e-12


def log_sum_loss(w, X, y):
    """Objective and gradient of  mean((y - log(EPS + Xw))^2)."""
    s = X @ w + EPS
    r = y - np.log(s)
    loss = np.mean(r ** 2)
    grad = -2.0 / len(y) * (X.T @ (r / s))
    return loss, grad


def solve_bounded(X, y, w0=None, tol=1e-14, maxiter=20000):
    """Minimise the log-sum MSE for w in [0, 1]^n with L-BFGS-B. Returns (w, final loss)."""
    n = X.shape[1]
    w0 = np.full(n, 1.0 / n) if w0 is None else np.asarray(w0, float)
    res = minimize(log_sum_loss, w0, args=(X, y), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, 1.0)] * n,
                   options={"maxiter": maxiter, "maxfun": 4 * maxiter, "ftol": tol, "gtol": 1e-12})
    return res.x, float(res.fun)


def fit_target(X, y, checkpoint=None, learning_rate=1e-3, epochs=100000, batch_size=128,
               patience=100, seed=None, verbose=0):
    """Train the single-layer network on one target (settings of the paper).

    Adam(lr), batch 128, up to 1e5 epochs; EarlyStopping(loss, patience 100, restore best);
    ReduceLROnPlateau(loss, x0.5, patience 50, min_lr 1e-8, min_delta 1e-8);
    optional ModelCheckpoint(save_best_only) to ``checkpoint``.
    Returns (weights, history dict).
    """
    import tensorflow as tf

    from .model import build_model

    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
    model = build_model(X.shape[1], learning_rate)
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=50, min_lr=1e-8,
                                             min_delta=1e-8, verbose=0),
    ]
    if checkpoint:
        cbs.append(tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="loss", mode="min",
                                                      save_best_only=True, verbose=0))
    hist = model.fit(X.astype("float32"), y.astype("float32"), epochs=epochs, batch_size=batch_size,
                     verbose=verbose, callbacks=cbs)
    return model.get_layer("basis_head").get_weights()[0].ravel(), hist.history
