"""Fitting the mixing ratios of one target spectrum.

``fit_target`` – the training procedure of the paper (Keras, Adam, mini-batches of energy points).
"""


def fit_target(X, y, checkpoint=None, learning_rate=1e-3, epochs=10000, batch_size=128,
               patience=100, seed=None, verbose=0):
    """Train the single-layer network on one target (settings of the paper).

    Adam(lr), batch 128, up to 1e4 epochs; EarlyStopping(loss, patience 100, restore best);
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
