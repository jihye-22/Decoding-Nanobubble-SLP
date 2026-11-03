"""The single-layer basis-mixing network (TensorFlow / Keras).

    s(E) = sum_i w_i DOS_i(E)            (one Dense unit, no bias, no activation)
    loss = < ( y(E) - log(EPS + relu(s(E))) )^2 >_E ,    0 <= w_i <= 1

Each input feature is one library spectrum evaluated at the energy E, so a training
"sample" is one energy point and the 76 kernel entries are the mixing ratios.
"""
import tensorflow as tf

EPS = 1e-12


class ClipConstraint(tf.keras.constraints.Constraint):
    """Projects the weights onto [min_val, max_val] after every optimizer step."""

    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_val, self.max_val)

    def get_config(self):
        return {"min_val": self.min_val, "max_val": self.max_val}


def log_after_sum_mse(y_true, s_pred):
    """MSE between the target log-DOS and the log of the weighted sum of basis spectra.

    The ``relu`` only matters if ``s`` becomes non-positive (e.g. right after a Glorot
    initialisation with negative entries); for ``w >= 0`` and positive spectra it is the identity.
    """
    y_pred_log = tf.math.log(EPS + tf.nn.relu(s_pred))
    return tf.reduce_mean(tf.square(y_true - y_pred_log))


def build_model(n_basis=76, learning_rate=1e-3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_basis,)),
        tf.keras.layers.Dense(1, use_bias=False, kernel_constraint=ClipConstraint(0.0, 1.0),
                              name="basis_head"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=log_after_sum_mse)
    return model


def custom_objects():
    return {"ClipConstraint": ClipConstraint, "log_after_sum_mse": log_after_sum_mse}
