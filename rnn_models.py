"""
Elman, Jordan and multi-recurrent (MRNN) networks as Keras RNN cells.

All three cells take one input vector per time step and return a single
linear output y.

L2 weight penalty: every cell takes l2 >= 0. When l2 > 0, l2 * sum(w^2) over
all weight matrices (input W, recurrent U / Uh / Uo, output V) is added to
the training loss. Biases are not penalised.
"""

import numpy as np
import keras
from keras import layers, ops
from sklearn.base import BaseEstimator, RegressorMixin


def _l2(l2):
    """
    L2 regulariser for weight matrices, or None when l2 == 0.
    """

    return keras.regularizers.L2(l2) if l2 else None


class RNN(BaseEstimator, RegressorMixin):
    """
    Makes a Keras RNN behave like a scikit-learn model.
    """

    def __init__(self, arch="elman", lags=12, units=8, lr=0.01, epochs=300, n_banks=4, l2=0.0, seed=0):
        self.arch, self.lags, self.units, self.lr = arch, lags, units, lr
        self.epochs, self.n_banks, self.seed, self.l2 = epochs, n_banks, seed, l2

    def _prep(self, X):
        X = np.asarray(X, dtype="float32")[:, -self.lags:]      # keep only the last `lags` values
        return ((X - self.mu_) / self.sd_)[..., None]           # scale, add a feature axis -> (n, lags, 1)

    def fit(self, X, y):
        keras.utils.set_random_seed(self.seed)
        y = np.asarray(y, dtype="float32").ravel()

        # scaling from the TRAINING fold only
        self.mu_, self.sd_ = y.mean(), y.std()

        self.model_ = build_model(self.arch, self.lags, self.units, self.lr, self.n_banks, self.l2)
        self.model_.fit(self._prep(X), (y - self.mu_) / self.sd_, epochs=self.epochs, batch_size=len(y), verbose=0)
        return self

    def predict(self, X):
        return self.model_.predict(self._prep(X), verbose=0).ravel()*self.sd_ + self.mu_


class ElmanCell(layers.Layer):
    """
    Elman network: the context layer is a copy of the previous hidden state.
    """

    def __init__(self, units, l2=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.l2 = l2
        self.state_size = units
        self.output_size = 1

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(shape=(d, self.units), initializer="glorot_uniform", name="W", regularizer=_l2(self.l2))
        self.U = self.add_weight(shape=(self.units, self.units), initializer="orthogonal", name="U", regularizer=_l2(self.l2))
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", name="b")
        self.V = self.add_weight(shape=(self.units, 1), initializer="glorot_uniform", name="V", regularizer=_l2(self.l2))

    def call(self, x, states):
        h_prev = states[0]
        h = ops.tanh(ops.matmul(x, self.W) + ops.matmul(h_prev, self.U) + self.b)
        y = ops.matmul(h, self.V)
        return y, [h]

    def get_config(self):
        return {**super().get_config(), "units": self.units, "l2": self.l2}


class JordanCell(layers.Layer):
    """
    Jordan network: the context layer is the previous network output.
    """

    def __init__(self, units, l2=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.l2 = l2
        self.state_size = 1
        self.output_size = 1

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(shape=(d, self.units), initializer="glorot_uniform", name="W", regularizer=_l2(self.l2))
        self.U = self.add_weight(shape=(1, self.units), initializer="glorot_uniform", name="U", regularizer=_l2(self.l2))
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", name="b")
        self.V = self.add_weight(shape=(self.units, 1), initializer="glorot_uniform", name="V", regularizer=_l2(self.l2))

    def call(self, x, states):
        y_prev = states[0]
        h = ops.tanh(ops.matmul(x, self.W) + ops.matmul(y_prev, self.U) + self.b)
        y = ops.matmul(h, self.V)
        return y, [y]

    def get_config(self):
        return {**super().get_config(), "units": self.units, "l2": self.l2}


class MRNNCell(layers.Layer):
    """
    Multi-recurrent network: context layer is previous hidden state and network output.
    """

    def __init__(self, units, l2=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.l2 = l2
        self.state_size = [units, 1]   # [previous hidden state, previous output]
        self.output_size = 1

    def build(self, input_shape):
        d, u = input_shape[-1], self.units
        self.W = self.add_weight(shape=(d, u), initializer="glorot_uniform", name="W", regularizer=_l2(self.l2))
        self.Uh = self.add_weight(shape=(u, u), initializer="glorot_uniform", name="Uh", regularizer=_l2(self.l2))
        self.Uo = self.add_weight(shape=(1, u), initializer="glorot_uniform", name="Uo", regularizer=_l2(self.l2))
        self.b = self.add_weight(shape=(u,), initializer="zeros", name="b")
        self.V = self.add_weight(shape=(u, 1), initializer="glorot_uniform", name="V", regularizer=_l2(self.l2))

    def call(self, x, states):
        h_prev, y_prev = states
        h = ops.tanh(ops.matmul(x, self.W) + ops.matmul(h_prev, self.Uh) + ops.matmul(y_prev, self.Uo) + self.b)
        y = ops.matmul(h, self.V)
        return y, [h, y]

    def get_config(self):
        return {**super().get_config(), "units": self.units, "l2": self.l2}


def build_model(arch, lags, units, lr=1e-2, n_banks=4, l2=0.0):
    """
    arch: 'elman' | 'jordan' | 'mrnn'. l2: L2 weight penalty (0 = off).
    Returns a compiled Keras model.

    With l2 > 0 the reported `loss` (and `val_loss`) includes the penalty, so
    the model also tracks plain `mse` for comparing prediction error.
    """

    cells = {
        "elman": lambda: ElmanCell(units, l2=l2),
        "jordan": lambda: JordanCell(units, l2=l2),
        "mrnn": lambda: MRNNCell(units, l2=l2)
    }
    input_layer = keras.Input(shape=(lags, 1))
    output_layer = layers.RNN(cells[arch]())(input_layer)
    model = keras.Model(input_layer, output_layer)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mse"])
    return model


def make_windows(series, lags):
    """
    Turn a 1-D series into (X, y): X[i] = series[i:i+lags], y[i] = series[i+lags].
    """

    s = np.asarray(series, dtype="float32")
    X = np.stack([s[i:i + lags] for i in range(len(s) - lags)])[..., None]
    y = s[lags:, None]
    return X, y
