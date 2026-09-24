"""
Elman, Jordan and multi-recurrent (MRNN) networks as Keras RNN cells.

All three cells take one input vector per time step and return a single
linear output y_t. The output layer sits inside each cell so that Jordan
and MRNN can feed y_t back as context.

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
    Multi-recurrent network: hidden-state and output feedback, each held in
    n_banks memory banks with fixed self-recurrent decay rates alphas.

    Default alphas for n_banks = k are 0, 1/k, 2/k, ..., (k-1)/k. For example,
    k = 4 gives 0, 0.25, 0.5, 0.75. A bank with alpha = 0 is an exact copy of
    the last step, so bank 0 reproduces Elman (hidden) and Jordan (output)
    feedback. Banks with larger alpha keep a slower, smoothed memory of the past.
    """

    def __init__(self, units, n_banks=4, alphas=None, l2=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.l2 = l2
        self.n_banks = n_banks
        self.alphas = list(alphas) if alphas is not None else [i / n_banks for i in range(n_banks)]
        assert len(self.alphas) == n_banks
        self.state_size = [units * n_banks, n_banks]   # [hidden memories, output memories]
        self.output_size = 1
        # decay vectors laid out bank by bank to match ops.tile ordering
        self._a_h = np.repeat(self.alphas, units).astype("float32")
        self._a_o = np.array(self.alphas, dtype="float32")

    def build(self, input_shape):
        d, u, k = input_shape[-1], self.units, self.n_banks
        self.W = self.add_weight(shape=(d, u), initializer="glorot_uniform", name="W", regularizer=_l2(self.l2))
        self.Uh = self.add_weight(shape=(u * k, u), initializer="glorot_uniform", name="Uh", regularizer=_l2(self.l2))
        self.Uo = self.add_weight(shape=(k, u), initializer="glorot_uniform", name="Uo", regularizer=_l2(self.l2))
        self.b = self.add_weight(shape=(u,), initializer="zeros", name="b")
        self.V = self.add_weight(shape=(u, 1), initializer="glorot_uniform", name="V", regularizer=_l2(self.l2))
        self.c = self.add_weight(shape=(1,), initializer="zeros", name="c")

    def call(self, x, states):
        mh, mo = states
        h = ops.tanh(ops.matmul(x, self.W) + ops.matmul(mh, self.Uh) + ops.matmul(mo, self.Uo) + self.b)
        y = ops.matmul(h, self.V) + self.c
        a_h = ops.convert_to_tensor(self._a_h)
        a_o = ops.convert_to_tensor(self._a_o)
        new_mh = a_h * mh + (1.0 - a_h) * ops.tile(h, [1, self.n_banks])
        new_mo = a_o * mo + (1.0 - a_o) * y          # y (batch,1) broadcasts across banks
        return y, [new_mh, new_mo]

    def get_config(self):
        return {**super().get_config(), "units": self.units,
                "n_banks": self.n_banks, "alphas": self.alphas, "l2": self.l2}


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
        "mrnn": lambda: MRNNCell(units, n_banks=n_banks, l2=l2)
    }
    input_layer = keras.Input(shape=(lags, 1))
    output_layer = layers.RNN(cells[arch]())(input_layer)
    model = keras.Model(input_layer, output_layer)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mse"])
    return model


def early_stopping(patience=50, min_delta=1e-5):
    """
    Stop training once validation loss has not improved by at least min_delta
    for `patience` epochs, then roll back to the best weights seen.
    """

    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
    )


def fit_model(model, X, y, val_data=None, val_frac=0.2, epochs=1000, batch_size=None, patience=50, verbose=0):
    """
    Fit with early stopping on validation loss.

    If val_data=(X_val, y_val) is not given, the LAST val_frac of (X, y) is held
    out as validation. The split is chronological, never shuffled, so the model
    is never validated on data that comes before what it trained on.
    Returns the Keras History; len(history.epoch) is the number of epochs run.
    """

    if val_data is None:
        n_val = max(1, int(round(len(X) * val_frac)))
        X, X_val = X[:-n_val], X[-n_val:]
        y, y_val = y[:-n_val], y[-n_val:]
        val_data = (X_val, y_val)

    return model.fit(
        X, y,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size or len(X),
        shuffle=False,
        callbacks=[early_stopping(patience)],
        verbose=verbose,
    )


def make_windows(series, lags):
    """
    Turn a 1-D series into (X, y): X[i] = series[i:i+lags], y[i] = series[i+lags].
    """

    s = np.asarray(series, dtype="float32")
    X = np.stack([s[i:i + lags] for i in range(len(s) - lags)])[..., None]
    y = s[lags:, None]
    return X, y


if __name__ == "__main__":
    # Smoke test on a noisy sine wave.
    keras.utils.set_random_seed(0)
    t = np.arange(300)
    series = np.sin(2 * np.pi * t / 12) + 0.1 * np.random.randn(len(t))
    X, y = make_windows(series, lags=12)
    for arch in ["elman", "jordan", "mrnn"]:
        for l2 in [0.0, 1e-4]:
            keras.utils.set_random_seed(0)   # same initial weights with and without L2
            m = build_model(arch, lags=12, units=8, lr=1e-2, l2=l2)
            # first 200 windows train, next 50 validate (early stopping), last 38 test
            hist = fit_model(m, X[:200], y[:200], val_data=(X[200:250], y[200:250]),
                             epochs=1000)
            stopped = len(hist.epoch)
            best = int(np.argmin(hist.history["val_loss"])) + 1
            # plain MSE, without the L2 penalty, so runs are comparable
            mse = m.evaluate(X[250:], y[250:], verbose=0, return_dict=True)["mse"]
            print(f"{arch:6s} l2={l2:<6g} params={m.count_params():4d}  epochs run={stopped:4d}  "
                  f"best epoch={best:4d}  test MSE={mse:.4f}")
