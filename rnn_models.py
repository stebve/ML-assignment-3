"""
Elman, Jordan and multi-recurrent (MRNN) networks as Keras RNN cells.

All three cells take one input vector per time step and return a single
linear output y_t. The output layer sits inside each cell so that Jordan
and MRNN can feed y_t back as context.

    Elman : h_t = tanh(W x_t + U h_{t-1} + b)                    context = previous hidden state
    Jordan: h_t = tanh(W x_t + U y_{t-1} + b)                    context = previous output
    MRNN  : h_t = tanh(W x_t + Σ_k U_k Mh_k + Σ_k R_k Mo_k + b)  context = memory banks of
                                                                 hidden AND output, each bank
                                                                 decaying at its own rate α_k
            Mh_k <- α_k Mh_k + (1 - α_k) h_t
            Mo_k <- α_k Mo_k + (1 - α_k) y_t
    all   : y_t = V h_t + c

Model input shape: (batch, lags, n_features). The model predicts the value
that follows the window.
"""

import numpy as np
import keras
from keras import layers, ops


class ElmanCell(layers.Layer):
    """
    Elman network: the context layer is a copy of the previous hidden state.
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.output_size = 1

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(shape=(d, self.units), initializer="glorot_uniform", name="W")
        self.U = self.add_weight(shape=(self.units, self.units), initializer="orthogonal", name="U")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", name="b")
        self.V = self.add_weight(shape=(self.units, 1), initializer="glorot_uniform", name="V")

    def call(self, x, states):
        h_prev = states[0]
        h = ops.tanh(ops.matmul(x, self.W) + ops.matmul(h_prev, self.U) + self.b)
        y = ops.matmul(h, self.V)
        return y, [h]

    def get_config(self):
        return {**super().get_config(), "units": self.units}


class JordanCell(layers.Layer):
    """
    Jordan network: the context layer is the previous network output.
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = 1
        self.output_size = 1

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(shape=(d, self.units), initializer="glorot_uniform", name="W")
        self.U = self.add_weight(shape=(1, self.units), initializer="glorot_uniform", name="U")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", name="b")
        self.V = self.add_weight(shape=(self.units, 1), initializer="glorot_uniform", name="V")

    def call(self, x, states):
        y_prev = states[0]
        h = ops.tanh(ops.matmul(x, self.W) + ops.matmul(y_prev, self.U) + self.b)
        y = ops.matmul(h, self.V)
        return y, [y]

    def get_config(self):
        return {**super().get_config(), "units": self.units}


class MRNNCell(layers.Layer):
    """
    Multi-recurrent network: hidden-state and output feedback, each held in
    n_banks memory banks with fixed self-recurrent decay rates alphas.

    Default alphas for n_banks = k are 0, 1/k, 2/k, ..., (k-1)/k. For example,
    k = 4 gives 0, 0.25, 0.5, 0.75. A bank with alpha = 0 is an exact copy of
    the last step, so bank 0 reproduces Elman (hidden) and Jordan (output)
    feedback. Banks with larger alpha keep a slower, smoothed memory of the past.
    """

    def __init__(self, units, n_banks=4, alphas=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
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
        self.W = self.add_weight(shape=(d, u), initializer="glorot_uniform", name="W")
        self.Uh = self.add_weight(shape=(u * k, u), initializer="glorot_uniform", name="Uh")
        self.Uo = self.add_weight(shape=(k, u), initializer="glorot_uniform", name="Uo")
        self.b = self.add_weight(shape=(u,), initializer="zeros", name="b")
        self.V = self.add_weight(shape=(u, 1), initializer="glorot_uniform", name="V")
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
                "n_banks": self.n_banks, "alphas": self.alphas}


def build_model(arch, lags, units, lr=1e-2, n_banks=4):
    """
    arch: 'elman' | 'jordan' | 'mrnn'. Returns a compiled Keras model.
    """

    cells = {
        "elman": lambda: ElmanCell(units),
        "jordan": lambda: JordanCell(units),
        "mrnn": lambda: MRNNCell(units, n_banks=n_banks)
    }
    input_layer = keras.Input(shape=(lags, 1))
    output_layer = layers.RNN(cells[arch]())(input_layer)
    model = keras.Model(input_layer, output_layer)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


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
        m = build_model(arch, lags=12, units=8, lr=1e-2)
        m.fit(X[:250], y[:250], epochs=200, batch_size=len(X), verbose=0)
        mse = m.evaluate(X[250:], y[250:], verbose=0)
        print(f"{arch:6s} params={m.count_params():4d}  test MSE={mse:.4f}")
