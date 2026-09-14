"""Двухслойная сеть D -> H -> 1 с обратным распространением ошибки вручную
(пункт 5 задания).

Прямой проход
-------------
    Z1 = X W1 + b1        (N, H)
    A1 = phi1(Z1)         (N, H)
    z2 = A1 w2 + b2       (N,)
    a2 = phi2(z2)         (N,)

Обратный проход (вывод формул)
------------------------------
    dL/dz2 = dL/da2 * phi2'(z2)                       (N,)
    dL/dw2 = A1^T (dL/dz2)                            (H,)
    dL/db2 = sum_i dL/dz2_i                           скаляр
    dL/dA1 = (dL/dz2) (x) w2      -- внешнее произведение, (N, H)
    dL/dZ1 = dL/dA1 * phi1'(Z1)                       (N, H)
    dL/dW1 = X^T (dL/dZ1)                             (D, H)
    dL/db1 = сумма dL/dZ1 по объектам                 (H,)
"""

from __future__ import annotations

import numpy as np

from .activations import get_activation
from .losses import get_loss


class MLP:
    """Персептрон с одним скрытым слоем и одним выходом.

    Parameters
    ----------
    n_inputs : int
        Размерность входа D.
    n_hidden : int
        Число нейронов скрытого слоя H (по заданию для XOR достаточно 2).
    hidden_activation, output_activation : str
        Имена активаций скрытого и выходного слоёв.
    loss : str
        'mse' или 'bce'.
    seed : int | None
        Зерно генератора весов.
    """

    def __init__(
        self,
        n_inputs: int = 2,
        n_hidden: int = 2,
        hidden_activation: str = "tanh",
        output_activation: str = "sigmoid",
        loss: str = "bce",
        seed: int | None = 0,
    ):
        self.n_inputs = int(n_inputs)
        self.n_hidden = int(n_hidden)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss_name = loss

        self.phi1, self.phi1_prime = get_activation(hidden_activation)
        self.phi2, self.phi2_prime = get_activation(output_activation)
        self.loss_fn, self.loss_grad = get_loss(loss)

        rng = np.random.default_rng(seed)
        # Инициализация Ксавье: std = sqrt(1 / fan_in) для каждого слоя.
        self.W1 = rng.normal(0.0, 1.0 / np.sqrt(self.n_inputs), (self.n_inputs, self.n_hidden))
        self.b1 = np.zeros(self.n_hidden)
        self.w2 = rng.normal(0.0, 1.0 / np.sqrt(self.n_hidden), self.n_hidden)
        self.b2 = 0.0

        self._cache: dict | None = None

    # ------------------------------------------------------------------
    # Прямой проход
    # ------------------------------------------------------------------
    def forward(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Z1 = X @ self.W1 + self.b1
        A1 = self.phi1(Z1)
        z2 = A1 @ self.w2 + self.b2
        a2 = self.phi2(z2)
        self._cache = {"X": X, "Z1": Z1, "A1": A1, "z2": z2, "a2": a2}
        return a2

    __call__ = forward

    # ------------------------------------------------------------------
    # Обратный проход
    # ------------------------------------------------------------------
    def backward(self, y):
        """Градиенты по всем параметрам. Требует предварительного forward."""
        if self._cache is None:
            raise RuntimeError("Сначала нужно вызвать forward()")
        c = self._cache
        X, Z1, A1, z2, a2 = c["X"], c["Z1"], c["A1"], c["z2"], c["a2"]
        y = np.asarray(y, dtype=np.float64).reshape(a2.shape)

        # --- выходной слой -------------------------------------------------
        if self.output_activation == "sigmoid" and self.loss_name == "bce":
            dz2 = (a2 - y) / a2.size          # устойчивая свёрнутая форма
        else:
            dz2 = self.loss_grad(a2, y) * self.phi2_prime(z2)

        grad_w2 = A1.T @ dz2                   # (H,)
        grad_b2 = float(np.sum(dz2))

        # --- скрытый слой --------------------------------------------------
        dA1 = np.outer(dz2, self.w2)           # (N, H)
        dZ1 = dA1 * self.phi1_prime(Z1)        # (N, H)
        grad_W1 = X.T @ dZ1                    # (D, H)
        grad_b1 = dZ1.sum(axis=0)              # (H,)

        return grad_W1, grad_b1, grad_w2, grad_b2

    # ------------------------------------------------------------------
    # Потери и предсказания
    # ------------------------------------------------------------------
    def loss(self, X, y):
        return self.loss_fn(self.forward(X), np.asarray(y, dtype=np.float64).ravel())

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X, threshold: float = 0.5):
        a = self.forward(X)
        thr = 0.0 if self.output_activation == "tanh" else threshold
        return (a > thr).astype(int)

    def accuracy(self, X, y):
        y = np.asarray(y).ravel()
        y_bin = (y > 0.5).astype(int) if y.min() >= 0 else (y > 0).astype(int)
        return float(np.mean(self.predict(X) == y_bin))

    # ------------------------------------------------------------------
    # Плоский вектор параметров (для численной проверки градиента)
    # ------------------------------------------------------------------
    def get_params(self):
        return np.concatenate([self.W1.ravel(), self.b1, self.w2, [self.b2]])

    def set_params(self, theta):
        theta = np.asarray(theta, dtype=np.float64).ravel()
        d, h = self.n_inputs, self.n_hidden
        i = 0
        self.W1 = theta[i:i + d * h].reshape(d, h).copy(); i += d * h
        self.b1 = theta[i:i + h].copy(); i += h
        self.w2 = theta[i:i + h].copy(); i += h
        self.b2 = float(theta[i])

    def analytic_grad(self, X, y):
        self.forward(X)
        gW1, gb1, gw2, gb2 = self.backward(np.asarray(y, dtype=np.float64).ravel())
        return np.concatenate([gW1.ravel(), gb1, gw2, [gb2]])

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------
    def fit(
        self,
        X,
        y,
        lr: float = 0.5,
        epochs: int = 5000,
        verbose_every: int | None = None,
    ):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()

        history = {"loss": [], "accuracy": []}
        for epoch in range(epochs):
            a = self.forward(X)
            history["loss"].append(self.loss_fn(a, y))
            history["accuracy"].append(self.accuracy(X, y))

            gW1, gb1, gw2, gb2 = self.backward(y)
            self.W1 -= lr * gW1
            self.b1 -= lr * gb1
            self.w2 -= lr * gw2
            self.b2 -= lr * gb2

            if verbose_every and (epoch % verbose_every == 0 or epoch == epochs - 1):
                print(
                    f"  эпоха {epoch:5d} | loss = {history['loss'][-1]:.6f} "
                    f"| acc = {history['accuracy'][-1]:.3f}"
                )

        history["loss"].append(self.loss(X, y))
        history["accuracy"].append(self.accuracy(X, y))
        return history

    def __repr__(self):
        return (
            f"MLP({self.n_inputs}->{self.n_hidden}->1, "
            f"hidden={self.hidden_activation!r}, out={self.output_activation!r}, "
            f"loss={self.loss_name!r})"
        )
