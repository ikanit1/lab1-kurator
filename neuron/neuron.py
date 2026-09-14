"""Класс одиночного искусственного нейрона (пункты 1, 3, 4 задания).

Модель:

    z = w . x + b          -- взвешенная сумма входов со смещением
    a = phi(z)             -- применение функции активации

Обучение -- полноградиентный спуск по всей выборке:

    dL/dw = X^T (dL/dz) ,   dL/db = sum(dL/dz) ,   dL/dz = dL/da * phi'(z)
    w <- w - lr * dL/dw ,   b <- b - lr * dL/db
"""

from __future__ import annotations

import numpy as np

from .activations import get_activation
from .losses import get_loss


class Neuron:
    """Один нейрон с настраиваемой активацией и функцией потерь.

    Parameters
    ----------
    n_inputs : int
        Размерность входного вектора.
    activation : str
        Имя активации: 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'linear'.
    loss : str
        Имя функции потерь: 'mse' или 'bce'.
    seed : int | None
        Зерно генератора для воспроизводимой инициализации весов.
    """

    def __init__(
        self,
        n_inputs: int,
        activation: str = "sigmoid",
        loss: str = "bce",
        seed: int | None = 0,
    ):
        self.n_inputs = int(n_inputs)
        self.activation_name = activation
        self.loss_name = loss
        self.phi, self.phi_prime = get_activation(activation)
        self.loss_fn, self.loss_grad = get_loss(loss)

        rng = np.random.default_rng(seed)
        # Инициализация Ксавье: дисперсия 1/n_inputs удерживает z в рабочей
        # зоне активации и не даёт сигмоиде сразу насытиться.
        scale = 1.0 / np.sqrt(self.n_inputs)
        self.w = rng.normal(0.0, scale, size=self.n_inputs)
        self.b = 0.0

        self._cache: dict | None = None

    # ------------------------------------------------------------------
    # Прямой проход
    # ------------------------------------------------------------------
    def forward(self, X):
        """Прямой проход. X: (N, n_inputs) -> a: (N,)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != self.n_inputs:
            raise ValueError(
                f"Ожидалось {self.n_inputs} признаков, получено {X.shape[1]}"
            )
        z = X @ self.w + self.b
        a = self.phi(z)
        self._cache = {"X": X, "z": z, "a": a}
        return a

    __call__ = forward

    # ------------------------------------------------------------------
    # Обратный проход
    # ------------------------------------------------------------------
    def _output_delta(self, a, z, y):
        """dL/dz для выходного нейрона.

        Для связки sigmoid + BCE множители a(1-a) в dL/da и phi'(z)
        взаимно сокращаются, поэтому используется устойчивая формула
        dL/dz = (a - y) / N вместо произведения двух почти нулевых чисел.
        """
        if self.activation_name == "sigmoid" and self.loss_name == "bce":
            return (a - y) / a.size
        return self.loss_grad(a, y) * self.phi_prime(z)

    def backward(self, y):
        """Аналитический градиент по (w, b). Требует предварительного forward."""
        if self._cache is None:
            raise RuntimeError("Сначала нужно вызвать forward()")
        X, z, a = self._cache["X"], self._cache["z"], self._cache["a"]
        y = np.asarray(y, dtype=np.float64).reshape(a.shape)

        dz = self._output_delta(a, z, y)          # (N,)
        grad_w = X.T @ dz                          # (n_inputs,)
        grad_b = float(np.sum(dz))
        return grad_w, grad_b

    # ------------------------------------------------------------------
    # Потери и предсказания
    # ------------------------------------------------------------------
    def loss(self, X, y):
        return self.loss_fn(self.forward(X), np.asarray(y, dtype=np.float64).ravel())

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X, threshold: float = 0.5):
        """Бинарная метка. Для tanh порог автоматически сдвигается в 0."""
        a = self.forward(X)
        thr = 0.0 if self.activation_name == "tanh" else threshold
        return (a > thr).astype(int)

    def accuracy(self, X, y):
        y = np.asarray(y).ravel()
        y_bin = (y > 0.5).astype(int) if y.min() >= 0 else (y > 0).astype(int)
        return float(np.mean(self.predict(X) == y_bin))

    # ------------------------------------------------------------------
    # Плоский вектор параметров (нужен для численной проверки градиента)
    # ------------------------------------------------------------------
    def get_params(self):
        return np.concatenate([self.w, [self.b]])

    def set_params(self, theta):
        theta = np.asarray(theta, dtype=np.float64).ravel()
        self.w = theta[:-1].copy()
        self.b = float(theta[-1])

    def analytic_grad(self, X, y):
        """Аналитический градиент в виде плоского вектора (w..., b)."""
        self.forward(X)
        gw, gb = self.backward(np.asarray(y, dtype=np.float64).ravel())
        return np.concatenate([gw, [gb]])

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------
    def fit(
        self,
        X,
        y,
        lr: float = 0.1,
        epochs: int = 1000,
        verbose_every: int | None = None,
    ):
        """Полный градиентный спуск. Возвращает историю обучения."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()

        history = {"loss": [], "accuracy": []}
        for epoch in range(epochs):
            a = self.forward(X)
            history["loss"].append(self.loss_fn(a, y))
            history["accuracy"].append(self.accuracy(X, y))

            grad_w, grad_b = self.backward(y)
            self.w -= lr * grad_w
            self.b -= lr * grad_b

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
            f"Neuron(n_inputs={self.n_inputs}, activation={self.activation_name!r}, "
            f"loss={self.loss_name!r})"
        )
