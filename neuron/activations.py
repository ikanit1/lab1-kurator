"""Функции активации и их производные.

Все функции работают поэлементно с массивами NumPy и возвращают float64.
Производные записаны в виде d/dz phi(z), т.е. как функции от самого z
(а не от значения активации), чтобы их можно было единообразно
подставлять в формулу обратного распространения ошибки:

    dL/dz = dL/da * phi'(z)
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Сигмоида:  phi(z) = 1 / (1 + exp(-z)),   phi'(z) = phi(z) * (1 - phi(z))
# --------------------------------------------------------------------------
def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    # Численно устойчивая форма: для z >= 0 и z < 0 используются разные
    # выражения, чтобы exp() никогда не переполнялся.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)


# --------------------------------------------------------------------------
# Гиперболический тангенс: phi(z) = tanh(z),  phi'(z) = 1 - tanh^2(z)
# --------------------------------------------------------------------------
def tanh(z):
    return np.tanh(np.asarray(z, dtype=np.float64))


def tanh_prime(z):
    return 1.0 - np.tanh(np.asarray(z, dtype=np.float64)) ** 2


# --------------------------------------------------------------------------
# ReLU: phi(z) = max(0, z),  phi'(z) = [z > 0]
# В точке z = 0 производная не существует; по соглашению берём 0.
# --------------------------------------------------------------------------
def relu(z):
    z = np.asarray(z, dtype=np.float64)
    return np.maximum(0.0, z)


def relu_prime(z):
    z = np.asarray(z, dtype=np.float64)
    return (z > 0).astype(np.float64)


# --------------------------------------------------------------------------
# LeakyReLU: phi(z) = z при z > 0, alpha*z иначе;  phi'(z) = 1 или alpha
# --------------------------------------------------------------------------
LEAKY_ALPHA = 0.01


def leaky_relu(z, alpha: float = LEAKY_ALPHA):
    z = np.asarray(z, dtype=np.float64)
    return np.where(z > 0, z, alpha * z)


def leaky_relu_prime(z, alpha: float = LEAKY_ALPHA):
    z = np.asarray(z, dtype=np.float64)
    return np.where(z > 0, 1.0, alpha)


# --------------------------------------------------------------------------
# Линейная (тождественная) активация — нужна для контрольных экспериментов.
# --------------------------------------------------------------------------
def linear(z):
    return np.asarray(z, dtype=np.float64)


def linear_prime(z):
    return np.ones_like(np.asarray(z, dtype=np.float64))


#: Реестр: имя -> (функция, производная). Используется классами Neuron и MLP.
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_prime),
    "tanh": (tanh, tanh_prime),
    "relu": (relu, relu_prime),
    "leaky_relu": (leaky_relu, leaky_relu_prime),
    "linear": (linear, linear_prime),
}


def get_activation(name: str):
    """Вернуть пару (phi, phi') по строковому имени активации."""
    try:
        return ACTIVATIONS[name]
    except KeyError:
        raise ValueError(
            f"Неизвестная активация {name!r}. Доступны: {sorted(ACTIVATIONS)}"
        ) from None
