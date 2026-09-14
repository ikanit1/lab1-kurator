"""Функции потерь и их градиенты по выходу сети.

Каждая функция потерь представлена парой:

    value(a, y) -> float           -- скалярное значение потерь (усреднённое по выборке)
    grad(a, y)  -> ndarray         -- dL/da, той же формы, что и a

Здесь a -- выход (активация) последнего слоя, y -- целевые значения.
Усреднение по N объектам входит в определение L, поэтому множитель 1/N
уже присутствует в градиенте: это позволяет не зависеть от размера батча
при подборе скорости обучения.

Вывод формул
------------
1. Среднеквадратичная ошибка (MSE)

       L = (1/N) * sum_i (a_i - y_i)^2
       dL/da_i = 2 (a_i - y_i) / N

2. Бинарная кросс-энтропия (BCE)

       L = -(1/N) * sum_i [ y_i * ln(a_i) + (1 - y_i) * ln(1 - a_i) ]
       dL/da_i = (a_i - y_i) / ( N * a_i * (1 - a_i) )

   Для сигмоидного выхода a = sigma(z) справедливо phi'(z) = a(1-a),
   поэтому множители сокращаются и

       dL/dz_i = (a_i - y_i) / N

   -- это и есть «красивая» формула, которую используют на практике
   (см. Neuron._output_delta и MLP.backward).
"""

from __future__ import annotations

import numpy as np

#: Отступ от границ отрезка [0, 1] при вычислении логарифмов в BCE.
EPS = 1e-12


# --------------------------------------------------------------------------
# MSE
# --------------------------------------------------------------------------
def mse(a, y):
    a = np.asarray(a, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((a - y) ** 2))


def mse_grad(a, y):
    a = np.asarray(a, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return 2.0 * (a - y) / a.size


# --------------------------------------------------------------------------
# Binary Cross-Entropy
# --------------------------------------------------------------------------
def bce(a, y):
    a = np.clip(np.asarray(a, dtype=np.float64), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(a) + (1.0 - y) * np.log(1.0 - a)))


def bce_grad(a, y):
    a = np.clip(np.asarray(a, dtype=np.float64), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return (a - y) / (a * (1.0 - a) * a.size)


#: Реестр: имя -> (значение, градиент по активации).
LOSSES = {
    "mse": (mse, mse_grad),
    "bce": (bce, bce_grad),
}


def get_loss(name: str):
    """Вернуть пару (L, dL/da) по строковому имени функции потерь."""
    try:
        return LOSSES[name]
    except KeyError:
        raise ValueError(
            f"Неизвестная функция потерь {name!r}. Доступны: {sorted(LOSSES)}"
        ) from None
