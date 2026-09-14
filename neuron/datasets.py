"""Синтетические наборы данных для лабораторной работы.

Все генераторы возвращают пару (X, y), где X имеет форму (N, 2),
а y -- вектор меток {0, 1} длины N.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

RANDOM_STATE = 42


def xor_dataset():
    """Логическая функция XOR: 4 точки, линейно неразделимые."""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    return X, y


def and_dataset():
    """Логическая функция AND: линейно разделима."""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 0.0, 0.0, 1.0])
    return X, y


def or_dataset():
    """Логическая функция OR: линейно разделима."""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 1.0])
    return X, y


def and_or_dataset():
    """Композиция AND-OR: y = (x1 AND x2) OR (NOT x1 AND NOT x2), т.е. XNOR.

    Классический пример нелинейно разделимой задачи, получаемой
    композицией пороговых функций.
    """
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    return X, y


def moons_dataset(n_samples: int = 500, noise: float = 0.2):
    """sklearn.datasets.make_moons -- две переплетённые дуги."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=RANDOM_STATE)
    return X.astype(np.float64), y.astype(np.float64)


def circles_dataset(n_samples: int = 500, noise: float = 0.1, factor: float = 0.5):
    """sklearn.datasets.make_circles -- вложенные окружности."""
    X, y = make_circles(
        n_samples=n_samples, noise=noise, factor=factor, random_state=RANDOM_STATE
    )
    return X.astype(np.float64), y.astype(np.float64)


def blobs_dataset(n_samples: int = 300, cluster_std: float = 1.0):
    """make_blobs -- два хорошо разделимых облака (линейно разделимый случай)."""
    X, y = make_blobs(
        n_samples=n_samples,
        centers=[(-2.5, -2.5), (2.5, 2.5)],
        cluster_std=cluster_std,
        random_state=RANDOM_STATE,
    )
    return X.astype(np.float64), y.astype(np.float64)


#: Реестр наборов данных для выбора по имени варианта.
DATASETS = {
    "xor": xor_dataset,
    "and": and_dataset,
    "or": or_dataset,
    "and_or": and_or_dataset,
    "moons": moons_dataset,
    "circles": circles_dataset,
    "blobs": blobs_dataset,
}


def get_dataset(name: str, **kwargs):
    try:
        return DATASETS[name](**kwargs)
    except KeyError:
        raise ValueError(
            f"Неизвестный набор данных {name!r}. Доступны: {sorted(DATASETS)}"
        ) from None


def standardize(X):
    """Z-нормировка признаков: нулевое среднее, единичная дисперсия."""
    X = np.asarray(X, dtype=np.float64)
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)


def to_bipolar(y):
    """Перевод меток {0, 1} -> {-1, +1} (нужно для выхода с tanh и MSE)."""
    return 2.0 * np.asarray(y, dtype=np.float64) - 1.0


def targets_for(output_activation: str, y):
    """Подогнать диапазон меток под область значений выходной активации."""
    y = np.asarray(y, dtype=np.float64).ravel()
    return to_bipolar(y) if output_activation == "tanh" else y
