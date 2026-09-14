"""Пункт 1. Класс Neuron и прямой проход.

Проверяем, что forward() воспроизводит формулу z = w.x + b, a = phi(z),
сравнивая результат с расчётом «на бумаге».
"""

from . import _bootstrap  # noqa: F401
import numpy as np

from neuron import Neuron
from neuron.activations import sigmoid


def main():
    print("=" * 70)
    print("ПУНКТ 1. Прямой проход одиночного нейрона")
    print("=" * 70)

    n = Neuron(n_inputs=3, activation="sigmoid", loss="bce", seed=0)
    # Зададим веса явно, чтобы результат можно было проверить вручную.
    n.w = np.array([0.5, -1.0, 2.0])
    n.b = 0.25
    print(f"\nМодель: {n}")
    print(f"  w = {n.w},  b = {n.b}")

    X = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [-1.0, 0.5, -2.0],
    ])
    a = n.forward(X)
    z = X @ n.w + n.b

    print("\n  x                     z = w.x + b      a = sigmoid(z)")
    for xi, zi, ai in zip(X, z, a):
        print(f"  {str(xi):<22}{zi:>12.6f}{ai:>18.8f}")

    # Контроль: независимый пересчёт по определению.
    expected = sigmoid(z)
    assert np.allclose(a, expected, atol=1e-15), "forward() не совпал с формулой"
    print("\n  Совпадение с прямым расчётом по формуле: max |разность| = "
          f"{np.max(np.abs(a - expected)):.3e}")

    # Прямой проход работает и для одного объекта.
    one = n.forward([1.0, 2.0, 3.0])
    print(f"  forward одного объекта [1, 2, 3] -> {one[0]:.8f}")

    print("\n  Вывод: forward выполняет взвешенное суммирование входов со")
    print("  смещением и поэлементно применяет функцию активации.")


if __name__ == "__main__":
    main()
