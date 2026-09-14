"""Пункт 4. Цикл обучения (градиентный спуск) одиночного нейрона
на линейно разделимой задаче (make_blobs).
"""

from . import _bootstrap  # noqa: F401
import numpy as np

from neuron import Neuron
from neuron.datasets import blobs_dataset, or_dataset, standardize
from neuron.plotting import plot_dataset, plot_decision_boundary, plot_loss_curves


def main(lr: float = 0.5, epochs: int = 400):
    print("=" * 70)
    print("ПУНКТ 4. Обучение одиночного нейрона на линейно разделимой задаче")
    print("=" * 70)

    X, y = blobs_dataset(n_samples=300, cluster_std=1.0)
    X = standardize(X)
    print(f"\nНабор make_blobs: X.shape = {X.shape}, классов = {len(np.unique(y))}")
    print(f"Гиперпараметры: lr = {lr}, эпох = {epochs}, активация sigmoid, потери BCE")

    plot_dataset(X, y, "task4_blobs.png", "make_blobs: линейно разделимые классы")

    n = Neuron(n_inputs=2, activation="sigmoid", loss="bce", seed=0)
    print(f"\nНачальные параметры: w = {np.round(n.w, 6)}, b = {n.b}")
    print(f"Потери до обучения: {n.loss(X, y):.6f}, точность: {n.accuracy(X, y):.4f}\n")

    history = n.fit(X, y, lr=lr, epochs=epochs, verbose_every=50)

    print(f"\nИтоговые параметры: w = {np.round(n.w, 6)}, b = {n.b:.6f}")
    print(f"Потери после обучения: {history['loss'][-1]:.8f}")
    print(f"Точность:              {history['accuracy'][-1]:.4f}")

    # Уравнение разделяющей прямой: w1*x1 + w2*x2 + b = 0
    w1, w2 = n.w
    print(f"\nРазделяющая прямая: {w1:.4f}*x1 + {w2:.4f}*x2 + {n.b:.4f} = 0")
    if abs(w2) > 1e-12:
        print(f"  или x2 = {-w1 / w2:.4f}*x1 + {-n.b / w2:.4f}")

    p1 = plot_loss_curves(
        {f"lr = {lr}": history["loss"]},
        filename="task4_loss.png",
        title="Сходимость одиночного нейрона (make_blobs, BCE)",
    )
    p2 = plot_decision_boundary(
        n, X, y, "task4_boundary.png",
        title="Одиночный нейрон: линейная разделяющая поверхность",
    )
    print(f"\nГрафики: {p1}\n         {p2}")

    # --- контрольный пример: логическая функция OR --------------------------
    print("\nКонтрольный пример -- логическая функция OR (линейно разделима):")
    Xo, yo = or_dataset()
    no = Neuron(2, activation="sigmoid", loss="bce", seed=0)
    ho = no.fit(Xo, yo, lr=1.0, epochs=3000)
    print(f"  потери {ho['loss'][-1]:.6f}, точность {ho['accuracy'][-1]:.2f}")
    for xi, yi, ai in zip(Xo, yo, no.forward(Xo)):
        print(f"    x = {xi}, y = {yi:.0f}, a = {ai:.4f} -> {int(ai > 0.5)}")

    print("\nВывод: для линейно разделимых данных одного нейрона достаточно --")
    print("потери монотонно убывают, достигается 100% точность.")
    return history


if __name__ == "__main__":
    main()
