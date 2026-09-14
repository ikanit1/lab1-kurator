"""Служебные функции построения графиков (Matplotlib)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")          # сохранение в файл без графической оболочки
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")


def _save(fig, filename: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_activations(names, filename="activations.png", lo=-6.0, hi=6.0, n=600):
    """Графики функций активации и их производных на отрезке [lo; hi]."""
    from .activations import get_activation

    z = np.linspace(lo, hi, n)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ReLU и LeakyReLU почти совпадают при z > 0, поэтому линии чередуются
    # по стилю и толщине -- иначе одна кривая полностью скрывает другую.
    styles = ["-", "--", "-.", ":"]
    for i, name in enumerate(names):
        phi, phi_prime = get_activation(name)
        style = styles[i % len(styles)]
        lw = 2.4 - 0.4 * (i % len(styles))
        axes[0].plot(z, phi(z), style, label=name, linewidth=lw)
        axes[1].plot(z, phi_prime(z), style, label=f"{name}'", linewidth=lw)

    axes[0].set_title("Функции активации")
    axes[1].set_title("Производные функций активации")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("z")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("phi(z)")
    axes[1].set_ylabel("phi'(z)")

    return _save(fig, filename)


def plot_loss_curves(histories, filename="loss_curves.png", title="Кривые потерь",
                     logy=True, xlabel="эпоха", ylabel="значение функции потерь"):
    """Сводный график нескольких кривых потерь. histories: {подпись: [значения]}."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, losses in histories.items():
        ax.plot(losses, label=label, linewidth=1.8)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, filename)


def plot_decision_boundary(model, X, y, filename, title="Разделяющая поверхность",
                           threshold=0.5, steps=300):
    """Карта выхода модели и разделяющая линия на плоскости признаков."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()

    pad = 0.5 * (X.max(axis=0) - X.min(axis=0)) + 0.5
    x_min, y_min = X.min(axis=0) - pad
    x_max, y_max = X.max(axis=0) + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps)
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = model.forward(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    mesh = ax.contourf(xx, yy, zz, levels=40, cmap="RdBu_r", alpha=0.7)
    ax.contour(xx, yy, zz, levels=[threshold], colors="black", linewidths=2)
    fig.colorbar(mesh, ax=ax, label="выход нейрона")

    y_bin = (y > threshold).astype(int) if y.min() >= 0 else (y > 0).astype(int)
    for cls, marker, color in ((0, "o", "#1f4e99"), (1, "^", "#b2182b")):
        m = y_bin == cls
        ax.scatter(
            X[m, 0], X[m, 1], marker=marker, c=color, edgecolors="white",
            s=55, linewidths=0.8, label=f"класс {cls}", zorder=3,
        )
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="best")
    return _save(fig, filename)


def plot_dataset(X, y, filename, title="Набор данных"):
    """Диаграмма рассеяния исходных данных."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, marker, color in ((0, "o", "#1f4e99"), (1, "^", "#b2182b")):
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], marker=marker, c=color, s=45,
                   edgecolors="white", linewidths=0.7, label=f"класс {cls}")
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, filename)
