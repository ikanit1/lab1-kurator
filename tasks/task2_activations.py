"""Пункт 2. Функции активации, их производные и графики на [-6; 6].

Дополнительно производные проверяются численно (центральная разность),
чтобы убедиться в правильности аналитических формул.
"""

from . import _bootstrap  # noqa: F401
import numpy as np

from neuron.activations import ACTIVATIONS, get_activation
from neuron.plotting import plot_activations

NAMES = ["sigmoid", "tanh", "relu", "leaky_relu"]


def check_derivative(name, lo=-6.0, hi=6.0, n=241, eps=1e-6):
    """Сравнить phi'(z) с центральной разностью, избегая точек излома."""
    phi, phi_prime = get_activation(name)
    z = np.linspace(lo, hi, n)
    z = z[np.abs(z) > 1e-3]          # ReLU/LeakyReLU не дифференцируемы в нуле
    numeric = (phi(z + eps) - phi(z - eps)) / (2.0 * eps)
    return float(np.max(np.abs(phi_prime(z) - numeric)))


def main():
    print("=" * 70)
    print("ПУНКТ 2. Функции активации и их производные")
    print("=" * 70)

    print("\nЗначения в характерных точках:")
    grid = np.array([-6.0, -2.0, -0.5, 0.0, 0.5, 2.0, 6.0])
    header = f"{'z':>6} " + " ".join(f"{n:>12}" for n in NAMES)
    print("\n  phi(z)")
    print("  " + header)
    for zi in grid:
        row = " ".join(f"{get_activation(n)[0](np.array([zi]))[0]:>12.6f}" for n in NAMES)
        print(f"  {zi:>6.1f} {row}")

    print("\n  phi'(z)")
    print("  " + header)
    for zi in grid:
        row = " ".join(f"{get_activation(n)[1](np.array([zi]))[0]:>12.6f}" for n in NAMES)
        print(f"  {zi:>6.1f} {row}")

    print("\nЧисленная проверка производных (центральная разность, eps=1e-6):")
    for name in NAMES:
        err = check_derivative(name)
        status = "OK" if err < 1e-6 else "ПРОВЕРИТЬ"
        print(f"  {name:<12} max |phi' - разностная| = {err:.3e}   {status}")

    path = plot_activations(NAMES, filename="task2_activations.png")
    print(f"\nГрафик функций и производных на [-6; 6] сохранён: {path}")

    print("\nСвойства, важные для обучения:")
    print("  * sigmoid: область значений (0, 1), max phi' = 0.25 при z = 0;")
    print("    при |z| > 4 производная < 0.02 -- насыщение и затухание градиента.")
    print("  * tanh: область значений (-1, 1), центрирована в нуле, max phi' = 1,")
    print("    сходимость быстрее сигмоиды, но насыщение сохраняется.")
    print("  * ReLU: не насыщается при z > 0 (phi' = 1), но при z < 0 градиент")
    print("    строго нулевой -- эффект «мёртвых» нейронов.")
    print("  * LeakyReLU: тот же режим при z > 0, но phi' = 0.01 при z < 0,")
    print("    что не даёт нейрону «умереть» окончательно.")
    print(f"\nВсего активаций в реестре: {len(ACTIVATIONS)} ({', '.join(sorted(ACTIVATIONS))})")


if __name__ == "__main__":
    main()
