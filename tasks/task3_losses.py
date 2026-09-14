"""Пункт 3. Функции потерь MSE и BCE, вывод и программирование градиентов.

Вывод формул (для одного нейрона a = phi(z), z = w.x + b, N объектов)
--------------------------------------------------------------------
MSE:
    L   = (1/N) sum_i (a_i - y_i)^2
    dL/da_i = 2 (a_i - y_i) / N
    dL/dz_i = dL/da_i * phi'(z_i)
    dL/dw   = X^T (dL/dz)          dL/db = sum_i dL/dz_i

BCE:
    L   = -(1/N) sum_i [ y_i ln a_i + (1 - y_i) ln(1 - a_i) ]
    dL/da_i = (a_i - y_i) / (N a_i (1 - a_i))
    при a = sigmoid(z):  phi'(z) = a(1-a)  =>  dL/dz_i = (a_i - y_i) / N
"""

from . import _bootstrap  # noqa: F401
import numpy as np

from neuron import Neuron
from neuron.losses import bce, bce_grad, mse, mse_grad


def main():
    print("=" * 70)
    print("ПУНКТ 3. Функции потерь и аналитический градиент")
    print("=" * 70)

    a = np.array([0.1, 0.4, 0.8, 0.95])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    print(f"\nВыходы a = {a}")
    print(f"Метки  y = {y}")
    print(f"  MSE = {mse(a, y):.8f}")
    print(f"  BCE = {bce(a, y):.8f}")

    # --- градиент по активации: аналитика против конечных разностей ----------
    print("\nПроверка dL/da центральной разностью (eps = 1e-6):")
    eps = 1e-6
    for name, fn, grad_fn in (("MSE", mse, mse_grad), ("BCE", bce, bce_grad)):
        numeric = np.empty_like(a)
        for k in range(a.size):
            ap, am = a.copy(), a.copy()
            ap[k] += eps
            am[k] -= eps
            numeric[k] = (fn(ap, y) - fn(am, y)) / (2 * eps)
        analytic = grad_fn(a, y)
        err = np.max(np.abs(analytic - numeric))
        print(f"  {name}: аналитический {np.array2string(analytic, precision=8)}")
        print(f"       численный    {np.array2string(numeric, precision=8)}")
        print(f"       max |разность| = {err:.3e}")

    # --- сокращение множителей для связки sigmoid + BCE ---------------------
    print("\nПроверка тождества  dL/dz = (a - y)/N  для sigmoid + BCE:")
    rng = np.random.default_rng(7)
    X = rng.normal(size=(6, 2))
    y6 = rng.integers(0, 2, size=6).astype(float)

    n = Neuron(2, activation="sigmoid", loss="bce", seed=1)
    a6 = n.forward(X)
    z6 = n._cache["z"]
    grad_general = bce_grad(a6, y6) * n.phi_prime(z6)
    grad_folded = (a6 - y6) / a6.size
    print(f"  общая формула dL/da * phi'(z): {np.array2string(grad_general, precision=10)}")
    print(f"  свёрнутая (a - y)/N:           {np.array2string(grad_folded, precision=10)}")
    print(f"  max |разность| = {np.max(np.abs(grad_general - grad_folded)):.3e}")

    # --- градиенты по весам и смещению --------------------------------------
    print("\nГрадиент по параметрам нейрона (sigmoid + BCE, 6 объектов):")
    gw, gb = n.backward(y6)
    print(f"  dL/dw = {np.array2string(gw, precision=10)}")
    print(f"  dL/db = {gb:.10f}")

    print("\nГрадиент по параметрам нейрона (tanh + MSE, те же данные):")
    n2 = Neuron(2, activation="tanh", loss="mse", seed=1)
    n2.forward(X)
    gw2, gb2 = n2.backward(2 * y6 - 1)      # для tanh метки в {-1, +1}
    print(f"  dL/dw = {np.array2string(gw2, precision=10)}")
    print(f"  dL/db = {gb2:.10f}")

    print("\nЗамечание. BCE определена только для a из (0, 1), поэтому она")
    print("применяется с сигмоидным выходом; MSE работает с любой активацией,")
    print("но в паре с сигмоидой даёт слабый градиент в зоне насыщения")
    print("(множитель phi'(z) -> 0), из-за чего обучение замедляется.")


if __name__ == "__main__":
    main()
