"""Пункт 6. Численная проверка градиента методом конечных разностей.

Требование задания: расхождение аналитического и численного градиентов
не превышает 1e-6.
"""

from . import _bootstrap  # noqa: F401
import numpy as np

from neuron import MLP, Neuron
from neuron.datasets import moons_dataset, standardize, targets_for, xor_dataset
from neuron.gradcheck import gradient_check, kink_count, randomize_params, report

TOL = 1e-6


def check(model, X, y, title, verbose=True):
    res = gradient_check(model, X, y, eps=1e-5, tol=TOL)
    if verbose:
        print()
        print(report(res, title))
    return res


def main(verbose=True):
    print("=" * 70)
    print("ПУНКТ 6. Численная проверка градиента (центральные конечные разности)")
    print("=" * 70)
    print(f"\nШаг eps = 1e-5, порог относительной ошибки {TOL:.0e}, вычисления в float64.")

    rng = np.random.default_rng(0)
    X_small = rng.normal(size=(8, 2))
    y_small = rng.integers(0, 2, size=8).astype(float)

    Xm, ym = moons_dataset(n_samples=120, noise=0.2)
    Xm = standardize(Xm)

    Xx, yx = xor_dataset()

    results = []

    # --- одиночный нейрон: все сочетания активации и потерь -----------------
    print("\n--- Одиночный нейрон ---")
    for act in ("sigmoid", "tanh", "relu", "leaky_relu"):
        for loss in ("mse", "bce"):
            if loss == "bce" and act != "sigmoid":
                continue           # BCE требует выхода из (0, 1)
            n = Neuron(2, activation=act, loss=loss, seed=5)
            y_t = targets_for(act, y_small)
            res = check(n, X_small, y_t, f"Neuron: {act} + {loss.upper()}", verbose)
            results.append((f"Neuron {act}+{loss}", res))

    # --- сеть со скрытым слоем ---------------------------------------------
    print("\n--- Сеть 2-H-1 ---")
    configs = [
        ("tanh", "sigmoid", "bce", Xx, yx, "MLP 2-2-1: tanh/sigmoid + BCE (XOR)"),
        ("tanh", "tanh", "mse", Xx, targets_for("tanh", yx), "MLP 2-2-1: tanh/tanh + MSE (XOR)"),
        ("sigmoid", "sigmoid", "mse", Xx, yx, "MLP 2-2-1: sigmoid/sigmoid + MSE (XOR)"),
        ("relu", "sigmoid", "bce", Xm, ym, "MLP 2-4-1: relu/sigmoid + BCE (make_moons)"),
        ("leaky_relu", "sigmoid", "bce", Xm, ym, "MLP 2-4-1: leaky_relu/sigmoid + BCE (moons)"),
    ]
    for hid, out, loss, Xc, yc, title in configs:
        h = 4 if Xc is Xm else 2
        net = MLP(2, h, hidden_activation=hid, output_activation=out, loss=loss, seed=5)
        res = check(net, Xc, yc, title, verbose)
        results.append((title, res))

    # --- проверка после нескольких шагов обучения ---------------------------
    print("\n--- Та же проверка для обученной сети (параметры далеки от нуля) ---")
    net = MLP(2, 2, "tanh", "sigmoid", "bce", seed=0)
    net.fit(Xx, yx, lr=0.5, epochs=2000)
    res = check(net, Xx, yx, "MLP 2-2-1 после 2000 эпох обучения на XOR", verbose)
    results.append(("MLP обученная", res))

    # --- демонстрация излома ReLU -------------------------------------------
    print("\n--- Что происходит на изломе ReLU ---")
    net_kink = MLP(2, 4, "relu", "sigmoid", "bce", seed=0)
    n_kinks = kink_count(net_kink, Xx)
    res_kink = gradient_check(net_kink, Xx, yx)
    print(f"  Сеть с ReLU на наборе XOR: смещения инициализированы нулём,")
    print(f"  а вход (0, 0) даёт z = 0 в точности -> {n_kinks} точек излома.")
    print(f"  max отн. ошибка = {res_kink['max_rel_err']:.3e} -> "
          f"{'ПРОЙДЕНО' if res_kink['passed'] else 'НЕ ПРОЙДЕНО (ожидаемо)'}")
    randomize_params(net_kink, seed=0)
    res_fixed = gradient_check(net_kink, Xx, yx)
    print(f"  После randomize_params(): точек излома {kink_count(net_kink, Xx)}, "
          f"max отн. ошибка = {res_fixed['max_rel_err']:.3e} -> "
          f"{'ПРОЙДЕНО' if res_fixed['passed'] else 'НЕ ПРОЙДЕНО'}")
    print("  Вывод: расхождение вызвано негладкостью ReLU, а не ошибкой в формулах.")

    # --- итоговая сводка ----------------------------------------------------
    print("\n" + "=" * 70)
    print("СВОДКА")
    print("=" * 70)
    print(f"{'конфигурация':<48}{'max отн. ошибка':>18}  статус")
    worst = 0.0
    for name, res in results:
        worst = max(worst, res["max_rel_err"])
        print(f"{name:<48}{res['max_rel_err']:>18.3e}  "
              f"{'ПРОЙДЕНО' if res['passed'] else 'НЕ ПРОЙДЕНО'}")
    print(f"\nНаибольшая относительная ошибка по всем проверкам: {worst:.3e}")
    print(f"Требование задания (<= {TOL:.0e}): "
          f"{'ВЫПОЛНЕНО' if worst <= TOL else 'НЕ ВЫПОЛНЕНО'}")

    print("\nЗамечание о ReLU и LeakyReLU. В точке z = 0 производная не")
    print("существует; если конечная разность «перешагивает» излом, численный")
    print("градиент отличается от аналитического на конечную величину -- см.")
    print("раздел выше. Это ограничение самого метода конечных разностей, а не")
    print("признак ошибки в выводе формул: достаточно увести параметры от излома.")

    assert worst <= TOL, "проверка градиента не пройдена"
    return results


if __name__ == "__main__":
    main(verbose=True)
