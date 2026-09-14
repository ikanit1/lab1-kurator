"""Выполнение индивидуального варианта задания.

Вариант определяется номером обучающегося в журнале (см. neuron/variants.py).

    python -m tasks.run_variant --number 7
    python -m tasks.run_variant --table
"""

from . import _bootstrap  # noqa: F401
import argparse

import numpy as np

from neuron import MLP, Neuron
from neuron.datasets import get_dataset, standardize, targets_for
from neuron.gradcheck import gradient_check, kink_count, randomize_params
from neuron.plotting import plot_activations, plot_decision_boundary, plot_loss_curves
from neuron.variants import get_variant, variant_table

LEARNING_RATES = [0.001, 0.01, 0.1, 1.0]


def run(number: int, epochs: int = 4000, hidden: int = 8, restarts: int = 5):
    v = get_variant(number)
    print("=" * 70)
    print(v.describe())
    print("=" * 70)

    # --- данные -------------------------------------------------------------
    X, y = get_dataset(v.task)
    if v.task in ("moons", "circles"):
        X = standardize(X)
    y_t = targets_for(v.output_activation, y)
    print(f"\nДанные: {v.task}, X.shape = {X.shape}, "
          f"метки в диапазоне [{y_t.min():.0f}; {y_t.max():.0f}]")

    # --- пункт 2: графики активаций варианта --------------------------------
    path = plot_activations(list(v.activations), filename=f"variant{number}_activations.png")
    print(f"Графики активаций варианта: {path}")

    # --- пункт 4: одиночный нейрон ------------------------------------------
    print("\n[A] Одиночный нейрон")
    single = Neuron(2, activation=v.output_activation, loss=v.loss, seed=0)
    h_single = single.fit(X, y_t, lr=0.5, epochs=epochs)
    print(f"  потери {h_single['loss'][-1]:.6f}, точность {h_single['accuracy'][-1]:.4f}")

    # --- пункт 5: сеть со скрытым слоем -------------------------------------
    print(f"\n[B] Сеть 2-{hidden}-1 ({v.hidden_activation} -> {v.output_activation}, "
          f"{v.loss.upper()}), {restarts} перезапусков")
    best, best_hist = None, None
    for seed in range(restarts):
        net = MLP(2, hidden, v.hidden_activation, v.output_activation, v.loss, seed=seed)
        h = net.fit(X, y_t, lr=0.5, epochs=epochs)
        if best_hist is None or h["loss"][-1] < best_hist["loss"][-1]:
            best, best_hist = net, h
    print(f"  потери {best_hist['loss'][-1]:.6f}, точность {best_hist['accuracy'][-1]:.4f}")

    # --- пункт 6: проверка градиента ----------------------------------------
    check_net = MLP(2, hidden, v.hidden_activation, v.output_activation, v.loss, seed=0)
    X_check, y_check = X[:32], y_t[:32]
    if kink_count(check_net, X_check) > 0:
        # ReLU-подобные активации не дифференцируемы в нуле; нулевые смещения
        # вместе с нулевым входным вектором дают z = 0 в точности. Перед
        # проверкой уводим параметры от излома случайным сдвигом.
        print(f"\n[C] Обнаружены предактивации на изломе ReLU "
              f"({kink_count(check_net, X_check)} шт.) -- параметры рандомизированы")
        randomize_params(check_net, seed=0)
    res = gradient_check(check_net, X_check, y_check)
    print(f"\n[C] Проверка градиента: max отн. ошибка = {res['max_rel_err']:.3e} "
          f"({'ПРОЙДЕНО' if res['passed'] else 'НЕ ПРОЙДЕНО'})")

    # --- пункт 7: скорость обучения -----------------------------------------
    print("\n[D] Влияние скорости обучения")
    curves = {}
    for lr in LEARNING_RATES:
        net = MLP(2, hidden, v.hidden_activation, v.output_activation, v.loss, seed=0)
        h = net.fit(X, y_t, lr=lr, epochs=epochs)
        curves[f"lr = {lr}"] = h["loss"]
        print(f"  lr = {lr:<6} итоговые потери {h['loss'][-1]:.6f}, "
              f"точность {h['accuracy'][-1]:.4f}")

    p_loss = plot_loss_curves(
        curves, filename=f"variant{number}_lr.png",
        title=f"Вариант {number}: кривые потерь ({v.task}, {v.loss.upper()})",
    )
    thr = 0.0 if v.output_activation == "tanh" else 0.5
    p_bnd = plot_decision_boundary(
        best, X, y_t, f"variant{number}_boundary.png",
        title=f"Вариант {number}: разделяющая поверхность сети 2-{hidden}-1",
        threshold=thr,
    )
    print(f"\nГрафики: {p_loss}\n         {p_bnd}")
    return v, h_single, best_hist


def main():
    parser = argparse.ArgumentParser(description="Индивидуальный вариант лабораторной работы № 1")
    parser.add_argument("--number", "-n", type=int, default=1, help="номер в журнале")
    parser.add_argument("--epochs", "-e", type=int, default=4000)
    parser.add_argument("--hidden", type=int, default=8, help="нейронов в скрытом слое")
    parser.add_argument("--table", action="store_true", help="показать таблицу вариантов")
    args = parser.parse_args()

    if args.table:
        print(variant_table(24))
        return
    run(args.number, epochs=args.epochs, hidden=args.hidden)


if __name__ == "__main__":
    main()
