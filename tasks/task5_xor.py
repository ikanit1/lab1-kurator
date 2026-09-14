"""Пункт 5. XOR: одиночный нейрон не справляется, сеть 2-2-1 справляется.

Обратное распространение ошибки в классе MLP реализовано вручную
(см. neuron/mlp.py, метод backward).

Аналитическое обоснование невозможности решить XOR одним нейроном
----------------------------------------------------------------
Одиночный нейрон задаёт предикат sign(w1 x1 + w2 x2 + b). Для XOR нужно:

    (0,0) -> 0 :  b          < 0
    (0,1) -> 1 :  w2 + b     > 0
    (1,0) -> 1 :  w1 + b     > 0
    (1,1) -> 0 :  w1 + w2 + b< 0

Сложив второе и третье неравенства: w1 + w2 + 2b > 0, откуда
w1 + w2 + b > -b > 0, что противоречит четвёртому неравенству.
Система несовместна -> линейного разделения не существует.
"""

from . import _bootstrap  # noqa: F401
import numpy as np

from neuron import MLP, Neuron
from neuron.datasets import xor_dataset
from neuron.plotting import plot_decision_boundary, plot_loss_curves


def restarts(build, X, y, n_restarts=10, lr=1.0, epochs=5000):
    """Обучить модель из n_restarts случайных начальных приближений.

    Возвращает лучшую по значению потерь модель, её историю и списки
    итоговых потерь и точностей всех запусков.
    """
    best = None
    losses, accs = [], []
    for seed in range(n_restarts):
        model = build(seed)
        h = model.fit(X, y, lr=lr, epochs=epochs)
        losses.append(h["loss"][-1])
        accs.append(h["accuracy"][-1])
        if best is None or h["loss"][-1] < best[1]["loss"][-1]:
            best = (model, h)
    return best[0], best[1], losses, accs


def main(lr=0.5, epochs=5000, hidden=2):
    print("=" * 70)
    print("ПУНКТ 5. Задача XOR: один нейрон против сети 2-2-1")
    print("=" * 70)

    X, y = xor_dataset()
    print("\nНабор XOR:")
    for xi, yi in zip(X, y):
        print(f"  x = {xi}, y = {yi:.0f}")

    # --- 5.1 одиночный нейрон ----------------------------------------------
    print("\n[5.1] Одиночный нейрон (sigmoid + BCE), 10 перезапусков:")
    n, h_single, l_single, accs = restarts(
        lambda s: Neuron(2, activation="sigmoid", loss="bce", seed=s),
        X, y, n_restarts=10, lr=1.0, epochs=5000,
    )
    print(f"  точность по перезапускам: {accs}")
    print(f"  лучшая достигнутая точность: {max(accs):.2f}")
    print(f"  потери лучшего запуска:      {h_single['loss'][-1]:.6f} (= ln 2 = {np.log(2):.6f})")
    print("  выходы лучшего нейрона:")
    for xi, yi, ai in zip(X, y, n.forward(X)):
        print(f"    x = {xi}, y = {yi:.0f}, a = {ai:.4f} -> {int(ai > 0.5)}")
    assert max(accs) < 1.0, "одиночный нейрон не должен решать XOR"
    print("  => 100% точность недостижима: XOR линейно неразделим")
    print("     (доказательство несовместности системы -- в docstring модуля).")
    print("  Минимум BCE достигается при w -> 0, b -> 0: нейрон выдаёт 0.5 на всех")
    print("  четырёх точках, потери равны ln 2. Разделяющая прямая, дающая 75%")
    print("  верных ответов, существует, но она не является минимумом потерь.")

    plot_decision_boundary(
        n, X, y, "task5_xor_single.png",
        title="Одиночный нейрон на XOR: прямая не разделяет классы",
    )

    # --- 5.2 сеть со скрытым слоем -----------------------------------------
    print(f"\n[5.2] Сеть 2-{hidden}-1 (tanh скрытый + sigmoid выход, BCE),")
    print(f"      обратное распространение вручную, lr = {lr}, эпох = {epochs},")
    print("      10 перезапусков из разных начальных приближений:")
    net, h_mlp, l_mlp, accs_mlp = restarts(
        lambda s: MLP(2, hidden, hidden_activation="tanh",
                      output_activation="sigmoid", loss="bce", seed=s),
        X, y, n_restarts=10, lr=lr, epochs=epochs,
    )
    solved = sum(a == 1.0 for a in accs_mlp)
    print(f"  точность по перезапускам: {accs_mlp}")
    print(f"  XOR решён в {solved} запусках из {len(accs_mlp)}")
    print("  Неудачные запуски застревают в локальном минимуме BCE ~ 0.347:")
    print("  две противоположные вершины квадрата классифицируются уверенно,")
    print("  а две оставшиеся сеть оставляет ровно на границе (a ~ 0.5), то есть")
    print("  фактически отказывается их различать. Это известное свойство")
    print("  минимальной сети 2-2-1: ландшафт потерь невыпуклый и содержит плато,")
    print("  из которого градиентный спуск без импульса не выходит.")

    print(f"\n  итоговые потери:   {h_mlp['loss'][-1]:.8f}")
    print(f"  итоговая точность: {h_mlp['accuracy'][-1]:.2f}")
    print("  выходы сети:")
    for xi, yi, ai in zip(X, y, net.forward(X)):
        print(f"    x = {xi}, y = {yi:.0f}, a = {ai:.6f} -> {int(ai > 0.5)}")
    assert h_mlp["accuracy"][-1] == 1.0, "сеть 2-2-1 обязана решить XOR"

    print("\n  Обученные параметры:")
    print(f"    W1 =\n{np.round(net.W1, 4)}")
    print(f"    b1 = {np.round(net.b1, 4)}")
    print(f"    w2 = {np.round(net.w2, 4)}")
    print(f"    b2 = {net.b2:.4f}")
    print("\n  Активации скрытого слоя (каждый нейрон -- своя полуплоскость):")
    net.forward(X)
    A1 = net._cache["A1"]
    for xi, a1 in zip(X, A1):
        print(f"    x = {xi} -> h = {np.round(a1, 4)}")

    p1 = plot_loss_curves(
        {"одиночный нейрон": h_single["loss"], "сеть 2-2-1": h_mlp["loss"]},
        filename="task5_xor_loss.png",
        title="XOR: сходимость одиночного нейрона и сети со скрытым слоем",
    )
    p2 = plot_decision_boundary(
        net, X, y, "task5_xor_mlp.png",
        title="Сеть 2-2-1 на XOR: нелинейная разделяющая поверхность",
    )
    print(f"\nГрафики: {p1}\n         {p2}")

    # --- 5.3 влияние ширины скрытого слоя на устойчивость -------------------
    print("\n[5.3] Надёжность обучения в зависимости от ширины скрытого слоя:")
    for h_size in (2, 3, 4, 8):
        _, _, _, a_list = restarts(
            lambda s, hs=h_size: MLP(2, hs, hidden_activation="tanh",
                                     output_activation="sigmoid", loss="bce", seed=s),
            X, y, n_restarts=10, lr=lr, epochs=epochs,
        )
        ok = sum(a == 1.0 for a in a_list)
        print(f"  H = {h_size}: XOR решён в {ok} запусках из 10")
    print("  Расширение скрытого слоя убирает плато и делает сходимость устойчивой.")

    print("\nВывод: скрытый слой переводит входы в новое представление, в котором")
    print("классы становятся линейно разделимыми; выходной нейрон проводит в нём")
    print("обычную разделяющую гиперплоскость.")
    return h_single, h_mlp


if __name__ == "__main__":
    main()
