"""Пункт 7. Влияние скорости обучения на сходимость.

Значения по заданию: 0.001, 0.01, 0.1, 1.0. Дополнительно показан режим
расходимости при заведомо завышенном шаге.
"""

from . import _bootstrap  # noqa: F401
import numpy as np

from neuron import MLP, Neuron
from neuron.datasets import blobs_dataset, moons_dataset, standardize
from neuron.plotting import plot_loss_curves

LEARNING_RATES = [0.001, 0.01, 0.1, 1.0]


def summarize(histories, epochs):
    """Таблица: итоговые потери, точность и эпоха выхода на порог."""
    print(f"\n  {'lr':>8} {'L(0)':>12} {'L(конец)':>14} {'точность':>10} "
          f"{'эпоха L<0.1':>13} {'убывание':>10}")
    for lr, h in histories.items():
        losses = np.asarray(h["loss"])
        below = np.argmax(losses < 0.1) if (losses < 0.1).any() else -1
        monotone = "да" if np.all(np.diff(losses) <= 1e-12) else "нет"
        reached = str(below) if below >= 0 else "не достигнут"
        print(f"  {lr:>8} {losses[0]:>12.6f} {losses[-1]:>14.6f} "
              f"{h['accuracy'][-1]:>10.4f} {reached:>13} {monotone:>10}")


def main(epochs: int = 500):
    print("=" * 70)
    print("ПУНКТ 7. Влияние скорости обучения на сходимость")
    print("=" * 70)

    # --- 7.1 одиночный нейрон на линейно разделимой задаче -------------------
    X, y = blobs_dataset(n_samples=300)
    X = standardize(X)
    print(f"\n[7.1] Одиночный нейрон, make_blobs, sigmoid + BCE, {epochs} эпох")

    hist_neuron = {}
    for lr in LEARNING_RATES:
        n = Neuron(2, activation="sigmoid", loss="bce", seed=0)
        hist_neuron[lr] = n.fit(X, y, lr=lr, epochs=epochs)
    summarize(hist_neuron, epochs)

    p1 = plot_loss_curves(
        {f"lr = {lr}": h["loss"] for lr, h in hist_neuron.items()},
        filename="task7_lr_neuron.png",
        title="Одиночный нейрон: кривые потерь при разной скорости обучения",
    )

    # --- 7.2 сеть со скрытым слоем на make_moons ----------------------------
    Xm, ym = moons_dataset(n_samples=500, noise=0.2)
    Xm = standardize(Xm)
    print(f"\n[7.2] Сеть 2-8-1 (tanh + sigmoid, BCE), make_moons, {epochs * 4} эпох")

    hist_mlp = {}
    for lr in LEARNING_RATES:
        net = MLP(2, 8, "tanh", "sigmoid", "bce", seed=0)
        hist_mlp[lr] = net.fit(Xm, ym, lr=lr, epochs=epochs * 4)
    summarize(hist_mlp, epochs * 4)

    p2 = plot_loss_curves(
        {f"lr = {lr}": h["loss"] for lr, h in hist_mlp.items()},
        filename="task7_lr_mlp.png",
        title="Сеть 2-8-1 на make_moons: кривые потерь при разной скорости обучения",
    )

    # --- 7.3 режим расходимости ---------------------------------------------
    # Для линейной активации и MSE функция потерь квадратична:
    #     L(theta) = (1/N) ||X~ theta - y||^2 ,   H = (2/N) X~^T X~ .
    # Градиентный спуск сходится тогда и только тогда, когда lr < 2 / lambda_max(H);
    # это даёт точную границу, которую можно проверить численно.
    print("\n[7.3] Граница устойчивости шага (линейная активация + MSE):")
    X_aug = np.column_stack([X, np.ones(len(X))])
    hessian = 2.0 / len(X) * X_aug.T @ X_aug
    lam_max = float(np.linalg.eigvalsh(hessian).max())
    lr_crit = 2.0 / lam_max
    print(f"  lambda_max(H) = {lam_max:.6f}  =>  критический шаг 2/lambda_max = {lr_crit:.6f}")

    hist_big = {}
    for lr in (0.1, 0.4, lr_crit * 1.01, lr_crit * 1.2):
        n = Neuron(2, activation="linear", loss="mse", seed=0)
        h = n.fit(X, y, lr=lr, epochs=200)
        losses = np.asarray(h["loss"])
        finite = np.isfinite(losses).all()
        state = "сходится" if finite and losses[-1] < losses[0] else "расходится"
        hist_big[round(lr, 4)] = h
        tail = f"{losses[-1]:.6e}" if finite else "переполнение"
        print(f"  lr = {lr:>8.4f} ({lr / lr_crit:>5.2f} от критического): "
              f"L(0) = {losses[0]:.6f} -> L(200) = {tail:>14}  {state}")

    p3 = plot_loss_curves(
        {f"lr = {lr}": np.clip(h["loss"], 1e-12, 1e12) for lr, h in hist_big.items()},
        filename="task7_lr_divergence.png",
        title="Линейный нейрон + MSE: потеря устойчивости при lr > 2/lambda_max",
    )

    print(f"\nГрафики: {p1}\n         {p2}\n         {p3}")

    print("\nВыводы:")
    print("  * lr = 0.001 -- шаг слишком мал: за отведённое число эпох потери")
    print("    убывают почти линейно и не успевают выйти на плато;")
    print("  * lr = 0.01 -- сходимость уверенная, но медленная;")
    print("  * lr = 0.1 -- разумный компромисс между скоростью и устойчивостью;")
    print("  * lr = 1.0 -- самая быстрая сходимость на этих данных, кривая")
    print("    остаётся монотонной, поскольку градиент усреднён по выборке;")
    print("  * при lr > 2/lambda_max (см. 7.3) обновление «перепрыгивает» минимум,")
    print("    и потери растут по геометрической прогрессии -- расходимость.")
    print("    Для сигмоидного нейрона такой жёсткой границы нет: при большом")
    print("    шаге активация уходит в зону насыщения, градиент затухает, и")
    print("    обучение не взрывается, а просто останавливается.")
    print("\n  Общее правило: шаг подбирают как наибольший, при котором кривая")
    print("  потерь ещё убывает монотонно.")
    return hist_neuron, hist_mlp


if __name__ == "__main__":
    main()
