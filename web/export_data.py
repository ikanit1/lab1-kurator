"""Экспорт данных и эталонных значений из NumPy-реализации в веб-интерфейс.

Веб-страница обучается ровно на тех же наборах, что и лабораторная работа,
а не на данных, сгенерированных заново в браузере. Дополнительно
выгружается фикстура: фиксированные параметры сети и посчитанные NumPy
потери и градиент -- страница сравнивает с ними свой результат и
показывает расхождение.

    python web/export_data.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron import MLP                                                # noqa: E402
from neuron.datasets import (                                         # noqa: E402
    and_or_dataset, circles_dataset, moons_dataset, standardize, xor_dataset,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.js")


def pack(X, y):
    # Координаты выгружаются с полной точностью float64: json.dump печатает
    # repr(float), который читается обратно бит в бит. Округление здесь
    # сделало бы расхождение JS с NumPy на уровне 1e-8 вместо 1e-16.
    X = np.asarray(X, dtype=np.float64)
    return {
        "x": [float(v) for v in X[:, 0]],
        "y": [float(v) for v in X[:, 1]],
        "label": [int(v) for v in np.asarray(y).ravel()],
    }


def main():
    Xc, yc = circles_dataset(n_samples=500, noise=0.1, factor=0.5)
    Xm, ym = moons_dataset(n_samples=500, noise=0.2)
    Xx, yx = xor_dataset()
    Xa, ya = and_or_dataset()

    datasets = {
        "circles": {
            "title": "make_circles",
            "subtitle": "n = 500, noise = 0.1, factor = 0.5",
            "note": "Набор варианта 12. Классы вложены друг в друга, линейной границы не существует.",
            **pack(standardize(Xc), yc),
        },
        "moons": {
            "title": "make_moons",
            "subtitle": "n = 500, noise = 0.2",
            "note": "Две переплетённые дуги: граница нелинейная, но классы не вложены.",
            **pack(standardize(Xm), ym),
        },
        "xor": {
            "title": "XOR",
            "subtitle": "4 точки",
            "note": "Минимальный линейно неразделимый пример.",
            **pack(Xx, yx),
        },
        "and_or": {
            "title": "AND-OR (XNOR)",
            "subtitle": "4 точки",
            "note": "Композиция пороговых функций, также линейно неразделима.",
            **pack(Xa, ya),
        },
    }

    # --- фикстура для сверки JS-порта с NumPy -------------------------------
    Xs = standardize(Xc)
    net = MLP(2, 8, "tanh", "sigmoid", "bce", seed=0)
    rng = np.random.default_rng(20240501)
    theta = rng.normal(0.0, 0.6, size=net.get_params().size)
    net.set_params(theta)

    grad = net.analytic_grad(Xs, yc)
    fixture = {
        "n_hidden": 8,
        "hidden_activation": "tanh",
        "output_activation": "sigmoid",
        "loss_name": "bce",
        "dataset": "circles",
        "theta": [float(v) for v in theta],
        "loss": float(net.loss(Xs, yc)),
        "grad": [float(v) for v in grad],
        "forward_head": [float(v) for v in net.forward(Xs)[:8]],
    }

    payload = {
        "generated_by": "web/export_data.py",
        "numpy_version": np.__version__,
        "datasets": datasets,
        "fixture": fixture,
    }

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("// Сгенерировано web/export_data.py -- не редактировать вручную.\n")
        f.write("// Наборы данных выгружены из scikit-learn и стандартизованы тем же\n")
        f.write("// кодом, что используется в neuron/datasets.py.\n")
        f.write("window.LAB_DATA = ")
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")

    size = os.path.getsize(OUT)
    print(f"записано {OUT} ({size / 1024:.1f} КБ)")
    print(f"  наборов: {len(datasets)}")
    print(f"  фикстура: {len(theta)} параметров, loss = {fixture['loss']:.12f}")


if __name__ == "__main__":
    main()
