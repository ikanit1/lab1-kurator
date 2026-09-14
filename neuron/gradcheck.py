"""Численная проверка градиента методом центральных конечных разностей
(пункт 6 задания).

Для каждого параметра theta_k:

    dL/dtheta_k  ~=  ( L(theta_k + eps) - L(theta_k - eps) ) / (2 eps)

Центральная разность имеет погрешность O(eps^2) против O(eps) у односторонней,
поэтому при eps = 1e-5 и вычислениях в float64 достигается точность ~1e-10,
что с запасом укладывается в требуемые 1e-6.

Сравнение ведётся по относительной ошибке

    rel = |g_analytic - g_numeric| / max(1e-12, |g_analytic| + |g_numeric|)
"""

from __future__ import annotations

import numpy as np


def numeric_grad(model, X, y, eps: float = 1e-5):
    """Численный градиент функции потерь модели по всем её параметрам."""
    theta0 = model.get_params().astype(np.float64)
    grad = np.zeros_like(theta0)

    for k in range(theta0.size):
        theta_plus = theta0.copy()
        theta_plus[k] += eps
        model.set_params(theta_plus)
        l_plus = model.loss(X, y)

        theta_minus = theta0.copy()
        theta_minus[k] -= eps
        model.set_params(theta_minus)
        l_minus = model.loss(X, y)

        grad[k] = (l_plus - l_minus) / (2.0 * eps)

    model.set_params(theta0)          # восстановить исходные параметры
    return grad


def gradient_check(model, X, y, eps: float = 1e-5, tol: float = 1e-6):
    """Сравнить аналитический и численный градиенты.

    Возвращает словарь с обоими градиентами, максимальной абсолютной и
    относительной ошибкой и флагом прохождения проверки.
    """
    g_analytic = model.analytic_grad(X, y).astype(np.float64)
    g_numeric = numeric_grad(model, X, y, eps=eps)

    abs_err = np.abs(g_analytic - g_numeric)
    denom = np.maximum(1e-12, np.abs(g_analytic) + np.abs(g_numeric))
    rel_err = abs_err / denom

    return {
        "analytic": g_analytic,
        "numeric": g_numeric,
        "max_abs_err": float(abs_err.max()),
        "max_rel_err": float(rel_err.max()),
        "tol": tol,
        "passed": bool(rel_err.max() <= tol),
        "kinks": kink_count(model, X, eps),
    }


def report(result: dict, title: str = "Проверка градиента") -> str:
    """Человекочитаемый отчёт о результате gradient_check."""
    lines = [
        f"{title}",
        "-" * len(title),
        f"{'#':>3} {'аналитический':>18} {'численный':>18} {'|разность|':>14} {'отн. ошибка':>14}",
    ]
    for k, (ga, gn) in enumerate(zip(result["analytic"], result["numeric"])):
        lines.append(
            f"{k:>3} {ga:>18.12f} {gn:>18.12f} "
            f"{abs(ga - gn):>14.3e} {abs(ga - gn) / max(1e-12, abs(ga) + abs(gn)):>14.3e}"
        )
    lines.append("")
    lines.append(f"max |разность|      = {result['max_abs_err']:.3e}")
    lines.append(f"max отн. ошибка     = {result['max_rel_err']:.3e}")
    lines.append(
        f"порог {result['tol']:.0e}: "
        + ("ПРОЙДЕНО" if result["passed"] else "НЕ ПРОЙДЕНО")
    )
    kinks = result.get("kinks", 0)
    if kinks:
        lines.append(
            f"ВНИМАНИЕ: {kinks} предактиваций попало в eps-окрестность излома "
            "ReLU (z = 0);\n         в этих точках производная не существует, и "
            "расхождение с конечной\n         разностью ожидаемо. Используйте "
            "randomize_params(), чтобы увести\n         параметры от излома."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Диагностика изломов ReLU-подобных активаций
# --------------------------------------------------------------------------
KINKED_ACTIVATIONS = ("relu", "leaky_relu")


def kink_count(model, X, eps: float = 1e-5) -> int:
    """Сколько предактиваций z попало в eps-окрестность излома z = 0.

    ReLU и LeakyReLU не дифференцируемы в нуле. Если |z| < eps, конечная
    разность «перешагивает» излом и численный градиент систематически
    расходится с аналитическим -- проверка провалится не из-за ошибки в
    коде, а из-за негладкости самой функции.

    Типичный источник таких точек -- нулевая инициализация смещений в паре
    с нулевым входным вектором (например, точка (0, 0) в наборах XOR и
    AND-OR): тогда z = 0 в точности.
    """
    model.forward(X)
    cache = getattr(model, "_cache", None) or {}
    total = 0
    if getattr(model, "hidden_activation", None) in KINKED_ACTIVATIONS and "Z1" in cache:
        total += int(np.sum(np.abs(cache["Z1"]) < eps))
    out_act = getattr(model, "output_activation", None) or getattr(model, "activation_name", None)
    if out_act in KINKED_ACTIVATIONS:
        z_out = cache.get("z2", cache.get("z"))
        if z_out is not None:
            total += int(np.sum(np.abs(z_out) < eps))
    return total


def randomize_params(model, seed: int = 0, scale: float = 0.7):
    """Задать параметрам модели случайные значения.

    Используется перед проверкой градиента, чтобы увести предактивации от
    изломов ReLU (в частности, от точного нуля при b = 0) и проверять
    именно правильность формул, а не поведение функции в точке негладкости.
    """
    rng = np.random.default_rng(seed)
    theta = model.get_params()
    model.set_params(rng.normal(0.0, scale, size=theta.size))
    return model
