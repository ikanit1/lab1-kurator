"""Индивидуальные варианты заданий.

Вариант определяется номером обучающегося в журнале. По условию варьируются
три компонента:

* набор функций активации: sigmoid+tanh / relu+leaky_relu / tanh+relu;
* тип задачи: xor / and_or / moons / circles;
* функция потерь: mse / bce.

Правило раскладки номера N (нумерация с единицы):

    activation_set = ACTIVATION_SETS[(N - 1) % 3]
    task           = TASKS[(N - 1) % 4]
    loss           = LOSSES[(N - 1) % 2]

Периоды 3, 4 и 2 дают 12 различных сочетаний, повторяющихся с шагом 12.
"""

from __future__ import annotations

from dataclasses import dataclass

ACTIVATION_SETS = [
    ("sigmoid", "tanh"),
    ("relu", "leaky_relu"),
    ("tanh", "relu"),
]

TASKS = ["xor", "and_or", "moons", "circles"]

LOSS_NAMES = ["mse", "bce"]


@dataclass(frozen=True)
class Variant:
    number: int
    activations: tuple
    task: str
    loss: str

    @property
    def hidden_activation(self) -> str:
        """Активация скрытого слоя -- первая из набора варианта."""
        return self.activations[0]

    @property
    def output_activation(self) -> str:
        """Активация выхода.

        BCE определена только для выходов из (0, 1), поэтому при loss='bce'
        выход всегда сигмоидный. Для MSE берём вторую активацию набора, если
        её область значений подходит для классификации (sigmoid или tanh),
        иначе -- сигмоиду.
        """
        if self.loss == "bce":
            return "sigmoid"
        second = self.activations[1]
        return second if second in ("sigmoid", "tanh") else "sigmoid"

    def describe(self) -> str:
        return (
            f"Вариант {self.number}: активации {self.activations[0]}+{self.activations[1]}, "
            f"задача {self.task}, функция потерь {self.loss.upper()}\n"
            f"  скрытый слой: {self.hidden_activation}, выход: {self.output_activation}"
        )


def get_variant(number: int) -> Variant:
    """Собрать описание варианта по номеру в журнале (начиная с 1)."""
    if number < 1:
        raise ValueError("Номер варианта должен быть положительным")
    i = number - 1
    return Variant(
        number=number,
        activations=ACTIVATION_SETS[i % len(ACTIVATION_SETS)],
        task=TASKS[i % len(TASKS)],
        loss=LOSS_NAMES[i % len(LOSS_NAMES)],
    )


def variant_table(n: int = 12) -> str:
    """Текстовая таблица первых n вариантов."""
    head = f"{'№':>3} | {'активации':<22} | {'задача':<8} | {'потери':<6} | {'скрытый':<10} | {'выход':<8}"
    lines = [head, "-" * len(head)]
    for number in range(1, n + 1):
        v = get_variant(number)
        lines.append(
            f"{v.number:>3} | {v.activations[0] + '+' + v.activations[1]:<22} | "
            f"{v.task:<8} | {v.loss:<6} | {v.hidden_activation:<10} | {v.output_activation:<8}"
        )
    return "\n".join(lines)
