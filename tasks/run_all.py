"""Последовательный запуск всех пунктов хода работы (1-7).

    python -m tasks.run_all
"""

from . import _bootstrap  # noqa: F401

from . import (
    task1_forward,
    task2_activations,
    task3_losses,
    task4_train_linear,
    task5_xor,
    task6_gradcheck,
    task7_learning_rate,
)

STEPS = [
    ("Пункт 1. Класс Neuron и прямой проход", task1_forward.main),
    ("Пункт 2. Функции активации и производные", task2_activations.main),
    ("Пункт 3. Функции потерь и градиенты", task3_losses.main),
    ("Пункт 4. Обучение на линейно разделимой задаче", task4_train_linear.main),
    ("Пункт 5. XOR и скрытый слой", task5_xor.main),
    ("Пункт 6. Численная проверка градиента", task6_gradcheck.main),
    ("Пункт 7. Влияние скорости обучения", task7_learning_rate.main),
]


def main():
    for i, (title, fn) in enumerate(STEPS, start=1):
        print("\n\n")
        print("#" * 70)
        print(f"# {title}")
        print("#" * 70)
        fn()

    print("\n\n" + "=" * 70)
    print("ВСЕ ПУНКТЫ ВЫПОЛНЕНЫ. Графики сохранены в каталоге figures/.")
    print("=" * 70)


if __name__ == "__main__":
    main()
