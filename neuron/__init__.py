"""Лабораторная работа № 1: искусственный нейрон на NumPy с нуля.

Пакет содержит реализацию без использования фреймворков автоматического
дифференцирования: все градиенты выведены и запрограммированы вручную.
"""

from .activations import (
    ACTIVATIONS,
    get_activation,
    leaky_relu,
    leaky_relu_prime,
    relu,
    relu_prime,
    sigmoid,
    sigmoid_prime,
    tanh,
    tanh_prime,
)
from .datasets import DATASETS, get_dataset, standardize, targets_for
from .gradcheck import gradient_check, numeric_grad, report
from .losses import LOSSES, bce, bce_grad, get_loss, mse, mse_grad
from .mlp import MLP
from .neuron import Neuron
from .variants import get_variant, variant_table

__all__ = [
    "ACTIVATIONS", "DATASETS", "LOSSES", "MLP", "Neuron",
    "bce", "bce_grad", "get_activation", "get_dataset", "get_loss",
    "get_variant", "gradient_check", "leaky_relu", "leaky_relu_prime",
    "mse", "mse_grad", "numeric_grad", "relu", "relu_prime", "report",
    "sigmoid", "sigmoid_prime", "standardize", "tanh", "tanh_prime",
    "targets_for", "variant_table",
]
