"""Автотесты к лабораторной работе № 1.

Запуск:  python -m unittest discover -s tests -v
   либо:  python -m pytest tests -q
"""

import filecmp
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron import MLP, Neuron                                   # noqa: E402
from neuron.activations import ACTIVATIONS, get_activation       # noqa: E402
from neuron.datasets import (                                    # noqa: E402
    and_or_dataset, blobs_dataset, moons_dataset, standardize,
    targets_for, xor_dataset,
)
from neuron.gradcheck import (                                   # noqa: E402
    gradient_check, kink_count, numeric_grad, randomize_params,
)
from neuron.losses import bce, bce_grad, mse, mse_grad           # noqa: E402
from neuron.variants import get_variant                          # noqa: E402

TOL = 1e-6


class TestActivations(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(float(get_activation("sigmoid")[0](np.array([0.0]))[0]), 0.5)
        self.assertAlmostEqual(float(get_activation("tanh")[0](np.array([0.0]))[0]), 0.0)
        self.assertAlmostEqual(float(get_activation("relu")[0](np.array([-3.0]))[0]), 0.0)
        self.assertAlmostEqual(float(get_activation("relu")[0](np.array([3.0]))[0]), 3.0)
        self.assertAlmostEqual(float(get_activation("leaky_relu")[0](np.array([-1.0]))[0]), -0.01)

    def test_sigmoid_is_numerically_stable(self):
        z = np.array([-1e4, -800.0, 0.0, 800.0, 1e4])
        a = get_activation("sigmoid")[0](z)
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertTrue(np.all((a >= 0.0) & (a <= 1.0)))

    def test_derivatives_match_finite_differences(self):
        """phi'(z) совпадает с центральной разностью вне точек излома."""
        eps = 1e-6
        z = np.linspace(-6, 6, 241)
        z = z[np.abs(z) > 1e-3]
        for name in ACTIVATIONS:
            with self.subTest(activation=name):
                phi, phi_prime = get_activation(name)
                numeric = (phi(z + eps) - phi(z - eps)) / (2 * eps)
                self.assertLess(float(np.max(np.abs(phi_prime(z) - numeric))), TOL)

    def test_unknown_activation_raises(self):
        with self.assertRaises(ValueError):
            get_activation("softmax")


class TestLosses(unittest.TestCase):
    def setUp(self):
        self.a = np.array([0.1, 0.4, 0.8, 0.95])
        self.y = np.array([0.0, 0.0, 1.0, 1.0])

    def test_perfect_prediction_gives_zero_loss(self):
        y = np.array([0.0, 1.0])
        self.assertAlmostEqual(mse(y, y), 0.0)
        self.assertLess(bce(np.array([1e-12, 1 - 1e-12]), y), 1e-9)

    def test_gradients_match_finite_differences(self):
        eps = 1e-6
        for fn, grad_fn in ((mse, mse_grad), (bce, bce_grad)):
            with self.subTest(loss=fn.__name__):
                numeric = np.empty_like(self.a)
                for k in range(self.a.size):
                    ap, am = self.a.copy(), self.a.copy()
                    ap[k] += eps
                    am[k] -= eps
                    numeric[k] = (fn(ap, self.y) - fn(am, self.y)) / (2 * eps)
                self.assertLess(float(np.max(np.abs(grad_fn(self.a, self.y) - numeric))), TOL)

    def test_bce_handles_boundary_values(self):
        """Клиппинг не даёт логарифму уйти в бесконечность."""
        self.assertTrue(np.isfinite(bce(np.array([0.0, 1.0]), np.array([1.0, 0.0]))))


class TestNeuronForward(unittest.TestCase):
    def test_forward_matches_definition(self):
        n = Neuron(3, activation="sigmoid", seed=0)
        n.w = np.array([0.5, -1.0, 2.0])
        n.b = 0.25
        X = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, -2.0]])
        expected = 1.0 / (1.0 + np.exp(-(X @ n.w + n.b)))
        np.testing.assert_allclose(n.forward(X), expected, atol=1e-15)

    def test_single_sample_is_accepted(self):
        n = Neuron(2, seed=0)
        self.assertEqual(n.forward([1.0, 2.0]).shape, (1,))

    def test_wrong_feature_count_raises(self):
        n = Neuron(2, seed=0)
        with self.assertRaises(ValueError):
            n.forward(np.zeros((3, 5)))

    def test_backward_requires_forward(self):
        n = Neuron(2, seed=0)
        with self.assertRaises(RuntimeError):
            n.backward(np.zeros(2))


class TestGradients(unittest.TestCase):
    """Пункт 6: аналитический градиент против конечных разностей, порог 1e-6."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(8, 2))
        self.y = rng.integers(0, 2, size=8).astype(float)

    def test_neuron_gradients(self):
        for act in ("sigmoid", "tanh", "relu", "leaky_relu", "linear"):
            for loss in ("mse", "bce"):
                if loss == "bce" and act != "sigmoid":
                    continue
                with self.subTest(activation=act, loss=loss):
                    n = Neuron(2, activation=act, loss=loss, seed=5)
                    res = gradient_check(n, self.X, targets_for(act, self.y), tol=TOL)
                    self.assertTrue(res["passed"], f"max rel err = {res['max_rel_err']:.3e}")

    def test_mlp_gradients(self):
        combos = [
            ("tanh", "sigmoid", "bce"),
            ("tanh", "tanh", "mse"),
            ("sigmoid", "sigmoid", "mse"),
            ("relu", "sigmoid", "bce"),
            ("leaky_relu", "sigmoid", "bce"),
        ]
        for hid, out, loss in combos:
            with self.subTest(hidden=hid, out=out, loss=loss):
                net = MLP(2, 3, hid, out, loss, seed=5)
                res = gradient_check(net, self.X, targets_for(out, self.y), tol=TOL)
                self.assertTrue(res["passed"], f"max rel err = {res['max_rel_err']:.3e}")

    def test_gradients_hold_after_training(self):
        X, y = xor_dataset()
        net = MLP(2, 2, "tanh", "sigmoid", "bce", seed=0)
        net.fit(X, y, lr=0.5, epochs=1500)
        res = gradient_check(net, X, y, tol=TOL)
        self.assertTrue(res["passed"], f"max rel err = {res['max_rel_err']:.3e}")

    def test_gradcheck_restores_parameters(self):
        net = MLP(2, 3, "tanh", "sigmoid", "bce", seed=1)
        before = net.get_params().copy()
        numeric_grad(net, self.X, self.y)
        np.testing.assert_allclose(net.get_params(), before, atol=1e-15)

    def test_relu_kink_is_detected_and_repairable(self):
        """На изломе ReLU проверка обязана падать, после сдвига -- проходить."""
        X, y = xor_dataset()
        net = MLP(2, 4, "relu", "sigmoid", "bce", seed=0)
        self.assertGreater(kink_count(net, X), 0)
        self.assertFalse(gradient_check(net, X, y, tol=TOL)["passed"])
        randomize_params(net, seed=0)
        self.assertEqual(kink_count(net, X), 0)
        self.assertTrue(gradient_check(net, X, y, tol=TOL)["passed"])


class TestTraining(unittest.TestCase):
    def test_single_neuron_solves_linearly_separable(self):
        X, y = blobs_dataset(n_samples=200)
        X = standardize(X)
        n = Neuron(2, activation="sigmoid", loss="bce", seed=0)
        h = n.fit(X, y, lr=0.5, epochs=300)
        self.assertEqual(h["accuracy"][-1], 1.0)
        self.assertLess(h["loss"][-1], 0.05)

    def test_loss_decreases_monotonically(self):
        X, y = blobs_dataset(n_samples=200)
        X = standardize(X)
        n = Neuron(2, activation="sigmoid", loss="bce", seed=0)
        losses = np.asarray(n.fit(X, y, lr=0.1, epochs=300)["loss"])
        self.assertTrue(np.all(np.diff(losses) <= 1e-12))

    def test_single_neuron_cannot_solve_xor(self):
        X, y = xor_dataset()
        for seed in range(5):
            n = Neuron(2, activation="sigmoid", loss="bce", seed=seed)
            h = n.fit(X, y, lr=1.0, epochs=3000)
            self.assertLess(h["accuracy"][-1], 1.0)

    def test_mlp_solves_xor(self):
        X, y = xor_dataset()
        solved = False
        for seed in range(10):
            net = MLP(2, 2, "tanh", "sigmoid", "bce", seed=seed)
            if net.fit(X, y, lr=0.5, epochs=5000)["accuracy"][-1] == 1.0:
                solved = True
                break
        self.assertTrue(solved, "сеть 2-2-1 обязана решать XOR хотя бы при одной инициализации")

    def test_wider_hidden_layer_always_solves_xor(self):
        X, y = xor_dataset()
        for seed in range(5):
            net = MLP(2, 8, "tanh", "sigmoid", "bce", seed=seed)
            self.assertEqual(net.fit(X, y, lr=0.5, epochs=4000)["accuracy"][-1], 1.0)

    def test_mlp_solves_and_or_composition(self):
        X, y = and_or_dataset()
        net = MLP(2, 8, "tanh", "sigmoid", "bce", seed=0)
        self.assertEqual(net.fit(X, y, lr=0.5, epochs=4000)["accuracy"][-1], 1.0)

    def test_mlp_fits_moons(self):
        X, y = moons_dataset(n_samples=300, noise=0.2)
        X = standardize(X)
        net = MLP(2, 8, "tanh", "sigmoid", "bce", seed=0)
        self.assertGreater(net.fit(X, y, lr=1.0, epochs=2000)["accuracy"][-1], 0.95)

    def test_larger_lr_converges_faster(self):
        """Пункт 7: при большем шаге потери убывают быстрее."""
        X, y = blobs_dataset(n_samples=200)
        X = standardize(X)
        finals = []
        for lr in (0.001, 0.01, 0.1, 1.0):
            n = Neuron(2, activation="sigmoid", loss="bce", seed=0)
            finals.append(n.fit(X, y, lr=lr, epochs=300)["loss"][-1])
        self.assertTrue(all(finals[i] > finals[i + 1] for i in range(len(finals) - 1)), finals)


class TestVariants(unittest.TestCase):
    def test_variant_fields_are_valid(self):
        for number in range(1, 31):
            v = get_variant(number)
            self.assertIn(v.task, ("xor", "and_or", "moons", "circles"))
            self.assertIn(v.loss, ("mse", "bce"))
            self.assertIn(v.hidden_activation, ACTIVATIONS)
            self.assertIn(v.output_activation, ACTIVATIONS)

    def test_bce_always_paired_with_sigmoid_output(self):
        for number in range(1, 31):
            v = get_variant(number)
            if v.loss == "bce":
                self.assertEqual(v.output_activation, "sigmoid")

    def test_variants_cycle_with_period_12(self):
        for number in range(1, 13):
            a, b = get_variant(number), get_variant(number + 12)
            self.assertEqual((a.activations, a.task, a.loss), (b.activations, b.task, b.loss))

    def test_invalid_variant_raises(self):
        with self.assertRaises(ValueError):
            get_variant(0)


class TestWebBuild(unittest.TestCase):
    """Собранный public/ должен совпадать со свежей сборкой из web/."""

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _build_module(self):
        path = os.path.join(self.ROOT, "web", "build.py")
        spec = importlib.util.spec_from_file_location("lab_web_build", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_public_is_up_to_date(self):
        public = os.path.join(self.ROOT, "public")
        if not os.path.isdir(public):
            self.skipTest("public/ не собран: запустите python web/build.py")
        build = self._build_module()
        tmp = tempfile.mkdtemp()
        try:
            build.build(tmp)
            names = sorted(os.listdir(tmp))
            match, mismatch, errors = filecmp.cmpfiles(tmp, public, names, shallow=False)
            self.assertEqual(
                (mismatch, errors), ([], []),
                "public/ отстал от web/ -- выполните: python web/build.py",
            )
            self.assertEqual(sorted(match), names)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_document_is_complete(self):
        index = os.path.join(self.ROOT, "public", "index.html")
        if not os.path.exists(index):
            self.skipTest("public/index.html не собран")
        with open(index, encoding="utf-8") as f:
            html = f.read()
        head = html.split("</head>", 1)[0]
        for token in ("<!doctype html>", '<html lang="ru">', "<title>", "favicon.svg"):
            self.assertIn(token, html, token)
        # Всё, чему место в <head>, должно оказаться именно там.
        self.assertIn("<title>", head)
        self.assertIn("<style>", head)
        for asset in ("data.js", "engine.js", "app.js"):
            self.assertIn('src="' + asset + '"', html)
            self.assertTrue(os.path.exists(os.path.join(self.ROOT, "public", asset)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
