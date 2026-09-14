/* Порт NumPy-реализации из neuron/ на JavaScript.
 *
 * Формулы, порядок параметров в плоском векторе и численные приёмы
 * (устойчивая сигмоида, свёрнутый градиент для sigmoid + BCE, клиппинг
 * логарифма в BCE) повторяют neuron/activations.py, neuron/losses.py,
 * neuron/neuron.py и neuron/mlp.py один в один. Совпадение проверяется
 * на фикстуре из data.js -- см. verifyAgainstNumpy().
 */
(function (global) {
  "use strict";

  // ---------------------------------------------------------------- утилиты
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Нормальное распределение методом Бокса — Мюллера. */
  function makeNormal(seed) {
    const rand = mulberry32(seed);
    let spare = null;
    return function normal(mean, std) {
      if (spare !== null) {
        const v = spare;
        spare = null;
        return mean + std * v;
      }
      let u = 0;
      let v = 0;
      let s = 0;
      do {
        u = rand() * 2 - 1;
        v = rand() * 2 - 1;
        s = u * u + v * v;
      } while (s >= 1 || s === 0);
      const f = Math.sqrt((-2 * Math.log(s)) / s);
      spare = v * f;
      return mean + std * u * f;
    };
  }

  // ------------------------------------------------------------- активации
  function sigmoid(z) {
    // Устойчивая форма: exp() не переполняется ни при каком знаке z.
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const e = Math.exp(z);
    return e / (1 + e);
  }
  function sigmoidPrime(z) {
    const s = sigmoid(z);
    return s * (1 - s);
  }
  function tanhPrime(z) {
    const t = Math.tanh(z);
    return 1 - t * t;
  }
  const LEAKY_ALPHA = 0.01;

  const ACTIVATIONS = {
    sigmoid: { fn: sigmoid, prime: sigmoidPrime, label: "sigmoid", range: [0, 1] },
    tanh: { fn: Math.tanh, prime: tanhPrime, label: "tanh", range: [-1, 1] },
    relu: {
      fn: (z) => (z > 0 ? z : 0),
      prime: (z) => (z > 0 ? 1 : 0),
      label: "ReLU",
      range: [0, 6],
    },
    leaky_relu: {
      fn: (z) => (z > 0 ? z : LEAKY_ALPHA * z),
      prime: (z) => (z > 0 ? 1 : LEAKY_ALPHA),
      label: "LeakyReLU",
      range: [-0.1, 6],
    },
    linear: { fn: (z) => z, prime: () => 1, label: "linear", range: [-6, 6] },
  };

  // ----------------------------------------------------------------- потери
  const BCE_EPS = 1e-12;

  const LOSSES = {
    mse: {
      label: "MSE",
      value(a, y) {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += (a[i] - y[i]) * (a[i] - y[i]);
        return s / a.length;
      },
      grad(a, y) {
        const g = new Float64Array(a.length);
        for (let i = 0; i < a.length; i++) g[i] = (2 * (a[i] - y[i])) / a.length;
        return g;
      },
    },
    bce: {
      label: "BCE",
      value(a, y) {
        let s = 0;
        for (let i = 0; i < a.length; i++) {
          const ai = Math.min(1 - BCE_EPS, Math.max(BCE_EPS, a[i]));
          s += y[i] * Math.log(ai) + (1 - y[i]) * Math.log(1 - ai);
        }
        return -s / a.length;
      },
      grad(a, y) {
        const g = new Float64Array(a.length);
        for (let i = 0; i < a.length; i++) {
          const ai = Math.min(1 - BCE_EPS, Math.max(BCE_EPS, a[i]));
          g[i] = (ai - y[i]) / (ai * (1 - ai) * a.length);
        }
        return g;
      },
    },
  };

  // ------------------------------------------------------- одиночный нейрон
  class Neuron {
    constructor(opts) {
      const o = opts || {};
      this.nInputs = o.nInputs || 2;
      this.activationName = o.activation || "sigmoid";
      this.lossName = o.loss || "bce";
      this.act = ACTIVATIONS[this.activationName];
      this.lossDef = LOSSES[this.lossName];

      const normal = makeNormal(o.seed === undefined ? 0 : o.seed);
      const scale = 1 / Math.sqrt(this.nInputs);
      this.w = new Float64Array(this.nInputs);
      for (let d = 0; d < this.nInputs; d++) this.w[d] = normal(0, scale);
      this.b = 0;
      this._cache = null;
    }

    forward(X) {
      const n = X.length;
      const z = new Float64Array(n);
      const a = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let s = this.b;
        for (let d = 0; d < this.nInputs; d++) s += X[i][d] * this.w[d];
        z[i] = s;
        a[i] = this.act.fn(s);
      }
      this._cache = { X, z, a };
      return a;
    }

    backward(y) {
      const { X, z, a } = this._cache;
      const n = a.length;
      const dz = new Float64Array(n);
      if (this.activationName === "sigmoid" && this.lossName === "bce") {
        // Множители a(1-a) сокращаются -- та же свёрнутая форма, что в Python.
        for (let i = 0; i < n; i++) dz[i] = (a[i] - y[i]) / n;
      } else {
        const da = this.lossDef.grad(a, y);
        for (let i = 0; i < n; i++) dz[i] = da[i] * this.act.prime(z[i]);
      }
      const gw = new Float64Array(this.nInputs);
      let gb = 0;
      for (let i = 0; i < n; i++) {
        for (let d = 0; d < this.nInputs; d++) gw[d] += X[i][d] * dz[i];
        gb += dz[i];
      }
      return { gw, gb };
    }

    loss(X, y) {
      return this.lossDef.value(this.forward(X), y);
    }

    accuracy(X, y) {
      const a = this.forward(X);
      const thr = this.activationName === "tanh" ? 0 : 0.5;
      let ok = 0;
      for (let i = 0; i < a.length; i++) {
        const pred = a[i] > thr ? 1 : 0;
        const target = y[i] > thr ? 1 : 0;
        if (pred === target) ok++;
      }
      return ok / a.length;
    }

    step(X, y, lr) {
      this.forward(X);
      const { gw, gb } = this.backward(y);
      for (let d = 0; d < this.nInputs; d++) this.w[d] -= lr * gw[d];
      this.b -= lr * gb;
    }

    getParams() {
      const t = new Float64Array(this.nInputs + 1);
      t.set(this.w, 0);
      t[this.nInputs] = this.b;
      return t;
    }

    setParams(theta) {
      for (let d = 0; d < this.nInputs; d++) this.w[d] = theta[d];
      this.b = theta[this.nInputs];
    }

    analyticGrad(X, y) {
      this.forward(X);
      const { gw, gb } = this.backward(y);
      const g = new Float64Array(this.nInputs + 1);
      g.set(gw, 0);
      g[this.nInputs] = gb;
      return g;
    }
  }

  // ----------------------------------------------------- сеть D -> H -> 1
  class MLP {
    constructor(opts) {
      const o = opts || {};
      this.nInputs = o.nInputs || 2;
      this.nHidden = o.nHidden || 8;
      this.hiddenActivation = o.hiddenActivation || "tanh";
      this.outputActivation = o.outputActivation || "sigmoid";
      this.lossName = o.loss || "bce";
      this.phi1 = ACTIVATIONS[this.hiddenActivation];
      this.phi2 = ACTIVATIONS[this.outputActivation];
      this.lossDef = LOSSES[this.lossName];

      const normal = makeNormal(o.seed === undefined ? 0 : o.seed);
      const D = this.nInputs;
      const H = this.nHidden;
      // Инициализация Ксавье, как в neuron/mlp.py: std = 1/sqrt(fan_in).
      this.W1 = new Float64Array(D * H);
      for (let k = 0; k < D * H; k++) this.W1[k] = normal(0, 1 / Math.sqrt(D));
      this.b1 = new Float64Array(H);
      this.w2 = new Float64Array(H);
      for (let j = 0; j < H; j++) this.w2[j] = normal(0, 1 / Math.sqrt(H));
      this.b2 = 0;
      this._cache = null;
      this.epoch = 0;
    }

    forward(X) {
      const n = X.length;
      const D = this.nInputs;
      const H = this.nHidden;
      const Z1 = new Float64Array(n * H);
      const A1 = new Float64Array(n * H);
      const z2 = new Float64Array(n);
      const a2 = new Float64Array(n);
      const phi1 = this.phi1.fn;
      const phi2 = this.phi2.fn;

      for (let i = 0; i < n; i++) {
        const xi = X[i];
        let s2 = this.b2;
        for (let j = 0; j < H; j++) {
          let s = this.b1[j];
          for (let d = 0; d < D; d++) s += xi[d] * this.W1[d * H + j];
          Z1[i * H + j] = s;
          const a = phi1(s);
          A1[i * H + j] = a;
          s2 += a * this.w2[j];
        }
        z2[i] = s2;
        a2[i] = phi2(s2);
      }
      this._cache = { X, Z1, A1, z2, a2 };
      return a2;
    }

    backward(y) {
      const { X, Z1, A1, z2, a2 } = this._cache;
      const n = a2.length;
      const D = this.nInputs;
      const H = this.nHidden;

      const dz2 = new Float64Array(n);
      if (this.outputActivation === "sigmoid" && this.lossName === "bce") {
        for (let i = 0; i < n; i++) dz2[i] = (a2[i] - y[i]) / n;
      } else {
        const da = this.lossDef.grad(a2, y);
        for (let i = 0; i < n; i++) dz2[i] = da[i] * this.phi2.prime(z2[i]);
      }

      const gW1 = new Float64Array(D * H);
      const gb1 = new Float64Array(H);
      const gw2 = new Float64Array(H);
      let gb2 = 0;
      const prime1 = this.phi1.prime;

      for (let i = 0; i < n; i++) {
        const xi = X[i];
        const d2 = dz2[i];
        gb2 += d2;
        for (let j = 0; j < H; j++) {
          gw2[j] += A1[i * H + j] * d2;
          const dz1 = d2 * this.w2[j] * prime1(Z1[i * H + j]);
          gb1[j] += dz1;
          for (let d = 0; d < D; d++) gW1[d * H + j] += xi[d] * dz1;
        }
      }
      return { gW1, gb1, gw2, gb2 };
    }

    step(X, y, lr) {
      this.forward(X);
      const g = this.backward(y);
      for (let k = 0; k < this.W1.length; k++) this.W1[k] -= lr * g.gW1[k];
      for (let j = 0; j < this.nHidden; j++) {
        this.b1[j] -= lr * g.gb1[j];
        this.w2[j] -= lr * g.gw2[j];
      }
      this.b2 -= lr * g.gb2;
      this.epoch++;
    }

    loss(X, y) {
      return this.lossDef.value(this.forward(X), y);
    }

    accuracy(X, y) {
      const a = this.forward(X);
      const thr = this.outputActivation === "tanh" ? 0 : 0.5;
      let ok = 0;
      for (let i = 0; i < a.length; i++) {
        const pred = a[i] > thr ? 1 : 0;
        const target = y[i] > thr ? 1 : 0;
        if (pred === target) ok++;
      }
      return ok / a.length;
    }

    /** Плоский вектор параметров: [W1 по строкам, b1, w2, b2] -- как в Python. */
    getParams() {
      const D = this.nInputs;
      const H = this.nHidden;
      const t = new Float64Array(D * H + H + H + 1);
      t.set(this.W1, 0);
      t.set(this.b1, D * H);
      t.set(this.w2, D * H + H);
      t[D * H + 2 * H] = this.b2;
      return t;
    }

    setParams(theta) {
      const D = this.nInputs;
      const H = this.nHidden;
      for (let k = 0; k < D * H; k++) this.W1[k] = theta[k];
      for (let j = 0; j < H; j++) this.b1[j] = theta[D * H + j];
      for (let j = 0; j < H; j++) this.w2[j] = theta[D * H + H + j];
      this.b2 = theta[D * H + 2 * H];
    }

    analyticGrad(X, y) {
      this.forward(X);
      const g = this.backward(y);
      const D = this.nInputs;
      const H = this.nHidden;
      const out = new Float64Array(D * H + H + H + 1);
      out.set(g.gW1, 0);
      out.set(g.gb1, D * H);
      out.set(g.gw2, D * H + H);
      out[D * H + 2 * H] = g.gb2;
      return out;
    }
  }

  // ------------------------------------------- численная проверка градиента
  function numericGrad(model, X, y, eps) {
    eps = eps || 1e-5;
    const theta0 = Float64Array.from(model.getParams());
    const grad = new Float64Array(theta0.length);
    const probe = Float64Array.from(theta0);

    for (let k = 0; k < theta0.length; k++) {
      probe[k] = theta0[k] + eps;
      model.setParams(probe);
      const lPlus = model.loss(X, y);

      probe[k] = theta0[k] - eps;
      model.setParams(probe);
      const lMinus = model.loss(X, y);

      probe[k] = theta0[k];
      grad[k] = (lPlus - lMinus) / (2 * eps);
    }
    model.setParams(theta0);
    return grad;
  }

  function gradientCheck(model, X, y, eps, tol) {
    eps = eps || 1e-5;
    tol = tol || 1e-6;
    const analytic = model.analyticGrad(X, y);
    const numeric = numericGrad(model, X, y, eps);
    let maxAbs = 0;
    let maxRel = 0;
    const rows = [];
    for (let k = 0; k < analytic.length; k++) {
      const abs = Math.abs(analytic[k] - numeric[k]);
      const rel = abs / Math.max(1e-12, Math.abs(analytic[k]) + Math.abs(numeric[k]));
      if (abs > maxAbs) maxAbs = abs;
      if (rel > maxRel) maxRel = rel;
      rows.push({ k, analytic: analytic[k], numeric: numeric[k], abs, rel });
    }
    return { rows, maxAbs, maxRel, tol, passed: maxRel <= tol, kinks: kinkCount(model, X, eps) };
  }

  const KINKED = ["relu", "leaky_relu"];

  /** Сколько предактиваций попало в eps-окрестность излома z = 0. */
  function kinkCount(model, X, eps) {
    eps = eps || 1e-5;
    model.forward(X);
    const c = model._cache || {};
    let total = 0;
    if (KINKED.indexOf(model.hiddenActivation) >= 0 && c.Z1) {
      for (let k = 0; k < c.Z1.length; k++) if (Math.abs(c.Z1[k]) < eps) total++;
    }
    const outAct = model.outputActivation || model.activationName;
    const zOut = c.z2 || c.z;
    if (KINKED.indexOf(outAct) >= 0 && zOut) {
      for (let k = 0; k < zOut.length; k++) if (Math.abs(zOut[k]) < eps) total++;
    }
    return total;
  }

  function randomizeParams(model, seed, scale) {
    const normal = makeNormal(seed === undefined ? 0 : seed);
    const theta = model.getParams();
    const next = new Float64Array(theta.length);
    for (let k = 0; k < next.length; k++) next[k] = normal(0, scale === undefined ? 0.7 : scale);
    model.setParams(next);
    return model;
  }

  // ------------------------------------------------ сверка с эталоном NumPy
  /**
   * Считает те же величины, что и NumPy, на фикстуре из data.js.
   * Возвращает расхождение по значению потерь и по каждой компоненте градиента.
   */
  function verifyAgainstNumpy(fixture, dataset) {
    const X = dataset.points;
    const y = dataset.labels;
    const net = new MLP({
      nInputs: 2,
      nHidden: fixture.n_hidden,
      hiddenActivation: fixture.hidden_activation,
      outputActivation: fixture.output_activation,
      loss: fixture.loss_name,
      seed: 0,
    });
    net.setParams(Float64Array.from(fixture.theta));

    const loss = net.loss(X, y);
    const grad = net.analyticGrad(X, y);

    let maxGradDiff = 0;
    for (let k = 0; k < grad.length; k++) {
      maxGradDiff = Math.max(maxGradDiff, Math.abs(grad[k] - fixture.grad[k]));
    }
    const head = net.forward(X);
    let maxFwdDiff = 0;
    for (let i = 0; i < fixture.forward_head.length; i++) {
      maxFwdDiff = Math.max(maxFwdDiff, Math.abs(head[i] - fixture.forward_head[i]));
    }

    return {
      lossJs: loss,
      lossPy: fixture.loss,
      lossDiff: Math.abs(loss - fixture.loss),
      maxGradDiff,
      maxFwdDiff,
      nParams: grad.length,
      grad,
    };
  }

  // ------------------------------------------------------- работа с данными
  /** Развернуть набор из data.js в массив точек [[x1, x2], ...] и метки. */
  function loadDataset(raw) {
    const n = raw.label.length;
    const points = new Array(n);
    const labels = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      points[i] = [raw.x[i], raw.y[i]];
      labels[i] = raw.label[i];
    }
    return {
      points,
      labels,
      title: raw.title,
      subtitle: raw.subtitle,
      note: raw.note,
      n,
    };
  }

  /** Метки под область значений выходной активации: tanh требует {-1, +1}. */
  function targetsFor(outputActivation, labels) {
    if (outputActivation !== "tanh") return labels;
    const out = new Float64Array(labels.length);
    for (let i = 0; i < labels.length; i++) out[i] = 2 * labels[i] - 1;
    return out;
  }

  global.LabEngine = {
    ACTIVATIONS,
    LOSSES,
    Neuron,
    MLP,
    numericGrad,
    gradientCheck,
    kinkCount,
    randomizeParams,
    verifyAgainstNumpy,
    loadDataset,
    targetsFor,
    makeNormal,
  };
})(window);
