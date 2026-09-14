/* Логика страницы: графики, карта решений и семь станций хода работы.
 * Все вычисления идут через LabEngine -- порт NumPy-реализации из neuron/.
 */
(function () {
  "use strict";

  var E = window.LabEngine;
  var DATA = window.LAB_DATA;
  var $ = function (id) { return document.getElementById(id); };

  // =====================================================================
  // Палитра: читается из CSS-токенов, чтобы графики следовали теме страницы
  // =====================================================================
  var css = function (name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  };
  var isDark = function () { return css("--is-dark") === "1"; };

  function palette() {
    var dark = isDark();
    return {
      dark: dark,
      ink: css("--ink"),
      ink2: css("--ink-2"),
      ink3: css("--ink-3"),
      line: css("--line"),
      lineStrong: css("--line-strong"),
      panel: css("--panel"),
      c0: css("--c0"),
      c1: css("--c1"),
      // Последовательная шкала для упорядоченных значений eta: один тон,
      // светлый -> тёмный в светлой теме и наоборот в тёмной.
      ramp: dark
        ? ["#0C5A53", "#12897E", "#35BFB1", "#93E3D8"]
        : ["#A6D6D0", "#4DB3A8", "#0A8C80", "#00534C"],
      // Расходящаяся шкала для карты решений: два тона и нейтральная
      // середина на уровне 0.5 -- там, где сеть не уверена.
      diverging: dark
        ? [[0, "#57DACA"], [0.25, "#17A899"], [0.42, "#12615A"], [0.5, "#232E2C"],
           [0.58, "#6B4A24"], [0.75, "#D97528"], [1, "#F4A86A"]]
        : [[0, "#006B62"], [0.25, "#35A99D"], [0.42, "#AFD8D2"], [0.5, "#E6E4DE"],
           [0.58, "#F2CFAB"], [0.75, "#E08A3A"], [1, "#A64D00"]]
    };
  }
  var P = palette();

  // ------------------------------------------------------------ цвет карты
  function hexRgb(h) {
    return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  }
  function divergingRgb(t) {
    var stops = P.diverging;
    t = Math.max(0, Math.min(1, t));
    for (var i = 1; i < stops.length; i++) {
      if (t <= stops[i][0]) {
        var a = stops[i - 1];
        var b = stops[i];
        var f = (t - a[0]) / (b[0] - a[0] || 1);
        var ca = hexRgb(a[1]);
        var cb = hexRgb(b[1]);
        return [
          Math.round(ca[0] + (cb[0] - ca[0]) * f),
          Math.round(ca[1] + (cb[1] - ca[1]) * f),
          Math.round(ca[2] + (cb[2] - ca[2]) * f)
        ];
      }
    }
    return hexRgb(stops[stops.length - 1][1]);
  }
  function scaleGradient() {
    return "linear-gradient(90deg," + P.diverging.map(function (s) {
      return s[1] + " " + (s[0] * 100).toFixed(0) + "%";
    }).join(",") + ")";
  }

  // =====================================================================
  // Форматирование чисел
  // =====================================================================
  var SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³",
              4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };

  function sci(x, digits) {
    if (x === 0) return "0";
    if (!isFinite(x)) return "∞";
    digits = digits === undefined ? 1 : digits;
    var exp = Math.floor(Math.log10(Math.abs(x)));
    var mant = x / Math.pow(10, exp);
    var sup = String(exp).split("").map(function (c) { return SUP[c] || c; }).join("");
    return mant.toFixed(digits) + "·10" + sup;
  }
  function fx(x, n) { return Number(x).toFixed(n === undefined ? 3 : n); }
  function pct(x) { return (x * 100).toFixed(1) + " %"; }

  // =====================================================================
  // Линейный график (SVG) с перекрестием и подсказкой
  // =====================================================================
  function niceTicks(min, max, count) {
    if (!(max > min)) return [min];
    var span = max - min;
    var step = Math.pow(10, Math.floor(Math.log10(span / count)));
    var err = (span / count) / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var out = [];
    for (var v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) {
      out.push(Math.abs(v) < step * 1e-6 ? 0 : v);
    }
    return out;
  }

  var SVG_NS = "http://www.w3.org/2000/svg";
  function el(name, attrs) {
    var node = document.createElementNS(SVG_NS, name);
    for (var k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) node.setAttribute(k, attrs[k]);
    return node;
  }

  /**
   * cfg = { series:[{name,color,dash,points:[[x,y]..],width}], xDomain, yDomain,
   *         yScale:'log'|'linear', xFormat, yFormat, height, xLabel, yLabel,
   *         markers:[{x,y,color}], hover:bool, tipFormat }
   */
  function lineChart(container, cfg) {
    // viewBox повторяет реальную ширину контейнера: иначе SVG масштабируется
    // и подписи осей уезжают до нечитаемых 7 пикселей.
    var W = Math.round(Math.max(300, Math.min(560, container.clientWidth || 520)));
    var H = cfg.height || 300;
    var m = {
      t: cfg.yLabel ? 24 : 12,
      r: cfg.directLabels ? 18 : 16,
      b: cfg.xLabel ? 42 : 28,
      l: 52
    };
    var log = cfg.yScale === "log";

    var xs = cfg.xDomain;
    var ys = cfg.yDomain;
    var ty = function (v) { return log ? Math.log10(Math.max(v, 1e-12)) : v; };
    var y0 = ty(ys[0]);
    var y1 = ty(ys[1]);

    var px = function (x) { return m.l + ((x - xs[0]) / (xs[1] - xs[0] || 1)) * (W - m.l - m.r); };
    var py = function (y) { return H - m.b - ((ty(y) - y0) / (y1 - y0 || 1)) * (H - m.t - m.b); };

    container.textContent = "";
    var svg = el("svg", {
      viewBox: "0 0 " + W + " " + H,
      role: "img",
      "aria-label": cfg.ariaLabel || "график"
    });

    // --- сетка и оси (подчёркнуто второстепенны) --------------------------
    var yTicks = log
      ? (function () {
          var out = [];
          var e;
          for (e = Math.ceil(y0); e <= Math.floor(y1); e++) out.push(Math.pow(10, e));
          if (out.length < 3) {
            // Меньше двух декад: добавляем промежуточные 2 и 5 внутри каждой.
            out = [];
            for (e = Math.floor(y0); e <= Math.ceil(y1); e++) {
              [1, 2, 5].forEach(function (mnt) {
                var v = mnt * Math.pow(10, e);
                if (ty(v) >= y0 - 1e-9 && ty(v) <= y1 + 1e-9) out.push(v);
              });
            }
          }
          return out.length ? out : [ys[0], ys[1]];
        })()
      : niceTicks(ys[0], ys[1], 5);
    var xTicks = cfg.xTicks || niceTicks(xs[0], xs[1], 5);

    yTicks.forEach(function (v) {
      var y = py(v);
      if (y < m.t - 1 || y > H - m.b + 1) return;
      svg.appendChild(el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: P.line, "stroke-width": 1 }));
      var t = el("text", { x: m.l - 8, y: y + 3.5, "text-anchor": "end",
        fill: P.ink3, "font-size": 10.5, "font-family": "var(--mono)" });
      t.textContent = (cfg.yFormat || function (d) { return log ? sci(d, 0) : String(+d.toFixed(6)); })(v);
      svg.appendChild(t);
    });

    xTicks.forEach(function (v) {
      var x = px(v);
      if (x < m.l - 1 || x > W - m.r + 1) return;
      svg.appendChild(el("line", { x1: x, x2: x, y1: m.t, y2: H - m.b, stroke: P.line, "stroke-width": 1 }));
      var t = el("text", { x: x, y: H - m.b + 16, "text-anchor": "middle",
        fill: P.ink3, "font-size": 10.5, "font-family": "var(--mono)" });
      t.textContent = (cfg.xFormat || function (d) { return String(+d.toFixed(6)); })(v);
      svg.appendChild(t);
    });

    // базовая линия
    svg.appendChild(el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b,
      stroke: P.lineStrong, "stroke-width": 1 }));

    if (cfg.yLabel) {
      var yl = el("text", { x: 0, y: 10, fill: P.ink3, "font-size": 10.5,
        "font-family": "var(--mono)" });
      yl.textContent = cfg.yLabel;
      svg.appendChild(yl);
    }
    if (cfg.xLabel) {
      var xl = el("text", { x: m.l + (W - m.l - m.r) / 2, y: H - 6, "text-anchor": "middle",
        fill: P.ink3, "font-size": 10.5, "font-family": "var(--mono)" });
      xl.textContent = cfg.xLabel;
      svg.appendChild(xl);
    }

    // --- сами линии -------------------------------------------------------
    cfg.series.forEach(function (s) {
      if (!s.points.length) return;
      var d = "";
      var open = false;
      s.points.forEach(function (p) {
        var vy = p[1];
        if (!isFinite(vy) || (log && vy <= 0)) { open = false; return; }
        var X = px(p[0]);
        var Y = py(vy);
        if (Y < m.t - 40 || Y > H - m.b + 40) { open = false; return; }
        d += (open ? "L" : "M") + X.toFixed(2) + " " + Y.toFixed(2) + " ";
        open = true;
      });
      svg.appendChild(el("path", {
        d: d, fill: "none", stroke: s.color, "stroke-width": s.width || 2,
        "stroke-dasharray": s.dash || null, "stroke-linecap": "round",
        "stroke-linejoin": "round"
      }));
    });

    // --- прямые подписи у конца линий ------------------------------------
    if (cfg.directLabels) {
      var labels = [];
      cfg.series.forEach(function (s) {
        var last = null;
        for (var i = s.points.length - 1; i >= 0; i--) {
          var v = s.points[i][1];
          if (isFinite(v) && (!log || v > 0)) { last = s.points[i]; break; }
        }
        if (last) labels.push({ y: py(last[1]), name: s.name, color: s.color });
      });
      // Разводим подписи по вертикали, иначе близкие кривые слипаются.
      labels.sort(function (a, b) { return a.y - b.y; });
      var MIN_GAP = 13;
      for (var i = 1; i < labels.length; i++) {
        if (labels[i].y - labels[i - 1].y < MIN_GAP) labels[i].y = labels[i - 1].y + MIN_GAP;
      }
      var overflow = labels.length ? labels[labels.length - 1].y - (H - m.b - 2) : 0;
      if (overflow > 0) labels.forEach(function (l) { l.y -= overflow; });
      labels.forEach(function (l) {
        var t = el("text", { x: W - m.r - 2, y: Math.max(m.t + 8, l.y) + 3, fill: l.color,
          "font-size": 10.5, "font-weight": 600, "font-family": "var(--mono)",
          "text-anchor": "end" });
        t.textContent = l.name;
        svg.appendChild(t);
      });
    }

    // --- отдельные маркеры ------------------------------------------------
    (cfg.markers || []).forEach(function (mk) {
      svg.appendChild(el("circle", {
        cx: px(mk.x), cy: py(mk.y), r: 5, fill: mk.color,
        stroke: P.panel, "stroke-width": 2
      }));
    });
    if (cfg.vline !== undefined) {
      svg.appendChild(el("line", {
        x1: px(cfg.vline), x2: px(cfg.vline), y1: m.t, y2: H - m.b,
        stroke: P.ink3, "stroke-width": 1, "stroke-dasharray": "3 3"
      }));
    }

    container.appendChild(svg);

    // --- слой наведения ---------------------------------------------------
    if (cfg.hover !== false && cfg.series.length) {
      var tip = document.createElement("div");
      tip.className = "tip";
      container.appendChild(tip);

      var cross = el("line", { y1: m.t, y2: H - m.b, stroke: P.ink3,
        "stroke-width": 1, "stroke-dasharray": "2 3", opacity: 0 });
      svg.appendChild(cross);
      var dots = cfg.series.map(function (s) {
        var c = el("circle", { r: 3.5, fill: s.color, stroke: P.panel, "stroke-width": 2, opacity: 0 });
        svg.appendChild(c);
        return c;
      });

      var hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b,
        fill: "transparent", style: "cursor:crosshair" });
      svg.appendChild(hit);

      var hide = function () {
        tip.dataset.show = "0";
        cross.setAttribute("opacity", 0);
        dots.forEach(function (d) { d.setAttribute("opacity", 0); });
      };
      var move = function (ev) {
        var r = svg.getBoundingClientRect();
        var sx = ((ev.clientX - r.left) / r.width) * W;
        var xv = xs[0] + ((sx - m.l) / (W - m.l - m.r)) * (xs[1] - xs[0]);
        var rows = [];
        var anchorY = null;
        cfg.series.forEach(function (s, i) {
          var best = null;
          var bd = Infinity;
          for (var k = 0; k < s.points.length; k++) {
            var dd = Math.abs(s.points[k][0] - xv);
            if (dd < bd) { bd = dd; best = s.points[k]; }
          }
          if (!best || !isFinite(best[1]) || (log && best[1] <= 0)) {
            dots[i].setAttribute("opacity", 0);
            return;
          }
          dots[i].setAttribute("cx", px(best[0]));
          dots[i].setAttribute("cy", py(best[1]));
          dots[i].setAttribute("opacity", 1);
          if (anchorY === null) anchorY = py(best[1]);
          rows.push('<span class="tip__k">' + s.name + "</span> " +
            (cfg.tipFormat ? cfg.tipFormat(best[1]) : String(+best[1].toPrecision(5))));
        });
        if (!rows.length) { hide(); return; }
        cross.setAttribute("x1", px(xv));
        cross.setAttribute("x2", px(xv));
        cross.setAttribute("opacity", 1);
        tip.innerHTML = '<div class="tip__k">' +
          (cfg.tipX ? cfg.tipX(xv) : String(+xv.toPrecision(4))) + "</div>" + rows.join("<br>");
        tip.dataset.show = "1";
        var cw = container.clientWidth;
        var left = (px(xv) / W) * cw + 12;
        if (left + tip.offsetWidth > cw) left = (px(xv) / W) * cw - tip.offsetWidth - 12;
        tip.style.left = Math.max(0, left) + "px";
        tip.style.top = Math.max(0, ((anchorY || m.t) / H) * container.clientHeight - 10) + "px";
      };
      hit.addEventListener("pointermove", move);
      hit.addEventListener("pointerleave", hide);
    }
    return svg;
  }

  function legend(container, items) {
    container.textContent = "";
    items.forEach(function (it) {
      var s = document.createElement("span");
      s.className = "legend__item";
      var sw = document.createElement("span");
      if (it.dash) {
        sw.className = "legend__swatch legend__swatch--dash";
        sw.style.color = it.color;
      } else {
        sw.className = "legend__swatch";
        sw.style.background = it.color;
      }
      s.appendChild(sw);
      s.appendChild(document.createTextNode(it.name));
      container.appendChild(s);
    });
  }

  // =====================================================================
  // Карта решений: тепловая подложка + линия уровня 0.5 + точки выборки
  // =====================================================================
  var GRID = 100;

  function domainOf(ds) {
    var xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    ds.points.forEach(function (p) {
      if (p[0] < xmin) xmin = p[0];
      if (p[0] > xmax) xmax = p[0];
      if (p[1] < ymin) ymin = p[1];
      if (p[1] > ymax) ymax = p[1];
    });
    var padx = (xmax - xmin) * 0.18 + 0.25;
    var pady = (ymax - ymin) * 0.18 + 0.25;
    return [xmin - padx, xmax + padx, ymin - pady, ymax + pady];
  }

  function renderBoundary(canvas, model, ds, dom) {
    var ctx = canvas.getContext("2d");
    var w = canvas.width;
    var h = canvas.height;
    var d = dom || domainOf(ds);

    // --- сетка значений --------------------------------------------------
    var grid = new Float64Array(GRID * GRID);
    var probe = new Array(GRID);
    for (var r = 0; r < GRID; r++) {
      var gy = d[3] - ((r + 0.5) / GRID) * (d[3] - d[2]);
      for (var c = 0; c < GRID; c++) {
        probe[c] = [d[0] + ((c + 0.5) / GRID) * (d[1] - d[0]), gy];
      }
      var out = model.forward(probe);
      for (var c2 = 0; c2 < GRID; c2++) grid[r * GRID + c2] = out[c2];
    }

    // --- подложка --------------------------------------------------------
    var off = renderBoundary._off || (renderBoundary._off = document.createElement("canvas"));
    off.width = GRID;
    off.height = GRID;
    var octx = off.getContext("2d");
    var img = octx.createImageData(GRID, GRID);
    for (var i = 0; i < GRID * GRID; i++) {
      var rgb = divergingRgb(grid[i]);
      img.data[i * 4] = rgb[0];
      img.data[i * 4 + 1] = rgb[1];
      img.data[i * 4 + 2] = rgb[2];
      img.data[i * 4 + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(off, 0, 0, GRID, GRID, 0, 0, w, h);

    // --- линия уровня 0.5 методом marching squares ------------------------
    var sx = function (gx) { return (gx / (GRID - 1)) * w; };
    var sy = function (gy) { return (gy / (GRID - 1)) * h; };
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = P.dark ? "#E9F1EF" : "#0F1917";
    var LEVEL = 0.5;
    for (var rr = 0; rr < GRID - 1; rr++) {
      for (var cc = 0; cc < GRID - 1; cc++) {
        var v00 = grid[rr * GRID + cc];
        var v10 = grid[rr * GRID + cc + 1];
        var v01 = grid[(rr + 1) * GRID + cc];
        var v11 = grid[(rr + 1) * GRID + cc + 1];
        var pts = [];
        var edge = function (a, b, ax, ay, bx, by) {
          if ((a - LEVEL) * (b - LEVEL) < 0) {
            var t = (LEVEL - a) / (b - a);
            pts.push([ax + (bx - ax) * t, ay + (by - ay) * t]);
          }
        };
        edge(v00, v10, cc, rr, cc + 1, rr);
        edge(v10, v11, cc + 1, rr, cc + 1, rr + 1);
        edge(v01, v11, cc, rr + 1, cc + 1, rr + 1);
        edge(v00, v01, cc, rr, cc, rr + 1);
        if (pts.length >= 2) {
          ctx.moveTo(sx(pts[0][0]), sy(pts[0][1]));
          ctx.lineTo(sx(pts[1][0]), sy(pts[1][1]));
        }
      }
    }
    ctx.stroke();

    // --- точки выборки ----------------------------------------------------
    var tx = function (x) { return ((x - d[0]) / (d[1] - d[0])) * w; };
    var ty = function (y) { return h - ((y - d[2]) / (d[3] - d[2])) * h; };
    var big = ds.n <= 8;
    var R = big ? 11 : (ds.n > 300 ? 4.8 : 5.8);
    for (var k = 0; k < ds.n; k++) {
      var X = tx(ds.points[k][0]);
      var Y = ty(ds.points[k][1]);
      var cls = ds.labels[k] > 0.5 ? 1 : 0;
      ctx.beginPath();
      if (cls === 0) {
        ctx.arc(X, Y, R, 0, Math.PI * 2);
      } else {
        // треугольник: форма дублирует цвет, чтобы класс читался без него
        ctx.moveTo(X, Y - R * 1.12);
        ctx.lineTo(X + R, Y + R * 0.78);
        ctx.lineTo(X - R, Y + R * 0.78);
        ctx.closePath();
      }
      ctx.fillStyle = cls === 0 ? P.c0 : P.c1;
      ctx.fill();
      // Кольцо цвета подложки отделяет марки друг от друга; на плотных
      // наборах оно тоньше, иначе съедает сам цвет точки.
      ctx.lineWidth = big ? 2.5 : (ds.n > 300 ? 1.1 : 1.6);
      ctx.strokeStyle = P.panel;
      ctx.stroke();
    }
  }

  /** Только точки выборки: показывает саму задачу, без решения. */
  function renderScatter(canvas, ds) {
    var ctx = canvas.getContext("2d");
    var w = canvas.width;
    var h = canvas.height;
    var d = domainOf(ds);

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = P.panel;
    ctx.fillRect(0, 0, w, h);

    var tx = function (x) { return ((x - d[0]) / (d[1] - d[0])) * w; };
    var ty = function (y) { return h - ((y - d[2]) / (d[3] - d[2])) * h; };

    // Оси через начало координат -- только чтобы задать масштаб взгляду.
    ctx.strokeStyle = P.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(tx(0), 0); ctx.lineTo(tx(0), h);
    ctx.moveTo(0, ty(0)); ctx.lineTo(w, ty(0));
    ctx.stroke();

    var R = ds.n > 300 ? 4.4 : 6;
    for (var k = 0; k < ds.n; k++) {
      var X = tx(ds.points[k][0]);
      var Y = ty(ds.points[k][1]);
      var cls = ds.labels[k] > 0.5 ? 1 : 0;
      ctx.beginPath();
      if (cls === 0) {
        ctx.arc(X, Y, R, 0, Math.PI * 2);
      } else {
        ctx.moveTo(X, Y - R * 1.12);
        ctx.lineTo(X + R, Y + R * 0.78);
        ctx.lineTo(X - R, Y + R * 0.78);
        ctx.closePath();
      }
      ctx.fillStyle = cls === 0 ? P.c0 : P.c1;
      ctx.fill();
      ctx.lineWidth = 1.1;
      ctx.strokeStyle = P.panel;
      ctx.stroke();
    }
  }

  function fitCanvas(canvas) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var size = Math.round(canvas.clientWidth * dpr);
    if (size > 0 && canvas.width !== size) {
      canvas.width = size;
      canvas.height = size;
    }
  }

  // =====================================================================
  // Данные
  // =====================================================================
  var DS = {};
  Object.keys(DATA.datasets).forEach(function (k) { DS[k] = E.loadDataset(DATA.datasets[k]); });

  // =====================================================================
  // Станция 01 -- прямой проход
  // =====================================================================
  var s1 = { w1: 1.2, w2: -0.8, b: 0.3, x1: 0.9, x2: -1.4, act: "tanh" };

  function drawS1() {
    var a = E.ACTIVATIONS[s1.act];
    var z = s1.w1 * s1.x1 + s1.w2 * s1.x2 + s1.b;
    var out = a.fn(z);

    $("s1-w1-v").textContent = fx(s1.w1, 2);
    $("s1-w2-v").textContent = fx(s1.w2, 2);
    $("s1-b-v").textContent = fx(s1.b, 2);
    $("s1-x1-v").textContent = fx(s1.x1, 2);
    $("s1-x2-v").textContent = fx(s1.x2, 2);

    $("s1-trace").innerHTML = [
      row("w₁·x₁", fx(s1.w1, 2) + " · " + fx(s1.x1, 2) + " = " + fx(s1.w1 * s1.x1, 4)),
      row("w₂·x₂", fx(s1.w2, 2) + " · " + fx(s1.x2, 2) + " = " + fx(s1.w2 * s1.x2, 4)),
      row("смещение b", fx(s1.b, 4)),
      row("z", fx(z, 6), true),
      row("a = " + a.label + "(z)", fx(out, 6), true)
    ].join("");

    var pts = [];
    for (var t = -6; t <= 6.0001; t += 0.04) pts.push([t, a.fn(t)]);
    var lo = Math.min(-1.2, a.range[0]);
    var hi = Math.max(1.2, Math.min(a.range[1], 6));
    lineChart($("s1-chart"), {
      height: 190,
      series: [{ name: a.label, color: P.c0, points: pts }],
      xDomain: [-6, 6],
      yDomain: [lo, hi],
      markers: [{ x: Math.max(-6, Math.min(6, z)), y: out, color: P.c1 }],
      vline: Math.max(-6, Math.min(6, z)),
      xLabel: "z",
      hover: false,
      ariaLabel: "Кривая активации " + a.label + " с отмеченной текущей точкой"
    });
  }
  function row(k, v, strong) {
    return '<div class="trace__row' + (strong ? " trace__row--total" : "") +
      '"><span>' + k + "</span><b>" + v + "</b></div>";
  }

  ["w1", "w2", "b", "x1", "x2"].forEach(function (key) {
    $("s1-" + key).addEventListener("input", function (e) {
      s1[key] = parseFloat(e.target.value);
      drawS1();
    });
  });
  segmented("s1-act", "act", function (v) { s1.act = v; drawS1(); });

  function segmented(id, attr, onPick) {
    var root = $(id);
    root.addEventListener("click", function (ev) {
      var btn = ev.target.closest("button");
      if (!btn) return;
      Array.prototype.forEach.call(root.querySelectorAll("button"), function (b) {
        b.setAttribute("aria-pressed", String(b === btn));
      });
      onPick(btn.dataset[attr]);
    });
  }

  // =====================================================================
  // Станция 02 -- активации варианта
  // =====================================================================
  var s2z = 1.0;

  function drawS2() {
    var names = ["tanh", "relu"];
    var colors = [P.c0, P.c1];
    var series = [];
    names.forEach(function (n, i) {
      var a = E.ACTIVATIONS[n];
      var f = [], g = [];
      for (var t = -6; t <= 6.0001; t += 0.03) {
        f.push([t, a.fn(t)]);
        g.push([t, a.prime(t)]);
      }
      series.push({ name: a.label, color: colors[i], points: f, width: 2 });
      series.push({ name: a.label + "′", color: colors[i], points: g, width: 2, dash: "5 4" });
    });

    lineChart($("s2-chart"), {
      height: 300,
      series: series,
      xDomain: [-6, 6],
      yDomain: [-1.4, 6],
      vline: s2z,
      xLabel: "z",
      yLabel: "φ(z), φ′(z)",
      tipX: function (x) { return "z = " + x.toFixed(2); },
      tipFormat: function (v) { return v.toFixed(4); },
      ariaLabel: "Графики tanh и ReLU и их производных на отрезке от минус шести до шести"
    });

    legend($("s2-legend"), [
      { name: "tanh", color: P.c0 },
      { name: "tanh′", color: P.c0, dash: true },
      { name: "ReLU", color: P.c1 },
      { name: "ReLU′", color: P.c1, dash: true }
    ]);

    $("s2-z-v").textContent = fx(s2z, 2);
    $("s2-table").innerHTML = names.map(function (n) {
      var a = E.ACTIVATIONS[n];
      return "<tr><td>" + a.label + "</td><td>" + a.fn(s2z).toFixed(6) +
        "</td><td>" + a.prime(s2z).toFixed(6) + "</td></tr>";
    }).join("");
  }
  $("s2-z").addEventListener("input", function (e) { s2z = parseFloat(e.target.value); drawS2(); });

  // =====================================================================
  // Станция 03 -- функция потерь
  // =====================================================================
  var s3 = { a: 0.8, y: 1 };

  function drawS3() {
    var bce = [], mse = [];
    for (var t = 0.004; t <= 0.996; t += 0.004) {
      var av = [t];
      var yv = [s3.y];
      bce.push([t, E.LOSSES.bce.value(av, yv)]);
      mse.push([t, E.LOSSES.mse.value(av, yv)]);
    }
    var lb = E.LOSSES.bce.value([s3.a], [s3.y]);
    var lm = E.LOSSES.mse.value([s3.a], [s3.y]);
    var gb = E.LOSSES.bce.grad([s3.a], [s3.y])[0];
    var gm = E.LOSSES.mse.grad([s3.a], [s3.y])[0];

    lineChart($("s3-chart"), {
      height: 280,
      series: [
        { name: "BCE", color: P.c0, points: bce, width: 2.4 },
        { name: "MSE", color: P.c1, points: mse, width: 2, dash: "5 4" }
      ],
      xDomain: [0, 1],
      yDomain: [0, 5],
      vline: s3.a,
      markers: [{ x: s3.a, y: Math.min(lb, 5), color: P.c0 }],
      xLabel: "выход a",
      yLabel: "L",
      tipX: function (x) { return "a = " + x.toFixed(3); },
      tipFormat: function (v) { return v.toFixed(4); },
      ariaLabel: "Значение BCE и MSE в зависимости от выхода нейрона"
    });
    legend($("s3-legend"), [
      { name: "BCE при y = " + s3.y, color: P.c0 },
      { name: "MSE при y = " + s3.y, color: P.c1, dash: true }
    ]);

    $("s3-a-v").textContent = fx(s3.a, 2);
    $("s3-kv").innerHTML =
      kv("BCE", lb.toFixed(6)) +
      kv("∂BCE/∂a", gb.toFixed(6)) +
      kv("MSE", lm.toFixed(6)) +
      kv("∂MSE/∂a", gm.toFixed(6)) +
      kv("(a − y)/N", (s3.a - s3.y).toFixed(6));
  }
  function kv(k, v) { return "<dt>" + k + "</dt><dd>" + v + "</dd>"; }

  $("s3-a").addEventListener("input", function (e) { s3.a = parseFloat(e.target.value); drawS3(); });
  segmented("s3-y", "y", function (v) { s3.y = parseInt(v, 10); drawS3(); });

  // =====================================================================
  // Станция 04 -- одиночный нейрон
  // =====================================================================
  var s4 = { seed: 0, net: null, hist: [], loss: 0 };

  function runS4() {
    var ds = DS.circles;
    var n = new E.Neuron({ nInputs: 2, activation: "sigmoid", loss: "bce", seed: s4.seed });
    s4.hist = [];
    for (var e = 0; e < 3000; e++) {
      if (e % 15 === 0) s4.hist.push([e, n.loss(ds.points, ds.labels)]);
      n.step(ds.points, ds.labels, 0.5);
    }
    s4.loss = n.loss(ds.points, ds.labels);
    s4.hist.push([3000, s4.loss]);
    s4.net = n;
    drawS4();
  }

  function drawS4() {
    if (!s4.net) return;
    var ds = DS.circles;
    fitCanvas($("s4-canvas"));
    renderBoundary($("s4-canvas"), s4.net, ds);
    $("s4-scale").style.background = scaleGradient();

    lineChart($("s4-chart"), {
      height: 220,
      series: [{ name: "BCE", color: P.c0, points: s4.hist, width: 2 }],
      xDomain: [0, 3000],
      yDomain: [0.5, 0.8],
      xLabel: "эпоха",
      yLabel: "BCE",
      tipX: function (x) { return "эпоха " + Math.round(x); },
      tipFormat: function (v) { return v.toFixed(6); },
      ariaLabel: "Кривая потерь одиночного нейрона выходит на плато"
    });

    $("s4-kv").innerHTML =
      kv("инициализация", "seed " + s4.seed) +
      kv("потери в конце", s4.loss.toFixed(6)) +
      kv("ln 2", Math.LN2.toFixed(6)) +
      kv("точность", pct(s4.net.accuracy(ds.points, ds.labels))) +
      kv("эпох", "3000, η = 0.5");
  }

  // Каждое нажатие -- новое начальное приближение. Результат один и тот же,
  // и это как раз то, что нужно показать: дело не в неудачном старте.
  $("s4-run").addEventListener("click", function () {
    s4.seed = (s4.seed + 1) % 12;
    runS4();
  });

  // =====================================================================
  // Станция 05 -- сеть со скрытым слоем
  // =====================================================================
  var s5 = { ds: "circles", act: "tanh", H: 8, lr: 1, seed: 3, net: null, hist: [], playing: false, raf: 0 };
  var EPOCHS_PER_FRAME = 15;
  var MAX_EPOCHS = 6000;

  function buildS5() {
    s5.net = new E.MLP({
      nInputs: 2, nHidden: s5.H, hiddenActivation: s5.act,
      outputActivation: "sigmoid", loss: "bce", seed: s5.seed
    });
    s5.hist = [];
    recordS5();
  }
  function recordS5() {
    var ds = DS[s5.ds];
    s5.hist.push([s5.net.epoch, s5.net.loss(ds.points, ds.labels)]);
    if (s5.hist.length > 900) s5.hist = s5.hist.filter(function (_, i) { return i % 2 === 0; });
  }

  function drawS5() {
    var ds = DS[s5.ds];
    fitCanvas($("s5-canvas"));
    renderBoundary($("s5-canvas"), s5.net, ds);
    $("s5-scale").style.background = scaleGradient();

    var maxE = Math.max(200, s5.net.epoch);
    var losses = s5.hist.map(function (p) { return p[1]; });
    var lo = Math.max(1e-4, Math.min.apply(null, losses) * 0.7);
    var hi = Math.max.apply(null, losses) * 1.15;

    lineChart($("s5-chart"), {
      height: 220,
      series: [{ name: "BCE", color: P.c0, points: s5.hist, width: 2 }],
      xDomain: [0, maxE],
      yDomain: [lo, hi],
      yScale: "log",
      xLabel: "эпоха",
      yLabel: "BCE",
      tipX: function (x) { return "эпоха " + Math.round(x); },
      tipFormat: function (v) { return v.toFixed(6); },
      ariaLabel: "Кривая потерь обучаемой сети"
    });

    var acc = s5.net.accuracy(ds.points, ds.labels);
    var loss = s5.hist[s5.hist.length - 1][1];
    $("s5-kv").innerHTML =
      kv("эпоха", String(s5.net.epoch)) +
      kv("потери", loss.toFixed(6)) +
      kv("точность", pct(acc)) +
      kv("параметров", String(s5.net.getParams().length)) +
      kv("объектов", String(ds.n));
    $("s5-h-v").textContent = String(s5.H);
    $("s5-seed-v").textContent = String(s5.seed);

    var note = DS[s5.ds].note + " ";
    if (s5.H === 2) {
      note += "При H = 2 ландшафт потерь содержит плато: часть начальных приближений " +
        "застревает, и точность останавливается около 50 %. Переберите seed, чтобы увидеть это.";
    } else if (s5.act === "relu" && loss > 0.3 && s5.net.epoch > 400) {
      note += "ReLU в скрытом слое может «умереть»: если все предактивации ушли в " +
        "отрицательную область, градиент строго нулевой и обучение стоит.";
    } else if (acc === 1) {
      note += "Сеть разделила выборку полностью.";
    } else {
      note += "Скрытый слой строит новое представление, в котором классы становятся линейно разделимыми.";
    }
    $("s5-note").textContent = note;

    $("rd-acc").textContent = pct(acc);
  }

  function frame() {
    if (!s5.playing) return;
    var ds = DS[s5.ds];
    for (var i = 0; i < EPOCHS_PER_FRAME && s5.net.epoch < MAX_EPOCHS; i++) {
      s5.net.step(ds.points, ds.labels, s5.lr);
    }
    recordS5();
    drawS5();
    if (s5.net.epoch >= MAX_EPOCHS) { setPlaying(false); return; }
    s5.raf = requestAnimationFrame(frame);
  }
  function setPlaying(on) {
    s5.playing = on;
    $("s5-play").textContent = on ? "Пауза" : "Обучать";
    if (on) s5.raf = requestAnimationFrame(frame);
    else cancelAnimationFrame(s5.raf);
  }

  $("s5-play").addEventListener("click", function () {
    if (s5.net.epoch >= MAX_EPOCHS) { buildS5(); drawS5(); }
    setPlaying(!s5.playing);
  });
  $("s5-reset").addEventListener("click", function () {
    setPlaying(false);
    buildS5();
    drawS5();
  });
  $("s5-h").addEventListener("input", function (e) {
    s5.H = parseInt(e.target.value, 10);
    setPlaying(false); buildS5(); drawS5();
  });
  $("s5-seed").addEventListener("input", function (e) {
    s5.seed = parseInt(e.target.value, 10);
    setPlaying(false); buildS5(); drawS5();
  });
  segmented("s5-data", "ds", function (v) {
    s5.ds = v; setPlaying(false); buildS5(); drawS5();
  });
  segmented("s5-act", "act", function (v) {
    s5.act = v; setPlaying(false); buildS5(); drawS5();
  });
  segmented("s5-lr", "lr", function (v) { s5.lr = parseFloat(v); });

  // =====================================================================
  // Станция 06 -- проверка градиента
  // =====================================================================
  function runS6(useKink) {
    var ds = DS[s5.ds];
    var net;
    if (useKink) {
      // ReLU + точка (0,0) при нулевых смещениях: z = 0 в точности.
      net = new E.MLP({
        nInputs: 2, nHidden: 4, hiddenActivation: "relu",
        outputActivation: "sigmoid", loss: "bce", seed: 0
      });
      ds = DS.xor;
    } else {
      net = s5.net;
    }
    var sub = subsample(ds, 48);
    var res = E.gradientCheck(net, sub.points, sub.labels, 1e-5, 1e-6);

    $("s6-verdict").innerHTML = '<span class="verdict verdict--' + (res.passed ? "pass" : "fail") +
      '"><span class="verdict__icon">' + (res.passed ? "✓" : "✗") + "</span>" +
      (res.passed ? "ПРОЙДЕНО" : "НЕ ПРОЙДЕНО") + " · порог 1·10⁻⁶</span>";

    $("s6-kv").innerHTML =
      kv("max отн. ошибка", sci(res.maxRel, 2)) +
      kv("max |разность|", sci(res.maxAbs, 2)) +
      kv("параметров", String(res.rows.length)) +
      kv("шаг ε", "1·10⁻⁵") +
      kv("точек излома", String(res.kinks));

    var rows = res.rows.slice().sort(function (a, b) { return b.rel - a.rel; }).slice(0, 10);
    $("s6-table").innerHTML = rows.map(function (r) {
      return "<tr><td>θ[" + r.k + "]</td><td>" + r.analytic.toFixed(10) +
        "</td><td>" + r.numeric.toFixed(10) + "</td><td>" + sci(r.rel, 2) + "</td></tr>";
    }).join("");

    $("s6-note").innerHTML = useKink
      ? "Смещения инициализированы нулём, а набор XOR содержит точку (0, 0): значит " +
        "<b>z = 0 в точности</b>. В этой точке производная ReLU не существует, конечная " +
        "разность перешагивает излом, и проверка честно падает. Это ограничение метода " +
        "конечных разностей, а не ошибка в формулах — сдвиньте параметры от нуля, и " +
        "расхождение вернётся к 10⁻¹⁰."
      : "Показаны 10 компонент с наибольшей относительной ошибкой из " + res.rows.length +
        ". Градиент считается по формулам, выведенным вручную, и сверяется с центральной " +
        "разностью — никакого автоматического дифференцирования.";

    if (!useKink) $("rd-grad").textContent = sci(res.maxRel, 1);
    return res;
  }
  function subsample(ds, k) {
    if (ds.n <= k) return ds;
    var pts = [], lab = new Float64Array(k);
    var stride = ds.n / k;
    for (var i = 0; i < k; i++) {
      var idx = Math.floor(i * stride);
      pts.push(ds.points[idx]);
      lab[i] = ds.labels[idx];
    }
    return { points: pts, labels: lab, n: k };
  }
  $("s6-run").addEventListener("click", function () { runS6(false); });
  $("s6-kink").addEventListener("click", function () { runS6(true); });

  // =====================================================================
  // Станция 07 -- скорость обучения
  // =====================================================================
  var LRS = [0.001, 0.01, 0.1, 1];
  var s7runs = [];
  var s7conf = { H: 8, act: "tanh", ds: "circles" };

  function runS7() {
    // Конфигурация берётся со станции 05: кнопка «Пересчитать» прогоняет
    // выбранные там набор данных, активацию и ширину скрытого слоя.
    var ds = DS[s5.ds];
    s7conf = { H: s5.H, act: s5.act, ds: s5.ds };
    s7runs = LRS.map(function (lr, i) {
      var net = new E.MLP({
        nInputs: 2, nHidden: s7conf.H, hiddenActivation: s7conf.act,
        outputActivation: "sigmoid", loss: "bce", seed: 3
      });
      var pts = [];
      var reached = -1;
      for (var e = 0; e < 2000; e++) {
        if (e % 10 === 0) {
          var L = net.loss(ds.points, ds.labels);
          pts.push([e, L]);
          if (reached < 0 && L < 0.1) reached = e;
        }
        net.step(ds.points, ds.labels, lr);
      }
      var last = net.loss(ds.points, ds.labels);
      pts.push([2000, last]);
      return {
        lr: lr, points: pts, color: P.ramp[i],
        last: last, acc: net.accuracy(ds.points, ds.labels), reached: reached
      };
    });
    drawS7();
  }

  function drawS7() {
    s7runs.forEach(function (r, i) { r.color = P.ramp[i]; });
    var all = [];
    s7runs.forEach(function (r) { r.points.forEach(function (p) { all.push(p[1]); }); });

    lineChart($("s7-chart"), {
      height: 300,
      series: s7runs.map(function (r) {
        return { name: "η " + r.lr, color: r.color, points: r.points, width: 2 };
      }),
      xDomain: [0, 2000],
      yDomain: [Math.max(1e-3, Math.min.apply(null, all) * 0.7), Math.max.apply(null, all) * 1.1],
      yScale: "log",
      xLabel: "эпоха",
      yLabel: "BCE",
      directLabels: true,
      tipX: function (x) { return "эпоха " + Math.round(x); },
      tipFormat: function (v) { return v.toFixed(6); },
      ariaLabel: "Кривые потерь для четырёх значений скорости обучения"
    });

    legend($("s7-legend"), s7runs.map(function (r) {
      return { name: "η = " + r.lr, color: r.color };
    }));

    $("s7-cap").textContent = "2000 эпох · набор " + DS[s7conf.ds].title +
      " · сеть 2-" + s7conf.H + "-1 · скрытый слой " +
      E.ACTIVATIONS[s7conf.act].label;

    $("s7-table").innerHTML = s7runs.map(function (r) {
      return "<tr><td>" + r.lr + "</td><td>" + r.last.toFixed(6) + "</td><td>" +
        pct(r.acc) + "</td><td>" + (r.reached >= 0 ? r.reached : "—") + "</td></tr>";
    }).join("");
  }
  $("s7-run").addEventListener("click", runS7);

  // =====================================================================
  // Сверка с NumPy
  // =====================================================================
  function verify() {
    var v = E.verifyAgainstNumpy(DATA.fixture, DS.circles);
    var worst = Math.max(v.lossDiff, v.maxGradDiff, v.maxFwdDiff);
    $("rd-numpy").textContent = sci(worst, 1);
    $("colophon-verify").innerHTML =
      "Сверка на фикстуре из " + v.nParams + " параметров, посчитанной NumPy " +
      DATA.numpy_version + ": потери расходятся на <code>" + sci(v.lossDiff, 1) +
      "</code>, прямой проход — на <code>" + sci(v.maxFwdDiff, 1) +
      "</code>, градиент — на <code>" + sci(v.maxGradDiff, 1) +
      "</code>. Это уровень машинного эпсилон float64, то есть браузер и NumPy " +
      "считают одно и то же.";
    return v;
  }

  // =====================================================================
  // Запуск и реакция на смену темы
  // =====================================================================
  function drawOverview() {
    fitCanvas($("ov-canvas"));
    renderScatter($("ov-canvas"), DS.circles);
    var ref = new E.MLP({
      nInputs: 2, nHidden: 8, hiddenActivation: "tanh",
      outputActivation: "sigmoid", loss: "bce", seed: 0
    });
    $("ov-params").textContent = String(ref.getParams().length);
  }

  function redrawAll() {
    P = palette();
    drawOverview();
    drawS1();
    drawS2();
    drawS3();
    drawS4();
    drawS5();
    if (s7runs.length) drawS7();
  }

  function boot() {
    verify();
    drawOverview();
    drawS1();
    drawS2();
    drawS3();
    buildS5();
    // Сеть открывается уже обученной: страница должна показывать результат
    // сразу, а не пустую заготовку.
    var ds = DS.circles;
    for (var e = 0; e < 1500; e++) {
      s5.net.step(ds.points, ds.labels, 1);
      if (e % 25 === 0) recordS5();
    }
    recordS5();
    drawS5();
    runS4();
    runS6(false);
    runS7();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { requestAnimationFrame(boot); });
  } else {
    requestAnimationFrame(boot);
  }

  var mq = window.matchMedia("(prefers-color-scheme: dark)");
  (mq.addEventListener ? mq.addEventListener.bind(mq, "change") : mq.addListener.bind(mq))(redrawAll);
  new MutationObserver(redrawAll).observe(document.documentElement, {
    attributes: true, attributeFilter: ["data-theme"]
  });

  var rt;
  window.addEventListener("resize", function () {
    clearTimeout(rt);
    rt = setTimeout(function () {
      fitCanvas($("ov-canvas"));
      fitCanvas($("s4-canvas"));
      fitCanvas($("s5-canvas"));
      redrawAll();
    }, 160);
  });
})();
