# -*- coding: utf-8 -*-
"""Функциональные тесты веб-интерфейса: каждый орган управления нажимается,
и проверяется, что состояние страницы действительно изменилось.

Требуется Playwright и браузер Chromium:

    pip install playwright && playwright install chromium
    python -m unittest tests.test_ui -v

Без них тест пропускается, поэтому основной набор (tests/test_lab1.py)
остаётся запускаемым на голом Python с NumPy.
"""

from __future__ import annotations

import asyncio
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "public", "index.html")
URL = "file://" + INDEX

#: Возможные места, где лежит заранее установленный Chromium.
CHROMIUM_HINTS = ["/opt/pw-browsers/chromium"]


def _playwright_available():
    try:
        import playwright.async_api  # noqa: F401
    except ImportError:
        return False
    return True


def _executable_path():
    for path in CHROMIUM_HINTS:
        if os.path.exists(path):
            return path
    return None


class Probe:
    """Накопитель результатов: собирает все сбои, а не падает на первом."""

    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append((bool(ok), name, detail))

    @property
    def failures(self):
        return [(n, d) for ok, n, d in self.rows if not ok]

    def report(self):
        lines = [f"  {'OK  ' if ok else 'СБОЙ'}  {n}" + (f" — {d}" if d else "")
                 for ok, n, d in self.rows]
        lines.append(f"  пройдено {len(self.rows) - len(self.failures)} из {len(self.rows)}")
        return "\n".join(lines)


async def _exercise(page, probe):
    from_kv = "() => document.querySelector('#s5-kv dd').textContent"
    txt = lambda sel: page.eval_on_selector(sel, "e => e.textContent.trim()")
    attr = lambda sel, a: page.eval_on_selector(sel, f"e => e.getAttribute('{a}')")

    # --- 01: ползунки и выбор активации ----------------------------------
    for sid in ("s1-w1", "s1-w2", "s1-b", "s1-x1", "s1-x2"):
        before = await txt("#s1-trace")
        await page.fill(f"#{sid}", "2.05")
        await page.dispatch_event(f"#{sid}", "input")
        await page.wait_for_timeout(120)
        probe.check(f"01 ползунок {sid}",
                    before != await txt("#s1-trace") and await txt(f"#{sid}-v") == "2.05")

    for act, marker in (("relu", "ReLU"), ("sigmoid", "sigmoid"), ("tanh", "tanh")):
        await page.click(f'#s1-act button[data-act="{act}"]')
        await page.wait_for_timeout(150)
        probe.check(f"01 активация {act}",
                    marker in await txt("#s1-trace")
                    and await attr(f'#s1-act button[data-act="{act}"]', "aria-pressed") == "true")

    # --- 02: ползунок z ---------------------------------------------------
    before = await txt("#s2-table")
    await page.fill("#s2-z", "-4.25")
    await page.dispatch_event("#s2-z", "input")
    await page.wait_for_timeout(150)
    after = await txt("#s2-table")
    probe.check("02 ползунок z", before != after and await txt("#s2-z-v") == "-4.25")
    probe.check("02 производная ReLU слева от нуля равна нулю", "0.000000" in after)

    # --- 03: выход a и метка y -------------------------------------------
    before = await txt("#s3-kv")
    await page.fill("#s3-a", "0.1")
    await page.dispatch_event("#s3-a", "input")
    await page.wait_for_timeout(150)
    probe.check("03 ползунок a", before != await txt("#s3-kv"))
    await page.click('#s3-y button[data-y="0"]')
    await page.wait_for_timeout(200)
    probe.check("03 переключение метки y", "y = 0" in await txt("#s3-legend"))
    await page.click('#s3-y button[data-y="1"]')
    await page.wait_for_timeout(200)

    # --- 04: другая инициализация ----------------------------------------
    before = await txt("#s4-kv")
    await page.click("#s4-run")
    await page.wait_for_timeout(700)
    after = await txt("#s4-kv")
    probe.check("04 кнопка «Другая инициализация»", before != after and "seed 1" in after)
    probe.check("04 потери остаются на уровне ln 2", "0.693" in after, after[:40])

    # --- 05: наборы данных ------------------------------------------------
    for ds, n in (("moons", "500"), ("xor", "4"), ("and_or", "4"), ("circles", "500")):
        await page.click(f'#s5-data button[data-ds="{ds}"]')
        await page.wait_for_timeout(450)
        probe.check(f"05 набор {ds}",
                    n in await txt("#s5-kv")
                    and await attr(f'#s5-data button[data-ds="{ds}"]', "aria-pressed") == "true")

    # --- 05: активация, ширина слоя, инициализация ------------------------
    for act in ("relu", "tanh"):
        await page.click(f'#s5-act button[data-act="{act}"]')
        await page.wait_for_timeout(400)
        probe.check(f"05 активация {act} сбрасывает обучение", await page.evaluate(from_kv) == "0")

    await page.fill("#s5-h", "3")
    await page.dispatch_event("#s5-h", "input")
    await page.wait_for_timeout(400)
    probe.check("05 ползунок H меняет число параметров",
                await txt("#s5-h-v") == "3" and "13" in await txt("#s5-kv"))
    await page.fill("#s5-seed", "7")
    await page.dispatch_event("#s5-seed", "input")
    await page.wait_for_timeout(400)
    probe.check("05 ползунок seed", await txt("#s5-seed-v") == "7")
    await page.fill("#s5-h", "8")
    await page.dispatch_event("#s5-h", "input")
    await page.wait_for_timeout(400)

    # --- 05: скорость обучения -------------------------------------------
    for lr in ("0.001", "0.01", "0.1", "1"):
        await page.click(f'#s5-lr button[data-lr="{lr}"]')
        await page.wait_for_timeout(80)
        chosen = await page.eval_on_selector_all(
            "#s5-lr button", "es => es.filter(e => e.getAttribute('aria-pressed') === 'true').length")
        probe.check(f"05 скорость обучения {lr}",
                    chosen == 1
                    and await attr(f'#s5-lr button[data-lr="{lr}"]', "aria-pressed") == "true")

    # --- 05: обучать / пауза / сброс --------------------------------------
    start = int(await page.evaluate(from_kv))
    await page.click("#s5-play")
    await page.wait_for_timeout(1200)
    running = int(await page.evaluate(from_kv))
    probe.check("05 кнопка «Обучать»",
                running > start and await txt("#s5-play") == "Пауза", f"эпоха {start} → {running}")
    await page.click("#s5-play")
    await page.wait_for_timeout(500)
    paused = await page.evaluate(from_kv)
    await page.wait_for_timeout(700)
    probe.check("05 кнопка «Пауза»",
                paused == await page.evaluate(from_kv) and await txt("#s5-play") == "Обучать")
    await page.click("#s5-reset")
    await page.wait_for_timeout(500)
    probe.check("05 кнопка «Сбросить»", await page.evaluate(from_kv) == "0")

    # --- 06: проверка градиента -------------------------------------------
    await page.click("#s6-run")
    await page.wait_for_timeout(900)
    rows = await page.eval_on_selector_all("#s6-table tr", "es => es.length")
    probe.check("06 кнопка «Проверить градиент»",
                "ПРОЙДЕНО" in await txt("#s6-verdict") and rows == 10, f"{rows} строк")
    await page.click("#s6-kink")
    await page.wait_for_timeout(900)
    probe.check("06 кнопка «Показать излом ReLU» роняет проверку",
                "НЕ ПРОЙДЕНО" in await txt("#s6-verdict"))
    await page.click("#s6-run")
    await page.wait_for_timeout(900)
    verdict = await txt("#s6-verdict")
    probe.check("06 возврат к норме", "ПРОЙДЕНО" in verdict and "НЕ ПРОЙДЕНО" not in verdict)

    # --- 07: пересчёт по настройкам станции 05 ----------------------------
    cap_before, table_before = await txt("#s7-cap"), await txt("#s7-table")
    await page.click('#s5-data button[data-ds="moons"]')
    await page.wait_for_timeout(400)
    await page.fill("#s5-h", "12")
    await page.dispatch_event("#s5-h", "input")
    await page.wait_for_timeout(400)
    await page.click("#s7-run")
    await page.wait_for_timeout(2600)
    cap_after = await txt("#s7-cap")
    probe.check("07 кнопка «Пересчитать» подхватывает настройки станции 05",
                cap_before != cap_after and table_before != await txt("#s7-table"), cap_after)
    probe.check("07 в таблице четыре строки",
                await page.eval_on_selector_all("#s7-table tr", "es => es.length") == 4)

    # --- переключатель темы (есть только в автономной сборке) -------------
    if await page.query_selector("#theme-toggle"):
        modes = []
        for _ in range(3):
            await page.click("#theme-toggle")
            await page.wait_for_timeout(500)
            modes.append(await page.evaluate("() => document.documentElement.dataset.theme || 'auto'"))
        probe.check("переключатель темы проходит полный цикл",
                    modes == ["light", "dark", "auto"], " → ".join(modes))

    # --- навигация по станциям --------------------------------------------
    height = await page.evaluate("() => window.innerHeight")
    for i in range(1, 8):
        await page.click(f'.routes a[href="#s{i}"]')
        await page.wait_for_timeout(1300)
        rect = await page.evaluate(
            f"() => {{ const r = document.getElementById('s{i}').getBoundingClientRect();"
            f" return {{ top: r.top, bottom: r.bottom }}; }}")
        # Станция 07 -- последняя: страница упирается в конец и не доводит её
        # до самого верха. Достаточно, что станция попала в видимую область.
        probe.check(f"навигация к станции 0{i}", rect["top"] < height and rect["bottom"] > 0,
                    f"top = {rect['top']:+.0f}px")

    # --- подсказки на графиках ---------------------------------------------
    # У станции 01 слоя наведения нет намеренно: там одна точка-маркер.
    for cid in ("s2-chart", "s3-chart", "s4-chart", "s5-chart", "s7-chart"):
        cid_sel = f"#{cid}"
        await page.eval_on_selector(
            cid_sel, "e => e.scrollIntoView({ block: 'center', behavior: 'instant' })")
        await page.wait_for_timeout(400)
        box = await page.eval_on_selector(
            f"{cid_sel} svg",
            "e => { const r = e.getBoundingClientRect();"
            " return { x: r.x, y: r.y, w: r.width, h: r.height }; }")
        await page.mouse.move(box["x"] + box["w"] * 0.45, box["y"] + box["h"] * 0.5)
        await page.wait_for_timeout(120)
        await page.mouse.move(box["x"] + box["w"] * 0.62, box["y"] + box["h"] * 0.5)
        await page.wait_for_timeout(350)
        shown = await page.eval_on_selector(f"{cid_sel} .tip", "e => e.dataset.show")
        text = await txt(f"{cid_sel} .tip")
        probe.check(f"подсказка на графике {cid}", shown == "1" and bool(text), text[:32])
        await page.mouse.move(3, 3)
        await page.wait_for_timeout(250)
        probe.check(f"подсказка {cid} скрывается при уходе курсора",
                    await page.eval_on_selector(f"{cid_sel} .tip", "e => e.dataset.show") == "0")

    # --- подписи для доступности --------------------------------------------
    unlabelled = await page.evaluate("""() => {
        const out = [];
        document.querySelectorAll('button').forEach(b => {
            if (!b.textContent.trim() && !b.getAttribute('aria-label')) out.push(b.id || b.className);
        });
        document.querySelectorAll('input[type=range]').forEach(i => {
            if (!document.querySelector('label[for="' + i.id + '"]')) out.push('без label: ' + i.id);
        });
        return out;
    }""")
    probe.check("все кнопки и ползунки подписаны", unlabelled == [], str(unlabelled))

    # --- узкий экран ---------------------------------------------------------
    await page.set_viewport_size({"width": 400, "height": 900})
    await page.wait_for_timeout(1200)
    width = await page.evaluate("() => document.documentElement.scrollWidth")
    probe.check("узкий экран 400px: нет горизонтальной прокрутки", width <= 400, f"scrollWidth = {width}")
    await page.click("#s5-play")
    await page.wait_for_timeout(900)
    probe.check("узкий экран: обучение запускается",
                int(await page.evaluate(from_kv)) > 0)


async def _run():
    from playwright.async_api import async_playwright

    probe = Probe()
    errors = []
    async with async_playwright() as p:
        launch = {}
        exe = _executable_path()
        if exe:
            launch["executable_path"] = exe
        browser = await p.chromium.launch(**launch)
        page = await browser.new_page(viewport={"width": 1280, "height": 1000})
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console: {m.text}")
                if m.type == "error" and "ERR_" not in m.text else None)
        await page.goto(URL)
        await page.wait_for_timeout(4000)
        await _exercise(page, probe)
        await browser.close()
    return probe, errors


@unittest.skipUnless(_playwright_available(), "не установлен playwright")
@unittest.skipUnless(os.path.exists(INDEX), "public/ не собран: python web/build.py")
class TestInterface(unittest.TestCase):
    """Один проход по странице: все органы управления и все графики."""

    @classmethod
    def setUpClass(cls):
        if _executable_path() is None:
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    p.chromium.launch().close()
            except Exception as exc:                      # noqa: BLE001
                raise unittest.SkipTest(f"нет браузера Chromium: {exc}") from None
        cls.probe, cls.errors = asyncio.run(_run())

    def test_no_javascript_errors(self):
        self.assertEqual(self.errors, [], "страница выдала ошибки JavaScript")

    def test_every_control_responds(self):
        self.assertEqual(
            self.probe.failures, [],
            "органы управления не отработали:\n" + self.probe.report(),
        )

    def test_report(self):
        print("\n" + self.probe.report())


if __name__ == "__main__":
    unittest.main(verbosity=2)
