"""Сборка статического сайта из исходников web/.

    python web/build.py              # собрать в public/
    python web/build.py --out dist   # собрать в другой каталог

web/artifact.html -- это фрагмент страницы: у него нет <!doctype>, <html>,
<head> и <body>, потому что хостинг артефактов добавляет обёртку сам. Для
Vercel (и любого другого статического хостинга) нужен полноценный документ,
поэтому сборка:

* переносит <title>, <link> и <style> из фрагмента в <head>;
* добавляет метаданные, favicon и переключатель темы -- на обычном сайте
  нет внешнего интерфейса, который задавал бы data-theme;
* копирует data.js, engine.js и app.js рядом с index.html.

Результат самодостаточен: ни сборщиков, ни зависимостей, ни серверного кода.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ASSETS = ["data.js", "engine.js", "app.js"]

DESCRIPTION = (
    "Лабораторная работа № 1, вариант 12: искусственный нейрон на NumPy с нуля. "
    "Прямой проход, tanh и ReLU, бинарная кросс-энтропия, обучение сети на "
    "make_circles и численная проверка градиента."
)

FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <rect width="32" height="32" rx="6" fill="#0F1917"/>
  <circle cx="16" cy="16" r="10" fill="none" stroke="#00968A" stroke-width="2.6"/>
  <circle cx="16" cy="16" r="4" fill="#CC5F00"/>
</svg>
"""

# Переключатель темы нужен только автономной сборке: страница-артефакт
# получает data-theme от хостинга и второй орган управления там лишний.
THEME_CSS = """
    .theme-toggle {
      position: fixed;
      right: 16px;
      bottom: 16px;
      z-index: 40;
      font-family: var(--mono, ui-monospace, monospace);
      font-size: 12px;
      padding: 7px 12px;
      border-radius: 3px;
      border: 1px solid var(--line-strong, #B4C3BF);
      background: var(--panel, #fff);
      color: var(--ink-2, #445350);
      cursor: pointer;
      box-shadow: var(--shadow, 0 2px 10px rgba(0, 0, 0, .12));
    }
    .theme-toggle:hover { border-color: var(--c0, #00968A); color: var(--ink, #0F1917); }
    @media print { .theme-toggle { display: none; } }
"""

THEME_JS = """
  (function () {
    var MODES = ["auto", "light", "dark"];
    var NAMES = { auto: "тема: авто", light: "тема: светлая", dark: "тема: тёмная" };
    var root = document.documentElement;
    var btn = document.getElementById("theme-toggle");

    function read() {
      try { return localStorage.getItem("lab1-theme") || "auto"; } catch (e) { return "auto"; }
    }
    function apply(mode) {
      if (mode === "auto") root.removeAttribute("data-theme");
      else root.setAttribute("data-theme", mode);
      btn.textContent = NAMES[mode];
      btn.setAttribute("aria-label", NAMES[mode] + ". Нажмите, чтобы сменить.");
    }

    var mode = read();
    apply(mode);
    btn.addEventListener("click", function () {
      mode = MODES[(MODES.indexOf(mode) + 1) % MODES.length];
      try { localStorage.setItem("lab1-theme", mode); } catch (e) { /* приватный режим */ }
      apply(mode);
    });
  })();
"""

DOCUMENT = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{description}">
<meta name="color-scheme" content="light dark">
<meta property="og:type" content="website">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
{head}
<style>
  /* Сброс, который на хостинге артефактов добавляется автоматически. */
  :root {{ color-scheme: light dark; }}
  body {{ margin: 0; }}
  img {{ max-width: 100%; }}
  [hidden] {{ display: none !important; }}
{theme_css}</style>
</head>
<body>
{body}
<button class="theme-toggle" id="theme-toggle" type="button">тема: авто</button>
<script>{theme_js}</script>
</body>
</html>
"""


def split_fragment(fragment: str):
    """Вынуть из фрагмента то, чему место в <head>."""
    head_parts = []

    def take(pattern, text):
        found = re.findall(pattern, text, flags=re.S | re.I)
        head_parts.extend(m.strip() for m in found)
        return re.sub(pattern, "", text, flags=re.S | re.I)

    body = take(r"<title>.*?</title>", fragment)
    body = take(r"<link\b[^>]*>", body)
    body = take(r"<style>.*?</style>", body)

    title_match = re.search(r"<title>(.*?)</title>", fragment, flags=re.S | re.I)
    title = title_match.group(1).strip() if title_match else "Лабораторная работа № 1"
    return "\n".join(head_parts), body.strip(), title


def build(out_dir: str) -> list[str]:
    with open(os.path.join(HERE, "artifact.html"), encoding="utf-8") as f:
        fragment = f.read()

    missing = [a for a in ASSETS if not os.path.exists(os.path.join(HERE, a))]
    if missing:
        raise SystemExit(
            "Не хватает файлов: " + ", ".join(missing) +
            "\nСначала выполните: python web/export_data.py"
        )

    head, body, title = split_fragment(fragment)
    document = DOCUMENT.format(
        description=DESCRIPTION,
        title=title,
        head=head,
        body=body,
        theme_css=THEME_CSS,
        theme_js=THEME_JS,
    )

    os.makedirs(out_dir, exist_ok=True)
    written = []

    index = os.path.join(out_dir, "index.html")
    with open(index, "w", encoding="utf-8") as f:
        f.write(document)
    written.append(index)

    favicon = os.path.join(out_dir, "favicon.svg")
    with open(favicon, "w", encoding="utf-8") as f:
        f.write(FAVICON)
    written.append(favicon)

    for asset in ASSETS:
        dst = os.path.join(out_dir, asset)
        shutil.copyfile(os.path.join(HERE, asset), dst)
        written.append(dst)

    return written


def main():
    parser = argparse.ArgumentParser(description="Сборка статического сайта")
    parser.add_argument("--out", "-o", default=os.path.join(ROOT, "public"),
                        help="каталог сборки (по умолчанию public/)")
    args = parser.parse_args()

    written = build(args.out)
    print(f"собрано в {os.path.relpath(args.out, ROOT)}/")
    for path in written:
        print(f"  {os.path.basename(path):<14} {os.path.getsize(path) / 1024:>7.1f} КБ")


if __name__ == "__main__":
    main()
