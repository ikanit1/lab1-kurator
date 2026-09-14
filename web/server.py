"""Локальный запуск веб-интерфейса -- без интернета и без зависимостей.

    python web/server.py            # http://127.0.0.1:8000
    python web/server.py --port 9000

Опубликованная версия страницы живёт в облаке, но на защите интернет может
быть недоступен. Этот сервер собирает ту же страницу из web/artifact.html,
оборачивая её в полноценный HTML-документ (в облаке обёртку добавляет
хостинг), и раздаёт рядом data.js, engine.js и app.js.

Используется только стандартная библиотека: ставить ничего не нужно.
"""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))

DOCUMENT = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin: 0; font: 14px system-ui, sans-serif; }}
  img {{ max-width: 100%; }}
  [hidden] {{ display: none !important; }}
</style>
</head>
<body>
{fragment}
</body>
</html>
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def do_GET(self):  # noqa: N802 -- имя задано базовым классом
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "artifact.html"), encoding="utf-8") as f:
                body = DOCUMENT.format(fragment=f.read()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, fmt, *args):
        pass          # тихий лог: на показе он только мешает


def main():
    parser = argparse.ArgumentParser(description="Веб-интерфейс лабораторной работы № 1")
    parser.add_argument("--port", "-p", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="открыть браузер сразу")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(HERE, "data.js")):
        raise SystemExit("Нет web/data.js -- сначала выполните: python web/export_data.py")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/"
        print(f"Веб-интерфейс запущен: {url}")
        print("Остановить -- Ctrl+C")
        if args.open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nОстановлено.")


if __name__ == "__main__":
    main()
