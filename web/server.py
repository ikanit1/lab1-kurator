"""Локальный просмотр собранного сайта -- без интернета и без зависимостей.

    python web/server.py            # http://127.0.0.1:8000
    python web/server.py --open --port 9000

Сервер сначала вызывает web/build.py, а затем раздаёт получившийся каталог
public/. Поэтому локально видно ровно то же, что отдаёт Vercel: один и тот же
index.html и те же ассеты, никакой отдельной ветки кода для разработки.

Используется только стандартная библиотека: ставить ничего не нужно.
"""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import webbrowser

from build import build

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PUBLIC = os.path.join(ROOT, "public")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PUBLIC, **kwargs)

    def end_headers(self):
        # На показе важнее свежесть, чем кеш: правки видны после перезагрузки.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Веб-интерфейс лабораторной работы № 1")
    parser.add_argument("--port", "-p", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="открыть браузер сразу")
    args = parser.parse_args()

    build(PUBLIC)

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
