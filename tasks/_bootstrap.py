"""Добавляет корень репозитория в sys.path.

Нужен, чтобы скрипты запускались и как модули (`python -m tasks.task1_forward`),
и напрямую (`python tasks/task1_forward.py`).
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
