"""Shared pytest config: put ``server/`` on ``sys.path`` so the vendored
``vm_metrics`` imports flat, the way it does inside the Modal images.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SERVER_DIR = str(Path(__file__).resolve().parent.parent / "server")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)
