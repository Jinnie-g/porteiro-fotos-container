"""Registro de reprodutibilidade: sementes, versões, commit."""
from __future__ import annotations

import platform
import random
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from . import __version__


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_info(root) -> dict:
    def sh(*cmd):
        try:
            return subprocess.check_output(cmd, cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {
        "data_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versao_pipeline": __version__,
        "git_commit": sh("git", "rev-parse", "HEAD"),
        "git_sujo": bool(sh("git", "status", "--porcelain")),
        "python": sys.version.split()[0], "plataforma": platform.platform(),
    }


def pip_freeze() -> str:
    return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
