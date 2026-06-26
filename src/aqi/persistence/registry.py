"""Model artefact registry: per-run directories with manifests."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from aqi.config import MODELS_DIR


def make_run_id() -> str:
    ts = time.strftime("%Y%m%dT%H%M%S")
    sha = "nogit"
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass
    return f"{ts}_{sha}"


def run_dir(run_id: str) -> Path:
    p = MODELS_DIR / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_dir(run_id: str, model_name: str) -> Path:
    p = run_dir(run_id) / model_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_run_manifest(run_id: str, info: dict) -> None:
    (run_dir(run_id) / "run.json").write_text(json.dumps({
        "run_id": run_id,
        "code_sha": _git_sha(),
        "pid": os.getpid(),
        **info,
    }, indent=2, default=str))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"
