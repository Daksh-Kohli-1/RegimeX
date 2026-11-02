"""Shared utilities: config loading, seeding, small IO helpers."""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULTS: dict = {
    "outputs_dir": "outputs",
    "data_dir": "data",
    "label_mode": "forward",
    "macd_normalize_by_price": True,
    "embargo_days": "auto",
    "use_extended_universe": False,
    "extended_tickers": [],
    "benchmarks": {"persistence": True, "garch": True, "garch_variant": "gjr"},
    "walk_forward": {
        "enabled": True,
        "min_train_days": 1000,
        "test_window_days": 252,
        "step_days": 252,
        "include_deep": True,
    },
    "cash_earns_rf": True,
    "transaction_cost_bps": 5,
    "vol_target": {"target_annual_vol": 0.12, "max_leverage": 1.0, "min_vol_floor": 0.03},
    "explain": {"enabled": True, "permutation_repeats": 5, "use_shap": True},
}


def _merge_defaults(cfg: dict, defaults: dict) -> dict:
    """Fill missing keys (recursively) so older config files keep working."""
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v.copy() if isinstance(v, (dict, list)) else v
        elif isinstance(v, dict) and isinstance(cfg[k], dict):
            _merge_defaults(cfg[k], v)
    return cfg


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml from the project root (or a given path)."""
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return _merge_defaults(cfg, DEFAULTS)


def active_tickers(cfg: dict) -> list[str]:
    """Base universe, plus the extended universe when switched on (README 9.10)."""
    tickers = list(cfg["tickers"])
    if cfg.get("use_extended_universe"):
        tickers += [t for t in cfg.get("extended_tickers", []) if t not in tickers]
    return tickers


def active_start_date(cfg: dict) -> str:
    if cfg.get("use_extended_universe") and cfg.get("extended_start_date"):
        return cfg["extended_start_date"]
    return cfg["start_date"]


def effective_gap(cfg: dict) -> int:
    """Number of rows purged between train and test (README 9.5)."""
    g = cfg.get("embargo_days", "auto")
    if g == "auto":
        return int(cfg["vol_window"]) if cfg.get("label_mode", "forward") == "forward" else 1
    return int(g)


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve an output/data dir from config relative to project root."""
    p = Path(cfg[key])
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def log(msg: str) -> None:
    print(f"[vrf] {msg}", flush=True)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False
