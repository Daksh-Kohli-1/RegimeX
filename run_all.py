#!/usr/bin/env python3
"""
Run the entire Volatility Regime Forecasting pipeline end-to-end:

  data -> features -> labels -> baselines -> benchmarks (persistence, GARCH)
       -> deep models -> walk-forward validation -> explainability -> report

Usage:
    python run_all.py                       # everything
    python run_all.py --config other.yaml   # use another config file
    python run_all.py --no-deep             # skip LSTM/Transformer (much faster)
    python run_all.py --no-walk-forward     # holdout only
    python run_all.py --no-explain          # skip feature-importance step
"""
from __future__ import annotations

import argparse
import time

from src import (data_pipeline, evaluate, explain, features, labeling, models_baseline,
                 models_benchmarks, models_deep, walkforward)
from src.utils import load_config, log, torch_available


def banner(text: str) -> None:
    log("=" * 70)
    log(text)
    log("=" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--no-deep", action="store_true", help="skip LSTM/Transformer everywhere")
    parser.add_argument("--no-walk-forward", action="store_true", help="skip walk-forward validation")
    parser.add_argument("--no-explain", action="store_true", help="skip feature importance")
    args = parser.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    if args.no_walk_forward:
        cfg["walk_forward"]["enabled"] = False
    if args.no_explain:
        cfg["explain"]["enabled"] = False
    if args.no_deep:
        cfg["walk_forward"]["include_deep"] = False

    banner("STAGE 1 - Data pipeline + feature engineering")
    data_pipeline.get_price_data(cfg, force_refresh=False)
    features.run(cfg)

    banner("STAGE 2 - Regime labels (forward target, labeler fit on TRAIN only)")
    labeling.run(cfg)

    banner("STAGE 3a - Baseline models (LogReg, GBT)")
    models_baseline.run(cfg)

    banner("STAGE 3b - Benchmarks (Persistence, GARCH)")
    models_benchmarks.run(cfg)

    banner("STAGE 3c - Deep models (LSTM, Transformer)")
    if args.no_deep:
        log("--no-deep given: skipping.")
    elif torch_available():
        models_deep.run(cfg)
    else:
        log("torch not installed - run `pip install torch` and re-run to include LSTM/Transformer.")

    banner("STAGE 4 - Walk-forward validation (purged, expanding window)")
    walkforward.run(cfg)

    banner("STAGE 5 - Explainability")
    explain.run(cfg)

    banner("STAGE 6 - Strategy backtests + evaluation report")
    report_path = evaluate.run(cfg)

    banner(f"DONE in {time.time() - t0:.1f}s. See {report_path} and outputs/figures/")


if __name__ == "__main__":
    main()
