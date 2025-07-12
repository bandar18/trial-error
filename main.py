#!/usr/bin/env python3
"""CLI entry for the end-to-end intraday ML pipeline."""
import argparse
import json
import os
from datetime import datetime

from data import download_data
from features import build_dataset
from model import generate_signals, train_models
from backtest import backtest


def run(days: int = 56, horizon: int = 30, eta: float = 0.0005):
    tickers = ["SPY", "^VIX", "DX-Y.NYB"]

    print("[1/5] Downloading data ...")
    raw = download_data(tickers, days)

    print("[2/5] Building dataset ...")
    X, y, price = build_dataset(raw, horizon_min=horizon, eta=eta)
    print(f"Features shape: {X.shape}, Labels: {y.value_counts().to_dict()}")

    print("[3/5] Training models ...")
    base_model, meta_model, res = train_models(X, y)

    print("[4/5] Generating signals ...")
    signals = generate_signals(res)

    print("[5/5] Backtesting ...")
    bt_df, metrics = backtest(price, signals)

    os.makedirs("reports", exist_ok=True)
    report = {
        "run_ts": datetime.utcnow().isoformat(),
        "params": {"days": days, "horizon": horizon, "eta": eta},
        "metrics": metrics,
    }

    with open("reports/report.json", "w") as fp:
        json.dump(report, fp, indent=2)

    print("Finished. Metrics:\n", json.dumps(metrics, indent=2))
    print("Plots saved to ./figs, report saved to ./reports/report.json")


def cli():
    parser = argparse.ArgumentParser(description="Intraday SPY ML Pipeline")
    parser.add_argument("--days", type=int, default=56, help="Lookback days (multiple of 7)")
    parser.add_argument("--horizon", type=int, default=30, help="Barrier horizon in minutes")
    parser.add_argument("--eta", type=float, default=0.0005, help="Barrier threshold (fraction)")
    args = parser.parse_args()

    run(days=args.days, horizon=args.horizon, eta=args.eta)


if __name__ == "__main__":
    cli()