from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _metrics(equity: pd.Series, returns: pd.Series) -> Dict[str, float]:
    periods_per_day = 78  # 5-minute bars in 6.5h session
    ann_factor = np.sqrt(252 * periods_per_day)

    sharpe = (returns.mean() * periods_per_day * 252) / (
        returns.std(ddof=0) * ann_factor + 1e-12
    )
    downside = returns.clip(upper=0)
    sortino = (returns.mean() * periods_per_day * 252) / (
        downside.std(ddof=0) * ann_factor + 1e-12
    )

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    hit_rate = (returns > 0).mean()

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "hit_rate": float(hit_rate),
    }


def backtest(
    price: pd.Series,
    signals: pd.Series,
    tick_size: float = 0.01,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Vectorised backtest applying 1-tick slippage and zero fees."""
    price = price.loc[signals.index]
    fwd_ret = price.pct_change().shift(-1).loc[signals.index]

    pnl = fwd_ret * signals

    # Slippage when position changes
    position_change = signals.diff().abs().fillna(0)
    slippage = (tick_size / price) * position_change
    pnl -= slippage

    equity = (1 + pnl.fillna(0)).cumprod()

    metrics = _metrics(equity, pnl.dropna())

    # Plot & save
    os.makedirs("figs", exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    equity.plot(ax=axes[0], title="Equity Curve")
    axes[0].set_ylabel("Equity")

    dd = equity / equity.cummax() - 1
    dd.plot(ax=axes[1], title="Drawdown")
    axes[1].set_ylabel("Drawdown")
    plt.tight_layout()
    fig.savefig("figs/equity_drawdown.png", dpi=150)
    plt.close(fig)

    out = pd.DataFrame({"pnl": pnl, "equity": equity})
    return out, metrics