"""用造数据本地验证回测引擎，不依赖真实快照（组员数据模块未完成前的替代方案）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.backtest.engine import run_backtest
from core.schemas import StrategyConfig


def make_synthetic_prices(symbol: str, start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """造一份带趋势+噪声的日线，足够触发双均线交叉。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    cycle = np.sin(np.linspace(0, 6 * np.pi, n)) * 15
    noise = rng.normal(0, 0.8, n).cumsum() * 0.1
    close = 100 + cycle + noise
    close = np.maximum(close, 1.0)
    open_ = close * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({
        "symbol": symbol, "date": dates, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume,
    })


def test_ma_cross_runs_end_to_end():
    prices = make_synthetic_prices("510300", "2023-01-01", "2024-12-31", seed=1)
    benchmark = make_synthetic_prices("510300_bench", "2023-01-01", "2024-12-31", seed=2)

    config = StrategyConfig(
        template="ma_cross",
        symbols=["510300"],
        start="2023-01-01",
        end="2024-12-31",
        params={"fast": 5, "slow": 20},
    )

    trades, daily, metrics = run_backtest(prices, config, benchmark)

    assert not daily.empty
    assert (daily["symbol"] == "__TOTAL__").any()
    assert metrics.n_trades == len(trades)
    assert metrics.total_return is not None
    print("trades:", len(trades))
    print("metrics:", metrics)


def test_short_history_skips_annualization():
    prices = make_synthetic_prices("510300", "2024-11-01", "2024-12-31", seed=3)
    config = StrategyConfig(
        template="ma_cross", symbols=["510300"],
        start="2024-11-01", end="2024-12-31", params={"fast": 5, "slow": 20},
    )
    _, _, metrics = run_backtest(prices, config)
    assert metrics.annual_return is None
    assert any("不做年化" in n for n in metrics.notes)


def test_no_trades_when_no_cross():
    dates = pd.bdate_range("2023-01-01", "2023-06-30")
    flat = pd.DataFrame({
        "symbol": "510300", "date": dates, "open": 100.0, "high": 100.5,
        "low": 99.5, "close": 100.0, "volume": 1_000_000,
    })
    config = StrategyConfig(
        template="ma_cross", symbols=["510300"],
        start="2023-01-01", end="2023-06-30", params={"fast": 5, "slow": 20},
    )
    trades, daily, metrics = run_backtest(flat, config)
    assert trades.empty
    assert metrics.n_trades == 0
    assert "期间无交易" in metrics.notes


if __name__ == "__main__":
    test_ma_cross_runs_end_to_end()
    test_short_history_skips_annualization()
    test_no_trades_when_no_cross()
    print("all tests passed")
