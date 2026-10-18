"""回测入口：接收 StrategyConfig + 行情，分发到对应模板，再计算 Metrics。"""
from __future__ import annotations

import pandas as pd

from core.data import load_snapshot
from core.schemas import Metrics, StrategyConfig
from core.backtest.signals import run_ma_cross
from core.backtest.metrics import compute_metrics


def run_backtest(prices: pd.DataFrame, config: StrategyConfig,
                  benchmark_prices: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, Metrics]:
    """prices 为 PRICE_COLUMNS 长表（可含多个 symbol，当前模板按需取用）。

    返回 (trades, daily, metrics)。
    """
    symbol_prices = prices[prices["symbol"] == config.symbols[0]]

    if config.template == "ma_cross":
        trades, daily = run_ma_cross(symbol_prices, config, benchmark_prices)
    elif config.template == "momentum_rotation":
        raise NotImplementedError("动量轮动模板尚未实现")
    else:
        raise ValueError(f"未知策略模板: {config.template}")

    metrics = compute_metrics(daily, trades)
    return trades, daily, metrics


def run_backtest_from_snapshot(config: StrategyConfig,
                                benchmark_symbol: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, Metrics]:
    """从 config.snapshot_id 读快照后跑回测，供界面/实验模块直接调用。

    benchmark_symbol: 可选，快照里若含该标的则作为基准净值对比；不传则不画基准。
    """
    if not config.snapshot_id:
        raise ValueError("config.snapshot_id 未设置，无法读取快照")

    prices, _meta, _report = load_snapshot(config.snapshot_id)

    benchmark_prices = None
    if benchmark_symbol:
        bench = prices[prices["symbol"] == benchmark_symbol]
        if not bench.empty:
            benchmark_prices = bench

    return run_backtest(prices, config, benchmark_prices)
