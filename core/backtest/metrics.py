"""从逐日净值表计算 Metrics。"""
from __future__ import annotations

import pandas as pd

from core.schemas import Metrics, TRADE_COLUMNS

TRADING_DAYS_PER_YEAR = 252
MIN_DAYS_FOR_ANNUALIZATION = 60


def compute_metrics(daily: pd.DataFrame, trades: pd.DataFrame) -> Metrics:
    """daily 为 DAILY_COLUMNS 长表，取 symbol == "__TOTAL__" 的汇总行计算。"""
    total = daily[daily["symbol"] == "__TOTAL__"].sort_values("date").reset_index(drop=True)
    notes: list[str] = []

    if total.empty:
        return Metrics(
            total_return=None, annual_return=None, annual_vol=None,
            max_drawdown=None, max_dd_start=None, max_dd_end=None,
            n_trades=len(trades), notes=["无逐日净值数据"],
        )

    equity = total["equity"]
    n_days = len(total)

    initial_equity = equity.iloc[0]
    final_equity = equity.iloc[-1]
    total_return = final_equity / initial_equity - 1 if initial_equity else None

    daily_ret = equity.pct_change().dropna()

    if n_days < MIN_DAYS_FOR_ANNUALIZATION:
        annual_return = None
        annual_vol = None
        notes.append(f"历史仅 {n_days} 个交易日，少于 {MIN_DAYS_FOR_ANNUALIZATION} 日不做年化")
    else:
        years = n_days / TRADING_DAYS_PER_YEAR
        annual_return = (1 + total_return) ** (1 / years) - 1 if total_return is not None else None
        annual_vol = daily_ret.std() * (TRADING_DAYS_PER_YEAR ** 0.5) if len(daily_ret) > 1 else None

    max_drawdown, dd_start, dd_end = _max_drawdown(total["date"], equity)

    benchmark_equity = total["benchmark_equity"]
    if benchmark_equity.notna().all() and benchmark_equity.iloc[0]:
        benchmark_return = benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1
        excess_return = (total_return - benchmark_return) if total_return is not None else None
    else:
        benchmark_return = None
        excess_return = None
        notes.append("基准净值缺失，无法计算超额收益")

    total_cost = float(trades["commission"].sum() + trades["slippage_cost"].sum()) if not trades.empty else 0.0
    turnover = _turnover(trades, equity)

    if trades.empty:
        notes.append("期间无交易")

    return Metrics(
        total_return=total_return,
        annual_return=annual_return,
        annual_vol=annual_vol,
        max_drawdown=max_drawdown,
        max_dd_start=dd_start,
        max_dd_end=dd_end,
        n_trades=len(trades),
        turnover=turnover,
        total_cost=total_cost,
        benchmark_return=benchmark_return,
        excess_return=excess_return,
        notes=notes,
    )


def _max_drawdown(dates: pd.Series, equity: pd.Series) -> tuple[float | None, str | None, str | None]:
    if equity.empty:
        return None, None, None
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    trough_idx = drawdown.idxmin()
    max_dd = float(drawdown.loc[trough_idx])
    peak_idx = equity.loc[:trough_idx].idxmax()
    dd_start = str(pd.Timestamp(dates.loc[peak_idx]).date())
    dd_end = str(pd.Timestamp(dates.loc[trough_idx]).date())
    return max_dd, dd_start, dd_end


def _turnover(trades: pd.DataFrame, equity: pd.Series) -> float | None:
    if trades.empty or equity.empty:
        return None
    avg_equity = equity.mean()
    if not avg_equity:
        return None
    total_traded = float(trades["gross"].sum())
    return total_traded / avg_equity
