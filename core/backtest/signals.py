"""策略模板：生成信号并模拟成交，产出逐笔交易表与逐日净值表。

规则统一为"收盘出信号，次日开盘成交"：t 日收盘价算出的信号，在 t+1 日开盘价成交。
当前只实现 ma_cross（双均线）；momentum_rotation 留空待补。
"""
from __future__ import annotations

import pandas as pd

from core.schemas import CostConfig, StrategyConfig, DAILY_COLUMNS, TRADE_COLUMNS


def run_ma_cross(prices: pd.DataFrame, config: StrategyConfig,
                  benchmark_prices: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """双均线策略：fast 上穿 slow 买入，下穿卖出。单标的，满仓操作。

    prices: 单个 symbol 的 PRICE_COLUMNS 表，按 date 升序。
    benchmark_prices: 可选，同样格式的基准行情，用于填充 DAILY_COLUMNS.benchmark_equity。
    返回 (trades, daily)，列分别严格等于 TRADE_COLUMNS / DAILY_COLUMNS。
    """
    fast = int(config.params.get("fast", 5))
    slow = int(config.params.get("slow", 20))
    if fast <= 0 or slow <= 0:
        raise ValueError(f"均线参数必须为正整数: fast={fast}, slow={slow}")
    if fast >= slow:
        raise ValueError(f"快线周期必须小于慢线周期: fast={fast}, slow={slow}")
    symbol = config.symbols[0]
    cost = config.cost

    df = prices.sort_values("date").reset_index(drop=True).copy()
    df["ma_fast"] = df["close"].rolling(fast).mean()
    df["ma_slow"] = df["close"].rolling(slow).mean()

    prev_diff = (df["ma_fast"] - df["ma_slow"]).shift(1)
    curr_diff = df["ma_fast"] - df["ma_slow"]
    cross_up = (prev_diff <= 0) & (curr_diff > 0)
    cross_down = (prev_diff >= 0) & (curr_diff < 0)

    df["signal"] = None
    df.loc[cross_up, "signal"] = "buy"
    df.loc[cross_down, "signal"] = "sell"

    bench = None
    if benchmark_prices is not None and not benchmark_prices.empty:
        bench = benchmark_prices.sort_values("date").set_index("date")["close"]

    cash = config.initial_cash
    shares = 0.0
    position_open = False
    trades: list[dict] = []
    daily_rows: list[dict] = []
    trade_id = 0

    bench_initial = bench.iloc[0] if bench is not None and len(bench) else None

    n = len(df)
    for i in range(n):
        row = df.iloc[i]
        date = row["date"]

        if i > 0:
            signal_row = df.iloc[i - 1]
            pending_signal = signal_row["signal"]
            if pending_signal == "buy" and not position_open:
                exec_price = row["open"]
                gross = cash
                shares = _apply_buy_cost(gross, exec_price, cost)
                if shares > 0:
                    commission, slippage_cost = _buy_costs(shares, exec_price, cost)
                    spent = shares * exec_price + commission + slippage_cost
                    cash -= spent
                    position_open = True
                    trade_id += 1
                    trades.append({
                        "trade_id": trade_id, "signal_date": signal_row["date"], "exec_date": date,
                        "symbol": symbol, "side": "buy", "shares": shares, "price": exec_price,
                        "gross": shares * exec_price, "commission": commission,
                        "slippage_cost": slippage_cost, "rule_trigger": f"ma{fast}_cross_up_ma{slow}",
                    })
            elif pending_signal == "sell" and position_open:
                exec_price = row["open"]
                gross = shares * exec_price
                commission, slippage_cost = _sell_costs(shares, exec_price, cost)
                cash += gross - commission - slippage_cost
                trade_id += 1
                trades.append({
                    "trade_id": trade_id, "signal_date": signal_row["date"], "exec_date": date,
                    "symbol": symbol, "side": "sell", "shares": shares, "price": exec_price,
                    "gross": gross, "commission": commission,
                    "slippage_cost": slippage_cost, "rule_trigger": f"ma{fast}_cross_down_ma{slow}",
                })
                shares = 0.0
                position_open = False

        close = row["close"]
        market_value = shares * close
        equity = cash + market_value

        benchmark_equity = None
        if bench is not None and date in bench.index and bench_initial:
            benchmark_equity = config.initial_cash * (bench.loc[date] / bench_initial)

        daily_rows.append({
            "date": date, "symbol": symbol, "shares": shares, "close": close,
            "market_value": market_value, "cash": cash, "equity": equity,
            "benchmark_equity": benchmark_equity, "daily_pnl": None,
        })

    daily = pd.DataFrame(daily_rows, columns=DAILY_COLUMNS)
    daily["daily_pnl"] = daily["equity"].diff()

    total = daily.copy()
    total["symbol"] = "__TOTAL__"
    daily_full = pd.concat([daily, total], ignore_index=True)[DAILY_COLUMNS]

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return trades_df, daily_full


def _buy_costs(shares: float, price: float, cost: CostConfig) -> tuple[float, float]:
    gross = shares * price
    commission = max(gross * cost.commission_rate, cost.min_commission)
    slippage_cost = gross * cost.slippage_bps / 10_000
    return commission, slippage_cost


def _sell_costs(shares: float, price: float, cost: CostConfig) -> tuple[float, float]:
    gross = shares * price
    commission = max(gross * cost.commission_rate, cost.min_commission)
    slippage_cost = gross * cost.slippage_bps / 10_000
    return commission, slippage_cost


def _apply_buy_cost(cash: float, price: float, cost: CostConfig) -> float:
    """用全部现金满仓买入时，倒推在扣除佣金和滑点后能买到的股数。"""
    if cash <= 0 or price <= 0:
        return 0.0

    commission_rate = max(float(cost.commission_rate), 0.0)
    min_commission = max(float(cost.min_commission), 0.0)
    slippage_rate = max(float(cost.slippage_bps), 0.0) / 10_000

    # 总支出 = gross + max(gross * commission_rate, min_commission)
    #          + gross * slippage_rate。按最低佣金与比例佣金的分界点分段求解。
    if commission_rate > 0:
        threshold_gross = min_commission / commission_rate
        threshold_cash = threshold_gross * (1 + commission_rate + slippage_rate)
    else:
        threshold_cash = float("inf")

    if cash >= threshold_cash:
        gross = cash / (1 + commission_rate + slippage_rate)
    else:
        gross = max(cash - min_commission, 0.0) / (1 + slippage_rate)
    return gross / price
