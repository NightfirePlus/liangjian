"""策略模板：生成信号并模拟成交，产出逐笔交易表与逐日净值表。

规则统一为"收盘出信号，次日开盘成交"：t 日收盘价算出的信号，在 t+1 日开盘价成交。
当前实现 ma_cross（双均线）、breakout（价格通道突破）和 rsi_reversion（RSI 均值回归）。
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

    return _run_long_only_signals(
        df, config, benchmark_prices,
        buy_trigger=f"ma{fast}_cross_up_ma{slow}",
        sell_trigger=f"ma{fast}_cross_down_ma{slow}",
    )


def run_breakout(
    prices: pd.DataFrame,
    config: StrategyConfig,
    benchmark_prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """价格通道突破：突破此前 entry 日最高价买入，跌破此前 exit 日最低价卖出。"""
    entry = int(config.params.get("entry_window", 20))
    exit_ = int(config.params.get("exit_window", 10))
    if entry <= 0 or exit_ <= 0:
        raise ValueError(f"突破和退出周期必须为正整数: entry_window={entry}, exit_window={exit_}")
    if entry <= exit_:
        raise ValueError(f"突破周期必须大于退出周期: entry_window={entry}, exit_window={exit_}")

    df = prices.sort_values("date").reset_index(drop=True).copy()
    df["entry_high"] = df["high"].rolling(entry).max().shift(1)
    df["exit_low"] = df["low"].rolling(exit_).min().shift(1)
    df["signal"] = None
    df.loc[df["close"] > df["entry_high"], "signal"] = "buy"
    df.loc[df["close"] < df["exit_low"], "signal"] = "sell"
    return _run_long_only_signals(
        df, config, benchmark_prices,
        buy_trigger=f"close_breaks_{entry}d_high",
        sell_trigger=f"close_breaks_{exit_}d_low",
    )


def run_rsi_reversion(
    prices: pd.DataFrame,
    config: StrategyConfig,
    benchmark_prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """RSI 均值回归：RSI 进入超卖阈值买入，进入超买阈值卖出。"""
    period = int(config.params.get("rsi_period", 14))
    buy_threshold = float(config.params.get("buy_threshold", 30))
    sell_threshold = float(config.params.get("sell_threshold", 70))
    if period <= 0:
        raise ValueError(f"RSI 周期必须为正整数: rsi_period={period}")
    if not 0 < buy_threshold < sell_threshold < 100:
        raise ValueError(
            "RSI 阈值必须满足 0 < 买入阈值 < 卖出阈值 < 100: "
            f"buy_threshold={buy_threshold}, sell_threshold={sell_threshold}"
        )

    df = prices.sort_values("date").reset_index(drop=True).copy()
    delta = df["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    average_loss = losses.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    relative_strength = average_gain / average_loss
    df["rsi"] = 100 - (100 / (1 + relative_strength))
    df.loc[(average_loss == 0) & (average_gain > 0), "rsi"] = 100.0

    df["signal"] = None
    df.loc[df["rsi"] <= buy_threshold, "signal"] = "buy"
    df.loc[df["rsi"] >= sell_threshold, "signal"] = "sell"
    return _run_long_only_signals(
        df, config, benchmark_prices,
        buy_trigger=f"rsi{period}_at_or_below_{buy_threshold:g}",
        sell_trigger=f"rsi{period}_at_or_above_{sell_threshold:g}",
    )


def _run_long_only_signals(
    df: pd.DataFrame,
    config: StrategyConfig,
    benchmark_prices: pd.DataFrame | None,
    *,
    buy_trigger: str,
    sell_trigger: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按统一的次日开盘、单标的满仓规则执行已生成的 buy/sell 信号。"""
    symbol = config.symbols[0]
    cost = config.cost

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
                        "slippage_cost": slippage_cost, "rule_trigger": buy_trigger,
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
                    "slippage_cost": slippage_cost, "rule_trigger": sell_trigger,
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
