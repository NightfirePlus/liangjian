"""数据质量检查。原则:能整理的整理并记录,可疑的只标记,绝不机械删除真实行情波动。"""
from __future__ import annotations

import pandas as pd

from core.data.universe import anomaly_limit, category_of
from core.schemas import IssueKind, PRICE_COLUMNS, QualityIssue, QualityReport, Severity

# 严重度约定:
# - 排序/去重/起止不一致是整理类问题,记录 info;
# - 缺失交易日、涨跌幅异常是可疑信号,标记 warn;
# - 非正价格、OHLC 不自洽属于数据错误,标 error。

_PRICE_FIELDS = ("open", "high", "low", "close")
_MIN_BARS = 30     # 双均线最慢参数 20 至少需要 21 根日线,留缓冲
_MAX_DATES = 10    # 单个 issue 的 dates 最多列出数,其余折叠进 message


def _fmt_dates(dates: pd.Series) -> list[str]:
    return dates.dt.strftime("%Y-%m-%d").tolist()[:_MAX_DATES]


def _add(issues: list[QualityIssue], symbol: str, kind: IssueKind, severity: Severity,
         message: str, dates: pd.Series | None = None, action: str = "flagged") -> None:
    issues.append(QualityIssue(
        symbol=symbol, kind=kind, severity=severity, message=message,
        dates=_fmt_dates(dates) if dates is not None else [],
        action=action,
    ))


def validate_prices(prices: pd.DataFrame, snapshot_id: str,
                    anomaly_threshold: float = 0.11) -> tuple[pd.DataFrame, QualityReport]:
    """检查并整理长表行情,返回 (整理后的表, 报告)。

    必须覆盖 schemas.IssueKind 的全部类型:
    - 排序、去重:整理并记录(action="dropped"/"flagged")。
    - 缺失交易日:以资产池各标的日期并集作为日历,找出个别标的缺失。
    - 价格异常:单日涨跌幅绝对值超过阈值只标记。注意 ETF 涨跌停为 10%,
      科创/创业板 20%;跨境 ETF 另有溢价影响。阈值需按类别区分并写明依据。
    - 非正价格、OHLC 不自洽:标 error。
    - 历史不足、各标的起止不一致:写入 short_history / misaligned。
    """
    missing_cols = [c for c in PRICE_COLUMNS if c not in prices.columns]
    if missing_cols:
        raise ValueError(f"行情表缺少契约列 {missing_cols}(要求 {PRICE_COLUMNS})")

    df = prices[PRICE_COLUMNS].copy()
    df["symbol"] = df["symbol"].astype(str)
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as err:
        raise ValueError(f"date 列无法解析为日期: {err}") from err
    for col in _PRICE_FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    issues: list[QualityIssue] = []

    # 1) 非正/空价格:客观上不可能存在,标 error 并删除该行(不是删除真实波动)
    price_cols = list(_PRICE_FIELDS)
    bad = df[price_cols].isna().any(axis=1) | (df[price_cols] <= 0).any(axis=1)
    if bad.any():
        for sym, grp in df[bad].groupby("symbol", sort=False):
            _add(issues, sym, "nonpositive_price", "error",
                 f"{len(grp)} 行价格为 0/负/空,已删除", grp["date"], action="dropped")
        df = df[~bad]

    # 2) 逐标的:排序、去重、OHLC 自洽、涨跌幅异常、历史长度
    cleaned: list[pd.DataFrame] = []
    for sym, grp in df.groupby("symbol", sort=True):
        if not grp["date"].is_monotonic_increasing:
            _add(issues, sym, "unsorted", "info", "日期未升序,已按日期整理")
        grp = grp.sort_values("date").reset_index(drop=True)

        dup_mask = grp["date"].duplicated(keep="last")
        if dup_mask.any():
            _add(issues, sym, "duplicate", "info",
                 f"{int(dup_mask.sum())} 个日期重复,已去重(保留最后一条)",
                 grp.loc[dup_mask, "date"], action="dropped")
            grp = grp[~dup_mask].reset_index(drop=True)

        n = len(grp)
        if n == 0:
            _add(issues, sym, "short_history", "error", "整理后无有效数据")
            continue
        if n < _MIN_BARS:
            _add(issues, sym, "short_history", "warn",
                 f"仅 {n} 根日线,不足以支撑慢线参数 20 的双均线回测(至少 {_MIN_BARS} 根)")

        bad_ohlc = ((grp["high"] < grp["low"])
                    | (grp["close"] < grp["low"]) | (grp["close"] > grp["high"])
                    | (grp["open"] < grp["low"]) | (grp["open"] > grp["high"]))
        if bad_ohlc.any():
            _add(issues, sym, "ohlc_inconsistent", "error",
                 f"{int(bad_ohlc.sum())} 行 OHLC 不自洽(high<low 或 open/close 超出 [low, high]),仅标记",
                 grp.loc[bad_ohlc, "date"])

        # 涨跌幅异常:阈值按类别区分(见 universe.ANOMALY_LIMIT)。
        # 依据:境内股票资产 ETF 涨跌停 10%、创业/科创 20%,各加 1pp 缓冲;
        # 跨境/商品 ETF 无涨跌幅限制,跳过检查。
        lim = anomaly_limit(sym)
        if lim is not None:
            eff = anomaly_threshold if lim == 0.11 else lim  # 参数对 10% 类别生效,star 固定 0.21
            pct = grp["close"].pct_change()
            hit = pct.abs() > eff
            if hit.any():
                _add(issues, sym, "price_anomaly", "warn",
                     f"{int(hit.sum())} 个交易日涨跌幅超过 {eff:.1%}(类别 {category_of(sym)} 阈值),仅标记",
                     grp.loc[hit, "date"])

        cleaned.append(grp)

    out = pd.concat(cleaned, ignore_index=True) if cleaned else pd.DataFrame(columns=PRICE_COLUMNS)

    # 3) 缺失交易日:以资产池各标的日期并集作为日历
    if len(out):
        calendar = sorted(out["date"].unique())
        calendar_set = set(calendar)
        for sym, grp in out.groupby("symbol", sort=True):
            missing = [d for d in calendar if d not in set(grp["date"])]
            if missing:
                _add(issues, sym, "missing_days", "warn",
                     f"相对资产池交易日历缺失 {len(missing)} 个交易日(可能因停牌或数据源缺失)",
                     pd.Series(pd.to_datetime(missing)))

    # 4) 各标的起止不一致(资产池级问题,symbol 留空,明细在 message 里)
    if out["symbol"].nunique() > 1:
        ranges = out.groupby("symbol", sort=True)["date"].agg(["min", "max"])
        if ranges["min"].nunique() > 1 or ranges["max"].nunique() > 1:
            detail = []
            for sym, r in ranges.iterrows():
                n = int((out["symbol"] == sym).sum())
                detail.append(f"{sym}: {r['min']:%Y-%m-%d}~{r['max']:%Y-%m-%d}({n} 根)")
            _add(issues, "", "misaligned", "info",
                 "各标的起止日期不一致: " + "; ".join(detail))

    return out, QualityReport(snapshot_id=snapshot_id, issues=issues)
