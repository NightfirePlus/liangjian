"""从外部接口获取原始日线。只负责取数和统一列名，不做质量判断。"""
from __future__ import annotations

import pandas as pd

from core.schemas import PRICE_COLUMNS  # noqa: F401


def fetch_etf_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """取单只 ETF 日线。

    返回列严格等于 PRICE_COLUMNS，date 为 datetime64[ns] 升序。
    不做去重/异常处理，这些由 validate 记录。
    接口失败直接抛异常（不要吞掉、不要返回空表），由上层决定是否回退缓存。

    实现提示：akshare.fund_etf_hist_em(symbol, period="daily",
    start_date="YYYYMMDD", end_date="YYYYMMDD", adjust="qfq")，
    需把中文列名映射为 PRICE_COLUMNS。要加超时与有限次重试。
    """
    raise NotImplementedError


def source_name() -> str:
    """写入 SnapshotMeta.source 的来源标识。"""
    return "akshare/fund_etf_hist_em"
