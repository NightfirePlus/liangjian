"""从外部接口获取原始日线。只负责取数和统一列名,不做质量判断。

数据源链(东财限流时自动降级,实际用到的源记录在 df.attrs["source"],
快照元信息会如实标注):
    akshare/fund_etf_hist_em  →  akshare/stock_zh_a_hist_tx
两源前复权口径已交叉验证一致(510300 在 2025-09-26 收盘价均为 4.528),
成交量统一换算为手(腾讯源原始单位为股,除以 100)。
"""
from __future__ import annotations

import socket
import time
from contextlib import contextmanager
from datetime import date

import akshare as ak
import pandas as pd

from core.schemas import PRICE_COLUMNS

# fund_etf_hist_em 的中文列名 → 契约列名
_EM_COLS = {
    "日期": "date", "开盘": "open", "最高": "high",
    "最低": "low", "收盘": "close", "成交量": "volume",
}
# stock_zh_a_hist_tx 的英文列名 → 契约列名(成交量原始单位为股,换算见 _to_standard_tx)
_TX_COLS = {"date": "date", "open": "open", "close": "close",
            "high": "high", "low": "low", "volume": "volume"}
_RETRIES = 3             # 东财源重试次数
_TX_RETRIES = 2          # 腾讯源重试次数(降级源,少一次)
_BACKOFF_SECONDS = 2.0   # 重试间隔按次数递增:2s / 4s
_TIMEOUT_SECONDS = 30.0  # akshare 接口不带 HTTP 超时参数,socket 级兜底

SOURCE_EM = "akshare/fund_etf_hist_em"
SOURCE_TX = "akshare/stock_zh_a_hist_tx"


@contextmanager
def _socket_timeout(seconds: float):
    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(old)


def _to_yyyymmdd(value: str | None) -> str:
    """兼容 "YYYY-MM-DD" 与 "YYYYMMDD" 两种写法,输出接口要求的 8 位数字。"""
    if value is None:
        value = date.today().isoformat()
    digits = str(value).replace("-", "")
    if len(digits) != 8 or not digits.isdigit():
        raise ValueError(f"日期格式应为 YYYY-MM-DD 或 YYYYMMDD,收到: {value!r}")
    return digits


def _normalize_symbol(symbol: str) -> str:
    """容忍 "sh510300" 这类带市场前缀的写法,只保留数字。"""
    digits = "".join(ch for ch in str(symbol) if ch.isdigit())
    if not digits:
        raise ValueError(f"无法识别的标的代码: {symbol!r}")
    return digits


def _market_of(symbol: str) -> str:
    """6/5 开头为沪市,其余按深市处理(与资产池代码规律一致)。"""
    return "sh" if symbol.startswith(("5", "6")) else "sz"


def _to_standard(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    """东财源:中文列名映射为 PRICE_COLUMNS;date 转 datetime64[ns];按日期升序。"""
    df = raw.rename(columns=_EM_COLS)
    missing = [c for c in _EM_COLS.values() if c not in df.columns]
    if missing:
        raise RuntimeError(f"东财源返回列不符合预期,缺少: {missing}(接口可能已改版,请升级 akshare)")
    df = df[list(_EM_COLS.values())].copy()
    return _finish(df, code)


def _to_standard_tx(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    """腾讯源:列名映射;成交量 股 → 手(与东财单位一致,已交叉验证)。"""
    df = raw.rename(columns=_TX_COLS)
    missing = [c for c in _TX_COLS.values() if c not in df.columns]
    if missing:
        raise RuntimeError(f"腾讯源返回列不符合预期,缺少: {missing}(接口可能已改版,请升级 akshare)")
    df = df[list(_TX_COLS.values())].copy()
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce") / 100
    return _finish(df, code)


def _finish(df: pd.DataFrame, code: str) -> pd.DataFrame:
    df["symbol"] = code
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[PRICE_COLUMNS].sort_values("date").reset_index(drop=True)


def _retry(call, tries: int, label: str) -> pd.DataFrame:
    """有限次重试 + socket 超时;空表视为失败;最终失败抛 RuntimeError。"""
    last_err: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            with _socket_timeout(_TIMEOUT_SECONDS):
                df = call()
            if df.empty:
                raise RuntimeError(f"{label}: 区间无数据(代码错误或区间无交易)")
            return df
        except Exception as err:  # 限流/断连/空表都重试,最终失败才抛
            last_err = err
            if attempt < tries:
                time.sleep(_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"{label} 失败(已重试 {tries} 次): {last_err}") from last_err


def fetch_etf_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """取单只 ETF 日线。

    返回列严格等于 PRICE_COLUMNS,date 为 datetime64[ns] 升序。
    不做去重/异常处理,这些由 validate 记录。
    接口失败直接抛异常(不要吞掉、不要返回空表),由上层决定是否回退缓存。

    实现提示:akshare.fund_etf_hist_em(symbol, period="daily",
    start_date="YYYYMMDD", end_date="YYYYMMDD", adjust="qfq"),
    需把中文列名映射为 PRICE_COLUMNS。要加超时与有限次重试。

    数据源链:东财为主,失败自动降级腾讯源(同为前复权口径),
    实际用到的源记录在返回表的 df.attrs["source"]。
    """
    code = _normalize_symbol(symbol)
    adjust_em = "" if adjust == "none" else adjust  # 两源接口都用空串表示不复权

    try:
        df = _retry(
            lambda: _to_standard(ak.fund_etf_hist_em(
                symbol=code, period="daily",
                start_date=_to_yyyymmdd(start), end_date=_to_yyyymmdd(end),
                adjust=adjust_em), code),
            _RETRIES, f"东财源标的 {code}")
        df.attrs["source"] = SOURCE_EM
        return df
    except RuntimeError:
        df = _retry(
            lambda: _to_standard_tx(ak.stock_zh_a_hist_tx(
                symbol=f"{_market_of(code)}{code}",
                start_date=_to_yyyymmdd(start), end_date=_to_yyyymmdd(end),
                adjust=adjust_em), code),
            _TX_RETRIES, f"腾讯源标的 {code}")
        df.attrs["source"] = SOURCE_TX
        return df


def source_name() -> str:
    """写入 SnapshotMeta.source 的来源标识(主源;降级时快照元信息会如实补充)。"""
    return SOURCE_EM
