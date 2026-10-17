"""本次支持的 ETF 资产池。

类别决定质量检查的涨跌停阈值,依据交易所规则:
- broad / industry: 跟踪境内股票资产,日涨跌幅限制 10%,检查阈值 0.11(10% + 1pp 缓冲)
- star: 跟踪创业板/科创板指数,日涨跌幅限制 20%,检查阈值 0.21(20% + 1pp 缓冲)
- no_limit: 跨境(QDII)与商品 ETF 无涨跌幅限制,不做涨跌幅异常检查
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Category = Literal["broad", "industry", "star", "no_limit"]


@dataclass(frozen=True)
class EtfInfo:
    symbol: str       # 6 位交易所代码
    name: str
    category: Category


# 与 validate 的涨跌幅异常检查阈值对应,依据见模块 docstring
ANOMALY_LIMIT: dict[Category, float | None] = {
    "broad": 0.11,
    "industry": 0.11,
    "star": 0.21,
    "no_limit": None,
}

DEFAULT_UNIVERSE: tuple[EtfInfo, ...] = (
    EtfInfo("510300", "沪深300ETF", "broad"),
    EtfInfo("510500", "中证500ETF", "broad"),
    EtfInfo("512100", "中证1000ETF", "broad"),
    EtfInfo("159915", "创业板ETF", "star"),
    EtfInfo("588000", "科创50ETF", "star"),
    EtfInfo("512880", "证券ETF", "industry"),
    EtfInfo("512690", "酒ETF", "industry"),
    EtfInfo("512010", "医药ETF", "industry"),
    EtfInfo("159920", "恒生ETF", "no_limit"),
    EtfInfo("513100", "纳指ETF", "no_limit"),
    EtfInfo("518880", "黄金ETF", "no_limit"),
)

_INDEX: dict[str, EtfInfo] = {e.symbol: e for e in DEFAULT_UNIVERSE}


def get_universe() -> list[EtfInfo]:
    """返回支持的 ETF 资产池(按代码排序)。"""
    return sorted(DEFAULT_UNIVERSE, key=lambda e: e.symbol)


def get_etf(symbol: str) -> EtfInfo | None:
    """查资产池;未收录返回 None。"""
    return _INDEX.get(str(symbol).strip())


def category_of(symbol: str) -> Category:
    """标的所属类别;未收录按 broad(10% 涨跌停)处理,偏保守。"""
    info = _INDEX.get(str(symbol).strip())
    return info.category if info else "broad"


def anomaly_limit(symbol: str) -> float | None:
    """涨跌幅异常检查阈值;None 表示该类别无涨跌幅限制,跳过检查。"""
    return ANOMALY_LIMIT[category_of(symbol)]
