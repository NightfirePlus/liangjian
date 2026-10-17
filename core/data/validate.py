"""数据质量检查。原则：能整理的整理并记录，可疑的只标记，绝不机械删除真实行情波动。"""
from __future__ import annotations

import pandas as pd

from core.schemas import QualityReport


def validate_prices(prices: pd.DataFrame, snapshot_id: str,
                    anomaly_threshold: float = 0.11) -> tuple[pd.DataFrame, QualityReport]:
    """检查并整理长表行情，返回 (整理后的表, 报告)。

    必须覆盖 schemas.IssueKind 的全部类型：
    - 排序、去重：整理并记录（action="dropped"/"flagged"）。
    - 缺失交易日：以资产池各标的日期并集作为日历，找出个别标的缺失。
    - 价格异常：单日涨跌幅绝对值超过阈值只标记。注意 ETF 涨跌停为 10%，
      科创/创业板 20%；跨境 ETF 另有溢价影响。阈值需按类别区分并写明依据。
    - 非正价格、OHLC 不自洽：标 error。
    - 历史不足、各标的起止不一致：写入 short_history / misaligned。
    """
    raise NotImplementedError
