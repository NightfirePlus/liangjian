"""快照：接口获取 → 校验整理 → 固定保存 → 回测只读快照。

目录约定（不要改，回测和界面都依赖）：
    data/snapshots/<snapshot_id>/prices.csv   长表，列为 PRICE_COLUMNS
    data/snapshots/<snapshot_id>/meta.json    SnapshotMeta
    data/snapshots/<snapshot_id>/quality.json QualityReport
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.schemas import QualityReport, SnapshotMeta

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "data" / "snapshots"


def build_snapshot(symbols: list[str], start: str, end: str, adjust: str = "qfq",
                   allow_cache_fallback: bool = True) -> tuple[SnapshotMeta, QualityReport]:
    """联网取数 → validate → 保存。snapshot_id 由内容哈希生成，同数据同 id。

    某只标的取数失败：若 allow_cache_fallback 且存在已验证快照覆盖该标的，用缓存，
    meta.from_cache=True，actual_end 如实反映缓存截止日；否则抛异常。
    绝不在失败时静默改用模拟数据。
    """
    raise NotImplementedError


def load_snapshot(snapshot_id: str) -> tuple[pd.DataFrame, SnapshotMeta, QualityReport]:
    """读取快照；回测引擎只通过这个函数拿数据，不直接联网。"""
    raise NotImplementedError


def list_snapshots() -> list[SnapshotMeta]:
    """列出本地所有快照（含演示快照），按 fetched_at 倒序。"""
    raise NotImplementedError


def make_synthetic_snapshot(symbols: list[str], start: str, end: str, seed: int = 0) -> SnapshotMeta:
    """仅供无网络开发测试。meta.is_synthetic 必须为 True，source 写明 "synthetic"。"""
    raise NotImplementedError
