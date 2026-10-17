"""数据模块公共入口:资产池 → 取数 → 质检 → 快照。

回测引擎只通过 load_snapshot 拿数据,不直接联网。
"""
from core.data.snapshot import (
    SNAPSHOT_DIR, build_snapshot, list_snapshots, load_snapshot, make_synthetic_snapshot,
)
from core.data.sources import fetch_etf_daily, source_name
from core.data.universe import (
    ANOMALY_LIMIT, DEFAULT_UNIVERSE, EtfInfo, anomaly_limit, category_of, get_etf, get_universe,
)
from core.data.validate import validate_prices

__all__ = [
    "ANOMALY_LIMIT", "DEFAULT_UNIVERSE", "EtfInfo", "SNAPSHOT_DIR",
    "anomaly_limit", "build_snapshot", "category_of",
    "fetch_etf_daily", "get_etf", "get_universe",
    "list_snapshots", "load_snapshot", "make_synthetic_snapshot",
    "source_name", "validate_prices",
]
