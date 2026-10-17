"""数据模块公共入口:资产池 → 取数 → 质检 → 快照。

回测引擎只通过 load_snapshot 拿数据,不直接联网。
"""
from core.data.snapshot import (
    SNAPSHOT_DIR, build_snapshot, cache_size_bytes, clean_snapshots, list_snapshots,
    load_snapshot, make_synthetic_snapshot, pin_snapshot, unpin_snapshot,
)
from core.data.sources import fetch_etf_daily, source_name
from core.data.universe import (
    ANOMALY_LIMIT, DEFAULT_UNIVERSE, EtfInfo, anomaly_limit, category_of, get_etf, get_universe,
)
from core.data.validate import validate_prices

__all__ = [
    "ANOMALY_LIMIT", "DEFAULT_UNIVERSE", "EtfInfo", "SNAPSHOT_DIR",
    "anomaly_limit", "build_snapshot", "cache_size_bytes", "category_of",
    "clean_snapshots", "fetch_etf_daily", "get_etf", "get_universe",
    "list_snapshots", "load_snapshot", "make_synthetic_snapshot",
    "pin_snapshot", "source_name", "unpin_snapshot", "validate_prices",
]
