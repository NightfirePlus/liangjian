"""快照:接口获取 → 校验整理 → 固定保存 → 回测只读快照。

目录约定(不要改,回测和界面都依赖):
    data/snapshots/<snapshot_id>/prices.csv   长表,列为 PRICE_COLUMNS
    data/snapshots/<snapshot_id>/meta.json    SnapshotMeta
    data/snapshots/<snapshot_id>/quality.json QualityReport
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from core.data.sources import fetch_etf_daily, source_name
from core.data.validate import validate_prices
from core.schemas import PRICE_COLUMNS, QualityIssue, QualityReport, SnapshotMeta

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "data" / "snapshots"

_VOLUME_UNIT_NOTE = "成交量单位为手(每手 100 份)"
# 缓存覆盖判断的容差:请求区间起点可能落在周末/长假,
# 数据源返回的是首个实际交易日,最长假期约 10 个自然日
_COVER_TOLERANCE_DAYS = 10


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_iso(value: str) -> str:
    """统一 meta 里请求区间的写法为 YYYY-MM-DD。"""
    s = str(value)
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def _hash_prices(prices: pd.DataFrame) -> str:
    """规范化序列化后取 SHA256 前 8 位:同数据同 id,不依赖取数顺序。"""
    payload = prices[PRICE_COLUMNS].sort_values(["symbol", "date"]).to_csv(
        index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                    encoding="utf-8", newline="\n")


def _read_meta(snapshot_dir: Path) -> SnapshotMeta | None:
    """读快照元信息;缺失或损坏返回 None(损坏的元信息不影响其它快照)。"""
    meta_file = snapshot_dir / "meta.json"
    if not meta_file.exists():
        return None
    try:
        return SnapshotMeta(**json.loads(meta_file.read_text(encoding="utf-8")))
    except Exception:
        return None


def _find_cached_prices(symbol: str, start: str, end: str,
                        adjust: str) -> tuple[pd.DataFrame, SnapshotMeta] | None:
    """在已有非模拟、同复权口径的快照里找覆盖 [start, end] 的该标的,取最新一个。

    模拟数据(is_synthetic)绝不作为回退来源,避免与真实快照混淆。
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    best: tuple[pd.DataFrame, SnapshotMeta] | None = None
    for meta_file in SNAPSHOT_DIR.glob("*/meta.json"):
        meta = _read_meta(meta_file.parent)
        if meta is None or meta.is_synthetic or meta.adjust != adjust:
            continue
        prices_file = meta_file.parent / "prices.csv"
        if not prices_file.exists():
            continue
        part = pd.read_csv(prices_file, parse_dates=["date"], dtype={"symbol": str})
        part = part[part["symbol"] == symbol]
        if part.empty:
            continue
        tol = pd.Timedelta(days=_COVER_TOLERANCE_DAYS)
        if part["date"].min() > start_ts + tol or part["date"].max() < end_ts - tol:
            continue
        if best is None or meta.fetched_at > best[1].fetched_at:
            window = part[(part["date"] >= start_ts) & (part["date"] <= end_ts)]
            best = (window, meta)
    return best


def build_snapshot(symbols: list[str], start: str, end: str, adjust: str = "qfq",
                   allow_cache_fallback: bool = True) -> tuple[SnapshotMeta, QualityReport]:
    """联网取数 → validate → 保存。snapshot_id 由内容哈希生成,同数据同 id。

    某只标的取数失败:若 allow_cache_fallback 且存在已验证快照覆盖该标的,用缓存,
    meta.from_cache=True,actual_end 如实反映缓存截止日;否则抛异常。
    绝不在失败时静默改用模拟数据。
    """
    symbols = list(dict.fromkeys(str(s).strip() for s in symbols if str(s).strip()))
    if not symbols:
        raise ValueError("symbols 不能为空")

    parts: list[pd.DataFrame] = []
    cached_symbols: list[str] = []
    cached_metas: list[SnapshotMeta] = []
    for sym in symbols:
        try:
            parts.append(fetch_etf_daily(sym, start, end, adjust=adjust))
        except Exception as err:
            if not allow_cache_fallback:
                raise RuntimeError(f"标的 {sym} 取数失败(已禁用缓存回退): {err}") from err
            hit = _find_cached_prices(sym, start, end, adjust)
            if hit is None:
                raise RuntimeError(f"标的 {sym} 取数失败,且没有同口径缓存快照可回退: {err}") from err
            frame, cached_meta = hit
            parts.append(frame)
            cached_symbols.append(sym)
            cached_metas.append(cached_meta)

    raw = pd.concat(parts, ignore_index=True)
    cleaned, draft_report = validate_prices(raw, snapshot_id="")
    if cleaned.empty:
        raise RuntimeError("清洗后无有效数据,快照未生成(见 quality.json 的 error 项)")

    actual_start = cleaned["date"].min().strftime("%Y-%m-%d")
    actual_end = cleaned["date"].max().strftime("%Y-%m-%d")
    snapshot_id = f"etf_{actual_end.replace('-', '')}_{_hash_prices(cleaned)}"
    report = QualityReport(snapshot_id=snapshot_id, issues=draft_report.issues)

    # 全部标的都来自缓存时,fetched_at 如实反映原始获取时间
    if cached_symbols and len(cached_symbols) == len(symbols):
        fetched_at = max(m.fetched_at for m in cached_metas)
    else:
        fetched_at = _now()
    notes = [_VOLUME_UNIT_NOTE]
    if cached_symbols:
        notes.append("缓存回退标的: " + ", ".join(sorted(cached_symbols))
                     + f"(来源快照 {', '.join(sorted({m.snapshot_id for m in cached_metas}))})")

    meta = SnapshotMeta(
        snapshot_id=snapshot_id, symbols=symbols,
        start=_to_iso(start), end=_to_iso(end),
        actual_start=actual_start, actual_end=actual_end,
        source=source_name(), fetched_at=fetched_at, adjust=adjust,
        is_synthetic=False, from_cache=bool(cached_symbols),
        redistributable=None, notes=notes,
    )

    out_dir = SNAPSHOT_DIR / snapshot_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_dir / "prices.csv", index=False, lineterminator="\n")
    _write_json(out_dir / "meta.json", asdict(meta))
    _write_json(out_dir / "quality.json", asdict(report))
    return meta, report


def load_snapshot(snapshot_id: str) -> tuple[pd.DataFrame, SnapshotMeta, QualityReport]:
    """读取快照;回测引擎只通过这个函数拿数据,不直接联网。"""
    out_dir = SNAPSHOT_DIR / snapshot_id
    if not out_dir.exists():
        raise FileNotFoundError(f"快照不存在: {snapshot_id}({out_dir})")
    prices = pd.read_csv(out_dir / "prices.csv", parse_dates=["date"], dtype={"symbol": str})
    meta_data = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    quality_data = json.loads((out_dir / "quality.json").read_text(encoding="utf-8"))
    meta = SnapshotMeta(**meta_data)
    report = QualityReport(
        snapshot_id=quality_data.get("snapshot_id", snapshot_id),
        issues=[QualityIssue(**i) for i in quality_data.get("issues", [])],
    )
    return prices, meta, report


def list_snapshots() -> list[SnapshotMeta]:
    """列出本地所有快照(含演示快照),按 fetched_at 倒序。"""
    metas = []
    for meta_file in SNAPSHOT_DIR.glob("*/meta.json"):
        meta = _read_meta(meta_file.parent)
        if meta is not None:
            metas.append(meta)
    return sorted(metas, key=lambda m: m.fetched_at, reverse=True)


def make_synthetic_snapshot(symbols: list[str], start: str, end: str, seed: int = 0) -> SnapshotMeta:
    """仅供无网络开发测试。meta.is_synthetic 必须为 True,source 写明 "synthetic"。

    几何随机游走生成日线,OHLC 内部自洽;同一 (symbols, 区间, seed) 生成同 id,
    重复调用会覆盖同一份文件,不会越积越多。
    """
    symbols = list(dict.fromkeys(str(s).strip() for s in symbols if str(s).strip()))
    if not symbols:
        raise ValueError("symbols 不能为空")
    days = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    if len(days) == 0:
        raise ValueError(f"区间无交易日: {start} ~ {end}")

    rows: list[pd.DataFrame] = []
    for sym in symbols:
        # 每只标的一把独立的随机数生成器,与标的在列表中的顺序无关
        seed_i = int(hashlib.sha256(sym.encode("utf-8")).hexdigest()[:8], 16) + seed
        rng = np.random.default_rng(seed_i)
        ret = rng.normal(0.0003, 0.01, len(days))
        close = 1.0 * np.cumprod(1 + ret)
        open_ = np.concatenate([[close[0]], close[:-1]])  # 开盘 = 昨收
        wiggle = np.abs(rng.normal(0, 0.004, len(days)))
        high = np.maximum(open_, close) * (1 + wiggle)
        low = np.minimum(open_, close) * (1 - wiggle)
        volume = rng.integers(1_000_000, 50_000_000, len(days))
        rows.append(pd.DataFrame({
            "symbol": sym, "date": days, "open": open_, "high": high,
            "low": low, "close": close, "volume": volume,
        }))

    prices = pd.concat(rows, ignore_index=True)[PRICE_COLUMNS]
    snapshot_id = f"etf_syn_{pd.Timestamp(end).strftime('%Y%m%d')}_{_hash_prices(prices)}"
    meta = SnapshotMeta(
        snapshot_id=snapshot_id, symbols=symbols,
        start=_to_iso(start), end=_to_iso(end),
        actual_start=str(prices["date"].min().date()), actual_end=str(prices["date"].max().date()),
        source="synthetic", fetched_at=_now(), adjust="none",
        is_synthetic=True, from_cache=False, redistributable=None,
        notes=["模拟数据,仅供开发测试,不代表任何真实行情"],
    )

    out_dir = SNAPSHOT_DIR / snapshot_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out_dir / "prices.csv", index=False, lineterminator="\n")
    _write_json(out_dir / "meta.json", asdict(meta))
    _write_json(out_dir / "quality.json", asdict(QualityReport(snapshot_id=snapshot_id)))
    return meta
