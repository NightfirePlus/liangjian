"""快照:接口获取 → 校验整理 → 固定保存 → 回测只读快照。

目录约定(不要改,回测和界面都依赖):
    data/snapshots/<snapshot_id>/prices.csv   长表,列为 PRICE_COLUMNS
    data/snapshots/<snapshot_id>/meta.json    SnapshotMeta
    data/snapshots/<snapshot_id>/quality.json QualityReport

缓存容量管理(快照即缓存,只增不删会越积越多):
    超过容量上限时自动回收,顺序为 模拟数据优先 → 最久未访问优先。
    保护规则:总数 ≤ min_keep 不动;最近被 load_snapshot 访问过的不动;
    pins.json 标记保留的不动;回收明细写入 .eviction_log.jsonl,不静默。
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timedelta
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

# ---------------------------------------------------------------- 容量管理
# ETF 日线单标的约 50KB/年,500MB 约合数千个快照,阈值设得宽松
_CACHE_DEFAULT_MAX_BYTES = 500 * 1024 * 1024
_CACHE_DEFAULT_MIN_IDLE_DAYS = 7   # 最近 7 天内被访问过的快照不回收
_CACHE_DEFAULT_MIN_KEEP = 3        # 快照总数低于此值时不动
_PINS_FILE = "pins.json"
_ACCESS_FILE = ".accessed_at"
_EVICT_LOG_FILE = ".eviction_log.jsonl"


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


def _snapshot_dirs() -> list[Path]:
    """快照目录列表(排除 .开头 的管理文件)。"""
    if not SNAPSHOT_DIR.exists():
        return []
    return [p for p in SNAPSHOT_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")]


def _touch_access(snapshot_dir: Path) -> None:
    """记录访问时间(load/构建/回退命中时调用),供回收时的"最久未访问"排序用。"""
    (snapshot_dir / _ACCESS_FILE).write_text(_now(), encoding="utf-8")


def _last_access(snapshot_dir: Path, meta: SnapshotMeta) -> datetime:
    """最近访问时间;没有访问记录时退回快照生成时间。"""
    access_file = snapshot_dir / _ACCESS_FILE
    if access_file.exists():
        try:
            return datetime.fromisoformat(access_file.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    return datetime.fromisoformat(meta.fetched_at)


def cache_size_bytes() -> int:
    """数据快照目录当前占用字节数。"""
    total = 0
    for d in _snapshot_dirs():
        for f in d.iterdir():
            if f.is_file():
                total += f.stat().st_size
    return total


def _read_pins() -> set[str]:
    pins_file = SNAPSHOT_DIR / _PINS_FILE
    if not pins_file.exists():
        return set()
    try:
        return set(json.loads(pins_file.read_text(encoding="utf-8")))
    except Exception:
        return set()


def _write_pins(pins: set[str]) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    (SNAPSHOT_DIR / _PINS_FILE).write_text(
        json.dumps(sorted(pins), ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def pin_snapshot(snapshot_id: str) -> None:
    """标记快照为"保留",自动清理永不回收(如:想长期留存的实验数据)。"""
    pins = _read_pins()
    pins.add(snapshot_id)
    _write_pins(pins)


def unpin_snapshot(snapshot_id: str) -> None:
    pins = _read_pins()
    pins.discard(snapshot_id)
    _write_pins(pins)


def _log_eviction(details: list[dict], dry_run: bool) -> None:
    """回收明细写日志(追加一行 JSON),删了什么、为什么删全部可查。"""
    if not details:
        return
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"at": _now(), "dry_run": dry_run, "details": details}, ensure_ascii=False)
    with open(SNAPSHOT_DIR / _EVICT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def clean_snapshots(max_bytes: int = _CACHE_DEFAULT_MAX_BYTES,
                    min_idle_days: float = _CACHE_DEFAULT_MIN_IDLE_DAYS,
                    min_keep: int = _CACHE_DEFAULT_MIN_KEEP,
                    dry_run: bool = True) -> dict:
    """缓存回收:占用超过 max_bytes 时,按"模拟数据优先 → 最久未访问优先"释放。

    保护规则(满足其一就不回收):
    - 快照总数 ≤ min_keep;
    - 真实数据且最近 min_idle_days 天内被 load_snapshot 读过或刚生成;
    - 在 pins.json 里标记保留(pin_snapshot)。

    dry_run=True 只返回方案不真删(默认)。回收明细同时追加到
    data/snapshots/.eviction_log.jsonl。返回 {evicted, freed_bytes, remaining_bytes, details}。
    """
    dirs = _snapshot_dirs()
    if len(dirs) <= min_keep:
        return {"evicted": 0, "freed_bytes": 0, "remaining_bytes": cache_size_bytes(), "details": []}

    total = cache_size_bytes()
    if total <= max_bytes:
        return {"evicted": 0, "freed_bytes": 0, "remaining_bytes": total, "details": []}

    pins = _read_pins()
    now = datetime.now()
    idle_limit = timedelta(days=min_idle_days)

    entries = []
    for d in dirs:
        meta = _read_meta(d)
        if meta is not None:
            entries.append((d, meta, _last_access(d, meta)))
    # 模拟数据排最前(可零成本重建,且豁免闲置保护),其余按最久未访问排序
    entries.sort(key=lambda t: (not t[1].is_synthetic, t[2]))

    budget = len(dirs) - min_keep
    details: list[dict] = []
    # planned/freed 分开:预演模式下 planned 和 remaining 也要推进,才能算出完整方案
    planned, freed = 0, 0
    remaining = total
    for d, meta, last_access in entries:
        if planned >= budget or remaining <= max_bytes:
            break
        if meta.snapshot_id in pins:
            details.append({"snapshot_id": meta.snapshot_id, "action": "kept", "reason": "pinned"})
            continue
        if not meta.is_synthetic and now - last_access < idle_limit:
            details.append({"snapshot_id": meta.snapshot_id, "action": "kept", "reason": "recently_accessed"})
            continue
        size = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
        details.append({"snapshot_id": meta.snapshot_id,
                        "action": "evict" if not dry_run else "would_evict",
                        "reason": "synthetic" if meta.is_synthetic else "idle",
                        "freed_bytes": size})
        planned += 1
        remaining -= size
        if not dry_run:
            shutil.rmtree(d)
            freed += size

    _log_eviction(details, dry_run)
    return {"evicted": 0 if dry_run else planned, "freed_bytes": freed,
            "remaining_bytes": total - freed, "details": details}


def _maybe_evict() -> dict:
    """建快照后自动检查容量,超限按默认策略回收(保护规则兜底,明细写日志)。"""
    try:
        return clean_snapshots(dry_run=False)
    except Exception:
        return {"evicted": 0, "freed_bytes": 0, "remaining_bytes": cache_size_bytes(), "details": []}


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
            _touch_access(meta_file.parent)  # 缓存回退命中 = 访问,参与回收时的闲置排序
    return best


def find_covering_snapshot(
    symbols: list[str], start: str, end: str, adjust: str = "qfq",
) -> SnapshotMeta | None:
    """查找完整覆盖请求区间的真实本地快照。"""
    requested_symbols = set(str(symbol).strip() for symbol in symbols if str(symbol).strip())
    if not requested_symbols:
        return None
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    tolerance = pd.Timedelta(days=_COVER_TOLERANCE_DAYS)
    matches: list[tuple[SnapshotMeta, Path]] = []
    for meta_file in SNAPSHOT_DIR.glob("*/meta.json"):
        meta = _read_meta(meta_file.parent)
        if (
            meta is None
            or meta.is_synthetic
            or meta.adjust != adjust
            or not requested_symbols.issubset(set(meta.symbols))
        ):
            continue
        prices_file = meta_file.parent / "prices.csv"
        if not prices_file.exists():
            continue
        prices = pd.read_csv(prices_file, parse_dates=["date"], dtype={"symbol": str})
        covered = True
        for symbol in requested_symbols:
            part = prices[prices["symbol"] == symbol]
            if (
                part.empty
                or part["date"].min() > start_ts + tolerance
                or part["date"].max() < end_ts - tolerance
            ):
                covered = False
                break
        if covered:
            matches.append((meta, meta_file.parent))
    if not matches:
        return None
    meta, directory = max(matches, key=lambda item: item[0].fetched_at)
    _touch_access(directory)
    return meta


def ensure_market_snapshot(
    symbols: list[str], start: str, end: str, adjust: str = "qfq",
) -> tuple[SnapshotMeta, QualityReport, bool]:
    """优先复用本地真实行情；无覆盖数据时联网获取并固化快照。

    返回 ``(meta, quality, cache_hit)``。回测仍只读取固定快照，保证结果可复现。
    """
    cached = find_covering_snapshot(symbols, start, end, adjust)
    if cached is not None:
        _prices, meta, report = load_snapshot(cached.snapshot_id)
        return meta, report, True
    meta, report = build_snapshot(symbols, start, end, adjust=adjust, allow_cache_fallback=True)
    return meta, report, False


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
    used_sources: dict[str, str] = {}  # 每个标的实际用到的数据源,写入 meta.notes 如实标注
    for sym in symbols:
        try:
            frame = fetch_etf_daily(sym, start, end, adjust=adjust)
            used_sources[sym] = frame.attrs.get("source", source_name())
            parts.append(frame)
        except Exception as err:
            if not allow_cache_fallback:
                raise RuntimeError(f"标的 {sym} 取数失败(已禁用缓存回退): {err}") from err
            hit = _find_cached_prices(sym, start, end, adjust)
            if hit is None:
                raise RuntimeError(f"标的 {sym} 取数失败,且没有同口径缓存快照可回退: {err}") from err
            frame, cached_meta = hit
            used_sources[sym] = cached_meta.source
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
    # 数据源如实标注:全部来自同一源就写该源,混合时写主源+降级说明
    distinct_sources = set(used_sources.values())
    source = next(iter(distinct_sources)) if len(distinct_sources) == 1 else f"{source_name()}(+腾讯降级)"
    if len(distinct_sources) > 1:
        notes.append("各标的实际数据源: " + "; ".join(f"{s}: {used_sources[s]}" for s in symbols))

    meta = SnapshotMeta(
        snapshot_id=snapshot_id, symbols=symbols,
        start=_to_iso(start), end=_to_iso(end),
        actual_start=actual_start, actual_end=actual_end,
        source=source, fetched_at=fetched_at, adjust=adjust,
        is_synthetic=False, from_cache=bool(cached_symbols),
        redistributable=None, notes=notes,
    )

    out_dir = SNAPSHOT_DIR / snapshot_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_dir / "prices.csv", index=False, lineterminator="\n")
    _write_json(out_dir / "meta.json", asdict(meta))
    _write_json(out_dir / "quality.json", asdict(report))
    _touch_access(out_dir)
    _maybe_evict()  # 超限自动回收,保护规则兜底
    return meta, report


def load_snapshot(snapshot_id: str) -> tuple[pd.DataFrame, SnapshotMeta, QualityReport]:
    """读取快照;回测引擎只通过这个函数拿数据,不直接联网。"""
    out_dir = SNAPSHOT_DIR / snapshot_id
    if not out_dir.exists():
        raise FileNotFoundError(f"快照不存在: {snapshot_id}({out_dir})")
    _touch_access(out_dir)  # 读取 = 访问,写访问时间供回收排序
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
    _touch_access(out_dir)
    return meta
