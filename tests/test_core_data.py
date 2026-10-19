"""数据模块自测:资产池、质检、快照读写、合成快照。

默认不联网;RUN_LIVE=1 时附加真实取数与真实建快照:
    RUN_LIVE=1 python -m pytest tests/ -q
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import core.data.snapshot as snap
import core.data.sources as src
from core.data.sources import fetch_etf_daily
from core.data.universe import anomaly_limit, category_of, get_etf, get_universe
from core.data.validate import validate_prices
from core.schemas import PRICE_COLUMNS

LIVE = os.environ.get("RUN_LIVE") == "1"


@pytest.fixture
def snap_dir(tmp_path, monkeypatch):
    """把快照目录指到临时目录,测试不污染仓库。"""
    monkeypatch.setattr(snap, "SNAPSHOT_DIR", tmp_path / "snapshots")
    return tmp_path / "snapshots"


def make_bars(symbol: str, start: str = "2025-01-02", n: int = 60) -> pd.DataFrame:
    """构造结构良好的长表行情(单标的)。"""
    dates = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(42)
    close = 1 + np.cumsum(rng.normal(0, 0.01, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"symbol": symbol, "date": dates, "open": open_,
                         "high": high, "low": low, "close": close,
                         "volume": np.full(n, 1000, dtype=np.int64)})


# ---------------------------------------------------------------- 资产池
def test_universe_lookup():
    assert get_etf("510300").category == "broad"
    assert get_etf("159915").category == "star"
    assert get_etf("999999") is None
    assert category_of("588000") == "star"
    assert category_of("999999") == "broad"  # 未收录按 10% 涨跌停处理,偏保守
    assert anomaly_limit("510300") == 0.11
    assert anomaly_limit("159915") == 0.21
    assert anomaly_limit("513100") is None  # 跨境无涨跌幅限制,跳过检查
    symbols = {e.symbol for e in get_universe()}
    assert {"510300", "159915", "588000", "513100", "518880"} <= symbols


# ---------------------------------------------------------------- 质检
def test_validate_clean_table_passes():
    bars = make_bars("510300")
    cleaned, report = validate_prices(bars, "snap_x")
    assert report.issues == []
    pd.testing.assert_frame_equal(cleaned, bars)


def test_validate_sorts_and_dedupes():
    bars = make_bars("510300")
    shuffled = bars.sample(frac=1.0, random_state=7)
    duplicated = pd.concat([shuffled, bars.iloc[[-1]]], ignore_index=True)
    cleaned, report = validate_prices(duplicated, "snap_x")
    kinds = {i.kind for i in report.issues}
    assert {"unsorted", "duplicate"} <= kinds
    assert len(cleaned) == len(bars)
    pd.testing.assert_frame_equal(cleaned, bars)


def test_validate_nonpositive_and_ohlc():
    bars = make_bars("510300")
    bad_price = bars.iloc[[5]].copy()
    bad_price.loc[:, "date"] = pd.Timestamp("2025-06-01")
    bad_price.loc[:, "close"] = -1.0
    bad_ohlc = bars.iloc[[10]].copy()
    bad_ohlc.loc[:, "date"] = pd.Timestamp("2025-06-02")
    bad_ohlc.loc[:, "high"] = bad_ohlc["low"] - 0.01
    dirty = pd.concat([bars, bad_price, bad_ohlc], ignore_index=True)
    cleaned, report = validate_prices(dirty, "snap_x")
    by_kind = {i.kind: i for i in report.issues}
    assert by_kind["nonpositive_price"].severity == "error"
    assert by_kind["nonpositive_price"].action == "dropped"
    assert by_kind["ohlc_inconsistent"].severity == "error"
    assert by_kind["ohlc_inconsistent"].action == "flagged"
    assert len(cleaned) == len(bars) + 1  # 非正价格行被删,OHLC 行保留仅标记


def test_validate_anomaly_threshold_by_category():
    # 15% 单日涨幅:宽基(10% 涨跌停)触发,创业(20%)与跨境(无限制)不触发
    bars = make_bars("510300")
    bars.loc[30:, ["open", "high", "low", "close"]] = \
        bars.loc[30:, ["open", "high", "low", "close"]] * 1.15
    _, report = validate_prices(bars, "s")
    assert "price_anomaly" in {i.kind for i in report.issues}

    star = bars.copy()
    star.loc[:, "symbol"] = "159915"
    _, report_star = validate_prices(star, "s")
    assert "price_anomaly" not in {i.kind for i in report_star.issues}

    cross = bars.copy()
    cross.loc[:, "symbol"] = "513100"
    _, report_cross = validate_prices(cross, "s")
    assert "price_anomaly" not in {i.kind for i in report_cross.issues}


def test_validate_missing_days_misaligned_short_history():
    a = make_bars("510300", n=60)
    b = make_bars("159915", n=60).drop([10, 11, 12]).head(50)
    c = make_bars("513100", n=5)
    dirty = pd.concat([a, b, c], ignore_index=True)
    cleaned, report = validate_prices(dirty, "s")
    kinds = {i.kind for i in report.issues}
    assert {"missing_days", "misaligned", "short_history"} <= kinds
    assert len(cleaned) == len(dirty)  # 三类问题都只标记,不删行


# ---------------------------------------------------------------- 快照
def test_synthetic_snapshot_roundtrip(snap_dir):
    meta = snap.make_synthetic_snapshot(["510300", "159915"], "2025-06-01", "2025-09-30", seed=7)
    assert meta.is_synthetic is True
    assert meta.source == "synthetic"
    prices, meta2, report = snap.load_snapshot(meta.snapshot_id)
    assert list(prices.columns) == PRICE_COLUMNS
    assert set(prices["symbol"]) == {"510300", "159915"}
    assert meta == meta2
    assert report.snapshot_id == meta.snapshot_id
    assert [m.snapshot_id for m in snap.list_snapshots()] == [meta.snapshot_id]


def test_synthetic_snapshot_deterministic_id(snap_dir):
    m1 = snap.make_synthetic_snapshot(["510300"], "2025-06-01", "2025-09-30", seed=7)
    m2 = snap.make_synthetic_snapshot(["510300"], "2025-06-01", "2025-09-30", seed=7)
    assert m1.snapshot_id == m2.snapshot_id
    assert len(snap.list_snapshots()) == 1  # 重复构建覆盖同一份,不越积越多


def test_build_snapshot_raises_without_cache(snap_dir, monkeypatch):
    def boom(symbol, start, end, adjust="qfq"):
        raise RuntimeError("模拟网络故障")

    monkeypatch.setattr(snap, "fetch_etf_daily", boom)
    with pytest.raises(RuntimeError, match="模拟网络故障"):
        snap.build_snapshot(["510300"], "2025-06-01", "2025-09-30")


def test_build_snapshot_cache_fallback(snap_dir, monkeypatch):
    # 先正常建一个真实快照
    monkeypatch.setattr(snap, "fetch_etf_daily",
                        lambda sym, start, end, adjust="qfq": make_bars(sym, start="2025-06-01", n=100))
    meta, _ = snap.build_snapshot(["510300"], "2025-06-01", "2025-09-30")

    # 再模拟 510300 接口故障,应回退到刚建好的快照
    def flaky(symbol, start, end, adjust="qfq"):
        if symbol == "510300":
            raise RuntimeError("模拟网络故障")
        return make_bars(symbol, start="2025-06-01", n=100)

    monkeypatch.setattr(snap, "fetch_etf_daily", flaky)
    meta2, _ = snap.build_snapshot(["510300", "159915"], "2025-06-01", "2025-09-30")
    assert meta2.from_cache is True
    assert "510300" in "".join(meta2.notes)
    prices, _, _ = snap.load_snapshot(meta2.snapshot_id)
    assert set(prices["symbol"]) == {"510300", "159915"}
    assert meta2.snapshot_id != meta.snapshot_id  # 数据不同,id 不同


def test_ensure_market_snapshot_reuses_covering_real_cache(snap_dir, monkeypatch):
    calls = []

    def fetch(symbol, start, end, adjust="qfq"):
        calls.append((symbol, start, end, adjust))
        frame = make_bars(symbol, start="2025-01-02", n=130)
        frame.attrs["source"] = "test/source"
        return frame

    monkeypatch.setattr(snap, "fetch_etf_daily", fetch)
    original, _ = snap.build_snapshot(["510300"], "2025-01-01", "2025-06-30")
    calls.clear()

    meta, report, cache_hit = snap.ensure_market_snapshot(
        ["510300"], "2025-01-01", "2025-06-30",
    )

    assert cache_hit is True
    assert meta.snapshot_id == original.snapshot_id
    assert not report.has_error
    assert calls == []


def test_ensure_market_snapshot_fetches_and_caches_when_missing(snap_dir, monkeypatch):
    calls = []

    def fetch(symbol, start, end, adjust="qfq"):
        calls.append((symbol, start, end, adjust))
        frame = make_bars(symbol, start="2025-01-02", n=130)
        frame.attrs["source"] = "test/source"
        return frame

    monkeypatch.setattr(snap, "fetch_etf_daily", fetch)
    meta, report, cache_hit = snap.ensure_market_snapshot(
        ["510300"], "2025-01-01", "2025-06-30",
    )

    assert cache_hit is False
    assert calls == [("510300", "2025-01-01", "2025-06-30", "qfq")]
    assert not report.has_error
    assert snap.find_covering_snapshot(["510300"], "2025-01-01", "2025-06-30") == meta


# ---------------------------------------------------------------- 数据源降级
def test_fetch_falls_back_to_tencent(monkeypatch):
    # 东财限流 → 自动降级腾讯源;成交量 股→手;来源写入 df.attrs
    def em_boom(**kwargs):
        raise RuntimeError("东财限流")

    monkeypatch.setattr(src.ak, "fund_etf_hist_em", em_boom)
    fake_tx = pd.DataFrame({
        "date": ["2025-09-26", "2025-09-29"], "open": [4.55, 4.52],
        "close": [4.528, 4.605], "high": [4.57, 4.63], "low": [4.52, 4.51],
        "volume": [715305900, 800000000],
    })
    monkeypatch.setattr(src.ak, "stock_zh_a_hist_tx", lambda **kwargs: fake_tx)
    df = src.fetch_etf_daily("510300", "2025-09-26", "2025-09-29")
    assert list(df.columns) == PRICE_COLUMNS
    assert df["volume"].iloc[0] == pytest.approx(7153059)  # 股 → 手
    assert df.attrs["source"] == "akshare/stock_zh_a_hist_tx"


def test_fetch_prefers_eastmoney_when_available(monkeypatch):
    # 东财正常时直接用东财,不降级,不换算成交量
    fake_em = pd.DataFrame({
        "日期": ["2025-09-26"], "开盘": [4.55], "收盘": [4.528],
        "最高": [4.57], "最低": [4.52], "成交量": [7153059],
    })
    monkeypatch.setattr(src.ak, "fund_etf_hist_em", lambda **kwargs: fake_em)
    df = src.fetch_etf_daily("510300", "2025-09-26", "2025-09-26")
    assert df.attrs["source"] == "akshare/fund_etf_hist_em"
    assert df["volume"].iloc[0] == 7153059  # 东财本就是手,不换算


# ---------------------------------------------------------------- 缓存回收
def _make_snapshot_files(snap_dir, snap_id, *, size=0, synthetic=False, fetched_days_ago=10):
    """直接造一个快照目录(绕过取数),用于测试回收策略。"""
    d = snap_dir / snap_id
    d.mkdir(parents=True)
    (d / "prices.csv").write_text("symbol,date,open,high,low,close,volume\n", encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({
        "snapshot_id": snap_id, "symbols": ["510300"],
        "start": "2025-01-01", "end": "2025-06-30",
        "actual_start": "2025-01-02", "actual_end": "2025-06-30",
        "source": "test", "adjust": "qfq", "is_synthetic": synthetic,
        "from_cache": False, "redistributable": None, "notes": [],
        "fetched_at": (datetime.now() - timedelta(days=fetched_days_ago)).isoformat(timespec="seconds"),
    }), encoding="utf-8")
    (d / "quality.json").write_text(json.dumps({"snapshot_id": snap_id, "issues": []}), encoding="utf-8")
    if size:
        (d / "big.bin").write_bytes(b"\0" * size)
    return d


def test_cache_under_limit_does_nothing(snap_dir):
    _make_snapshot_files(snap_dir, "snap_a", fetched_days_ago=30)
    assert snap.cache_size_bytes() > 0
    result = snap.clean_snapshots(max_bytes=10**9, min_keep=1, dry_run=False)
    assert result["evicted"] == 0
    assert (snap_dir / "snap_a").exists()


def test_clean_snapshots_evicts_oldest_idle(snap_dir):
    old = _make_snapshot_files(snap_dir, "snap_old", size=1000, fetched_days_ago=30)
    recent = _make_snapshot_files(snap_dir, "snap_recent", size=1000, fetched_days_ago=0)
    result = snap.clean_snapshots(max_bytes=500, min_keep=1, dry_run=False)
    assert result["evicted"] == 1
    assert not old.exists() and recent.exists()


def test_clean_snapshots_synthetic_evicted_first(snap_dir):
    syn = _make_snapshot_files(snap_dir, "syn", size=1000, synthetic=True, fetched_days_ago=0)
    real = _make_snapshot_files(snap_dir, "real", size=1000, fetched_days_ago=30)
    result = snap.clean_snapshots(max_bytes=500, min_keep=1, dry_run=False)
    assert not syn.exists() and real.exists()  # 模拟数据可零成本重建,优先删


def test_clean_snapshots_pin_protection(snap_dir):
    _make_snapshot_files(snap_dir, "pinned_snap", size=1000, fetched_days_ago=30)
    _make_snapshot_files(snap_dir, "other", size=1000, fetched_days_ago=30)
    snap.pin_snapshot("pinned_snap")
    result = snap.clean_snapshots(max_bytes=500, min_keep=1, dry_run=False)
    assert (snap_dir / "pinned_snap").exists()
    assert not (snap_dir / "other").exists()
    snap.unpin_snapshot("pinned_snap")
    assert "pinned_snap" not in json.loads((snap_dir / "pins.json").read_text(encoding="utf-8"))


def test_load_snapshot_updates_access_time(snap_dir):
    _make_snapshot_files(snap_dir, "old_idle", size=1000, fetched_days_ago=30)
    _make_snapshot_files(snap_dir, "used", size=1000, fetched_days_ago=30)
    snap.load_snapshot("used")  # 刚访问过,应受闲置保护
    result = snap.clean_snapshots(max_bytes=500, min_keep=1, dry_run=False)
    assert (snap_dir / "used").exists()
    assert not (snap_dir / "old_idle").exists()


def test_clean_snapshots_dry_run_deletes_nothing(snap_dir):
    _make_snapshot_files(snap_dir, "snap_x", size=1000, fetched_days_ago=30)
    _make_snapshot_files(snap_dir, "snap_y", size=1000, fetched_days_ago=30)
    result = snap.clean_snapshots(max_bytes=500, min_keep=1, dry_run=True)
    assert result["evicted"] == 0
    assert any(d["action"] == "would_evict" for d in result["details"])
    assert (snap_dir / "snap_x").exists() and (snap_dir / "snap_y").exists()
@pytest.mark.skipif(not LIVE, reason="RUN_LIVE=1 时联网跑真实取数")
def test_live_fetch_etf_daily():
    df = fetch_etf_daily("510300", "2025-06-01", "2025-09-30")
    assert list(df.columns) == PRICE_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].is_monotonic_increasing
    assert not df.empty
    assert (df[["open", "high", "low", "close"]] > 0).all().all()


@pytest.mark.skipif(not LIVE, reason="RUN_LIVE=1 时联网建真实快照")
def test_live_build_and_load_snapshot(snap_dir):
    meta, report = snap.build_snapshot(["510300", "159915"], "2025-06-01", "2025-09-30")
    prices, meta2, _ = snap.load_snapshot(meta.snapshot_id)
    assert set(prices["symbol"]) == {"510300", "159915"}
    assert meta == meta2
    assert report.has_error is False
