"""测试版完整验收：真实/模拟快照 → 界面回测 → 实验 → 无 LLM 报告 → ZIP。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from streamlit.testing.v1 import AppTest

import config
from core.data import build_snapshot, load_snapshot, make_synthetic_snapshot
from core.report.export import evidence_filename


def run(snapshot_id: str, output: Path) -> dict:
    prices, meta, quality = load_snapshot(snapshot_id)
    assert not quality.has_error, "快照存在错误级质量问题"
    # 禁用所有 HTTP 请求：界面及计算阶段只能读取已冻结快照。
    with patch.object(config, "LLM_AVAILABLE", False), patch(
        "requests.sessions.Session.request", side_effect=AssertionError("验收期间不应调用模型或联网")
    ):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        assert not app.exception, [e.value for e in app.exception]
        app.selectbox[0].select(snapshot_id).run()
        next(b for b in app.button if b.label == "运行回测").click().run()
        assert not app.exception, [e.value for e in app.exception]
        backtest = app.session_state["backtest_result"]
        assert backtest["config"].snapshot_id == snapshot_id
        trades, daily = backtest["trades"], backtest["daily"]
        total = daily[daily["symbol"] == "__TOTAL__"]
        assert total["equity"].notna().all() and (total["cash"] >= -1e-6).all()
        if not trades.empty:
            assert (trades["exec_date"] > trades["signal_date"]).all()
        computed_return = total["equity"].iloc[-1] / total["equity"].iloc[0] - 1
        assert abs(computed_return - backtest["metrics"].total_return) < 1e-10

        next(b for b in app.button if b.label == "运行完整验证实验").click().run()
        assert not app.exception, [e.value for e in app.exception]
        suite = app.session_state["validation_result"]["report"]
        assert suite.verdict is None
        assert abs(suite.baseline["total_return"] - computed_return) < 1e-10
        assert all(r.get("error") is None for exp in suite.experiments.values() for r in exp.rows)
        assert next(b for b in app.button if b.label == "生成 AI 分析").disabled
        saved = app.session_state["evidence_result"]
        pack, report, bundle = saved["pack"], saved["analysis"], saved["bundle"]
        assert report["mode"] == "facts" and not report["tool_log"]
        assert len(report["findings"]) == 4
        assert pack["entries"][f"{pack['run_id']}:context"]["data"]["snapshot"]["is_synthetic"] == meta.is_synthetic
        for finding in report["findings"]:
            assert all(eid in pack["entries"] for eid in finding["evidence_refs"])
        charts = len(app.get("plotly_chart"))
        assert charts == 8, f"应展示原始回测及七张证据图，实际 {charts}"
        assert len(app.get("download_button")) == 3

        with ZipFile(BytesIO(bundle)) as archive:
            assert archive.testzip() is None
            names = archive.namelist()
            assert len([n for n in names if n.endswith(".html")]) == 7
            for eid, entry in pack["entries"].items():
                assert evidence_filename(eid, entry) in names
            for name in ("report.json", "evidence.json"):
                json.loads(archive.read(name), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            markdown = archive.read("report.md").decode("utf-8")
            links = re.findall(r"\]\((evidence/[^)]+)\)", markdown)
            assert links and all(link in names for link in links)
            exported_trades = pd.read_csv(BytesIO(archive.read("evidence/table_trades.csv")))
            assert len(exported_trades) == len(trades)

        # 配置变化后旧报告不能继续显示或下载。
        app.number_input[0].set_value(6).run()
        assert not app.exception
        assert len(app.get("download_button")) == 0

    output.mkdir(parents=True, exist_ok=True)
    filename = f"{snapshot_id}_report.zip"
    (output / filename).write_bytes(bundle)
    summary = {
        "version": config.APP_VERSION,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "llm": "skipped; HTTP forbidden during UI run",
        "snapshot_id": snapshot_id, "source": meta.source,
        "is_synthetic": meta.is_synthetic, "price_rows": len(prices),
        "actual_start": meta.actual_start, "actual_end": meta.actual_end,
        "trade_rows": len(trades), "total_return": computed_return,
        "experiments": {key: len(value.rows) for key, value in suite.experiments.items()},
        "findings": len(report["findings"]), "rendered_charts": charts,
        "archive_files": len(names), "archive": filename,
        "evidence_hash": pack["evidence_hash"], "stale_result_hidden": True,
    }
    (output / f"{snapshot_id}_acceptance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--snapshot-id")
    mode.add_argument("--live", action="store_true", help="联网创建真实快照，失败时不改用模拟数据")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "acceptance")
    args = parser.parse_args()
    sid = args.snapshot_id
    if not sid:
        if args.live:
            meta, _ = build_snapshot(["510300"], "2023-01-01", "2024-12-31", allow_cache_fallback=False)
        else:
            meta = make_synthetic_snapshot(["510300"], "2023-01-01", "2024-12-31", seed=21)
        sid = meta.snapshot_id
    print(json.dumps(run(sid, args.output), ensure_ascii=False, indent=2))
