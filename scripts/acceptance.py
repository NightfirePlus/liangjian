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
import core.nlp.parser as parser_mod
from core.report.export import evidence_filename


def run(snapshot_id: str, output: Path) -> dict:
    prices, meta, quality = load_snapshot(snapshot_id)
    assert not quality.has_error, "快照存在错误级质量问题"
    # 禁用所有 HTTP 请求：界面及计算阶段只能读取已冻结快照。
    with patch.object(config, "LLM_AVAILABLE", False), patch.object(
        parser_mod, "LLM_AVAILABLE", False,
    ), patch(
        "requests.sessions.Session.request", side_effect=AssertionError("验收期间不应调用模型或联网")
    ):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        assert not app.exception, [e.value for e in app.exception]
        llm_fields = {field.label for field in app.text_input}
        assert {
            "API Key", "OpenAI 兼容接口 URL", "解析与问答模型", "报告模型（可选）",
        }.issubset(llm_fields)
        fields = {field.label: field for field in app.text_input}
        fields["API Key"].set_value("session-test-key")
        fields["OpenAI 兼容接口 URL"].set_value("https://example.test/v1")
        fields["解析与问答模型"].set_value("test-model")
        fields["报告模型（可选）"].set_value("test-report-model")
        next(b for b in app.button if b.label == "保存").click().run()
        assert app.session_state["runtime_llm_config"] == {
            "api_key": "session-test-key",
            "base_url": "https://example.test/v1",
            "model": "test-model",
            "report_model": "test-report-model",
        }
        next(b for b in app.button if b.label == "恢复服务器配置").click().run()
        assert "runtime_llm_config" not in app.session_state
        next(b for b in app.button if b.label == "用双均线开始").click().run()
        assert app.session_state["main_nav"] == "量化策略验证"
        assert app.session_state["strategy_draft"].template == "ma_cross"
        next(b for b in app.button if b.label.endswith("使用文档")).click().run()
        assert any(title.value == "使用文档" for title in app.title)
        next(b for b in app.button if b.label.endswith("AI 金融问答")).click().run()
        next(b for b in app.button if b.label == "双均线策略有什么局限？").click().run()
        assert len(app.session_state.get("finance_chat", [])) == 2
        next(b for b in app.button if b.label == "用双均线创建验证草案").click().run()
        assert app.session_state["main_nav"] == "量化策略验证"
        assert app.session_state["strategy_draft"].template == "ma_cross"
        snapshot_select = next(s for s in app.selectbox if s.label == "当前实验数据")
        snapshot_select.select(snapshot_id).run()
        # 三种入口都写入同一份草案；先验证模板与快捷创建，再用多轮对话形成最终草案。
        next(b for b in app.button if b.label == "10日 / 30日").click().run()
        assert app.session_state["strategy_draft"].origin == "preset"
        assert (app.session_state["strategy_draft"].fast, app.session_state["strategy_draft"].slow) == (10, 30)
        next(b for b in app.button if b.label == "生成策略草案").click().run()
        assert app.session_state["strategy_draft"].origin == "quick"
        assert (app.session_state["strategy_draft"].fast, app.session_state["strategy_draft"].slow) == (5, 20)

        app.chat_input[0].set_value("用沪深300ETF测试双均线策略").run()
        assert app.session_state["strategy_draft"].origin == "quick"  # 信息不足时不覆盖现有草案
        requested_start = (pd.Timestamp(meta.actual_start) - pd.Timedelta(days=1)).date().isoformat()
        app.chat_input[0].set_value(
            f"使用5日和20日，回测{requested_start}到{meta.actual_end}"
        ).run()
        draft = app.session_state["strategy_draft"]
        assert draft.origin == "conversation" and (draft.fast, draft.slow) == (5, 20)
        assert draft.symbol == "510300"
        assert draft.start == meta.actual_start and draft.normalization_notes

        next(b for b in app.button if b.label == "确认这份策略").click().run()
        assert app.session_state["confirmed_draft_hash"] == draft.execution_hash(snapshot_id)
        next(b for b in app.button if b.label == "开始验证并生成报告").click().run()
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

        suite = app.session_state["validation_result"]["report"]
        assert suite.verdict is None
        assert abs(suite.baseline["total_return"] - computed_return) < 1e-10
        assert all(r.get("error") is None for exp in suite.experiments.values() for r in exp.rows)
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

        # 报告完成后的普通追问只读取现有证据，不重新解析策略或运行回测。
        strategy_inputs_before = list(app.session_state["strategy_input_history"])
        validation_id_before = app.session_state["validation_result"]["report"].validation_id
        app.chat_input[0].set_value("参数实验说明了什么？").run()
        assert app.session_state["strategy_input_history"] == strategy_inputs_before
        assert len(app.session_state.get("report_qa_history", [])) == 1, {
            "qa_history": app.session_state.get("report_qa_history"),
            "last_chat": app.session_state["strategy_chat"][-1],
            "strategy_inputs": app.session_state.get("strategy_input_history"),
        }
        assert app.session_state["validation_result"]["report"].validation_id == validation_id_before
        assert "证据附件" in app.session_state["strategy_chat"][-1]["content"]

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

        # 第二种可执行模板也必须走完整 Agent 链路，而不只通过底层单元测试。
        next(b for b in app.button if b.label == "新建对话").click().run()
        next(b for b in app.button if b.label == "20日突破 / 10日退出").click().run()
        breakout_draft = app.session_state["strategy_draft"]
        assert breakout_draft.template == "breakout"
        assert (breakout_draft.entry_window, breakout_draft.exit_window) == (20, 10)
        next(b for b in app.button if b.label == "确认这份策略").click().run()
        assert app.session_state["confirmed_draft_hash"] == breakout_draft.execution_hash(snapshot_id)
        next(b for b in app.button if b.label == "开始验证并生成报告").click().run()
        assert not app.exception, [e.value for e in app.exception]
        breakout_backtest = app.session_state["backtest_result"]
        breakout_suite = app.session_state["validation_result"]["report"]
        assert breakout_backtest["config"].template == "breakout"
        breakout_parameter_rows = breakout_suite.experiments[
            "strategy_parameter_robustness"
        ].rows
        assert len(breakout_parameter_rows) == 9
        assert all(
            row["entry_window"] > row["exit_window"]
            and row.get("error") is None
            for row in breakout_parameter_rows
        )
        assert len(app.get("plotly_chart")) == 8

        # RSI 均值回归同样贯通确认、回测、参数热力图和报告。
        next(b for b in app.button if b.label == "新建对话").click().run()
        next(b for b in app.button if b.label == "14日 RSI / 30买入 / 70卖出").click().run()
        rsi_draft = app.session_state["strategy_draft"]
        assert rsi_draft.template == "rsi_reversion"
        assert (rsi_draft.rsi_period, rsi_draft.buy_threshold, rsi_draft.sell_threshold) == (
            14, 30, 70,
        )
        next(b for b in app.button if b.label == "确认这份策略").click().run()
        rsi_draft = app.session_state["strategy_draft"]
        assert app.session_state["confirmed_draft_hash"] == rsi_draft.execution_hash(snapshot_id)
        next(b for b in app.button if b.label == "开始验证并生成报告").click().run()
        assert not app.exception, [e.value for e in app.exception]
        rsi_backtest = app.session_state["backtest_result"]
        rsi_suite = app.session_state["validation_result"]["report"]
        assert rsi_backtest["config"].template == "rsi_reversion"
        rsi_parameter_rows = rsi_suite.experiments[
            "strategy_parameter_robustness"
        ].rows
        assert len(rsi_parameter_rows) == 9
        assert all(
            0 < row["buy_threshold"] < row["sell_threshold"] < 100
            and row.get("error") is None
            for row in rsi_parameter_rows
        )
        assert len(app.get("plotly_chart")) == 8

    output.mkdir(parents=True, exist_ok=True)
    filename = f"{snapshot_id}_report.zip"
    (output / filename).write_bytes(bundle)
    summary = {
        "version": config.APP_VERSION,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "llm": "skipped; HTTP forbidden during UI run",
        "strategy_entry_points": ["preset", "quick", "conversation"],
        "strategy_templates": ["ma_cross", "breakout", "rsi_reversion"],
        "strategy_confirmation_required": True,
        "report_evidence_qa": True,
        "snapshot_id": snapshot_id, "source": meta.source,
        "is_synthetic": meta.is_synthetic, "price_rows": len(prices),
        "actual_start": meta.actual_start, "actual_end": meta.actual_end,
        "trade_rows": len(trades), "total_return": computed_return,
        "experiments": {key: len(value.rows) for key, value in suite.experiments.items()},
        "findings": len(report["findings"]), "rendered_charts": charts,
        "archive_files": len(names), "archive": filename,
        "evidence_hash": pack["evidence_hash"], "stale_result_hidden": True,
        "breakout_trade_rows": len(breakout_backtest["trades"]),
        "breakout_parameter_points": len(breakout_parameter_rows),
        "rsi_trade_rows": len(rsi_backtest["trades"]),
        "rsi_parameter_points": len(rsi_parameter_rows),
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
