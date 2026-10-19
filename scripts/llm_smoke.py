"""检查 .env、OpenAI 兼容接口、自然语言解析和可选的证据报告调用。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from core.nlp import parse_user_input
from core.nlp.llm_client import complete_json
from core.report import build_evidence, generate_report
from core.schemas import ExperimentResult, ValidationReport


def _report_pack() -> dict:
    def experiment(rows: list[dict]) -> ExperimentResult:
        return ExperimentResult("llm_smoke", rows, "mixed")

    rows = [
        {"fast": 3, "slow": 15, "excess_return": 0.03, "total_return": 0.04, "n_trades": 10},
        {
            "fast": 5, "slow": 20, "excess_return": -0.01,
            "total_return": -0.02, "n_trades": 8, "is_baseline": True,
        },
        {"fast": 7, "slow": 25, "excess_return": 0.01, "total_return": 0.02, "n_trades": 6},
    ]
    suite = ValidationReport(
        "live_llm_smoke",
        {"excess_return": -0.01, "total_return": -0.02, "n_trades": 8},
        {
            "cost_sensitivity": experiment([
                {**row, "slippage_bps": index} for index, row in enumerate(rows)
            ]),
            "strategy_parameter_robustness": experiment(rows),
            "market_period_robustness": experiment([
                {**row, "period": str(index)} for index, row in enumerate(rows[:2])
            ]),
        },
        context={"snapshot": {"is_synthetic": True}, "limitations": ["仅用于接口检查。"]},
        baseline_daily=[
            {"date": "2024-01-01", "equity": 100, "benchmark_equity": 100},
            {"date": "2024-01-02", "equity": 98, "benchmark_equity": 99},
        ],
        baseline_trades=[
            {
                "side": "buy", "exec_date": "2024-01-01", "gross": 100,
                "commission": 1, "slippage_cost": 1,
            },
            {
                "side": "sell", "exec_date": "2024-01-02", "gross": 98,
                "commission": 1, "slippage_cost": 1,
            },
        ],
    )
    return build_evidence(suite)


def run(*, include_report: bool) -> dict:
    if not config.LLM_AVAILABLE:
        raise RuntimeError("LLM 配置不完整，请检查仓库根目录的 .env")

    ping = complete_json(
        "只返回合法 JSON object，不要代码块。",
        '返回 {"ok": true, "purpose": "connection_test"}',
    )
    if ping.get("ok") is not True:
        raise RuntimeError("兼容接口没有返回预期的连通性结果")

    parsed = parse_user_input(
        "用沪深300ETF做双均线策略，快线5日，慢线20日，回测2023-01-03到2024-12-31"
    )
    strategy = parsed.strategy_config
    if parsed.needs_clarification or parsed.degraded or strategy is None:
        raise RuntimeError(f"自然语言解析没有使用模型成功完成：{parsed.degraded_reason or parsed.questions}")

    result = {
        "status": "passed",
        "parser_model": config.LLM_MODEL,
        "report_model": config.LLM_REPORT_MODEL,
        "timeout_seconds": config.LLM_TIMEOUT_SECONDS,
        "connection": True,
        "parser": {
            "template": strategy.template,
            "symbols": strategy.symbols,
            "params": strategy.params,
            "start": strategy.start,
            "end": strategy.end,
        },
        "report": "skipped",
    }
    if include_report:
        report = generate_report(_report_pack(), use_llm=True)
        if report["mode"] != "ai":
            raise RuntimeError(f"{report['notice']} {report.get('error', '')}".strip())
        result["report"] = {
            "mode": report["mode"],
            "findings": len(report["findings"]),
            "dimensions": sorted({item["dimension"] for item in report["findings"]}),
            "tool_calls": len(report["tool_log"]),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="同时验证较慢的 AI 证据报告")
    args = parser.parse_args()
    print(json.dumps(run(include_report=args.report), ensure_ascii=False, indent=2))
