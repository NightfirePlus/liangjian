from copy import deepcopy
from io import BytesIO
from zipfile import ZipFile

import pytest

from core.schemas import ExperimentResult, ValidationReport
from core.report import (
    answer_report_question, build_evidence, chart_for, export_bundle, generate_report,
)
from core.report.analysis import fact_report, validate_findings
from core.report.evidence import EvidenceTools, dumps
from core.report.qa import format_report_answer, validate_report_answer
from core.nlp.llm_client import LLMError


@pytest.fixture
def pack():
    def experiment(rows):
        return ExperimentResult("exp", rows, "mixed")
    rows = [{"fast": 3, "slow": 15, "excess_return": .03, "total_return": .04, "n_trades": 10},
            {"fast": 5, "slow": 20, "excess_return": -.01, "total_return": -.02, "n_trades": 0, "is_baseline": True},
            {"fast": 7, "slow": 25, "excess_return": None, "error": "missing"}]
    suite = ValidationReport("run_abc", {"excess_return": -.01}, {
        "cost_sensitivity": experiment([{**r, "slippage_bps": i} for i, r in enumerate(rows)]),
        "strategy_parameter_robustness": experiment(rows),
        "market_period_robustness": experiment([{**r, "period": str(i)} for i, r in enumerate(rows)]),
    }, context={"snapshot": {"is_synthetic": True}, "limitations": ["局部测试。"]},
       baseline_daily=[{"date": "2024-01-01", "equity": 100, "benchmark_equity": 100},
                       {"date": "2024-01-02", "equity": 90, "benchmark_equity": 95}],
       baseline_trades=[{"side": "buy", "exec_date": "2024-01-01", "gross": 100, "commission": 1, "slippage_cost": 1},
                        {"side": "sell", "exec_date": "2024-01-02", "gross": 90, "commission": 1, "slippage_cost": 1}])
    return build_evidence(suite)


def test_summary_denominator_missing_and_no_trades(pack):
    summary = pack["entries"]["run_abc:summary:parameters"]["data"]
    assert summary["valid_count"] == 2
    assert summary["positive_ratio"] == .5
    assert summary["missing_count"] == 1
    assert summary["zero_trade_count"] == 1
    assert "verdict" not in dumps(pack)


def test_fact_report_and_offline_export(pack):
    report = generate_report(pack)
    assert report["mode"] == "facts"
    assert "-1.00%" in report["findings"][0]["statement"]
    archive = ZipFile(BytesIO(export_bundle(pack, report)))
    assert "report.md" in archive.namelist()
    assert "evidence/chart_parameters.html" in archive.namelist()
    markdown = archive.read("report.md").decode()
    assert "模拟数据" in markdown
    assert "evidence/table_parameters.csv" in markdown


def test_charts_preserve_negative_and_missing(pack):
    heatmap = chart_for(pack, "run_abc:chart:parameters")
    values = [x for row in heatmap.data[0].z for x in row]
    assert -.01 in values and .03 in values
    assert len(chart_for(pack, "run_abc:chart:cost").data) == 2
    assert len(chart_for(pack, "run_abc:chart:periods").data) == 3
    assert pack["entries"]["run_abc:table:roundtrips"]["data"][0]["net_pnl"] == -14
    assert chart_for(pack, "run_abc:chart:drawdown").data[0].y[1] == pytest.approx(-.1)


def test_breakout_parameter_heatmap_and_cell_tool(pack):
    breakout_pack = deepcopy(pack)
    table = breakout_pack["entries"]["run_abc:table:parameters"]
    table["data"] = [
        {"entry_window": 15, "exit_window": 8, "excess_return": .03,
         "total_return": .04, "is_baseline": False},
        {"entry_window": 20, "exit_window": 10, "excess_return": -.01,
         "total_return": -.02, "is_baseline": True},
    ]

    heatmap = chart_for(breakout_pack, "run_abc:chart:parameters")
    cell = EvidenceTools(breakout_pack).call(
        "get_breakout_parameter_cell",
        {"entry_window": 20, "exit_window": 10},
    )

    assert heatmap.layout.xaxis.title.text == "退出周期（日）"
    assert heatmap.layout.yaxis.title.text == "突破周期（日）"
    assert cell["rows"][0]["excess_return"] == -.01


def test_rsi_parameter_heatmap_and_cell_tool(pack):
    rsi_pack = deepcopy(pack)
    table = rsi_pack["entries"]["run_abc:table:parameters"]
    table["data"] = [
        {"rsi_period": 14, "buy_threshold": 25.0, "sell_threshold": 65.0,
         "excess_return": .02, "total_return": .03, "is_baseline": False},
        {"rsi_period": 14, "buy_threshold": 30.0, "sell_threshold": 70.0,
         "excess_return": -.01, "total_return": -.02, "is_baseline": True},
    ]

    heatmap = chart_for(rsi_pack, "run_abc:chart:parameters")
    cell = EvidenceTools(rsi_pack).call(
        "get_rsi_parameter_cell",
        {"rsi_period": 14, "buy_threshold": 30.0, "sell_threshold": 70.0},
    )

    assert heatmap.layout.xaxis.title.text == "超买卖出阈值"
    assert heatmap.layout.yaxis.title.text == "超卖买入阈值"
    assert cell["rows"][0]["excess_return"] == -.01


@pytest.mark.parametrize("mutation", ["reference", "path", "number", "dimension"])
def test_reject_bad_model_evidence(pack, mutation):
    candidate = {"findings": fact_report(pack)}
    first = candidate["findings"][0]
    if mutation == "reference":
        first["evidence_refs"] = ["other_run:baseline"]
    elif mutation == "path":
        first["values"]["v0"]["path"] = ["invented_field"]
    elif mutation == "number":
        first["statement_template"] = "收益高达99%。"
        first["values"] = {}
    else:
        first["dimension"] = "parameters"
    with pytest.raises(ValueError):
        validate_findings(candidate, pack)


def test_llm_tool_roundtrip(pack):
    answers = iter([
        {"tool_calls": [{"name": "get_evidence", "arguments": {"evidence_id": "run_abc:table:parameters"}}]},
        {"findings": fact_report(pack)},
    ])
    seen = []
    def complete(system, user):
        seen.append(user)
        return next(answers)
    report = generate_report(pack, use_llm=True, complete=complete)
    assert report["mode"] == "ai"
    assert len(report["tool_log"]) == 1
    assert '"tool_results"' in seen[-1] and '"fast": 3' in seen[-1]


def test_llm_retries_after_evidence_validation_error(pack):
    invalid = {"findings": deepcopy(fact_report(pack))}
    invalid["findings"][0]["statement_template"] = "收益高达99%。"
    invalid["findings"][0]["values"] = {}
    answers = iter([invalid, {"findings": fact_report(pack)}])
    seen = []

    def complete(system, user):
        seen.append(user)
        return next(answers)

    report = generate_report(pack, use_llm=True, complete=complete)
    assert report["mode"] == "ai"
    assert report["validation_retries"] == 1
    assert '"validation_feedback"' in seen[-1]


def test_llm_failure_and_budget_fall_back(pack):
    def failing(*args):
        raise LLMError("unavailable")
    assert generate_report(pack, use_llm=True, complete=failing)["mode"] == "facts"
    calls = []
    def looping(*args):
        calls.append(1)
        return {"tool_calls": [{"name": "get_evidence", "arguments": {"evidence_id": "run_abc:baseline"}}]}
    assert generate_report(pack, use_llm=True, complete=looping)["mode"] == "facts"
    assert len(calls) == 4


def test_unknown_tool_and_cross_run_export_rejected(pack):
    with pytest.raises(ValueError):
        EvidenceTools(pack).call("exec", {"code": "anything"})
    with pytest.raises(KeyError):
        EvidenceTools(pack).call("get_evidence", {"evidence_id": "other:baseline"})
    report = generate_report(pack)
    report["evidence_hash"] = "other"
    with pytest.raises(ValueError):
        export_bundle(pack, report)


def test_no_data_report_does_not_claim_support(pack):
    missing = deepcopy(pack)
    missing["entries"]["run_abc:baseline"]["data"] = {"error": "no prices"}
    report = generate_report(missing)
    assert "未获得" in report["findings"][0]["statement"]


def test_report_answer_requires_bound_values_and_valid_evidence(pack):
    candidate = {
        "level": "diagnosis",
        "answer_template": "原始策略的超额收益为{v0}，只能描述本次历史区间。",
        "values": {
            "v0": {
                "evidence_id": "run_abc:baseline",
                "path": ["excess_return"],
                "format": "percent",
            },
        },
        "evidence_refs": ["run_abc:baseline"],
        "limitations": ["尚未进行样本外验证。"],
    }

    answer = validate_report_answer(candidate, pack)

    assert "-1.00%" in answer["answer"]
    bad = deepcopy(candidate)
    bad["answer_template"] = "原始策略收益高达99%。"
    bad["values"] = {}
    with pytest.raises(ValueError):
        validate_report_answer(bad, pack)

    unsupported_dimension = deepcopy(candidate)
    unsupported_dimension["answer_template"] += " 参数实验也表现稳定。"
    with pytest.raises(ValueError, match="parameters"):
        validate_report_answer(unsupported_dimension, pack)

    with pytest.raises(ValueError, match="原因类问题"):
        validate_report_answer(candidate, pack, question="为什么回撤这么大？")


def test_report_question_tool_roundtrip_and_evidence_attachment(pack):
    report = generate_report(pack)
    answers = iter([
        {"tool_calls": [{
            "name": "get_evidence",
            "arguments": {"evidence_id": "run_abc:table:daily"},
        }]},
        {
            "level": "diagnosis",
            "answer_template": "本次回撤需要结合逐日净值检查，现有证据只覆盖已测区间。",
            "values": {},
            "evidence_refs": ["run_abc:baseline", "run_abc:table:daily"],
            "limitations": ["没有事件数据，不能归因于新闻或政策。"],
        },
    ])
    seen = []

    def complete(_system, user):
        seen.append(user)
        return next(answers)

    answer = answer_report_question(
        "为什么回撤这么大？", pack, report,
        answer_mode="技术细节", complete=complete,
    )
    rendered = format_report_answer(answer, pack)

    assert answer["mode"] == "ai"
    assert len(answer["tool_log"]) == 1
    assert "原始策略逐日净值" in rendered
    assert "run_abc:table:daily" in rendered
    assert '"answer_mode": "技术细节"' in seen[0]


def test_report_question_retries_after_invalid_numbers(pack):
    report = generate_report(pack)
    answers = iter([
        {
            "level": "diagnosis",
            "answer_template": "收益高达99%。",
            "values": {},
            "evidence_refs": ["run_abc:baseline"],
            "limitations": [],
        },
        {
            "level": "fact",
            "answer_template": "原始策略的超额收益为{v0}。",
            "values": {"v0": {
                "evidence_id": "run_abc:baseline",
                "path": ["excess_return"],
                "format": "percent",
            }},
            "evidence_refs": ["run_abc:baseline"],
            "limitations": ["仅适用于本次历史区间。"],
        },
    ])

    answer = answer_report_question(
        "表现如何？", pack, report, complete=lambda *_args: next(answers),
    )

    assert answer["mode"] == "ai"
    assert answer["validation_retries"] == 1
    assert answer["answer"] == "原始策略的超额收益为-1.00%。"


def test_report_question_falls_back_without_llm(pack):
    report = generate_report(pack)
    answer = answer_report_question("成本影响大吗？", pack, report, use_llm=False)

    assert answer["mode"] == "facts"
    assert "滑点" in answer["answer"]
    assert answer["evidence_refs"]
