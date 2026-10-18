from copy import deepcopy
from io import BytesIO
from zipfile import ZipFile

import pytest

from core.schemas import ExperimentResult, ValidationReport
from core.report import build_evidence, chart_for, generate_report, export_bundle
from core.report.analysis import fact_report, validate_findings
from core.report.evidence import EvidenceTools, dumps
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
