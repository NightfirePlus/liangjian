"""验证实验：NLP 空规格补全、三类扫描、判定与可复现哈希。"""
from __future__ import annotations

from dataclasses import replace

import pytest

import core.experiments.runner as experiment_runner
import core.nlp.parser as parser_mod
from core.data import make_synthetic_snapshot
from core.experiments import resolve_experiment_spec, run_experiment, run_validation_suite
from core.schemas import ExperimentSpec, StrategyConfig


@pytest.fixture(scope="module")
def base_config() -> StrategyConfig:
    meta = make_synthetic_snapshot(
        ["510300", "510500"], "2023-01-01", "2024-12-31", seed=21,
    )
    return StrategyConfig(
        template="ma_cross",
        symbols=["510300"],
        start="2023-01-01",
        end="2024-12-31",
        params={"fast": 5, "slow": 20},
        snapshot_id=meta.snapshot_id,
    )


def make_spec(kind: str, *, confirmed: bool = True) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="",
        kind=kind,
        claim_text="验证这个策略是否稳健",
        metric="net_excess_return",
        grid={},
        support_rule="",
        confirmed_by_user=confirmed,
    )


def test_resolve_nlp_spec_fills_defaults(base_config):
    resolved = resolve_experiment_spec(make_spec("cost_sweep"), base_config)

    assert resolved.experiment_id.startswith("exp_cost_sweep_")
    assert resolved.metric == "excess_return"
    assert resolved.grid["slippage_bps"] == [0.0, 1.0, 2.0, 5.0, 10.0]
    assert resolved.support_rule == "all(excess_return > 0 where slippage_bps <= 5)"


def test_experiment_requires_user_confirmation(base_config):
    with pytest.raises(ValueError, match="尚未经过用户确认"):
        run_experiment(base_config, make_spec("cost_sweep", confirmed=False))


def test_cost_sweep_runs_and_cost_only_increases(base_config):
    result = run_experiment(base_config, make_spec("cost_sweep"))

    assert len(result.rows) == 5
    assert [row["slippage_bps"] for row in result.rows] == [0.0, 1.0, 2.0, 5.0, 10.0]
    assert all(row["error"] is None for row in result.rows)
    returns = [row["total_return"] for row in result.rows]
    assert returns == sorted(returns, reverse=True)
    assert sum(row["is_baseline"] for row in result.rows) == 1
    assert result.verdict in {"supported", "not_supported", "mixed"}
    assert len(result.config_hash) == 64


def test_param_neighborhood_contains_baseline_and_valid_pairs(base_config):
    resolved = resolve_experiment_spec(make_spec("param_neighborhood"), base_config)
    result = run_experiment(base_config, resolved)

    pairs = {(row["fast"], row["slow"]) for row in result.rows}
    assert (5, 20) in pairs
    assert len(pairs) == 9
    assert all(fast < slow for fast, slow in pairs)
    assert all(row["error"] is None for row in result.rows)
    assert sum(row["is_baseline"] for row in result.rows) == 1
    assert result.total_points == 9
    assert result.pass_rate == result.passed_points / 9
    assert result.rule_satisfied == (result.pass_rate >= 0.70)


def test_breakout_validation_suite_scans_breakout_parameters(base_config):
    config = replace(
        base_config,
        template="breakout",
        params={"entry_window": 20, "exit_window": 10},
    )

    report = run_validation_suite(config)

    rows = report.experiments["strategy_parameter_robustness"].rows
    pairs = {(row["entry_window"], row["exit_window"]) for row in rows}
    assert (20, 10) in pairs
    assert len(pairs) == 9
    assert all(entry > exit_ for entry, exit_ in pairs)
    assert all(row["error"] is None for row in rows)
    assert sum(row["is_baseline"] for row in rows) == 1
    assert report.context["config"]["template"] == "breakout"


def test_rsi_validation_suite_scans_thresholds(base_config):
    config = replace(
        base_config,
        template="rsi_reversion",
        params={"rsi_period": 14, "buy_threshold": 30, "sell_threshold": 70},
    )

    report = run_validation_suite(config)

    rows = report.experiments["strategy_parameter_robustness"].rows
    pairs = {(row["buy_threshold"], row["sell_threshold"]) for row in rows}
    assert (30, 70) in pairs
    assert len(pairs) == 9
    assert all(0 < buy < sell < 100 for buy, sell in pairs)
    assert all(row["rsi_period"] == 14 and row["error"] is None for row in rows)
    assert sum(row["is_baseline"] for row in rows) == 1


def test_period_split_covers_two_non_overlapping_halves(base_config):
    resolved = resolve_experiment_spec(make_spec("period_split"), base_config)
    result = run_experiment(base_config, resolved)

    assert len(result.rows) == 2
    first, second = result.rows
    assert first["period"] == "前半段"
    assert second["period"] == "后半段"
    assert first["end"] < second["start"]
    assert first["error"] is None and second["error"] is None
    assert result.rule_satisfied == (result.passed_points >= 1)


def test_config_hash_is_reproducible(base_config):
    spec = make_spec("cost_sweep")

    first = run_experiment(base_config, spec)
    second = run_experiment(base_config, spec)

    assert first.experiment_id == second.experiment_id
    assert first.config_hash == second.config_hash
    assert first.rows == second.rows


def test_hash_changes_when_benchmark_changes(base_config):
    spec = make_spec("cost_sweep")

    same_asset = run_experiment(base_config, spec, benchmark_symbol="510300")
    other_asset = run_experiment(base_config, spec, benchmark_symbol="510500")

    assert same_asset.config_hash != other_asset.config_hash


def test_experiment_id_does_not_depend_on_confirmation_order(base_config):
    raw = make_spec("cost_sweep", confirmed=False)
    resolved_before_confirmation = resolve_experiment_spec(raw, base_config)
    confirmed_first = replace(raw, confirmed_by_user=True)
    resolved_after_confirmation = resolve_experiment_spec(confirmed_first, base_config)

    assert resolved_before_confirmation.experiment_id == resolved_after_confirmation.experiment_id


def test_unknown_custom_rule_is_rejected(base_config):
    spec = replace(make_spec("cost_sweep"), support_rule="mean(excess_return) > 0")

    with pytest.raises(ValueError, match="当前仅支持判断规则"):
        resolve_experiment_spec(spec, base_config)


def test_validation_suite_separates_strategy_and_market_changes(base_config):
    report = run_validation_suite(base_config)

    assert report.validation_id.startswith("validation_")
    assert report.baseline["is_baseline"] is True
    assert set(report.experiments) == {
        "cost_sensitivity",
        "strategy_parameter_robustness",
        "market_period_robustness",
    }
    parameter_rows = report.experiments["strategy_parameter_robustness"].rows
    assert any(row["fast"] != 5 or row["slow"] != 20 for row in parameter_rows)
    period_rows = report.experiments["market_period_robustness"].rows
    assert all("fast" not in row and "slow" not in row for row in period_rows)
    assert report.verdict is None
    assert report.reasons == []
    assert report.context["config"]["params"] == {"fast": 5, "slow": 20}
    assert report.context["snapshot"]["is_synthetic"] is True
    assert report.event_analysis is None
    assert len(report.config_hash) == 64


def test_cost_rule_only_requires_normal_cost_range(base_config):
    rows = [
        {"slippage_bps": 0.0, "passed": True},
        {"slippage_bps": 5.0, "passed": True},
        {"slippage_bps": 10.0, "passed": False},
    ]

    verdict, _failure, passed, total, rate, rule_satisfied = (
        experiment_runner._judge("cost_sweep", rows)
    )

    assert verdict == "mixed"
    assert (passed, total, rate) == (2, 3, 2 / 3)
    assert rule_satisfied is True


def test_nlp_result_can_run_experiment_after_confirmation(base_config, monkeypatch):
    candidate = {
        "template": "ma_cross",
        "symbol_mentions": ["沪深300ETF"],
        "symbols": ["510300"],
        "start": "2023-01-01",
        "end": "2024-12-31",
        "params": {"fast": 5, "slow": 20},
        "overrides": {},
        "claim_text": "我认为这个策略对交易成本不敏感",
        "ambiguities": [],
        "missing": [],
    }
    monkeypatch.setattr(parser_mod, "LLM_AVAILABLE", True)
    monkeypatch.setattr(parser_mod, "complete_json", lambda *_args: candidate)

    parsed = parser_mod.parse_user_input("测试双均线并验证成本敏感性")
    assert parsed.needs_clarification is False
    config = replace(parsed.strategy_config, snapshot_id=base_config.snapshot_id)
    spec = replace(parsed.experiment_spec, confirmed_by_user=True)

    result = run_experiment(config, spec)

    assert result.experiment_id.startswith("exp_cost_sweep_")
    assert len(result.rows) == 5
    assert all(row["metric"] == "excess_return" for row in result.rows)
