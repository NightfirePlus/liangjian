"""自然语言解析模块自测:校验、默认值覆盖、降级路径、端到端。

不联网、不依赖真实 LLM key,全部通过 monkeypatch 模拟大模型返回,
和 test_core_data.py 的风格保持一致。
"""
from __future__ import annotations

import pytest

import core.nlp.parser as parser_mod
from core.nlp.defaults import apply_defaults, DEFAULT_ASSUMPTIONS
from core.nlp.fallback import parse_with_rules
from core.nlp.validate import validate_candidate


# ---------------------------------------------------------------- validate
def test_validate_missing_template_and_symbols():
    result = validate_candidate({
        "template": None, "symbols": [], "ambiguities": [], "params": {},
        "start": None, "end": None, "missing": [],
    })
    assert not result.ok
    fields = {i.field for i in result.issues}
    assert "template" in fields
    assert "symbols" in fields
    assert "period" in fields


def test_validate_symbol_not_in_universe():
    result = validate_candidate({
        "template": "ma_cross", "symbols": ["999999"], "ambiguities": [],
        "params": {"fast": 5, "slow": 20}, "start": "2023-01-01", "end": "2024-01-01",
        "missing": [],
    })
    assert not result.ok
    assert any("999999" in i.message for i in result.issues)


def test_validate_ambiguous_symbol_not_resolved():
    result = validate_candidate({
        "template": "ma_cross", "symbols": [], "ambiguities": ["沪深300对应多个ETF"],
        "params": {"fast": 5, "slow": 20}, "start": "2023-01-01", "end": "2024-01-01",
        "missing": [],
    })
    assert not result.ok
    assert any("沪深300" in i.message for i in result.issues)


def test_validate_fast_slow_order():
    result = validate_candidate({
        "template": "ma_cross", "symbols": ["510300"], "ambiguities": [],
        "params": {"fast": 20, "slow": 5}, "start": "2023-01-01", "end": "2024-01-01",
        "missing": [],
    })
    assert not result.ok
    assert any("快线" in i.message for i in result.issues)


def test_validate_breakout_parameters():
    valid = validate_candidate({
        "template": "breakout", "symbols": ["510300"], "ambiguities": [],
        "params": {"entry_window": 20, "exit_window": 10},
        "start": "2023-01-01", "end": "2024-01-01", "missing": [],
    })
    invalid = validate_candidate({
        "template": "breakout", "symbols": ["510300"], "ambiguities": [],
        "params": {"entry_window": 10, "exit_window": 20},
        "start": "2023-01-01", "end": "2024-01-01", "missing": [],
    })

    assert valid.ok
    assert not invalid.ok
    assert any("突破周期" in issue.message for issue in invalid.issues)


def test_validate_rsi_parameters():
    valid = validate_candidate({
        "template": "rsi_reversion", "symbols": ["510300"], "ambiguities": [],
        "params": {"rsi_period": 14, "buy_threshold": 30, "sell_threshold": 70},
        "start": "2023-01-01", "end": "2024-01-01", "missing": [],
    })
    invalid = validate_candidate({
        "template": "rsi_reversion", "symbols": ["510300"], "ambiguities": [],
        "params": {"rsi_period": 14, "buy_threshold": 80, "sell_threshold": 20},
        "start": "2023-01-01", "end": "2024-01-01", "missing": [],
    })

    assert valid.ok
    assert not invalid.ok
    assert any("RSI 阈值" in issue.message for issue in invalid.issues)


def test_validate_period_order():
    result = validate_candidate({
        "template": "ma_cross", "symbols": ["510300"], "ambiguities": [],
        "params": {"fast": 5, "slow": 20}, "start": "2024-01-01", "end": "2023-01-01",
        "missing": [],
    })
    assert not result.ok
    assert any(i.field == "period" for i in result.issues)


def test_validate_complete_candidate_passes():
    result = validate_candidate({
        "template": "ma_cross", "symbols": ["510300"], "ambiguities": [],
        "params": {"fast": 5, "slow": 20}, "start": "2023-01-01", "end": "2024-01-01",
        "missing": [],
    })
    assert result.ok


# ---------------------------------------------------------------- defaults
def test_apply_defaults_no_override():
    merged, overridden = apply_defaults(None)
    assert overridden == []
    for key, (default, _note) in DEFAULT_ASSUMPTIONS.items():
        assert merged[key] == default


def test_apply_defaults_with_override():
    merged, overridden = apply_defaults({"exec_timing": "same_close", "slippage_bps": 3})
    assert set(overridden) == {"exec_timing", "slippage_bps"}
    assert merged["exec_timing"] == "same_close"
    assert merged["slippage_bps"] == 3
    # 未覆盖的项仍落默认值
    assert merged["adjust"] == DEFAULT_ASSUMPTIONS["adjust"][0]
    assert merged["commission_rate"] == DEFAULT_ASSUMPTIONS["commission_rate"][0]


def test_apply_defaults_ignores_unknown_key():
    merged, overridden = apply_defaults({"not_a_real_key": 123})
    assert "not_a_real_key" not in merged
    assert overridden == []


# ---------------------------------------------------------------- parser: 降级路径
def test_parser_degrades_when_llm_unavailable(monkeypatch):
    monkeypatch.setattr(parser_mod, "LLM_AVAILABLE", False)
    result = parser_mod.parse_user_input("510300双均线，5日和20日，2023-01-01到2024-01-01")
    assert result.degraded is True
    assert result.degraded_reason


def test_relative_years_use_market_snapshot_end_instead_of_model_date(monkeypatch):
    monkeypatch.setattr(parser_mod, "LLM_AVAILABLE", True)
    candidate = {
        "template": "ma_cross", "symbol_mentions": ["沪深300ETF"], "symbols": ["510300"],
        "start": "2022-09-26", "end": "2024-09-26", "params": {"fast": 5, "slow": 20},
        "overrides": {}, "claim_text": None, "ambiguities": [], "missing": [],
    }
    monkeypatch.setattr(parser_mod, "complete_json", _mock_complete_json(candidate))

    result = parser_mod.parse_user_input(
        "用沪深300ETF测试双均线，5日线上穿20日线买入，回测近两年。",
        reference_end="2024-12-31",
    )

    assert result.strategy_config.start == "2022-12-31"
    assert result.strategy_config.end == "2024-12-31"
    assert result.raw_candidate["_relative_period_years"] == 2


def test_latest_explicit_dates_override_earlier_relative_period(monkeypatch):
    monkeypatch.setattr(parser_mod, "LLM_AVAILABLE", False)
    result = parser_mod.parse_user_input(
        "510300双均线，5日和20日，回测近两年。\n"
        "改成2023-01-02到2024-12-31",
        reference_end="2024-12-31",
    )

    assert result.strategy_config.start == "2023-01-02"
    assert result.strategy_config.end == "2024-12-31"
    assert "_relative_period_years" not in result.raw_candidate


def test_parser_degrades_on_llm_error(monkeypatch):
    monkeypatch.setattr(parser_mod, "LLM_AVAILABLE", True)

    def _boom(system_prompt, user_prompt):
        raise parser_mod.LLMError("网络超时")

    monkeypatch.setattr(parser_mod, "complete_json", _boom)
    result = parser_mod.parse_user_input("随便测一个策略")
    assert result.degraded is True
    assert "网络超时" in result.degraded_reason


def test_rule_fallback_uses_latest_multi_turn_changes():
    candidate = parse_with_rules(
        "510300双均线，5日和20日，2023-01-01到2024-01-01\n"
        "改成510500，10日和30日，2022-01-01到2025-01-01"
    )

    assert candidate["symbols"] == ["510500"]
    assert candidate["params"] == {"fast": 10, "slow": 30}
    assert candidate["start"] == "2022-01-01"
    assert candidate["end"] == "2025-01-01"


def test_rule_fallback_parses_breakout_strategy():
    candidate = parse_with_rules(
        "510300价格通道突破，收盘突破过去20日最高价买入，"
        "跌破过去10日最低价卖出，2023-01-01到2024-01-01"
    )

    assert candidate["template"] == "breakout"
    assert candidate["params"] == {"entry_window": 20, "exit_window": 10}


def test_rule_fallback_parses_rsi_strategy():
    candidate = parse_with_rules(
        "510300使用14日RSI，低于30买入，高于70卖出，"
        "2023-01-01到2024-01-01"
    )

    assert candidate["template"] == "rsi_reversion"
    assert candidate["params"] == {
        "rsi_period": 14, "buy_threshold": 30.0, "sell_threshold": 70.0,
    }


# ---------------------------------------------------------------- parser: 端到端
def _mock_complete_json(candidate: dict):
    def _fn(system_prompt, user_prompt):
        return candidate
    return _fn


def test_parser_end_to_end_success(monkeypatch):
    monkeypatch.setattr(parser_mod, "LLM_AVAILABLE", True)
    candidate = {
        "template": "ma_cross", "symbol_mentions": ["沪深300ETF"], "symbols": ["510300"],
        "start": "2023-01-01", "end": "2024-12-31", "params": {"fast": 5, "slow": 20},
        "overrides": {}, "claim_text": None, "ambiguities": [], "missing": [],
    }
    monkeypatch.setattr(parser_mod, "complete_json", _mock_complete_json(candidate))

    result = parser_mod.parse_user_input("沪深300ETF，5日均线上穿20日均线买入，下穿卖出")

    assert result.needs_clarification is False
    assert result.degraded is False
    cfg = result.strategy_config
    assert cfg is not None
    assert cfg.template == "ma_cross"
    assert cfg.symbols == ["510300"]
    assert cfg.start == "2023-01-01" and cfg.end == "2024-12-31"
    assert cfg.params == {"fast": 5, "slow": 20}
    # 未显式覆盖，成本走默认值
    assert cfg.cost.commission_rate == DEFAULT_ASSUMPTIONS["commission_rate"][0]
    assert cfg.cost.slippage_bps == DEFAULT_ASSUMPTIONS["slippage_bps"][0]
    assert result.overridden_defaults == []
    assert "510300" in result.confirmation_text or "沪深300" in result.confirmation_text
    assert result.experiment_spec is None


def test_parser_end_to_end_with_override_and_claim(monkeypatch):
    monkeypatch.setattr(parser_mod, "LLM_AVAILABLE", True)
    candidate = {
        "template": "ma_cross", "symbol_mentions": ["沪深300ETF"], "symbols": ["510300"],
        "start": "2023-01-01", "end": "2024-12-31", "params": {"fast": 5, "slow": 20},
        "overrides": {"slippage_bps": 3}, "claim_text": "我认为这个策略对交易成本不敏感",
        "ambiguities": [], "missing": [],
    }
    monkeypatch.setattr(parser_mod, "complete_json", _mock_complete_json(candidate))

    result = parser_mod.parse_user_input(
        "沪深300ETF，5日均线上穿20日均线买入，下穿卖出，滑点按3个基点算。"
        "我认为这个策略对交易成本不敏感")

    assert result.needs_clarification is False
    assert result.overridden_defaults == ["slippage_bps"]
    assert result.strategy_config.cost.slippage_bps == 3
    assert "按你的要求" in result.confirmation_text
    assert result.experiment_spec is not None
    assert result.experiment_spec.kind == "cost_sweep"
    assert result.experiment_spec.claim_text == "我认为这个策略对交易成本不敏感"
    assert result.experiment_spec.metric == "excess_return"
    assert result.experiment_spec.grid == {}
    assert result.experiment_spec.support_rule == ""


def test_parser_needs_clarification_collects_all_issues(monkeypatch):
    monkeypatch.setattr(parser_mod, "LLM_AVAILABLE", True)
    candidate = {
        "template": None, "symbol_mentions": [], "symbols": [],
        "start": None, "end": None, "params": {},
        "overrides": {}, "claim_text": None, "ambiguities": [], "missing": [],
    }
    monkeypatch.setattr(parser_mod, "complete_json", _mock_complete_json(candidate))

    result = parser_mod.parse_user_input("帮我测一下策略")

    assert result.needs_clarification is True
    assert result.strategy_config is None
    # 一轮内尽量把模板/标的/区间缺失都收集到，而不是只问一个就返回
    assert len(result.questions) >= 3
    assert result.raw_candidate == candidate
