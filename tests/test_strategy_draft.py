from dataclasses import replace

from core.schemas import CostConfig, StrategyConfig
from core.strategy import StrategyDraft, align_draft_to_available_period, specified_fields_from_candidate


def test_draft_from_config_and_execution_hash_ignore_origin():
    config = StrategyConfig(
        template="ma_cross",
        symbols=["510300"],
        start="2023-01-01",
        end="2024-01-01",
        params={"fast": 5, "slow": 20},
        cost=CostConfig(slippage_bps=2),
    )
    draft = StrategyDraft.from_config(config, origin="conversation")

    assert draft.symbol == "510300"
    assert draft.slippage_bps == 2
    assert draft.to_config("snapshot_a").snapshot_id == "snapshot_a"
    assert draft.execution_hash("snapshot_a") == replace(
        draft, origin="preset", confirmation_text="different wording",
    ).execution_hash("snapshot_a")


def test_draft_validation_covers_market_range_and_parameters():
    draft = StrategyDraft(
        symbol="510300", start="2022-01-01", end="2025-01-01",
        fast=20, slow=5, initial_cash=500,
    )

    issues = draft.validate(
        available_symbols=["510500"],
        available_start="2023-01-01",
        available_end="2024-01-01",
    )

    assert any("不包含标的" in issue for issue in issues)
    assert any("早于" in issue for issue in issues)
    assert any("晚于" in issue for issue in issues)
    assert any("快线" in issue for issue in issues)
    assert any("1,000" in issue for issue in issues)


def test_nearby_non_trading_boundaries_are_aligned_to_available_period():
    draft = StrategyDraft(
        symbol="510300", start="2023-01-01", end="2025-01-01", fast=5, slow=20,
    )

    aligned = align_draft_to_available_period(draft, "2023-01-02", "2024-12-31")

    assert aligned.start == "2023-01-02"
    assert aligned.end == "2024-12-31"
    assert len(aligned.normalization_notes) == 2
    assert not aligned.validate(available_start="2023-01-02", available_end="2024-12-31")


def test_missing_market_history_is_not_silently_shortened():
    draft = StrategyDraft(
        symbol="510300", start="2022-01-01", end="2025-06-01", fast=5, slow=20,
    )

    aligned = align_draft_to_available_period(draft, "2023-01-02", "2024-12-31")
    issues = aligned.validate(available_start="2023-01-02", available_end="2024-12-31")

    assert aligned.start == draft.start and aligned.end == draft.end
    assert not aligned.normalization_notes
    assert any("早于" in issue for issue in issues)
    assert any("晚于" in issue for issue in issues)


def test_candidate_fields_distinguish_input_from_defaults():
    fields = specified_fields_from_candidate({
        "template": "ma_cross",
        "symbols": ["510300"],
        "start": "2023-01-01",
        "params": {"fast": 5, "slow": 20},
        "overrides": {"slippage_bps": 3},
    })

    assert fields == ["策略类型", "交易标的", "测试区间", "快线周期", "慢线周期", "滑点"]


def test_breakout_draft_roundtrip_and_validation():
    config = StrategyConfig(
        template="breakout",
        symbols=["510300"],
        start="2023-01-01",
        end="2024-01-01",
        params={"entry_window": 20, "exit_window": 10},
    )

    draft = StrategyDraft.from_config(config)
    restored = draft.to_config("snapshot_breakout")

    assert draft.entry_window == 20 and draft.exit_window == 10
    assert restored.template == "breakout"
    assert restored.params == {"entry_window": 20, "exit_window": 10}
    assert restored.snapshot_id == "snapshot_breakout"
    assert any(
        "突破周期" in issue
        for issue in replace(draft, entry_window=5, exit_window=10).validate()
    )


def test_rsi_draft_roundtrip_and_validation():
    config = StrategyConfig(
        template="rsi_reversion", symbols=["510300"],
        start="2023-01-01", end="2024-01-01",
        params={"rsi_period": 14, "buy_threshold": 30, "sell_threshold": 70},
    )

    draft = StrategyDraft.from_config(config)
    restored = draft.to_config("snapshot_rsi")

    assert restored.template == "rsi_reversion"
    assert restored.params == {
        "rsi_period": 14, "buy_threshold": 30.0, "sell_threshold": 70.0,
    }
    assert any(
        "RSI 阈值" in issue
        for issue in replace(draft, buy_threshold=80, sell_threshold=20).validate()
    )
