"""自然语言解析主流程：调用大模型 -> 程序校验 -> 追问或填默认值+生成复述。

对接下游（core/backtest 读 StrategyConfig，core/experiments 读 ExperimentSpec）
和上游（app.py 的输入框与确认卡）的唯一入口是 parse_user_input()。
"""
from __future__ import annotations

from config import LLM_AVAILABLE
from core.nlp.clarify import generate_questions
from core.nlp.confirm import build_confirmation_text
from core.nlp.defaults import apply_defaults, default_note
from core.nlp.fallback import parse_with_rules
from core.nlp.llm_client import LLMError, complete_json
from core.nlp.result import ParseResult
from core.nlp.schema_prompt import build_system_prompt, build_user_prompt
from core.nlp.validate import validate_candidate
from core.schemas import CostConfig, ExperimentSpec, StrategyConfig

_CLAIM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cost_sweep": ("成本", "佣金", "滑点", "手续费"),
    "period_split": ("牛市", "熊市", "阶段", "时期", "分时段", "不同时间"),
    "param_neighborhood": ("参数", "周期", "敏感"),
}


def parse_user_input(text: str, history: list[dict] | None = None) -> ParseResult:
    """text 为用户本轮输入；history 为之前的追问往返（见 schema_prompt.build_user_prompt）。

    返回 ParseResult：needs_clarification=True 时等待用户补充信息，
    False 时 strategy_config 已经可以直接送入 core/backtest。
    """
    candidate, degraded, degraded_reason = _get_candidate(text, history)

    validation = validate_candidate(candidate)
    if not validation.ok:
        return ParseResult(
            needs_clarification=True,
            questions=generate_questions(validation.issues),
            degraded=degraded,
            degraded_reason=degraded_reason,
            raw_candidate=candidate,
        )

    merged_defaults, overridden = apply_defaults(candidate.get("overrides"))
    strategy_config = _build_strategy_config(candidate, merged_defaults, overridden)
    experiment_spec = _build_experiment_spec(candidate)

    confirmation_text = build_confirmation_text(
        strategy_config, overridden, claim_text=candidate.get("claim_text"))

    return ParseResult(
        needs_clarification=False,
        strategy_config=strategy_config,
        experiment_spec=experiment_spec,
        confirmation_text=confirmation_text,
        overridden_defaults=overridden,
        degraded=degraded,
        degraded_reason=degraded_reason,
        raw_candidate=candidate,
    )


def _get_candidate(text: str, history: list[dict] | None) -> tuple[dict, bool, str]:
    """返回 (候选字典, 是否降级, 降级原因)。"""
    if not LLM_AVAILABLE:
        return parse_with_rules(text), True, "当前未配置大模型 API，仅做了最基础的关键词识别"

    try:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(text, history)
        return complete_json(system_prompt, user_prompt), False, ""
    except LLMError:
        try:
            system_prompt = build_system_prompt()
            user_prompt = build_user_prompt(text, history)
            return complete_json(system_prompt, user_prompt), False, ""
        except LLMError as err:
            return parse_with_rules(text), True, f"大模型解析暂时不可用（{err}），仅做了最基础的关键词识别"


def _build_strategy_config(candidate: dict, merged_defaults: dict, overridden: list[str]) -> StrategyConfig:
    assumptions = [_assumption_text(key, merged_defaults[key], key in overridden) for key in merged_defaults]
    cost = CostConfig(
        commission_rate=merged_defaults["commission_rate"],
        min_commission=merged_defaults["min_commission"],
        slippage_bps=merged_defaults["slippage_bps"],
    )
    return StrategyConfig(
        template=candidate["template"],
        symbols=list(candidate["symbols"]),
        start=candidate["start"],
        end=candidate["end"],
        params=dict(candidate.get("params") or {}),
        cost=cost,
        assumptions=assumptions,
    )


def _assumption_text(key: str, value, is_overridden: bool) -> str:
    note = default_note(key)
    suffix = "（按你的要求）" if is_overridden else "（默认假设）"
    return f"{note}，当前取值 {value} {suffix}"


def _build_experiment_spec(candidate: dict) -> ExperimentSpec | None:
    """claim_text 不存在时不生成实验（没有可验证的观点）。

    kind/metric 为粗略猜测，grid/support_rule 留空，由 core/experiments/ 实现时填充；
    这里只负责识别"用户是否表达了一个可验证的观点"，不涉及实验怎么跑。
    """
    claim_text = candidate.get("claim_text")
    if not claim_text:
        return None

    kind = "param_neighborhood"
    for candidate_kind, keywords in _CLAIM_KEYWORDS.items():
        if any(k in claim_text for k in keywords):
            kind = candidate_kind
            break

    return ExperimentSpec(
        experiment_id="",
        kind=kind,
        claim_text=claim_text,
        metric="excess_return",
        grid={},
        support_rule="",
    )
