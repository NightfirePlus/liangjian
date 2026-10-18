"""parse_user_input() 的返回类型，单独成文件避免 parser/clarify/confirm 相互循环引用。"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.schemas import ExperimentSpec, StrategyConfig


@dataclass
class ParseResult:
    """自然语言解析一轮的结果。

    needs_clarification=True 时：strategy_config/experiment_spec 为 None，
        questions 列出本轮需要用户补充/澄清的问题（尽量一次性列全），
        raw_candidate 保留本轮大模型的原始候选字段，供下一轮连同用户的回答
        一起喂给大模型，避免用户重新从头描述一遍。
    needs_clarification=False 时：strategy_config 必定非空，confirmation_text
        是给用户看的自然语言复述（代替直接甩 JSON），overridden_defaults 列出
        本次被用户显式覆盖的默认假设 key（对应 core/nlp/defaults.py）。
    """
    needs_clarification: bool
    questions: list[str] = field(default_factory=list)
    strategy_config: StrategyConfig | None = None
    experiment_spec: ExperimentSpec | None = None
    confirmation_text: str = ""
    overridden_defaults: list[str] = field(default_factory=list)
    degraded: bool = False          # True 表示本轮未走大模型，走的是规则兜底
    degraded_reason: str = ""        # 降级原因，供界面提示用户
    raw_candidate: dict | None = None
