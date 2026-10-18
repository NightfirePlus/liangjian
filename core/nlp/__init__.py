"""自然语言 -> StrategyConfig / ExperimentSpec，含追问与不支持需求的说明。

主入口：parse_user_input(text, history) -> ParseResult，见 core/nlp/parser.py。
默认假设清单：core/nlp/defaults.py（人读版本见 docs/assumptions.md）。
"""
from __future__ import annotations

from core.nlp.parser import parse_user_input
from core.nlp.result import ParseResult

__all__ = ["parse_user_input", "ParseResult"]
