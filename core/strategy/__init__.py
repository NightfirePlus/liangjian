"""自然语言、快捷创建和典型模板之间共享的策略草案。"""

from core.strategy.draft import (
    StrategyDraft,
    align_draft_to_available_period,
    specified_fields_from_candidate,
)

__all__ = ["StrategyDraft", "align_draft_to_available_period", "specified_fields_from_candidate"]
