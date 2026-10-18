"""把校验出的问题列表转成给用户的追问句，本地规则拼接，不调用大模型。

目标是尽量一次性把本轮能看到的所有问题都问出来，减少多轮来回。
"""
from __future__ import annotations

from core.nlp.validate import ValidationIssue


def generate_questions(issues: list[ValidationIssue]) -> list[str]:
    """issues 里同一 field 的多条会被分别保留（如多个 ambiguities），
    不同 field 之间不去重合并，让用户能逐条看清楚缺的是什么。
    """
    return [issue.message for issue in issues]
