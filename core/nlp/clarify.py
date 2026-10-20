"""把校验出的问题列表转成给用户的追问句，本地规则拼接，不调用大模型。

使用中文追问，并合并重复的问题。
"""
from __future__ import annotations

from core.nlp.validate import ValidationIssue


def generate_questions(issues: list[ValidationIssue]) -> list[str]:
    """保留不同问题，避免相同内容重复展示。"""
    return list(dict.fromkeys(issue.message.strip() for issue in issues if issue.message.strip()))
