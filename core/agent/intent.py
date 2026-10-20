"""用确定性规则区分策略修改与报告追问。"""
from __future__ import annotations

import re
from typing import Literal

AgentIntent = Literal["strategy", "report_question"]

_STRATEGY_CHANGE_PATTERNS = (
    r"(?:修改|调整|更改|改成|改为|换成|换一个|重新设置)",
    r"(?:快线|慢线|均线周期|佣金|滑点|初始资金|回测区间|交易标的).*(?:改|设|换|用)",
    r"(?:把|将).*(?:设为|改为|换成)",
    r"(?:重新|再)(?:回测|验证|测试|生成)",
    r"用.+(?:测试|回测).*(?:策略|均线)",
    r"(?:新建|创建|生成).*(?:策略|回测)",
)


def route_agent_message(text: str, *, report_ready: bool) -> AgentIntent:
    """报告完成后默认回答报告问题，明确修改指令才重新解析策略。"""
    if not report_ready:
        return "strategy"
    normalized = "".join(text.strip().split())
    if any(re.search(pattern, normalized) for pattern in _STRATEGY_CHANGE_PATTERNS):
        return "strategy"
    return "report_question"
