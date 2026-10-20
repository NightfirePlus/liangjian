"""金融入门问答：优先调用模型，失败时返回经过审校的本地基础解释。"""
from __future__ import annotations

from typing import Callable

from config import LLM_AVAILABLE
from core.nlp.llm_client import LLMError, complete_json


_MODE_GUIDANCE = {
    "新手讲解": "先用一两句大白话直接回答，必要时加一个简短例子。只讲当前问题，不堆砌术语；必须用术语时随口解释，不输出内部字段名或代码。",
    "简洁摘要": "用三到五句话直接回答，只保留理解问题必需的概念和限制。",
    "技术细节": "可以给出公式、统计口径和实现细节，但仍需解释关键术语。",
}

_LOCAL_TOPICS = (
    (("什么是量化", "量化投资", "量化是什么"),
     "量化是把投资想法写成明确规则，再用数据执行和检验。规则需要说清楚买什么、何时买卖、用多少资金以及如何计算成本。历史回测可以检查规则过去怎样运行，但不能保证未来收益。"),
    (("回测",),
     "回测是让一套明确策略在历史行情上重新执行，观察收益、回撤、交易次数和成本等结果。可信的回测要避免使用当时不可能知道的数据，并明确成交时点、费用和比较基准。"),
    (("最大回撤", "回撤"),
     "最大回撤表示资产净值从某个历史高点跌到随后低点时，出现过的最大跌幅。它帮助理解策略过去最难熬的一段，但不会告诉你下一次下跌何时发生。"),
    (("双均线", "均线"),
     "双均线策略比较短周期和长周期的平均价格。短均线上穿长均线时通常理解为趋势转强，下穿时理解为趋势转弱。它容易理解和验证，但在震荡行情中可能频繁产生无效交易。"),
    (("价格突破", "通道突破", "突破策略"),
     "价格通道突破策略在价格超过此前一段时间的高点时进入，跌破较短周期低点时退出。它试图跟随趋势，代价是可能在假突破或震荡阶段反复亏损。"),
    (("RSI", "rsi", "超买", "超卖", "相对强弱"),
     "RSI 是衡量近期上涨与下跌强弱的指标，常见范围为零到一百。均值回归策略会在 RSI 进入超卖区时买入、进入超买区时卖出；它假设价格会向常态回归，但在持续单边行情中可能过早入场或离场。"),
    (("滑点",),
     "滑点是策略假设的成交价与实际可成交价格之间的差异。交易越频繁、市场越不活跃，滑点对结果的影响通常越明显，因此报告会单独做成本敏感性实验。"),
    (("超额收益", "基准"),
     "基准是用来比较的简单方案，例如买入并持有同一只 ETF。超额收益是策略收益减去基准收益；策略赚钱但超额收益为负，表示它仍落后于这个比较方案。"),
)

_SYSTEM_PROMPT = """你是量见的金融入门助手，面向几乎不了解量化的用户。
只讲稳定的基础概念、常见策略原理、回测方法和风险。不要声称掌握实时行情、新闻或政策；
不要提供个性化买卖建议、收益承诺或具体证券推荐。涉及历史回测时说明它不代表未来表现。
当前产品可直接验证的策略有双均线 ma_cross、价格通道突破 breakout 和 RSI 均值回归 rsi_reversion。
展示给用户的回答使用简短中文，不要输出策略代码、内部字段名或程序路径。
仅输出 JSON：
{"answer":"中文回答","related_concepts":["最多三个概念"],
 "suggested_strategy":"ma_cross或breakout或rsi_reversion或null"}
suggested_strategy 只在问题明显与当前支持策略有关时填写，否则为 null。
"""


def answer_finance_question(
    question: str,
    *,
    mode: str = "新手讲解",
    history: list[dict] | None = None,
    use_llm: bool = LLM_AVAILABLE,
    complete: Callable[..., dict] = complete_json,
) -> dict:
    """返回可直接渲染的回答；模型不可用时仍能回答常见基础概念。"""
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空")
    mode = mode if mode in _MODE_GUIDANCE else "新手讲解"
    if use_llm:
        recent = (history or [])[-4:]
        user_prompt = (
            f"回答方式：{mode}。{_MODE_GUIDANCE[mode]}\n"
            f"最近对话：{recent}\n用户问题：{question}"
        )
        try:
            candidate = complete(_SYSTEM_PROMPT, user_prompt)
            return _validate(candidate, source="ai")
        except (LLMError, TypeError, ValueError):
            pass
    return _local_answer(question)


def _validate(candidate: dict, *, source: str) -> dict:
    if not isinstance(candidate, dict) or not isinstance(candidate.get("answer"), str):
        raise ValueError("问答结果格式错误")
    answer = candidate["answer"].strip()
    if not answer:
        raise ValueError("问答内容为空")
    concepts = candidate.get("related_concepts") or []
    if not isinstance(concepts, list) or any(not isinstance(item, str) for item in concepts):
        raise ValueError("相关概念格式错误")
    suggested = candidate.get("suggested_strategy")
    if suggested not in {None, "ma_cross", "breakout", "rsi_reversion"}:
        raise ValueError("建议策略不受支持")
    return {
        "answer": answer,
        "related_concepts": concepts[:3],
        "suggested_strategy": suggested,
        "source": source,
    }


def _local_answer(question: str) -> dict:
    for keywords, answer in _LOCAL_TOPICS:
        if any(keyword in question for keyword in keywords):
            strategy = None
            if any(keyword in question for keyword in ("双均线", "均线")):
                strategy = "ma_cross"
            elif any(keyword in question for keyword in ("价格突破", "通道突破", "突破策略")):
                strategy = "breakout"
            elif any(keyword in question for keyword in ("RSI", "rsi", "超买", "超卖", "相对强弱")):
                strategy = "rsi_reversion"
            return {
                "answer": answer,
                "related_concepts": [],
                "suggested_strategy": strategy,
                "source": "local",
            }
    return {
        "answer": (
            "目前离线知识只覆盖量化、回测、均线、价格突破、RSI、回撤、滑点和基准等基础概念。"
            "你可以换成其中一个具体概念提问；配置模型后，我也可以在不查询实时行情的范围内继续解释。"
        ),
        "related_concepts": ["回测", "最大回撤", "交易成本"],
        "suggested_strategy": None,
        "source": "local",
    }
