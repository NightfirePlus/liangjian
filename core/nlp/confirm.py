"""把最终的 StrategyConfig 转成自然语言复述文案，本地模板拼接，不调用大模型。

设计目的：用户不需要看懂字段名就能判断系统有没有理解歪。只简要提示本次被用户
显式覆盖的默认假设（见 core/nlp/defaults.py），未覆盖的默认值不逐条展开，
用户可查 docs/assumptions.md。
"""
from __future__ import annotations

from core.data.universe import get_etf
from core.nlp.defaults import default_note
from core.schemas import StrategyConfig

_TEMPLATE_NAMES = {
    "ma_cross": "双均线策略",
    "momentum_rotation": "动量轮动策略",
}


def build_confirmation_text(config: StrategyConfig, overridden_defaults: list[str],
                             claim_text: str | None = None) -> str:
    symbol_names = []
    for sym in config.symbols:
        info = get_etf(sym)
        symbol_names.append(f"{info.name}({sym})" if info else sym)

    template_name = _TEMPLATE_NAMES.get(config.template, config.template)
    lines = [
        f"我理解你想测试的是：{template_name}，标的为 {', '.join(symbol_names)}，"
        f"回测区间 {config.start} 至 {config.end}。"
    ]

    if config.template == "ma_cross":
        fast, slow = config.params.get("fast"), config.params.get("slow")
        lines.append(f"规则：{fast} 日均线上穿 {slow} 日均线买入，下穿卖出。")
    elif config.template == "momentum_rotation":
        lookback = config.params.get("lookback")
        top_n = config.params.get("top_n")
        rebalance = config.params.get("rebalance")
        lines.append(f"规则：按过去 {lookback} 日动量排序，持有前 {top_n} 名，{rebalance} 再平衡。")

    if overridden_defaults:
        notes = "；".join(f"{default_note(key)}（按你的要求）" for key in overridden_defaults)
        lines.append(f"按你的要求调整了以下默认设置：{notes}。")
    else:
        lines.append("其余执行细节（成交时点、复权口径、交易成本等）按默认假设处理，详见使用文档。")

    if claim_text:
        lines.append(f"同时会验证你提到的观点：“{claim_text}”。")

    lines.append("以上理解对吗？确认无误就可以开始回测，有问题请直接说明要修改的地方。")
    return "\n".join(lines)
