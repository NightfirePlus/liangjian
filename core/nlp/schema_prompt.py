"""约定大模型必须返回的 JSON 字段，及拼装 system/user prompt 的函数。"""
from __future__ import annotations

from core.data.universe import get_universe

_SCHEMA_DESCRIPTION = """\
你是一个量化策略需求解析器。你的任务是把用户的自然语言描述解析成下面这个 JSON object，
只输出这一个 JSON object，不要输出任何其他文字、不要用代码块包裹。

字段说明：
- template: "ma_cross"（双均线）、"breakout"（价格通道突破）、
  "rsi_reversion"（RSI 超卖超买均值回归）或
  "momentum_rotation"（动量轮动）之一；无法判断则为 null。
- symbol_mentions: 用户原话里提到的标的名称/指代（字符串数组），如 ["沪深300"]。
- symbols: 你从 symbol_mentions 对照下方资产池映射出的 6 位 ETF 代码数组；
  如果一个名称可能对应资产池中多个标的，不要猜，把这些候选全部写进 ambiguities，
  symbols 里不要包含该项。
- start: 回测起始日期 "YYYY-MM-DD"，用户未提及则为 null。
- end: 回测结束日期 "YYYY-MM-DD"，用户未提及则为 null。
- params: 策略参数对象。ma_cross 用 {"fast": 整数, "slow": 整数}；
  breakout 用 {"entry_window": 整数, "exit_window": 整数}，前者是向上突破的历史高点周期，
  后者是向下突破的历史低点周期，且 entry_window 必须大于 exit_window；
  rsi_reversion 用 {"rsi_period": 整数, "buy_threshold": 数字, "sell_threshold": 数字}，
  阈值必须满足 0 < buy_threshold < sell_threshold < 100；
  momentum_rotation 用 {"lookback": 整数, "top_n": 整数, "rebalance": 字符串}。
  用户没提到的参数不要编造，缺的就不要放进这个对象里。
- overrides: 用户明确提出的、与下方默认假设不同的执行细节，只在用户确实提及时才放进来，
  可能的 key: exec_timing（"next_open"/"same_close"）、adjust（"qfq"/"hfq"/"none"）、
  position_sizing（"full" 或具体仓位说明字符串）、commission_rate（小数）、
  slippage_bps（数字）。用户没特别要求的字段不要出现在这里。
- claim_text: 如果用户表达了一个关于策略表现的观点/猜测（比如"我认为这个策略对交易成本不敏感"），
  原样摘录这句话；没有则为 null。
- ambiguities: 字符串数组，仅列出真正有多种解释的点，用简短中文写清楚候选是什么。
  这些文字会展示给新手，不得使用内部字段名、代码路径或未解释的专业术语。
  缺少名称、日期、买卖条件等信息不属于歧义，不要放在这里。
- missing: 固定输出空数组。缺少哪些信息由程序统一检查，不需要重复列出。

理解新手表达时：
- “跌得多就买，涨回来就卖”等模糊想法，不足以确定使用哪种指标。
  用户没有提到指标或具体规则时，template 为 null；不要擅自选 RSI 或编造参数。
- 不要把你建议的策略、参数或日期当成用户已同意的设置。

以下是本次支持的 ETF 资产池（代码 - 名称），symbols 只能从这里面选：
{universe}

默认假设（用户没特别提及就按这个处理，不需要你在 missing/ambiguities 里提这些，
只有用户明确要求了不同取值时才放进 overrides）：
- 收盘形成信号，次日开盘价成交
- 价格使用前复权口径
- 单标的满仓操作，不使用杠杆
- 单边佣金率 0.01%，单边滑点 1 个基点
"""


def build_system_prompt() -> str:
    universe_lines = "\n".join(f"{e.symbol} - {e.name}" for e in get_universe())
    return _SCHEMA_DESCRIPTION.replace("{universe}", universe_lines)


def build_user_prompt(text: str, history: list[dict] | None = None) -> str:
    """history 为上一轮的追问往返记录，每项形如 {"questions": [...], "answer": "..."}。

    把历史追问和用户本轮的新回答拼在一起交给大模型，让它结合上下文重新给出完整候选，
    而不是只处理本轮这一句话（否则多轮累积的信息会丢失）。
    """
    if not history:
        return f"用户描述：{text}"

    parts = ["以下是之前几轮的澄清过程，请结合全部信息给出完整的 JSON 候选结果：\n"]
    for i, turn in enumerate(history, start=1):
        questions = turn.get("questions") or []
        answer = turn.get("answer", "")
        parts.append(f"第 {i} 轮，系统问: " + "；".join(questions))
        parts.append(f"第 {i} 轮，用户答: {answer}")
    parts.append(f"用户最初的描述：{text}")
    return "\n".join(parts)
