"""大模型不可用或调用失败时的最小规则兜底识别。

不追求准确，只做最基础的关键词/正则匹配，目的是在没有大模型的情况下也能跑通
"输入 -> 校验 -> 追问"流程，而不是直接报错退出。能力明显弱于大模型解析，
parser.py 会在 ParseResult.degraded 里标注，提示用户当前处于降级模式。
"""
from __future__ import annotations

import re

from core.data.universe import get_universe

_MA_CROSS_KEYWORDS = ("均线", "ma_cross", "双均线")
_BREAKOUT_KEYWORDS = ("突破", "新高", "通道", "breakout")
_RSI_KEYWORDS = ("RSI", "rsi", "相对强弱", "超买", "超卖")
_ROTATION_KEYWORDS = ("轮动", "动量")
_DATE_PATTERN = re.compile(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})")


def parse_with_rules(text: str) -> dict:
    """返回结构与 schema_prompt.py 约定的候选字典一致，但字段覆盖率低得多。"""
    candidate: dict = {
        "template": None, "symbol_mentions": [], "symbols": [], "start": None, "end": None,
        "params": {}, "overrides": {}, "claim_text": None, "ambiguities": [], "missing": [],
    }

    template_mentions = [
        (text.rfind(keyword), template)
        for template, keywords in (
            ("ma_cross", _MA_CROSS_KEYWORDS),
            ("breakout", _BREAKOUT_KEYWORDS),
            ("rsi_reversion", _RSI_KEYWORDS),
            ("momentum_rotation", _ROTATION_KEYWORDS),
        )
        for keyword in keywords
        if keyword in text
    ]
    if template_mentions:
        candidate["template"] = max(template_mentions)[1]

    # 当前执行器是单标的；多轮修改时以最后一次出现的受支持标的为准。
    symbol_mentions = []
    for etf in get_universe():
        positions = [
            text.rfind(alias)
            for alias in (etf.symbol, etf.name, etf.name.replace("ETF", ""))
            if alias and alias in text
        ]
        if positions:
            symbol_mentions.append((max(positions), etf))
    if symbol_mentions:
        _position, selected = max(symbol_mentions, key=lambda item: item[0])
        candidate["symbols"] = [selected.symbol]
        candidate["symbol_mentions"] = [selected.name]

    dates = _DATE_PATTERN.findall(text)
    if len(dates) >= 1:
        y, m, d = dates[-2] if len(dates) >= 2 else dates[-1]
        candidate["start"] = f"{y}-{int(m):02d}-{int(d):02d}"
    if len(dates) >= 2:
        y, m, d = dates[-1]
        candidate["end"] = f"{y}-{int(m):02d}-{int(d):02d}"

    pairs = re.findall(
        r"(\d+)\s*日(?:均线)?[^\d]{0,12}(\d+)\s*日(?:均线)?",
        text,
    )
    if candidate["template"] == "ma_cross" and pairs:
        fast, slow = pairs[-1]
        candidate["params"] = {"fast": int(fast), "slow": int(slow)}
    if candidate["template"] == "breakout":
        entry_matches = re.findall(r"(\d+)\s*日[^。；，,]{0,12}(?:新高|最高|向上突破)", text)
        exit_matches = re.findall(r"(\d+)\s*日[^。；，,]{0,12}(?:新低|最低|向下突破)", text)
        if entry_matches and exit_matches:
            candidate["params"] = {
                "entry_window": int(entry_matches[-1]),
                "exit_window": int(exit_matches[-1]),
            }
    if candidate["template"] == "rsi_reversion":
        period_matches = re.findall(
            r"(?:RSI\s*(?:周期)?\s*(?:为|是|=)?\s*(\d+)|(\d+)\s*日\s*RSI)",
            text,
            flags=re.IGNORECASE,
        )
        buy_matches = re.findall(
            r"(?:低于|小于|跌破|不高于|超卖(?:阈值|线)?(?:为|是)?)\s*(\d+(?:\.\d+)?)",
            text,
        )
        sell_matches = re.findall(
            r"(?:高于|大于|突破|不低于|超买(?:阈值|线)?(?:为|是)?)\s*(\d+(?:\.\d+)?)",
            text,
        )
        if period_matches:
            left, right = period_matches[-1]
            candidate["params"]["rsi_period"] = int(left or right)
        if buy_matches:
            candidate["params"]["buy_threshold"] = float(buy_matches[-1])
        if sell_matches:
            candidate["params"]["sell_threshold"] = float(sell_matches[-1])

    return candidate
