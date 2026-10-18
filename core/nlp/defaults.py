"""回测实现细节的默认假设清单。

这些是用户描述策略时通常不会特别说明的执行细节（成交时点、复权口径、仓位、
交易成本等）。约定：用户没有显式提及就按这里的默认值处理，不逐条追问、
不在确认卡里逐条复述；人读版本见 docs/assumptions.md。

但解析结果里如果用户明确要求了某一项（如"按收盘价成交"“滑点算2个基点”），
必须能覆盖对应默认值——这里提供 apply_defaults() 作为覆盖接口。

新增/修改默认值时：
1. 在 DEFAULT_ASSUMPTIONS 里加一条 (默认取值, 人读说明)。
2. 同步更新 docs/assumptions.md 的表格。
"""
from __future__ import annotations

from typing import Any

# key -> (默认取值, 人读说明)。说明文案会被 confirm.py 用在"按你的要求覆盖"提示里，
# 默认值本身不在确认卡里逐条展开，用户可查 docs/assumptions.md。
DEFAULT_ASSUMPTIONS: dict[str, tuple[Any, str]] = {
    "exec_timing": ("next_open", "收盘形成信号，次日开盘成交"),
    "adjust": ("qfq", "价格使用前复权口径"),
    "position_sizing": ("full", "单标的满仓操作，不使用杠杆"),
    "commission_rate": (0.0001, "单边佣金率 0.01%"),
    "min_commission": (0.0, "单笔最低佣金，ETF 简化为 0"),
    "slippage_bps": (1.0, "单边滑点 1 个基点"),
}


def default_value(key: str) -> Any:
    """取某一项的默认取值；key 不存在时抛 KeyError，调用方应保证 key 合法。"""
    return DEFAULT_ASSUMPTIONS[key][0]


def default_note(key: str) -> str:
    """取某一项默认值的人读说明。"""
    return DEFAULT_ASSUMPTIONS[key][1]


def apply_defaults(overrides: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """合并用户显式覆盖值与默认值。

    overrides: 用户/大模型解析出的显式取值，key 需落在 DEFAULT_ASSUMPTIONS 内，
               不认识的 key 会被忽略（不报错，由调用方决定是否记录为未识别项）。
    返回 (合并后的完整取值字典, 真正被覆盖的 key 列表)。
    未被覆盖的 key 一律落默认值，不出现在第二个返回值里。
    """
    overrides = overrides or {}
    merged: dict[str, Any] = {}
    overridden: list[str] = []
    for key, (default, _note) in DEFAULT_ASSUMPTIONS.items():
        if key in overrides and overrides[key] is not None:
            merged[key] = overrides[key]
            overridden.append(key)
        else:
            merged[key] = default
    return merged, overridden
