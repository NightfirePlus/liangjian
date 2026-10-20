"""对大模型（或规则兜底）给出的候选解析结果做程序校验。

只校验"核心内容"——缺了就无法构成一个可执行的策略，必须追问用户：
    template 是否在支持的模板枚举内；symbols 是否非空且都在资产池里，
    且候选阶段没有遗留 ambiguities；该模板必需的 params 是否齐全且合理；
    start/end 是否存在、格式正确、start < end。
"实现细节"（exec_timing/adjust/position_sizing/commission_rate/slippage_bps 等，
对应 core/nlp/defaults.py 的 DEFAULT_ASSUMPTIONS）不在这里校验，没提及就留给
defaults.apply_defaults() 落默认值，不触发追问。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.data.universe import get_etf

REQUIRED_PARAMS: dict[str, list[str]] = {
    "ma_cross": ["fast", "slow"],
    "breakout": ["entry_window", "exit_window"],
    "rsi_reversion": ["rsi_period", "buy_threshold", "sell_threshold"],
    "momentum_rotation": ["lookback", "top_n", "rebalance"],
}
SUPPORTED_TEMPLATES = tuple(REQUIRED_PARAMS.keys())

_PARAMETER_QUESTIONS = {
    "fast": "短期平均价格看最近多少个交易日",
    "slow": "长期平均价格看最近多少个交易日",
    "entry_window": "买入时，与最近多少个交易日的最高价比较",
    "exit_window": "卖出时，与最近多少个交易日的最低价比较",
    "rsi_period": "用最近多少个交易日计算涨跌强弱",
    "buy_threshold": "指标低于多少时买入",
    "sell_threshold": "指标高于多少时卖出",
    "lookback": "比较最近多少个交易日的涨幅",
    "top_n": "选涨幅靠前的几只基金",
    "rebalance": "多久重新挑选一次",
}


@dataclass
class ValidationIssue:
    field: str
    message: str   # 人能读懂的一句话，供 clarify.py 转成追问


@dataclass
class ValidationResult:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def validate_candidate(candidate: dict) -> ValidationResult:
    """candidate 为大模型/规则兜底给出的候选字典（结构见 schema_prompt.py）。"""
    issues: list[ValidationIssue] = []

    template = candidate.get("template")
    if template not in SUPPORTED_TEMPLATES:
        issues.append(ValidationIssue(
            "template", "你想在什么情况下买入、卖出？可以举个具体例子。"))

    symbols = candidate.get("symbols") or []
    ambiguities = candidate.get("ambiguities") or []
    if not symbols and not ambiguities:
        issues.append(ValidationIssue("symbols", "你想测试哪只基金？告诉我名称或代码就行，比如沪深300基金（510300）。"))
    else:
        unknown = [s for s in symbols if get_etf(s) is None]
        if unknown:
            issues.append(ValidationIssue(
                "symbols", f"暂时不支持代码 {', '.join(unknown)}，请从侧栏选择一只支持的基金。"))
    for amb in ambiguities:
        issues.append(ValidationIssue("symbols", amb))

    if template in REQUIRED_PARAMS:
        params = candidate.get("params") or {}
        missing_params = [p for p in REQUIRED_PARAMS[template] if p not in params or params[p] in (None, "")]
        if missing_params:
            questions = "；".join(_PARAMETER_QUESTIONS[p] for p in missing_params)
            explanation = (
                "这里的 RSI 是一个衡量近期涨跌强弱的指标，数值在0到100之间。"
                if template == "rsi_reversion" else ""
            )
            issues.append(ValidationIssue(
                "params", f"{explanation}还想确认：{questions}？"))
        elif template == "ma_cross":
            try:
                fast, slow = int(params["fast"]), int(params["slow"])
                if fast <= 0 or slow <= 0:
                    issues.append(ValidationIssue("params", "均线周期必须是正整数"))
                elif fast >= slow:
                    issues.append(ValidationIssue(
                        "params", f"快线周期({fast})必须小于慢线周期({slow})，请确认哪个是快线哪个是慢线"))
            except (TypeError, ValueError):
                issues.append(ValidationIssue("params", "均线周期必须是整数"))
        elif template == "breakout":
            try:
                entry = int(params["entry_window"])
                exit_ = int(params["exit_window"])
                if entry <= 0 or exit_ <= 0:
                    issues.append(ValidationIssue("params", "突破和退出周期必须是正整数"))
                elif entry <= exit_:
                    issues.append(ValidationIssue(
                        "params", f"突破周期({entry})必须大于退出周期({exit_})"))
            except (TypeError, ValueError):
                issues.append(ValidationIssue("params", "突破和退出周期必须是整数"))
        elif template == "rsi_reversion":
            try:
                period = int(params["rsi_period"])
                buy = float(params["buy_threshold"])
                sell = float(params["sell_threshold"])
                if period <= 0:
                    issues.append(ValidationIssue("params", "计算涨跌强弱的天数需要填大于0的整数，比如14。"))
                elif not 0 < buy < sell < 100:
                    issues.append(ValidationIssue(
                        "params", "买入时的指标数值要小于卖出时的数值，两个数都要在0到100之间。"))
            except (TypeError, ValueError):
                issues.append(ValidationIssue("params", "请用数字填写计算天数，以及买入、卖出时的指标数值。"))

    start, end = candidate.get("start"), candidate.get("end")
    if not start or not end:
        if start:
            question = "想测到哪一天？比如2025-12-31。"
        elif end:
            question = "想从哪一天开始测试？比如2023-01-01。"
        else:
            question = "想用哪段时间的历史行情测试？比如2023-01-01到2025-12-31。"
        issues.append(ValidationIssue("period", question))
    elif start >= end:
        issues.append(ValidationIssue("period", f"起始日期({start})不能晚于/等于结束日期({end})"))

    # 缺失项由上面的程序校验统一判断。模型的 missing 可能包含内部字段名、
    # 重复问题，甚至与实际已提供的信息矛盾，不能直接展示或阻止用户继续。

    return ValidationResult(issues=issues)
