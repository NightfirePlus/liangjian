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
            "template", "没看出你想用哪种策略模板（目前可执行双均线、价格突破和 RSI 均值回归）"))

    symbols = candidate.get("symbols") or []
    ambiguities = candidate.get("ambiguities") or []
    if not symbols and not ambiguities:
        issues.append(ValidationIssue("symbols", "没识别出具体的标的，需要补充想测试哪个 ETF"))
    else:
        unknown = [s for s in symbols if get_etf(s) is None]
        if unknown:
            issues.append(ValidationIssue(
                "symbols", f"标的代码 {', '.join(unknown)} 不在当前支持的资产池内"))
    for amb in ambiguities:
        issues.append(ValidationIssue("symbols", amb))

    if template in REQUIRED_PARAMS:
        params = candidate.get("params") or {}
        missing_params = [p for p in REQUIRED_PARAMS[template] if p not in params or params[p] in (None, "")]
        if missing_params:
            issues.append(ValidationIssue(
                "params", f"策略参数缺少 {', '.join(missing_params)}，需要补充具体数值"))
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
                    issues.append(ValidationIssue("params", "RSI 周期必须是正整数"))
                elif not 0 < buy < sell < 100:
                    issues.append(ValidationIssue(
                        "params", "RSI 阈值必须满足 0 < 买入阈值 < 卖出阈值 < 100"))
            except (TypeError, ValueError):
                issues.append(ValidationIssue("params", "RSI 周期和阈值必须是数字"))

    start, end = candidate.get("start"), candidate.get("end")
    if not start or not end:
        issues.append(ValidationIssue("period", "没说清楚回测的起止日期"))
    elif start >= end:
        issues.append(ValidationIssue("period", f"起始日期({start})不能晚于/等于结束日期({end})"))

    for extra_missing in candidate.get("missing") or []:
        # 大模型自己标注的缺失项，去重后一并纳入（规则兜底里基本不会有这个字段）
        if not any(extra_missing == i.message for i in issues):
            issues.append(ValidationIssue("missing", extra_missing))

    return ValidationResult(issues=issues)
