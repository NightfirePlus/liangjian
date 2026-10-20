"""三种创建入口共用的、尚未确认的策略草案。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Literal

from core.schemas import CostConfig, StrategyConfig

DraftOrigin = Literal["conversation", "quick", "preset", "manual"]


@dataclass
class StrategyDraft:
    """可编辑草案；只有用户确认后才转换成正式 ``StrategyConfig``。"""

    template: Literal["ma_cross", "breakout", "rsi_reversion"] = "ma_cross"
    symbol: str = ""
    start: str = ""
    end: str = ""
    fast: int = 5
    slow: int = 20
    entry_window: int = 20
    exit_window: int = 10
    rsi_period: int = 14
    buy_threshold: float = 30.0
    sell_threshold: float = 70.0
    benchmark_symbol: str | None = None
    initial_cash: float = 100_000.0
    commission_rate: float = 0.0001
    min_commission: float = 0.0
    slippage_bps: float = 1.0
    origin: DraftOrigin = "manual"
    specified_fields: list[str] = field(default_factory=list)
    confirmation_text: str = ""
    degraded: bool = False
    degraded_reason: str = ""
    normalization_notes: list[str] = field(default_factory=list)

    @classmethod
    def from_config(
        cls,
        config: StrategyConfig,
        *,
        benchmark_symbol: str | None = None,
        origin: DraftOrigin = "conversation",
        specified_fields: list[str] | None = None,
        confirmation_text: str = "",
        degraded: bool = False,
        degraded_reason: str = "",
    ) -> "StrategyDraft":
        if config.template not in {"ma_cross", "breakout", "rsi_reversion"}:
            raise ValueError("当前策略验证只支持双均线、价格突破和 RSI 均值回归策略")
        if len(config.symbols) != 1:
            raise ValueError("当前策略回测要求且只支持一个交易标的")
        return cls(
            template=config.template,
            symbol=config.symbols[0],
            start=config.start,
            end=config.end,
            fast=int(config.params.get("fast", 5)),
            slow=int(config.params.get("slow", 20)),
            entry_window=int(config.params.get("entry_window", 20)),
            exit_window=int(config.params.get("exit_window", 10)),
            rsi_period=int(config.params.get("rsi_period", 14)),
            buy_threshold=float(config.params.get("buy_threshold", 30)),
            sell_threshold=float(config.params.get("sell_threshold", 70)),
            benchmark_symbol=benchmark_symbol or config.symbols[0],
            initial_cash=float(config.initial_cash),
            commission_rate=float(config.cost.commission_rate),
            min_commission=float(config.cost.min_commission),
            slippage_bps=float(config.cost.slippage_bps),
            origin=origin,
            specified_fields=list(specified_fields or []),
            confirmation_text=confirmation_text,
            degraded=degraded,
            degraded_reason=degraded_reason,
        )

    def validate(
        self,
        *,
        available_symbols: list[str] | None = None,
        available_start: str | None = None,
        available_end: str | None = None,
    ) -> list[str]:
        issues: list[str] = []
        if not self.symbol:
            issues.append("请选择交易标的")
        elif available_symbols is not None and self.symbol not in available_symbols:
            issues.append(f"当前行情快照不包含标的 {self.symbol}")
        try:
            start, end = date.fromisoformat(self.start), date.fromisoformat(self.end)
            if start >= end:
                issues.append("开始日期必须早于结束日期")
            if available_start and start < date.fromisoformat(available_start):
                issues.append(f"开始日期早于当前快照可用日期 {available_start}")
            if available_end and end > date.fromisoformat(available_end):
                issues.append(f"结束日期晚于当前快照可用日期 {available_end}")
        except ValueError:
            issues.append("日期必须使用 YYYY-MM-DD 格式")
        if self.template == "ma_cross":
            if self.fast <= 0 or self.slow <= 0:
                issues.append("均线周期必须是正整数")
            elif self.fast >= self.slow:
                issues.append("快线周期必须小于慢线周期")
        elif self.template == "breakout":
            if self.entry_window <= 0 or self.exit_window <= 0:
                issues.append("突破和退出周期必须是正整数")
            elif self.entry_window <= self.exit_window:
                issues.append("突破周期必须大于退出周期")
        elif self.template == "rsi_reversion":
            if self.rsi_period <= 0:
                issues.append("RSI 周期必须是正整数")
            elif not 0 < self.buy_threshold < self.sell_threshold < 100:
                issues.append("RSI 阈值必须满足 0 < 买入阈值 < 卖出阈值 < 100")
        else:
            issues.append(f"当前不支持策略模板 {self.template}")
        if self.initial_cash < 1_000:
            issues.append("初始资金不能少于 1,000 元")
        if self.commission_rate < 0 or self.min_commission < 0 or self.slippage_bps < 0:
            issues.append("交易成本不能为负数")
        return issues

    def to_config(self, snapshot_id: str) -> StrategyConfig:
        issues = self.validate()
        if issues:
            raise ValueError("；".join(issues))
        if self.template == "ma_cross":
            params = {"fast": self.fast, "slow": self.slow}
        elif self.template == "breakout":
            params = {"entry_window": self.entry_window, "exit_window": self.exit_window}
        else:
            params = {
                "rsi_period": int(self.rsi_period),
                "buy_threshold": float(self.buy_threshold),
                "sell_threshold": float(self.sell_threshold),
            }
        return StrategyConfig(
            template=self.template,
            symbols=[self.symbol],
            start=self.start,
            end=self.end,
            initial_cash=self.initial_cash,
            params=params,
            cost=CostConfig(
                commission_rate=self.commission_rate,
                min_commission=self.min_commission,
                slippage_bps=self.slippage_bps,
            ),
            snapshot_id=snapshot_id,
            assumptions=[
                "收盘形成信号，次日开盘成交",
                "仅多头，满仓操作",
                "允许小数份额",
            ],
        )

    def execution_hash(self, snapshot_id: str) -> str:
        """确认状态只绑定实际计算输入，不受入口或说明文案影响。"""
        payload = asdict(self.to_config(snapshot_id))
        payload["benchmark_symbol"] = self.benchmark_symbol
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def align_draft_to_available_period(
    draft: StrategyDraft,
    available_start: str,
    available_end: str,
    *,
    tolerance_days: int = 10,
) -> StrategyDraft:
    """将紧邻行情边界的非交易日对齐到实际可用交易日。"""
    try:
        requested_start = date.fromisoformat(draft.start)
        requested_end = date.fromisoformat(draft.end)
        actual_start = date.fromisoformat(available_start)
        actual_end = date.fromisoformat(available_end)
    except ValueError:
        return draft

    start, end = requested_start, requested_end
    notes = list(draft.normalization_notes)
    if requested_start < actual_start and (actual_start - requested_start).days <= tolerance_days:
        start = actual_start
        notes.append(
            f"开始日期 {requested_start.isoformat()} 为非交易日或紧邻数据边界，"
            f"已对齐到首个可用交易日 {actual_start.isoformat()}。"
        )
    if requested_end > actual_end and (requested_end - actual_end).days <= tolerance_days:
        end = actual_end
        notes.append(
            f"结束日期 {requested_end.isoformat()} 为非交易日或紧邻数据边界，"
            f"已对齐到最后可用交易日 {actual_end.isoformat()}。"
        )
    return replace(draft, start=start.isoformat(), end=end.isoformat(), normalization_notes=notes)


def specified_fields_from_candidate(candidate: dict | None) -> list[str]:
    """记录自然语言中实际出现的字段，供确认卡区分用户输入与默认值。"""
    candidate = candidate or {}
    fields: list[str] = []
    if candidate.get("template"):
        fields.append("策略类型")
    if candidate.get("symbols"):
        fields.append("交易标的")
    if candidate.get("start") or candidate.get("end"):
        fields.append("测试区间")
    params = candidate.get("params") or {}
    if params.get("fast") is not None:
        fields.append("快线周期")
    if params.get("slow") is not None:
        fields.append("慢线周期")
    if params.get("entry_window") is not None:
        fields.append("突破周期")
    if params.get("exit_window") is not None:
        fields.append("退出周期")
    if params.get("rsi_period") is not None:
        fields.append("RSI 周期")
    if params.get("buy_threshold") is not None:
        fields.append("超卖买入阈值")
    if params.get("sell_threshold") is not None:
        fields.append("超买卖出阈值")
    override_labels = {
        "commission_rate": "佣金",
        "min_commission": "最低佣金",
        "slippage_bps": "滑点",
    }
    for key in (candidate.get("overrides") or {}):
        if key in override_labels:
            fields.append(override_labels[key])
    return fields
