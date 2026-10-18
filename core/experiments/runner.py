"""成本敏感性、参数邻域和分时段实验的确定性执行器。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import Callable

import pandas as pd

from core.backtest.engine import run_backtest
from core.data import load_snapshot
from core.schemas import (
    ExperimentResult,
    ExperimentSpec,
    StrategyConfig,
    ValidationReport,
)

_METRIC_ALIASES = {"net_excess_return": "excess_return"}
_SUPPORTED_METRICS = {"excess_return"}
_NORMAL_COST_MAX_BPS = 5.0
_PARAMETER_PASS_RATIO = 0.70
_DEFAULT_SUPPORT_RULES = {
    "cost_sweep": "all(excess_return > 0 where slippage_bps <= 5)",
    "param_neighborhood": "ratio(excess_return > 0) >= 0.70",
    "period_split": "count(excess_return > 0) >= 1",
}


def resolve_experiment_spec(spec: ExperimentSpec, config: StrategyConfig) -> ExperimentSpec:
    """补齐 NLP 留空的实验 ID、扫描网格、指标名和判断规则。

    三类实验使用各自固定、可解释的 v1 规则。后续若要支持用户自定义阈值，
    应扩展结构化契约，而不是执行任意表达式。
    """
    if config.template != "ma_cross":
        raise NotImplementedError("验证实验当前只支持已实现回测的 ma_cross 模板")

    metric = _METRIC_ALIASES.get(spec.metric, spec.metric or "excess_return")
    if metric not in _SUPPORTED_METRICS:
        raise ValueError(f"实验指标不是 Metrics 字段: {metric!r}")

    grid = spec.grid or _default_grid(spec.kind, config)
    expected_rule = _DEFAULT_SUPPORT_RULES.get(spec.kind)
    if expected_rule is None:
        raise ValueError(f"未知实验类型: {spec.kind!r}")
    support_rule = spec.support_rule or expected_rule
    if support_rule != expected_rule:
        raise ValueError(
            f"当前仅支持判断规则 {expected_rule!r}，收到: {support_rule!r}"
        )

    provisional = replace(
        spec,
        metric=metric,
        grid=grid,
        support_rule=support_rule,
    )
    experiment_id = spec.experiment_id or _experiment_id(provisional, config)
    return replace(provisional, experiment_id=experiment_id)


def run_experiment(
    config: StrategyConfig,
    spec: ExperimentSpec,
    benchmark_symbol: str | None = None,
) -> ExperimentResult:
    """从 ``config.snapshot_id`` 读取一次快照并执行完整扫描。

    未显式传入基准时，默认用策略首个标的的买入持有净值计算超额收益。
    实验必须经过用户确认，避免解析结果未经确认就触发批量回测。
    """
    if not spec.confirmed_by_user:
        raise ValueError("实验尚未经过用户确认，不能运行")
    if not config.snapshot_id:
        raise ValueError("config.snapshot_id 未设置，无法运行验证实验")
    if not config.symbols:
        raise ValueError("config.symbols 不能为空")

    resolved = resolve_experiment_spec(spec, config)
    prices, _meta, quality_report = load_snapshot(config.snapshot_id)
    if quality_report.has_error:
        raise ValueError("快照存在错误级数据质量问题，不能运行验证实验")

    effective_benchmark = benchmark_symbol or config.symbols[0]
    benchmark_prices = prices[prices["symbol"] == effective_benchmark]
    if benchmark_prices.empty:
        benchmark_prices = None

    runners: dict[str, Callable[..., list[dict]]] = {
        "cost_sweep": _run_cost_sweep,
        "param_neighborhood": _run_param_neighborhood,
        "period_split": _run_period_split,
    }
    try:
        runner = runners[resolved.kind]
    except KeyError as err:
        raise ValueError(f"未知实验类型: {resolved.kind!r}") from err

    rows = runner(prices, config, resolved, benchmark_prices)
    verdict, first_failure, passed_points, total_points, pass_rate, rule_satisfied = (
        _judge(resolved.kind, rows)
    )
    return ExperimentResult(
        experiment_id=resolved.experiment_id,
        rows=rows,
        verdict=verdict,
        first_failure=first_failure,
        config_hash=_config_hash(config, resolved, effective_benchmark),
        passed_points=passed_points,
        total_points=total_points,
        pass_rate=pass_rate,
        rule_satisfied=rule_satisfied,
    )


def run_validation_suite(
    config: StrategyConfig,
    benchmark_symbol: str | None = None,
) -> ValidationReport:
    """运行基准、成本、策略参数和市场时段四部分验证。

    ``event_analysis`` 在 v1 中保持为空，作为以后接入时事归因的输出接口；
    它不参与当前结论。
    """
    if not config.snapshot_id:
        raise ValueError("config.snapshot_id 未设置，无法运行验证实验")
    if not config.symbols:
        raise ValueError("config.symbols 不能为空")

    prices, _meta, quality_report = load_snapshot(config.snapshot_id)
    if quality_report.has_error:
        raise ValueError("快照存在错误级数据质量问题，不能运行验证实验")

    effective_benchmark = benchmark_symbol or config.symbols[0]
    benchmark_prices = prices[prices["symbol"] == effective_benchmark]
    if benchmark_prices.empty:
        raise ValueError(f"快照中没有比较基准 {effective_benchmark}")

    baseline = _run_point(
        prices,
        config,
        "excess_return",
        benchmark_prices,
        {"case": "baseline", "is_baseline": True},
        include_records=True,
    )

    specs = {
        "cost_sensitivity": ExperimentSpec(
            experiment_id="",
            kind="cost_sweep",
            claim_text="策略在合理交易成本下仍然有效",
            metric="excess_return",
            grid={},
            support_rule="",
            confirmed_by_user=True,
        ),
        "strategy_parameter_robustness": ExperimentSpec(
            experiment_id="",
            kind="param_neighborhood",
            claim_text="策略不依赖单个特殊参数组合",
            metric="excess_return",
            grid={},
            support_rule="",
            confirmed_by_user=True,
        ),
        "market_period_robustness": ExperimentSpec(
            experiment_id="",
            kind="period_split",
            claim_text="原策略在不同真实市场时段具有一定适用性",
            metric="excess_return",
            grid={},
            support_rule="",
            confirmed_by_user=True,
        ),
    }
    experiments = {
        name: run_experiment(config, spec, effective_benchmark)
        for name, spec in specs.items()
    }

    suite_payload = {
        "version": 2,
        "config": asdict(config),
        "benchmark_symbol": effective_benchmark,
        "rules": _DEFAULT_SUPPORT_RULES,
    }
    config_hash = _hash_payload(suite_payload)
    return ValidationReport(
        validation_id=f"validation_{config_hash[:10]}",
        baseline=baseline,
        experiments=experiments,
        config_hash=config_hash,
        event_analysis=None,
        baseline_daily=baseline.pop("daily_records", []),
        baseline_trades=baseline.pop("trade_records", []),
        context={
            "config": asdict(config),
            "benchmark_symbol": effective_benchmark,
            "snapshot": asdict(_meta),
            "quality": asdict(quality_report),
            "limitations": [
                "参数扫描仅覆盖局部范围，未经样本外验证。",
                "时间分段独立启动并重新预热，不能按收益率直接相加。",
                "基准为未扣交易成本的买入持有；策略收益已扣配置成本。",
                "成交笔数包含买入与卖出，不等于完整交易轮次。",
                "未接入事件证据，不能判断时事因果关系。",
            ],
        },
    )


def _default_grid(kind: str, config: StrategyConfig) -> dict:
    if kind == "cost_sweep":
        values = {0.0, 1.0, 2.0, 5.0, 10.0, float(config.cost.slippage_bps)}
        return {"slippage_bps": sorted(values)}
    if kind == "param_neighborhood":
        fast = int(config.params.get("fast", 5))
        slow = int(config.params.get("slow", 20))
        fast_values = sorted({max(1, fast - 2), fast, fast + 2})
        slow_values = sorted({max(2, slow - 5), slow, slow + 5})
        return {"fast": fast_values, "slow": slow_values}
    if kind == "period_split":
        start = pd.Timestamp(config.start)
        end = pd.Timestamp(config.end)
        if start >= end:
            raise ValueError("分时段实验要求开始日期早于结束日期")
        midpoint = (start + (end - start) / 2).normalize()
        second_start = midpoint + pd.Timedelta(days=1)
        return {
            "periods": [
                {"label": "前半段", "start": str(start.date()), "end": str(midpoint.date())},
                {"label": "后半段", "start": str(second_start.date()), "end": str(end.date())},
            ]
        }
    raise ValueError(f"未知实验类型: {kind!r}")


def _run_cost_sweep(
    prices: pd.DataFrame,
    config: StrategyConfig,
    spec: ExperimentSpec,
    benchmark_prices: pd.DataFrame | None,
) -> list[dict]:
    values = spec.grid.get("slippage_bps")
    if not isinstance(values, list) or not values:
        raise ValueError("cost_sweep.grid.slippage_bps 必须是非空列表")

    rows: list[dict] = []
    for value in values:
        bps = float(value)
        if bps < 0:
            raise ValueError("slippage_bps 不能为负数")
        point_config = replace(
            config,
            cost=replace(config.cost, slippage_bps=bps),
        )
        rows.append(_run_point(
            prices,
            point_config,
            spec.metric,
            benchmark_prices,
            {
                "slippage_bps": bps,
                "is_baseline": bps == float(config.cost.slippage_bps),
            },
        ))
    return rows


def _run_param_neighborhood(
    prices: pd.DataFrame,
    config: StrategyConfig,
    spec: ExperimentSpec,
    benchmark_prices: pd.DataFrame | None,
) -> list[dict]:
    fast_values = spec.grid.get("fast")
    slow_values = spec.grid.get("slow")
    if not isinstance(fast_values, list) or not fast_values:
        raise ValueError("param_neighborhood.grid.fast 必须是非空列表")
    if not isinstance(slow_values, list) or not slow_values:
        raise ValueError("param_neighborhood.grid.slow 必须是非空列表")

    rows: list[dict] = []
    for fast_value in fast_values:
        for slow_value in slow_values:
            fast, slow = int(fast_value), int(slow_value)
            if fast <= 0 or slow <= 0 or fast >= slow:
                continue
            point_config = replace(config, params={**config.params, "fast": fast, "slow": slow})
            rows.append(_run_point(
                prices,
                point_config,
                spec.metric,
                benchmark_prices,
                {
                    "fast": fast,
                    "slow": slow,
                    "is_baseline": (
                        fast == int(config.params.get("fast", 5))
                        and slow == int(config.params.get("slow", 20))
                    ),
                },
            ))
    if not rows:
        raise ValueError("参数邻域没有合法的 fast/slow 组合")
    return rows


def _run_period_split(
    prices: pd.DataFrame,
    config: StrategyConfig,
    spec: ExperimentSpec,
    benchmark_prices: pd.DataFrame | None,
) -> list[dict]:
    periods = spec.grid.get("periods")
    if not isinstance(periods, list) or not periods:
        raise ValueError("period_split.grid.periods 必须是非空列表")

    rows: list[dict] = []
    for index, period in enumerate(periods, start=1):
        if not isinstance(period, dict) or not period.get("start") or not period.get("end"):
            raise ValueError("每个 period 必须包含 start 和 end")
        start, end = str(period["start"]), str(period["end"])
        point_config = replace(config, start=start, end=end)
        rows.append(_run_point(
            prices,
            point_config,
            spec.metric,
            benchmark_prices,
            {"period": period.get("label") or f"时段{index}", "start": start, "end": end},
        ))
    return rows


def _run_point(
    prices: pd.DataFrame,
    config: StrategyConfig,
    metric: str,
    benchmark_prices: pd.DataFrame | None,
    dimensions: dict,
    include_records: bool = False,
) -> dict:
    try:
        trades, _daily, metrics = run_backtest(prices, config, benchmark_prices)
        metric_value = getattr(metrics, metric)
        return {
            **({
                "daily_records": json.loads(_daily[_daily["symbol"] == "__TOTAL__"].to_json(orient="records", date_format="iso")),
                "trade_records": json.loads(trades.to_json(orient="records", date_format="iso")),
            } if include_records else {}),
            **dimensions,
            **asdict(metrics),
            "metric": metric,
            "metric_value": metric_value,
            "passed": None if metric_value is None else bool(metric_value > 0),
            "error": None,
            "trade_rows": len(trades),
        }
    except (TypeError, ValueError) as error:
        return {
            **dimensions,
            "metric": metric,
            "metric_value": None,
            "passed": None,
            "error": str(error),
        }


def _judge(
    kind: str,
    rows: list[dict],
) -> tuple[str, dict | None, int, int, float | None, bool | None]:
    first_failure = next((row for row in rows if row.get("passed") is not True), None)
    if not rows or any(row.get("passed") is None for row in rows):
        passed_points = sum(row.get("passed") is True for row in rows)
        total_points = len(rows)
        pass_rate = passed_points / total_points if total_points else None
        return "insufficient", first_failure, passed_points, total_points, pass_rate, None

    passed = [bool(row["passed"]) for row in rows]
    passed_points = sum(passed)
    total_points = len(rows)
    pass_rate = passed_points / total_points

    if kind == "cost_sweep":
        evaluated = [
            bool(row["passed"])
            for row in rows
            if float(row["slippage_bps"]) <= _NORMAL_COST_MAX_BPS
        ]
        if not evaluated:
            return "insufficient", first_failure, passed_points, total_points, pass_rate, None
        rule_satisfied = all(evaluated)
    elif kind == "param_neighborhood":
        rule_satisfied = pass_rate >= _PARAMETER_PASS_RATIO
    elif kind == "period_split":
        rule_satisfied = passed_points >= 1
    else:
        raise ValueError(f"未知实验类型: {kind!r}")

    if rule_satisfied:
        verdict = "supported" if all(passed) else "mixed"
    elif any(passed):
        verdict = "mixed"
    else:
        verdict = "not_supported"
    return verdict, first_failure, passed_points, total_points, pass_rate, rule_satisfied




def _experiment_id(spec: ExperimentSpec, config: StrategyConfig) -> str:
    digest = _hash_payload({"config": asdict(config), "experiment": _experiment_payload(spec)})
    return f"exp_{spec.kind}_{digest[:8]}"


def _config_hash(config: StrategyConfig, spec: ExperimentSpec, benchmark_symbol: str) -> str:
    return _hash_payload({
        "config": asdict(config),
        "experiment": _experiment_payload(spec),
        "benchmark_symbol": benchmark_symbol,
    })


def _experiment_payload(spec: ExperimentSpec) -> dict:
    """只保留影响实验计算的字段，排除流程状态和生成后的 ID。"""
    payload = asdict(spec)
    payload["experiment_id"] = ""
    payload["confirmed_by_user"] = False
    return payload


def _hash_payload(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
