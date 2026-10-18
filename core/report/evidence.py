"""本次实验的只读证据目录、确定性统计和交互图表。"""
from __future__ import annotations

import hashlib
import json
import math

import pandas as pd
import plotly.graph_objects as go


GROUPS = {
    "cost_sensitivity": ("cost", "滑点敏感性"),
    "strategy_parameter_robustness": ("parameters", "参数邻域"),
    "market_period_robustness": ("periods", "时间分段"),
}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dumps(value):
    return json.dumps(clean(value), ensure_ascii=False, allow_nan=False, indent=2)


def build_evidence(report):
    """只导出事实，不向模型传入旧版 verdict / rule_satisfied。"""
    entries = {}
    prefix = report.validation_id

    def add(key, title, kind, data, parents=()):
        eid = f"{prefix}:{key}"
        entries[eid] = {
            "id": eid, "title": title, "kind": kind,
            "data": clean(data), "parents": list(parents),
        }
        return eid

    def facts(row):
        return {k: v for k, v in row.items() if k not in {
            "passed", "metric_value", "metric", "trade_rows"
        }}

    add("context", "策略、快照与实验限制", "context", report.context)
    add("baseline", "原始策略指标", "metrics", facts(report.baseline))
    daily_id = add("table:daily", "原始策略逐日净值", "table", report.baseline_daily)
    for slug, title in [("equity", "策略与基准净值"), ("drawdown", "原始策略回撤")]:
        add(f"chart:{slug}", title, "chart", {"table_id": daily_id, "chart_type": slug}, [daily_id])
    trades_id = add("table:trades", "原始策略成交记录", "table", report.baseline_trades)
    completed, pending = [], None
    for trade in report.baseline_trades:
        if trade["side"] == "buy":
            pending = trade
        elif trade["side"] == "sell" and pending is not None:
            entry_cost = pending["gross"] + pending["commission"] + pending["slippage_cost"]
            exit_value = trade["gross"] - trade["commission"] - trade["slippage_cost"]
            completed.append({"buy_date": pending["exec_date"], "sell_date": trade["exec_date"],
                              "net_pnl": exit_value - entry_cost})
            pending = None
    rounds_id = add("table:roundtrips", "已平仓交易盈亏（扣佣金及滑点）", "table", completed, [trades_id])
    add("summary:trades", "交易统计", "summary", {
        "completed_count": len(completed), "open_position_count": int(pending is not None),
        "realized_pnl": sum(r["net_pnl"] for r in completed),
        "note": "仅适用于当前单标的全仓进出；未平仓浮盈亏不计入已实现盈亏。",
    }, [rounds_id])
    for slug, title in [("trade_distribution", "已平仓交易盈亏分布"), ("trade_contribution", "累计已实现交易盈亏")]:
        add(f"chart:{slug}", title, "chart", {"table_id": rounds_id, "chart_type": slug}, [rounds_id])
    for key, (slug, title) in GROUPS.items():
        rows = [facts(row) for row in report.experiments[key].rows]
        table_id = add(f"table:{slug}", title + "数据表", "table", rows)
        valid = [r for r in rows if not r.get("error") and
                 r.get("excess_return") is not None and math.isfinite(r["excess_return"])]
        positive = sum(r["excess_return"] > 0 for r in valid)
        summary = {
            "tested_count": len(rows), "valid_count": len(valid),
            "missing_count": len(rows) - len(valid), "positive_count": positive,
            "positive_ratio": positive / len(valid) if valid else None,
            "minimum_excess_return": min((r["excess_return"] for r in valid), default=None),
            "maximum_excess_return": max((r["excess_return"] for r in valid), default=None),
            "zero_trade_count": sum(r.get("n_trades") == 0 for r in valid),
        }
        add(f"summary:{slug}", title + "统计", "summary", summary, [table_id])
        add(f"chart:{slug}", title + "图", "chart",
            {"table_id": table_id, "chart_type": slug}, [table_id])
    digest = hashlib.sha256(dumps(entries).encode("utf-8")).hexdigest()
    return {"run_id": prefix, "evidence_hash": digest, "entries": entries}


def chart_for(pack, evidence_id):
    entry = pack["entries"][evidence_id]
    data = entry["data"]
    frame = pd.DataFrame(pack["entries"][data["table_id"]]["data"])
    for col in ["excess_return", "total_return", "max_drawdown"]:
        if col not in frame:
            frame[col] = None
    kind = data["chart_type"]
    fig = go.Figure()
    if frame.empty:
        fig.add_annotation(text="没有可绘制的数据", showarrow=False)
    elif kind in {"equity", "drawdown"}:
        if kind == "equity":
            for col, title in [("equity", "策略净值"), ("benchmark_equity", "基准净值")]:
                fig.add_trace(go.Scatter(x=frame["date"], y=frame[col], mode="lines", name=title))
            fig.update_layout(yaxis_title="净值（元）")
        else:
            drawdown = frame["equity"] / frame["equity"].cummax() - 1
            fig.add_trace(go.Scatter(x=frame["date"], y=drawdown, fill="tozeroy", name="回撤"))
            fig.update_layout(yaxis_tickformat=".1%")
    elif kind == "trade_distribution":
        fig.add_trace(go.Histogram(x=frame["net_pnl"], name="已平仓交易"))
        fig.update_layout(xaxis_title="单次交易净盈亏（元）", yaxis_title="交易轮次")
    elif kind == "trade_contribution":
        fig.add_trace(go.Scatter(x=frame["sell_date"], y=frame["net_pnl"].cumsum(), mode="lines+markers", name="累计已实现盈亏"))
        fig.update_layout(yaxis_title="已实现盈亏（元，不含未平仓持仓）")
    elif kind == "parameters":
        grid = frame.pivot_table(index="fast", columns="slow", values="excess_return", dropna=False)
        fig.add_trace(go.Heatmap(x=grid.columns.tolist(), y=grid.index.tolist(),
                                z=grid.to_numpy().tolist(), zmid=0, colorscale="RdYlGn",
                                colorbar={"tickformat": ".1%"},
                                hovertemplate="快线=%{y}<br>慢线=%{x}<br>超额收益=%{z:.2%}<extra></extra>"))
        original = frame[frame.get("is_baseline", pd.Series(False, index=frame.index)) == True]
        fig.add_trace(go.Scatter(x=original["slow"], y=original["fast"], mode="markers",
                                 marker={"symbol": "star", "size": 15, "color": "black"}, name="原始参数"))
        fig.update_layout(xaxis_title="慢线周期（日）", yaxis_title="快线周期（日）")
    else:
        xcol = "slippage_bps" if kind == "cost" else "period"
        frame = frame.sort_values(xcol) if kind == "cost" else frame
        for metric, label in [("total_return", "策略收益"), ("excess_return", "超额收益")]:
            trace = go.Scatter(x=frame[xcol], y=frame[metric], name=label, mode="lines+markers") if kind == "cost" else go.Bar(x=frame[xcol], y=frame[metric], name=label)
            fig.add_trace(trace)
        if kind == "periods":
            fig.add_trace(go.Bar(x=frame[xcol], y=frame["max_drawdown"], name="最大回撤"))
        fig.update_layout(yaxis_tickformat=".1%", xaxis_title="单边滑点（bps）" if kind == "cost" else "独立回测时段", barmode="group")
    fig.update_layout(title=entry["title"])
    return fig


class EvidenceTools:
    """工具只能读取本次证据，不接收路径、代码或网络地址。"""

    def __init__(self, pack):
        self.pack = pack

    def catalog(self):
        return [{k: e[k] for k in ("id", "title", "kind")} for e in self.pack["entries"].values()]

    def call(self, name, arguments):
        if name == "get_evidence" and set(arguments) == {"evidence_id"}:
            return self.pack["entries"][arguments["evidence_id"]]
        if name == "get_parameter_cell" and set(arguments) == {"fast", "slow"}:
            entry = self.pack["entries"][f"{self.pack['run_id']}:table:parameters"]
            return {"evidence_id": entry["id"], "rows": [r for r in entry["data"] if
                    r["fast"] == arguments["fast"] and r["slow"] == arguments["slow"]]}
        raise ValueError("未知工具或参数")
