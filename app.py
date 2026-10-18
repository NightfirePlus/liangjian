"""量见 Streamlit 界面：手动配置策略并运行真实快照回测。"""
from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import APP_NAME, APP_TAGLINE, APP_VERSION, DISCLAIMER, LLM_AVAILABLE
from core.backtest.engine import run_backtest_from_snapshot
from core.data import get_etf, list_snapshots, load_snapshot
from core.experiments import run_validation_suite
from core.report import build_evidence, chart_for, generate_report, export_bundle
from core.schemas import CostConfig, Metrics, StrategyConfig, ValidationReport

st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")


def _symbol_label(symbol: str) -> str:
    info = get_etf(symbol)
    return f"{info.name}（{symbol}）" if info else symbol


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def _money(value: float | None) -> str:
    return "—" if value is None else f"¥{value:,.2f}"


def _render_metrics(metrics: Metrics) -> None:
    row1 = st.columns(5)
    row1[0].metric("总收益", _percent(metrics.total_return))
    row1[1].metric("年化收益", _percent(metrics.annual_return))
    row1[2].metric("年化波动", _percent(metrics.annual_vol))
    row1[3].metric("最大回撤", _percent(metrics.max_drawdown))
    row1[4].metric("交易次数", str(metrics.n_trades))

    row2 = st.columns(4)
    row2[0].metric("基准收益", _percent(metrics.benchmark_return))
    row2[1].metric("超额收益", _percent(metrics.excess_return))
    row2[2].metric("累计成本", _money(metrics.total_cost))
    row2[3].metric("换手率", _percent(metrics.turnover))


def _render_curve(daily: pd.DataFrame) -> None:
    total = daily[daily["symbol"] == "__TOTAL__"].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=total["date"], y=total["equity"], name="策略净值", mode="lines",
    ))
    if total["benchmark_equity"].notna().any():
        fig.add_trace(go.Scatter(
            x=total["date"], y=total["benchmark_equity"], name="基准净值", mode="lines",
        ))
    fig.update_layout(
        title="策略与基准净值",
        xaxis_title="日期",
        yaxis_title="资产净值（元）",
        hovermode="x unified",
        legend_title_text="",
    )
    st.plotly_chart(fig, width="stretch")


def _render_validation_report(report: ValidationReport) -> None:
    baseline = report.baseline
    st.subheader("原始策略基准")
    baseline_cols = st.columns(4)
    baseline_cols[0].metric("策略收益", _percent(baseline.get("total_return")))
    baseline_cols[1].metric("基准收益", _percent(baseline.get("benchmark_return")))
    baseline_cols[2].metric("超额收益", _percent(baseline.get("excess_return")))
    baseline_cols[3].metric("成交笔数", str(baseline.get("n_trades", 0)))

    sections = [
        (
            "cost_sensitivity",
            "成本敏感性（本版改变滑点）",
            ["slippage_bps", "is_baseline", "total_return", "excess_return", "total_cost", "n_trades", "error"],
        ),
        (
            "strategy_parameter_robustness",
            "策略参数稳健性（改变用户策略参数）",
            ["fast", "slow", "is_baseline", "total_return", "excess_return", "n_trades", "error"],
        ),
        (
            "market_period_robustness",
            "市场时段稳健性（保持策略，改变真实数据区间）",
            ["period", "start", "end", "total_return", "excess_return", "n_trades", "error"],
        ),
    ]
    for key, title, columns in sections:
        result = report.experiments[key]
        with st.expander(title, expanded=True):
            st.caption(
                f"共 {result.total_points} 个实验点，"
                f"其中 {result.passed_points} 个超额收益为正；不代表策略有效性的判定。"
            )
            frame = pd.DataFrame(result.rows)
            visible_columns = [column for column in columns if column in frame.columns]
            st.dataframe(frame[visible_columns], width="stretch", hide_index=True)

    st.caption(f"验证编号：{report.validation_id} · 配置哈希：{report.config_hash}")


st.title(APP_NAME)
st.caption(APP_TAGLINE)
st.caption(f"测试版 {APP_VERSION}")
st.warning(DISCLAIMER)

if not LLM_AVAILABLE:
    st.info("模型尚未配置。可以运行回测、验证实验，并查看图表与事实报告。")

st.header("1. 选择数据快照")
snapshots = list_snapshots()
if not snapshots:
    st.error("本地没有可用快照。请先通过数据模块创建真实或模拟快照。")
    st.stop()

snapshot_by_id = {meta.snapshot_id: meta for meta in snapshots}
snapshot_id = st.selectbox(
    "行情快照",
    options=list(snapshot_by_id),
    format_func=lambda sid: (
        f"{sid} · {snapshot_by_id[sid].actual_start} 至 {snapshot_by_id[sid].actual_end}"
        f" · {'模拟数据' if snapshot_by_id[sid].is_synthetic else '真实数据'}"
    ),
)
prices, snapshot_meta, quality_report = load_snapshot(snapshot_id)

if snapshot_meta.is_synthetic:
    st.warning("当前选择的是模拟数据，只能用于验证系统流程，不能用于判断策略有效性。")
if quality_report.has_error:
    st.error("该快照存在错误级数据质量问题，请先处理后再回测。")

meta_cols = st.columns(4)
meta_cols[0].metric("数据来源", snapshot_meta.source)
meta_cols[1].metric("标的数量", str(len(snapshot_meta.symbols)))
meta_cols[2].metric("实际开始", snapshot_meta.actual_start)
meta_cols[3].metric("数据截止", snapshot_meta.actual_end)

with st.expander("查看快照与质量详情"):
    st.json(asdict(snapshot_meta))
    if quality_report.issues:
        st.dataframe(
            pd.DataFrame([asdict(issue) for issue in quality_report.issues]),
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("未记录数据质量问题。")

st.header("2. 手动配置策略")
st.caption("自然语言模块完成后，它会自动填写这些字段；回测入口保持不变。")

symbol_col, benchmark_col = st.columns(2)
symbol = symbol_col.selectbox(
    "交易标的",
    options=snapshot_meta.symbols,
    format_func=_symbol_label,
)
benchmark_options = ["__NONE__", *snapshot_meta.symbols]
benchmark_symbol = benchmark_col.selectbox(
    "比较基准",
    options=benchmark_options,
    index=benchmark_options.index(symbol),
    format_func=lambda value: "不计算基准" if value == "__NONE__" else f"买入持有：{_symbol_label(value)}",
)
benchmark_symbol = None if benchmark_symbol == "__NONE__" else benchmark_symbol

symbol_prices = prices[prices["symbol"] == symbol]
min_date = symbol_prices["date"].min().date()
max_date = symbol_prices["date"].max().date()
date_col1, date_col2 = st.columns(2)
start = date_col1.date_input(
    "开始日期", value=min_date, min_value=min_date, max_value=max_date,
    key=f"start_{snapshot_id}_{symbol}",
)
end = date_col2.date_input(
    "结束日期", value=max_date, min_value=min_date, max_value=max_date,
    key=f"end_{snapshot_id}_{symbol}",
)

param_col1, param_col2, cash_col = st.columns(3)
fast = int(param_col1.number_input("快线周期（日）", min_value=1, value=5, step=1))
slow = int(param_col2.number_input("慢线周期（日）", min_value=2, value=20, step=1))
initial_cash = float(cash_col.number_input(
    "初始资金（元）", min_value=1_000.0, value=100_000.0, step=10_000.0,
))

with st.expander("交易成本", expanded=False):
    cost_col1, cost_col2, cost_col3 = st.columns(3)
    commission_bps = float(cost_col1.number_input(
        "单边佣金（bps）", min_value=0.0, value=1.0, step=0.1,
        help="1 bps = 0.01%",
    ))
    min_commission = float(cost_col2.number_input(
        "单笔最低佣金（元）", min_value=0.0, value=0.0, step=1.0,
    ))
    slippage_bps = float(cost_col3.number_input(
        "单边滑点（bps）", min_value=0.0, value=1.0, step=0.5,
    ))

date_invalid = start > end
param_invalid = fast >= slow
if date_invalid:
    st.error("开始日期不能晚于结束日期。")
if param_invalid:
    st.error("快线周期必须小于慢线周期。")

config = StrategyConfig(
    template="ma_cross",
    symbols=[symbol],
    start=start.isoformat(),
    end=end.isoformat(),
    initial_cash=initial_cash,
    params={"fast": fast, "slow": slow},
    cost=CostConfig(
        commission_rate=commission_bps / 10_000,
        min_commission=min_commission,
        slippage_bps=slippage_bps,
    ),
    snapshot_id=snapshot_id,
    assumptions=["收盘形成信号，次日开盘成交", "仅多头，满仓操作", "允许小数份额"],
)

st.subheader("策略确认卡")
confirm_left, confirm_right = st.columns([2, 1])
with confirm_left:
    st.json(asdict(config))
with confirm_right:
    st.markdown("**执行规则**")
    for assumption in config.assumptions:
        st.write(f"- {assumption}")
    st.write(f"- 行情口径：{snapshot_meta.adjust}")
    st.write(f"- 数据来源：{snapshot_meta.source}")
    st.write(f"- 比较基准：{_symbol_label(benchmark_symbol) if benchmark_symbol else '不计算'}")

run_disabled = date_invalid or param_invalid or quality_report.has_error
run_clicked = st.button("运行回测", type="primary", disabled=run_disabled)

result_key = (
    snapshot_id, symbol, benchmark_symbol, start.isoformat(), end.isoformat(),
    fast, slow, initial_cash, commission_bps, min_commission, slippage_bps,
)
if run_clicked:
    try:
        with st.spinner("正在读取快照并计算回测……"):
            trades, daily, metrics = run_backtest_from_snapshot(
                config, benchmark_symbol=benchmark_symbol,
            )
        st.session_state["backtest_result"] = {
            "key": result_key,
            "config": config,
            "trades": trades,
            "daily": daily,
            "metrics": metrics,
        }
    except Exception as error:
        st.exception(error)

stored = st.session_state.get("backtest_result")
if stored and stored["key"] != result_key:
    st.info("配置已经改变，请重新运行回测以更新结果。")
elif stored:
    st.header("3. 回测结果")
    metrics = stored["metrics"]
    trades = stored["trades"]
    daily = stored["daily"]

    _render_metrics(metrics)
    _render_curve(daily)

    if metrics.max_dd_start and metrics.max_dd_end:
        st.caption(f"最大回撤区间：{metrics.max_dd_start} 至 {metrics.max_dd_end}")
    for note in metrics.notes:
        st.info(note)

    st.subheader("交易记录")
    if trades.empty:
        st.info("当前区间没有触发交易。")
    else:
        display_trades = trades.copy()
        display_trades["signal_date"] = pd.to_datetime(display_trades["signal_date"]).dt.date
        display_trades["exec_date"] = pd.to_datetime(display_trades["exec_date"]).dt.date
        st.dataframe(display_trades, width="stretch", hide_index=True)

    download_col1, download_col2 = st.columns(2)
    download_col1.download_button(
        "下载交易记录 CSV",
        data=trades.to_csv(index=False).encode("utf-8-sig"),
        file_name="trades.csv",
        mime="text/csv",
    )
    download_col2.download_button(
        "下载逐日净值 CSV",
        data=daily.to_csv(index=False).encode("utf-8-sig"),
        file_name="daily_equity.csv",
        mime="text/csv",
    )

st.header("4. 验证实验")
st.caption(
    "先复核原始策略，再分别改变交易成本、策略参数和真实市场时段。"
    "时事归因已预留结果接口，本版不参与结论。"
)
if not stored or stored["key"] != result_key:
    st.info("请先用当前配置运行基准回测，再运行验证实验。")
else:
    st.write(
        "验证实验会使用同一份行情快照；参数实验只改变快慢线，"
        "时段实验保持原策略参数不变。"
    )
    if benchmark_symbol is None:
        st.caption("验证实验将使用交易标的的买入持有作为比较基准。")
    validation_clicked = st.button("运行完整验证实验", type="secondary")
    if validation_clicked:
        try:
            with st.spinner("正在运行基准、成本、参数和市场时段验证……"):
                validation_report = run_validation_suite(
                    stored["config"],
                    benchmark_symbol=benchmark_symbol or symbol,
                )
            st.session_state["validation_result"] = {
                "key": result_key,
                "report": validation_report,
            }
            pack = build_evidence(validation_report)
            factual = generate_report(pack)
            st.session_state["evidence_result"] = {
                "key": result_key, "pack": pack, "analysis": factual,
                "bundle": export_bundle(pack, factual),
            }
        except Exception as error:
            st.exception(error)

    validation_stored = st.session_state.get("validation_result")
    if validation_stored and validation_stored["key"] == result_key:
        _render_validation_report(validation_stored["report"])

st.header("5. 证据报告")
evidence_stored = st.session_state.get("evidence_result")
if evidence_stored and evidence_stored["key"] == result_key:
    pack = evidence_stored["pack"]
    context = pack["entries"][f"{pack['run_id']}:context"]["data"]
    if context.get("snapshot", {}).get("is_synthetic"):
        st.warning("本报告使用模拟数据，仅用于流程演示。")
    with st.expander("实验范围与限制"):
        for limitation in context.get("limitations", []):
            st.write(limitation)
    if st.button("生成 AI 分析", disabled=not LLM_AVAILABLE):
        with st.spinner("正在结合实验数据与证据生成分析……"):
            analysis = generate_report(pack, use_llm=True)
            evidence_stored["analysis"] = analysis
            evidence_stored["bundle"] = export_bundle(pack, analysis)
    analysis = evidence_stored["analysis"]
    st.info(analysis["notice"])
    for index, finding in enumerate(analysis["findings"]):
        st.subheader(finding["title"])
        st.caption("实验事实" if finding["level"] == "fact" else "AI 分析")
        st.write(finding["statement"])
        for limitation in finding["limitations"]:
            st.caption(limitation)
        for eid in finding["evidence_refs"]:
            entry = pack["entries"][eid]
            with st.expander("证据附件：" + entry["title"], expanded=entry["kind"] == "chart"):
                st.caption(eid)
                if entry["kind"] == "chart":
                    st.plotly_chart(chart_for(pack, eid), width="stretch", key=f"evidence_{index}_{eid}")
                elif entry["kind"] == "table":
                    st.dataframe(pd.DataFrame(entry["data"]), width="stretch", hide_index=True)
                else:
                    st.json(entry["data"])
    st.download_button("下载报告与证据附件 ZIP", data=evidence_stored["bundle"],
                       file_name=f"{pack['run_id']}.zip", mime="application/zip")
else:
    st.caption("运行验证实验后，将生成图表、事实摘要和可下载的证据附件。")
