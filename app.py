"""量见 Agent 界面：在对话中形成、确认并验证策略，在下方展示报告。"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import APP_NAME, APP_TAGLINE, APP_VERSION, DISCLAIMER, LLM_AVAILABLE
from core.backtest.engine import run_backtest_from_snapshot
from core.data import ensure_market_snapshot, get_etf, list_snapshots, load_snapshot
from core.experiments import run_validation_suite
from core.nlp import parse_user_input
from core.report import build_evidence, chart_for, export_bundle, generate_report
from core.schemas import Metrics, ValidationReport
from core.strategy import StrategyDraft, align_draft_to_available_period, specified_fields_from_candidate

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
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=total["date"], y=total["equity"], name="策略净值", mode="lines",
    ))
    if total["benchmark_equity"].notna().any():
        figure.add_trace(go.Scatter(
            x=total["date"], y=total["benchmark_equity"], name="基准净值", mode="lines",
        ))
    figure.update_layout(
        title="策略与基准净值",
        xaxis_title="日期",
        yaxis_title="资产净值（元）",
        hovermode="x unified",
        legend_title_text="",
    )
    st.plotly_chart(figure, width="stretch")


def _render_validation_tables(report: ValidationReport) -> None:
    baseline = report.baseline
    st.markdown("#### 原始策略基准")
    baseline_cols = st.columns(4)
    baseline_cols[0].metric("策略收益", _percent(baseline.get("total_return")))
    baseline_cols[1].metric("基准收益", _percent(baseline.get("benchmark_return")))
    baseline_cols[2].metric("超额收益", _percent(baseline.get("excess_return")))
    baseline_cols[3].metric("成交笔数", str(baseline.get("n_trades", 0)))
    sections = [
        (
            "cost_sensitivity",
            "成本敏感性",
            ["slippage_bps", "is_baseline", "total_return", "excess_return", "total_cost", "n_trades", "error"],
        ),
        (
            "strategy_parameter_robustness",
            "策略参数稳健性",
            ["fast", "slow", "is_baseline", "total_return", "excess_return", "n_trades", "error"],
        ),
        (
            "market_period_robustness",
            "市场时段稳健性",
            ["period", "start", "end", "total_return", "excess_return", "n_trades", "error"],
        ),
    ]
    for key, title, columns in sections:
        result = report.experiments[key]
        with st.expander(title):
            st.caption(
                f"共 {result.total_points} 个实验点，其中 {result.passed_points} 个超额收益为正；"
                "该数量不等于策略有效性的判定。"
            )
            frame = pd.DataFrame(result.rows)
            visible_columns = [column for column in columns if column in frame.columns]
            st.dataframe(frame[visible_columns], width="stretch", hide_index=True)
    st.caption(f"验证编号：{report.validation_id} · 配置哈希：{report.config_hash}")


_DRAFT_WIDGET_KEYS = {
    "symbol": "draft_symbol",
    "benchmark_symbol": "draft_benchmark",
    "start": "draft_start",
    "end": "draft_end",
    "fast": "draft_fast",
    "slow": "draft_slow",
    "initial_cash": "draft_cash",
    "commission_bps": "draft_commission_bps",
    "min_commission": "draft_min_commission",
    "slippage_bps": "draft_slippage_bps",
}


def _clear_calculated_results() -> None:
    for key in ("confirmed_draft_hash", "backtest_result", "validation_result", "evidence_result"):
        st.session_state.pop(key, None)


def _clear_draft() -> None:
    st.session_state.pop("strategy_draft", None)
    for key in _DRAFT_WIDGET_KEYS.values():
        st.session_state.pop(key, None)
    _clear_calculated_results()


def _append_chat(role: str, content: str) -> None:
    st.session_state.setdefault("strategy_chat", []).append({"role": role, "content": content})


def _symbol_period(prices: pd.DataFrame, symbol: str) -> tuple[str, str]:
    symbol_prices = prices[prices["symbol"] == symbol]
    if symbol_prices.empty:
        raise ValueError(f"行情快照中没有标的 {symbol}")
    return (
        str(pd.Timestamp(symbol_prices["date"].min()).date()),
        str(pd.Timestamp(symbol_prices["date"].max()).date()),
    )


def _default_draft(snapshot_meta, prices: pd.DataFrame) -> StrategyDraft:
    symbol = snapshot_meta.symbols[0]
    start, end = _symbol_period(prices, symbol)
    return StrategyDraft(
        symbol=symbol,
        benchmark_symbol=symbol,
        start=start,
        end=end,
        origin="manual",
    )


def _sync_draft_widgets(draft: StrategyDraft) -> None:
    values = {
        "symbol": draft.symbol,
        "benchmark_symbol": draft.benchmark_symbol or "__NONE__",
        "start": pd.Timestamp(draft.start).date(),
        "end": pd.Timestamp(draft.end).date(),
        "fast": draft.fast,
        "slow": draft.slow,
        "initial_cash": draft.initial_cash,
        "commission_bps": draft.commission_rate * 10_000,
        "min_commission": draft.min_commission,
        "slippage_bps": draft.slippage_bps,
    }
    for field, value in values.items():
        st.session_state[_DRAFT_WIDGET_KEYS[field]] = value


def _install_draft(draft: StrategyDraft, snapshot_id: str) -> None:
    st.session_state["strategy_draft"] = draft
    st.session_state["draft_snapshot_id"] = snapshot_id
    _sync_draft_widgets(draft)
    _clear_calculated_results()


def _period_start(actual_start: str, actual_end: str, years: int | None) -> str:
    if years is None:
        return actual_start
    start = pd.Timestamp(actual_end) - pd.DateOffset(years=years)
    return str(max(start, pd.Timestamp(actual_start)).date())


def _origin_label(origin: str) -> str:
    return {
        "conversation": "自然语言对话",
        "quick": "快捷创建",
        "preset": "典型策略模板",
        "manual": "手动调整",
    }.get(origin, origin)


# 侧栏只放环境与快捷入口，主页面保持为一个连续的 Agent 对话。
st.sidebar.title(APP_NAME)
st.sidebar.caption(f"{APP_TAGLINE} · {APP_VERSION}")
st.sidebar.header("验证环境")
snapshots = list_snapshots()
if not snapshots:
    st.error("本地没有可用快照。请先通过数据模块创建真实或模拟快照。")
    st.stop()

snapshot_by_id = {meta.snapshot_id: meta for meta in snapshots}
pending_snapshot_id = st.session_state.pop("_pending_snapshot_id", None)
if pending_snapshot_id in snapshot_by_id:
    st.session_state["selected_snapshot_id"] = pending_snapshot_id
elif st.session_state.get("selected_snapshot_id") not in snapshot_by_id:
    st.session_state.pop("selected_snapshot_id", None)
snapshot_id = st.sidebar.selectbox(
    "当前实验数据",
    options=list(snapshot_by_id),
    key="selected_snapshot_id",
    format_func=lambda sid: (
        f"{snapshot_by_id[sid].actual_start} 至 {snapshot_by_id[sid].actual_end}"
        f" · {'模拟' if snapshot_by_id[sid].is_synthetic else '真实'}"
    ),
)
prices, snapshot_meta, quality_report = load_snapshot(snapshot_id)
st.sidebar.caption(f"数据来源：{snapshot_meta.source} · {len(snapshot_meta.symbols)} 个标的")
st.sidebar.caption("对话需要其他区间时会先查本地缓存，缺失后自动联网获取。")

with st.sidebar.expander("数据详情"):
    st.json(asdict(snapshot_meta))
    if quality_report.issues:
        st.dataframe(
            pd.DataFrame([asdict(issue) for issue in quality_report.issues]),
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("未记录数据质量问题。")

if st.session_state.get("draft_snapshot_id") != snapshot_id:
    _clear_draft()
    st.session_state["draft_snapshot_id"] = snapshot_id
    st.session_state["strategy_chat"] = []
    for key in list(st.session_state):
        if key.startswith("quick_"):
            st.session_state.pop(key, None)

if st.sidebar.button("新建对话", use_container_width=True):
    _clear_draft()
    st.session_state["strategy_chat"] = []
    st.rerun()

st.sidebar.header("快捷创建")
st.sidebar.caption("快捷入口也会先生成草案，由你在对话中确认。")
st.sidebar.markdown("**典型双均线**")
for label, preset_fast, preset_slow in (
    ("5日 / 20日", 5, 20),
    ("10日 / 30日", 10, 30),
    ("20日 / 60日", 20, 60),
):
    if st.sidebar.button(label, key=f"preset_{preset_fast}_{preset_slow}", use_container_width=True):
        preset = _default_draft(snapshot_meta, prices)
        preset.fast, preset.slow, preset.origin = preset_fast, preset_slow, "preset"
        preset.specified_fields = ["典型策略模板"]
        _install_draft(preset, snapshot_id)
        _append_chat("assistant", f"我已载入 {label} 的典型双均线草案，请在对话中检查并确认。")
        st.rerun()

with st.sidebar.form("quick_strategy_form"):
    st.markdown("**按选项创建**")
    quick_symbol = st.selectbox(
        "交易标的", snapshot_meta.symbols, format_func=_symbol_label, key="quick_symbol",
        help="策略实际买卖的 ETF。",
    )
    quick_fast = st.selectbox(
        "快线周期", [5, 10, 20], format_func=lambda value: f"{value} 日",
        key="quick_fast", help="较短时间的平均价格，对近期变化更敏感。",
    )
    quick_slow = st.selectbox(
        "慢线周期", [20, 30, 60, 120], format_func=lambda value: f"{value} 日",
        key="quick_slow", help="较长时间的平均价格，用来表示长期趋势。",
    )
    quick_period = st.selectbox(
        "测试区间", ["全部数据", "近1年", "近3年", "近5年"],
        key="quick_period", help="选择近 N 年时会先查本地缓存，缺失后自动获取真实行情。",
    )
    quick_submitted = st.form_submit_button("生成策略草案", use_container_width=True)

if quick_submitted:
    try:
        years = {"全部数据": None, "近1年": 1, "近3年": 3, "近5年": 5}[quick_period]
        requested_start, requested_end = _symbol_period(prices, quick_symbol)
        data_reply = f"使用当前实验数据 `{snapshot_id}`。"
        if years is not None:
            requested_end = date.today().isoformat()
            requested_start = str(
                (pd.Timestamp(requested_end) - pd.DateOffset(years=years)).date()
            )
            resolved_meta, resolved_quality, cache_hit = ensure_market_snapshot(
                [quick_symbol], requested_start, requested_end,
            )
            if resolved_quality.has_error:
                raise RuntimeError("获取到的行情未通过数据质量检查")
            resolved_prices, resolved_meta, resolved_quality = load_snapshot(resolved_meta.snapshot_id)
            data_reply = (
                f"已复用本地真实行情缓存 `{resolved_meta.snapshot_id}`。"
                if cache_hit else
                f"已通过 {resolved_meta.source} 获取真实行情并缓存。"
            )
            if resolved_meta.snapshot_id != snapshot_id:
                st.session_state["_pending_snapshot_id"] = resolved_meta.snapshot_id
                snapshot_id = resolved_meta.snapshot_id
                prices = resolved_prices
                snapshot_meta = resolved_meta
                quality_report = resolved_quality
        available_start, available_end = _symbol_period(prices, quick_symbol)
        quick_draft = StrategyDraft(
            symbol=quick_symbol,
            benchmark_symbol=quick_symbol,
            start=requested_start,
            end=requested_end,
            fast=quick_fast,
            slow=quick_slow,
            origin="quick",
            specified_fields=["交易标的", "快线周期", "慢线周期", "测试区间"],
        )
        quick_draft = align_draft_to_available_period(
            quick_draft, available_start, available_end,
        )
        quick_issues = quick_draft.validate(
            available_symbols=snapshot_meta.symbols,
            available_start=available_start,
            available_end=available_end,
        )
        if quick_issues:
            st.sidebar.error("；".join(quick_issues))
        else:
            _install_draft(quick_draft, snapshot_id)
            _append_chat(
                "assistant", data_reply + "\n\n我已根据侧栏选项生成策略草案，请检查并确认。",
            )
            st.rerun()
    except (RuntimeError, ValueError) as error:
        st.sidebar.error(f"行情准备没有完成：{error}")

with st.sidebar.expander("参数是什么意思？"):
    st.write("**快线**：较短时间的平均价格，对近期变化更敏感。")
    st.write("**慢线**：较长时间的平均价格，用来表示长期趋势。")
    st.write("**上穿**：快线从慢线下方移动到上方。")
    st.write("**比较基准**：用来比较策略是否优于直接买入并持有。")


# chat_input 固定显示在页面底部，但先处理输入，确保新草案在本轮渲染前同步到编辑控件。
idea_text = st.chat_input(
    "告诉我你的策略想法，也可以继续补充或修改……",
    disabled=quality_report.has_error,
)
if idea_text and idea_text.strip():
    _append_chat("user", idea_text.strip())
    messages = st.session_state["strategy_chat"]
    combined_text = "\n".join(message["content"] for message in messages if message["role"] == "user")
    with st.spinner("正在理解你的想法……"):
        parsed = parse_user_input(combined_text, reference_end=date.today().isoformat())
    if parsed.needs_clarification:
        reply = "还需要你补充：\n\n" + "\n".join(f"- {question}" for question in parsed.questions)
        if parsed.degraded:
            reply += "\n\n当前使用基础规则识别，你也可以使用侧栏快捷创建。"
        _append_chat("assistant", reply)
    else:
        try:
            parsed_draft = StrategyDraft.from_config(
                parsed.strategy_config,
                benchmark_symbol=parsed.strategy_config.symbols[0],
                origin="conversation",
                specified_fields=specified_fields_from_candidate(parsed.raw_candidate),
                confirmation_text=parsed.confirmation_text,
                degraded=parsed.degraded,
                degraded_reason=parsed.degraded_reason,
            )
            relative_years = parsed.raw_candidate.get("_relative_period_years")
            if relative_years:
                parsed_draft.normalization_notes.append(
                    f"“近 {relative_years} 年”按当前日期 "
                    f"{parsed.raw_candidate['_relative_period_reference_end']} 计算。"
                )
            with st.spinner("正在检查本地行情，缺失时将从外部接口获取……"):
                resolved_meta, resolved_quality, cache_hit = ensure_market_snapshot(
                    [parsed_draft.symbol], parsed_draft.start, parsed_draft.end,
                )
            if resolved_quality.has_error:
                raise RuntimeError("获取到的行情未通过数据质量检查")
            resolved_prices, resolved_meta, resolved_quality = load_snapshot(
                resolved_meta.snapshot_id,
            )
            data_reply = (
                f"已复用本地真实行情缓存 `{resolved_meta.snapshot_id}`。"
                if cache_hit else
                f"本地没有覆盖该区间的真实行情，已通过 {resolved_meta.source} 获取并缓存。"
            )
            if resolved_meta.snapshot_id != snapshot_id:
                st.session_state["_pending_snapshot_id"] = resolved_meta.snapshot_id
                snapshot_id = resolved_meta.snapshot_id
                prices = resolved_prices
                snapshot_meta = resolved_meta
                quality_report = resolved_quality
            available_start, available_end = _symbol_period(prices, parsed_draft.symbol)
            parsed_draft = align_draft_to_available_period(
                parsed_draft, available_start, available_end,
            )
            parsed_issues = parsed_draft.validate(
                available_symbols=snapshot_meta.symbols,
                available_start=available_start,
                available_end=available_end,
            )
            if parsed_issues:
                _append_chat(
                    "assistant",
                    data_reply + "\n\n获取到的行情仍不能执行这份策略：\n\n"
                    + "\n".join(f"- {issue}" for issue in parsed_issues)
                    + "\n\n请继续告诉我如何修改区间或标的。",
                )
            else:
                _install_draft(parsed_draft, snapshot_id)
                reply = data_reply + "\n\n我已经把你的想法整理成可执行草案，请检查下面的策略确认卡。"
                if parsed.degraded:
                    reply += " 当前使用基础规则识别，请特别核对。"
                if parsed_draft.normalization_notes:
                    reply += "\n\n我对日期做了可执行调整：\n" + "\n".join(
                        f"- {note}" for note in parsed_draft.normalization_notes
                    )
                _append_chat("assistant", reply)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            _append_chat("assistant", f"行情准备或策略解析没有完成：{error}")
    st.rerun()


st.title("量化策略验证 Agent")
st.caption("用对话把投资想法变成可验证策略；图表、结论和证据会出现在对话下方。")
if snapshot_meta.is_synthetic:
    st.warning("当前使用模拟数据，只能验证系统流程，不能用于判断策略有效性。")
if quality_report.has_error:
    st.error("当前行情存在错误级质量问题，请在侧栏更换数据。")
if not LLM_AVAILABLE:
    st.info("模型尚未配置，Agent 将使用基础规则理解想法，并生成程序事实报告。")

with st.chat_message("assistant"):
    st.write(
        "你好，我是量见。直接说出你的投资想法，我会追问缺失条件、整理策略草案，"
        "经你确认后完成回测、验证实验和报告。"
    )
    st.caption("例如：用沪深300ETF测试双均线，5日线上穿20日线买入，回测近两年。")

for message in st.session_state.setdefault("strategy_chat", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


draft: StrategyDraft | None = st.session_state.get("strategy_draft")
result_key = None
if draft is not None:
    with st.chat_message("assistant"):
        st.markdown("### 策略确认卡")
        st.caption(f"草案来源：{_origin_label(draft.origin)}。任何修改都会使旧结果失效。")

        with st.expander("查看或调整策略", expanded=False):
            symbol_col, benchmark_col = st.columns(2)
            symbol_value = symbol_col.selectbox(
                "交易标的", snapshot_meta.symbols, key=_DRAFT_WIDGET_KEYS["symbol"],
                format_func=_symbol_label, help="策略实际买卖的 ETF。",
            )
            benchmark_options = ["__NONE__", *snapshot_meta.symbols]
            benchmark_value = benchmark_col.selectbox(
                "比较基准", benchmark_options, key=_DRAFT_WIDGET_KEYS["benchmark_symbol"],
                format_func=lambda value: "不比较" if value == "__NONE__" else f"买入持有：{_symbol_label(value)}",
            )
            date_col1, date_col2 = st.columns(2)
            start_value = date_col1.date_input(
                "开始日期", min_value=pd.Timestamp(snapshot_meta.actual_start).date(),
                max_value=pd.Timestamp(snapshot_meta.actual_end).date(), key=_DRAFT_WIDGET_KEYS["start"],
            )
            end_value = date_col2.date_input(
                "结束日期", min_value=pd.Timestamp(snapshot_meta.actual_start).date(),
                max_value=pd.Timestamp(snapshot_meta.actual_end).date(), key=_DRAFT_WIDGET_KEYS["end"],
            )
            param_col1, param_col2 = st.columns(2)
            fast_value = int(param_col1.number_input(
                "快线周期（日）", min_value=1, step=1, key=_DRAFT_WIDGET_KEYS["fast"],
                help="较短时间的平均价格，对近期变化更敏感。",
            ))
            slow_value = int(param_col2.number_input(
                "慢线周期（日）", min_value=2, step=1, key=_DRAFT_WIDGET_KEYS["slow"],
                help="较长时间的平均价格，用来表示长期趋势。",
            ))
            st.markdown("**资金与交易成本**")
            cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)
            cash_value = float(cost_col1.number_input(
                "初始资金（元）", min_value=1_000.0, step=10_000.0,
                key=_DRAFT_WIDGET_KEYS["initial_cash"],
            ))
            commission_bps_value = float(cost_col2.number_input(
                "单边佣金（bps）", min_value=0.0, step=0.1,
                key=_DRAFT_WIDGET_KEYS["commission_bps"], help="1 bps = 0.01%",
            ))
            min_commission_value = float(cost_col3.number_input(
                "最低佣金（元）", min_value=0.0, step=1.0,
                key=_DRAFT_WIDGET_KEYS["min_commission"],
            ))
            slippage_bps_value = float(cost_col4.number_input(
                "单边滑点（bps）", min_value=0.0, step=0.5,
                key=_DRAFT_WIDGET_KEYS["slippage_bps"],
            ))

        edited_draft = replace(
            draft,
            symbol=symbol_value,
            benchmark_symbol=None if benchmark_value == "__NONE__" else benchmark_value,
            start=start_value.isoformat(),
            end=end_value.isoformat(),
            fast=fast_value,
            slow=slow_value,
            initial_cash=cash_value,
            commission_rate=commission_bps_value / 10_000,
            min_commission=min_commission_value,
            slippage_bps=slippage_bps_value,
        )
        if (edited_draft.start, edited_draft.end) != (draft.start, draft.end):
            edited_draft = replace(edited_draft, normalization_notes=[])
        if asdict(edited_draft) != asdict(draft):
            st.session_state["strategy_draft"] = edited_draft
            _clear_calculated_results()
        draft = edited_draft

        available_start, available_end = _symbol_period(prices, draft.symbol)
        draft_issues = draft.validate(
            available_symbols=snapshot_meta.symbols,
            available_start=available_start,
            available_end=available_end,
        )
        st.write(
            f"当 **{draft.fast} 日均线向上穿过 {draft.slow} 日均线** 时买入 "
            f"**{_symbol_label(draft.symbol)}**；向下穿过时卖出。"
        )
        summary_cols = st.columns(3)
        summary_cols[0].markdown(f"**区间**  \n{draft.start} 至 {draft.end}")
        summary_cols[1].markdown(
            "**基准**  \n" + (_symbol_label(draft.benchmark_symbol) if draft.benchmark_symbol else "不比较")
        )
        summary_cols[2].markdown(f"**资金**  \n{_money(draft.initial_cash)}")
        for note in draft.normalization_notes:
            st.info(note)
        if draft.specified_fields:
            st.caption("来自你的输入或选择：" + "、".join(draft.specified_fields))
        st.caption("执行口径：收盘形成信号，下一交易日开盘成交；仅多头、满仓；允许小数份额。")
        if draft.degraded:
            st.warning(draft.degraded_reason or "该草案由基础规则识别，请仔细核对。")
        for issue in draft_issues:
            st.error(issue)

        draft_hash = None if draft_issues else draft.execution_hash(snapshot_id)
        if st.button("确认这份策略", disabled=bool(draft_issues), key=f"confirm_{draft_hash}"):
            st.session_state["confirmed_draft_hash"] = draft_hash
            _append_chat("assistant", "策略已确认。我会按这份配置执行，不会在实验中改写你的原始策略。")
        confirmed = bool(draft_hash and st.session_state.get("confirmed_draft_hash") == draft_hash)
        if confirmed:
            st.success("策略已确认，可以开始完整验证。")
        else:
            st.info("请先确认这份策略；调整任何字段后需要重新确认。")

        run_clicked = st.button(
            "开始验证并生成报告", type="primary",
            disabled=not confirmed or quality_report.has_error, key=f"run_agent_{draft_hash}",
        )
        result_key = draft_hash
        if run_clicked:
            config = draft.to_config(snapshot_id)
            benchmark_symbol = draft.benchmark_symbol or draft.symbol
            try:
                with st.status("Agent 正在验证策略……", expanded=True) as status:
                    st.write("正在读取固定行情快照并运行原始策略回测……")
                    trades, daily, metrics = run_backtest_from_snapshot(
                        config, benchmark_symbol=draft.benchmark_symbol,
                    )
                    st.session_state["backtest_result"] = {
                        "key": result_key, "config": config, "trades": trades,
                        "daily": daily, "metrics": metrics,
                    }
                    st.write("正在测试交易成本、邻近参数和不同时段……")
                    validation_report = run_validation_suite(config, benchmark_symbol=benchmark_symbol)
                    st.session_state["validation_result"] = {
                        "key": result_key, "report": validation_report,
                    }
                    st.write("正在整理热力图、对比图和可追溯证据……")
                    pack = build_evidence(validation_report)
                    if LLM_AVAILABLE:
                        st.write("正在让 AI 阅读结构化实验数据并撰写结论……")
                    else:
                        st.write("模型未配置，正在生成程序事实摘要……")
                    analysis = generate_report(pack, use_llm=LLM_AVAILABLE)
                    bundle = export_bundle(pack, analysis)
                    st.session_state["evidence_result"] = {
                        "key": result_key, "pack": pack, "analysis": analysis, "bundle": bundle,
                    }
                    status.update(label="验证和报告已经完成", state="complete", expanded=False)
                mode = "AI 证据报告" if analysis["mode"] == "ai" else "事实报告"
                completion = f"验证完成，已生成{mode}。图表、结论、证据和下载文件都在对话下方。"
                _append_chat("assistant", completion)
                st.success(completion)
            except Exception as error:
                _append_chat("assistant", f"验证没有完成：{error}")
                st.exception(error)


stored = st.session_state.get("backtest_result")
validation_stored = st.session_state.get("validation_result")
evidence_stored = st.session_state.get("evidence_result")
results_ready = bool(
    result_key
    and stored and stored.get("key") == result_key
    and validation_stored and validation_stored.get("key") == result_key
    and evidence_stored and evidence_stored.get("key") == result_key
)

st.divider()
st.header("报告与证据")
if not results_ready:
    st.caption("在上方对话中确认策略并开始验证后，这里会展示热力图、对比图、结论和完整报告。")
else:
    metrics = stored["metrics"]
    trades = stored["trades"]
    daily = stored["daily"]
    validation_report = validation_stored["report"]
    pack = evidence_stored["pack"]
    analysis = evidence_stored["analysis"]
    context = pack["entries"][f"{pack['run_id']}:context"]["data"]
    if context.get("snapshot", {}).get("is_synthetic"):
        st.warning("本报告使用模拟数据，仅用于流程演示。")

    overview_tab, charts_tab, findings_tab, data_tab = st.tabs([
        "表现概览", "热力图与对比图", "结论与证据", "完整数据与下载",
    ])
    with overview_tab:
        _render_metrics(metrics)
        _render_curve(daily)
        if metrics.max_dd_start and metrics.max_dd_end:
            st.caption(f"最大回撤区间：{metrics.max_dd_start} 至 {metrics.max_dd_end}")
        for note in metrics.notes:
            st.info(note)

    with charts_tab:
        chart_order = [
            "parameters", "cost", "periods", "equity", "drawdown",
            "trade_distribution", "trade_contribution",
        ]
        for slug in chart_order:
            evidence_id = f"{pack['run_id']}:chart:{slug}"
            if evidence_id not in pack["entries"]:
                continue
            entry = pack["entries"][evidence_id]
            st.markdown("#### " + entry["title"])
            st.plotly_chart(
                chart_for(pack, evidence_id), width="stretch", key=f"report_chart_{slug}",
            )

    with findings_tab:
        st.info(analysis["notice"])
        with st.expander("实验范围与限制"):
            for limitation in context.get("limitations", []):
                st.write("- " + limitation)
        for finding in analysis["findings"]:
            with st.container(border=True):
                st.markdown("### " + finding["title"])
                st.caption("实验事实" if finding["level"] == "fact" else "AI 分析")
                st.write(finding["statement"])
                for limitation in finding["limitations"]:
                    st.caption("局限：" + limitation)
                with st.expander("查看这条结论的证据附件"):
                    for evidence_id in finding["evidence_refs"]:
                        entry = pack["entries"][evidence_id]
                        st.markdown(f"**{entry['title']}**  \n`{evidence_id}`")
                        if entry["kind"] == "chart":
                            st.caption("对应图表已在“热力图与对比图”中展示。")
                        elif entry["kind"] == "table":
                            st.dataframe(pd.DataFrame(entry["data"]), width="stretch", hide_index=True)
                        else:
                            st.json(entry["data"])

    with data_tab:
        _render_validation_tables(validation_report)
        with st.expander("交易记录"):
            if trades.empty:
                st.info("当前区间没有触发交易。")
            else:
                display_trades = trades.copy()
                display_trades["signal_date"] = pd.to_datetime(display_trades["signal_date"]).dt.date
                display_trades["exec_date"] = pd.to_datetime(display_trades["exec_date"]).dt.date
                st.dataframe(display_trades, width="stretch", hide_index=True)
        download_col1, download_col2, download_col3 = st.columns(3)
        download_col1.download_button(
            "下载交易记录 CSV", data=trades.to_csv(index=False).encode("utf-8-sig"),
            file_name="trades.csv", mime="text/csv",
        )
        download_col2.download_button(
            "下载逐日净值 CSV", data=daily.to_csv(index=False).encode("utf-8-sig"),
            file_name="daily_equity.csv", mime="text/csv",
        )
        download_col3.download_button(
            "下载完整报告 ZIP", data=evidence_stored["bundle"],
            file_name=f"{pack['run_id']}.zip", mime="application/zip",
        )

st.caption(DISCLAIMER)
