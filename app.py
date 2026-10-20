"""量见 Agent 界面：在对话中形成、确认并验证策略，在下方展示报告。"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from config import APP_NAME, APP_TAGLINE, APP_VERSION, DISCLAIMER
from core.agent import route_agent_message
from core.backtest.engine import run_backtest_from_snapshot
from core.data import ensure_market_snapshot, get_etf, list_snapshots, load_snapshot
from core.experiments import run_validation_suite
from core.finance import answer_finance_question
from core.nlp import parse_user_input
from core.nlp.llm_client import LLMError, complete_json
from core.report import (
    answer_report_question, build_evidence, chart_for, export_bundle,
    format_report_answer, generate_report,
)
from core.schemas import Metrics, ValidationReport
from core.strategy import StrategyDraft, align_draft_to_available_period, specified_fields_from_candidate

st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f7f8fa; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e9edf2; }
    [data-testid="stSidebar"] .stButton > button {
      min-height: 2.85rem; border-radius: 14px; justify-content: flex-start;
      padding: .65rem .85rem; border: 1px solid #e4e8ee; font-weight: 650;
      box-shadow: none; transition: border-color .15s ease, background .15s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
      border-color: #ff6b66; color: #c93f3b; background: #fff7f6;
    }
    [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
      background: #fff1f0; color: #b93431; border-color: #ff6b66;
    }
    .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
    .qv-eyebrow { color: #e6534e; font-size: .82rem; font-weight: 700; letter-spacing: .08em; }
    .qv-hero { padding: 2.2rem 2.4rem; border-radius: 22px; color: white;
      background: linear-gradient(135deg, #17243a 0%, #263f63 68%, #375b82 100%);
      box-shadow: 0 16px 42px rgba(24, 43, 70, .16); margin-bottom: 1.3rem; }
    .qv-hero h1 { font-size: 2.5rem; line-height: 1.15; margin: .35rem 0 .8rem; }
    .qv-hero p { max-width: 740px; color: #dce6f1; font-size: 1.06rem; margin: 0; }
    .qv-step { color: #e6534e; font-weight: 750; font-size: .8rem; letter-spacing: .05em; }
    .qv-card-title { font-size: 1.05rem; font-weight: 720; margin-bottom: .35rem; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #e7ebf0;
      padding: .85rem 1rem; border-radius: 14px; }
    div[data-testid="stChatMessage"] { background: white; border: 1px solid #e8ebef;
      border-radius: 16px; padding: .25rem .55rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background: white; }
    .qv-muted { color: #687385; }
    </style>
    """,
    unsafe_allow_html=True,
)

_PAGES = ("首页", "使用文档", "量化策略验证", "AI 金融问答")
_ANSWER_MODES = ("新手讲解", "简洁摘要", "技术细节")


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
            [
                "fast", "slow", "entry_window", "exit_window", "rsi_period",
                "buy_threshold", "sell_threshold", "is_baseline",
                "total_return", "excess_return", "n_trades", "error",
            ],
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
    "entry_window": "draft_entry_window",
    "exit_window": "draft_exit_window",
    "rsi_period": "draft_rsi_period",
    "buy_threshold": "draft_buy_threshold",
    "sell_threshold": "draft_sell_threshold",
    "initial_cash": "draft_cash",
    "commission_bps": "draft_commission_bps",
    "min_commission": "draft_min_commission",
    "slippage_bps": "draft_slippage_bps",
}


def _clear_calculated_results() -> None:
    for key in ("confirmed_draft_hash", "backtest_result", "validation_result", "evidence_result"):
        st.session_state.pop(key, None)
    st.session_state.pop("report_qa_history", None)


def _clear_draft() -> None:
    st.session_state.pop("strategy_draft", None)
    for key in _DRAFT_WIDGET_KEYS.values():
        st.session_state.pop(key, None)
    _clear_calculated_results()


def _append_chat(role: str, content: str, *, evidence_refs: list[str] | None = None) -> None:
    message = {"role": role, "content": content}
    if evidence_refs:
        message["evidence_refs"] = list(dict.fromkeys(evidence_refs))
    st.session_state.setdefault("strategy_chat", []).append(message)


def _render_chat_evidence(message: dict, message_index: int) -> None:
    refs = message.get("evidence_refs") or []
    stored = st.session_state.get("evidence_result")
    if not refs or not stored:
        return
    pack = stored.get("pack") or {}
    entries = pack.get("entries") or {}
    valid_refs = [evidence_id for evidence_id in refs if evidence_id in entries]
    if not valid_refs:
        return
    with st.expander(f"查看证据附件（{len(valid_refs)}）"):
        for evidence_index, evidence_id in enumerate(valid_refs):
            entry = entries[evidence_id]
            st.markdown(f"**{entry['title']}**  \n`{evidence_id}`")
            if entry["kind"] == "chart":
                st.plotly_chart(
                    chart_for(pack, evidence_id), width="stretch",
                    key=f"chat_evidence_{message_index}_{evidence_index}",
                )
            elif entry["kind"] == "table":
                st.dataframe(pd.DataFrame(entry["data"]), width="stretch", hide_index=True)
            else:
                st.json(entry["data"])


def _symbol_period(prices: pd.DataFrame, symbol: str) -> tuple[str, str]:
    symbol_prices = prices[prices["symbol"] == symbol]
    if symbol_prices.empty:
        raise ValueError(f"行情快照中没有标的 {symbol}")
    return (
        str(pd.Timestamp(symbol_prices["date"].min()).date()),
        str(pd.Timestamp(symbol_prices["date"].max()).date()),
    )


def _snapshot_covers_request(
    prices: pd.DataFrame,
    symbol: str,
    requested_start: str,
    requested_end: str,
    *,
    tolerance_days: int = 10,
) -> bool:
    """判断当前已选快照能否覆盖请求；周末和节假日允许少量边界偏差。"""
    symbol_prices = prices[prices["symbol"] == symbol]
    if symbol_prices.empty:
        return False
    tolerance = pd.Timedelta(days=tolerance_days)
    return bool(
        symbol_prices["date"].min() <= pd.Timestamp(requested_start) + tolerance
        and symbol_prices["date"].max() >= pd.Timestamp(requested_end) - tolerance
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
        "entry_window": draft.entry_window,
        "exit_window": draft.exit_window,
        "rsi_period": draft.rsi_period,
        "buy_threshold": draft.buy_threshold,
        "sell_threshold": draft.sell_threshold,
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


def _current_report_ready(snapshot_id: str) -> bool:
    draft = st.session_state.get("strategy_draft")
    evidence = st.session_state.get("evidence_result")
    if draft is None or not evidence:
        return False
    try:
        return evidence.get("key") == draft.execution_hash(snapshot_id)
    except ValueError:
        return False


def _navigate(page: str) -> None:
    st.session_state["main_nav"] = page


def _set_session_choice(key: str, value: str) -> None:
    st.session_state[key] = value


def _sidebar_cards(
    title: str,
    options: tuple[str, ...],
    *,
    state_key: str,
    icons: tuple[str, ...],
) -> str:
    """用整块圆角按钮呈现侧栏选择，避免单选圆点的表单感。"""
    current = st.session_state.get(state_key)
    if current not in options:
        current = options[0]
        st.session_state[state_key] = current
    st.sidebar.markdown(f"**{title}**")
    for option, icon in zip(options, icons):
        st.sidebar.button(
            f"{icon}　{option}",
            key=f"{state_key}_{option}",
            type="primary" if option == current else "secondary",
            use_container_width=True,
            on_click=_set_session_choice,
            args=(state_key, option),
        )
    return current


def _effective_llm_settings() -> dict:
    saved = st.session_state.get("runtime_llm_config") or {}
    environment = {
        "api_key": config.LLM_API_KEY,
        "base_url": config.LLM_BASE_URL,
        "model": config.LLM_MODEL,
        "report_model": config.LLM_REPORT_MODEL or config.LLM_MODEL,
    } if config.LLM_AVAILABLE else {
        "api_key": "", "base_url": "", "model": "", "report_model": "",
    }
    return {
        "api_key": saved.get("api_key") or environment["api_key"],
        "base_url": saved.get("base_url") or environment["base_url"],
        "model": saved.get("model") or environment["model"],
        "report_model": (
            saved.get("report_model") or saved.get("model")
            or environment["report_model"] or environment["model"]
        ),
        "source": "session" if saved else ("environment" if config.LLM_AVAILABLE else "none"),
    }


def _llm_complete(settings: dict, *, report: bool = False):
    model = settings["report_model"] if report else settings["model"]

    def complete(system_prompt: str, user_prompt: str) -> dict:
        return complete_json(
            system_prompt,
            user_prompt,
            model=model,
            api_key=settings["api_key"],
            base_url=settings["base_url"],
        )

    return complete


def _clear_runtime_llm_config() -> None:
    st.session_state.pop("runtime_llm_config", None)
    for key in (
        "llm_api_key_input", "llm_base_url_input",
        "llm_model_input", "llm_report_model_input",
    ):
        st.session_state.pop(key, None)
    st.session_state["llm_connection_notice"] = (
        "info", "已恢复服务器默认配置。" if config.LLM_AVAILABLE else "已清除当前会话配置。"
    )


def _render_llm_connection() -> dict:
    """渲染会话级模型配置；API Key 不写文件、不进入报告和下载包。"""
    current = _effective_llm_settings()
    with st.sidebar.expander("模型连接", expanded=current["source"] == "none"):
        source_label = {
            "session": "当前会话配置",
            "environment": "服务器环境配置",
            "none": "尚未配置",
        }[current["source"]]
        st.caption(f"状态：{source_label}")
        with st.form("llm_connection_form"):
            key_input = st.text_input(
                "API Key",
                value=(st.session_state.get("runtime_llm_config") or {}).get("api_key", ""),
                key="llm_api_key_input",
                type="password",
                placeholder="sk-…",
                help="只保存在当前 Streamlit 会话中，不写入 .env 或仓库。",
            )
            url_input = st.text_input(
                "OpenAI 兼容接口 URL",
                value=current["base_url"],
                key="llm_base_url_input",
                placeholder="https://…/compatible-mode/v1",
            )
            model_input = st.text_input(
                "解析与问答模型",
                value=current["model"],
                key="llm_model_input",
                placeholder="例如 qwen-plus",
            )
            report_model_input = st.text_input(
                "报告模型（可选）",
                value=current["report_model"],
                key="llm_report_model_input",
                placeholder="留空则使用上面的模型",
            )
            save_col, test_col = st.columns(2)
            save_clicked = save_col.form_submit_button("保存", use_container_width=True)
            test_clicked = test_col.form_submit_button("测试并保存", use_container_width=True)

        if save_clicked or test_clicked:
            candidate = {
                "api_key": key_input.strip() or config.LLM_API_KEY,
                "base_url": url_input.strip(),
                "model": model_input.strip(),
                "report_model": report_model_input.strip() or model_input.strip(),
            }
            if not all((candidate["api_key"], candidate["base_url"], candidate["model"])):
                st.session_state["llm_connection_notice"] = (
                    "error", "请完整填写 API Key、接口 URL 和模型名。"
                )
            elif test_clicked:
                try:
                    response = complete_json(
                        "只输出 JSON 对象，不要输出其他文字。",
                        '请返回 {"ok": true}。',
                        model=candidate["model"],
                        api_key=candidate["api_key"],
                        base_url=candidate["base_url"],
                    )
                    if response.get("ok") is not True:
                        raise LLMError("模型返回了响应，但没有按测试格式返回")
                    st.session_state["runtime_llm_config"] = candidate
                    st.session_state["llm_connection_notice"] = ("success", "连接成功，配置已用于当前会话。")
                except LLMError as error:
                    st.session_state["llm_connection_notice"] = ("error", f"连接失败：{error}")
            else:
                st.session_state["runtime_llm_config"] = candidate
                st.session_state["llm_connection_notice"] = ("success", "配置已保存到当前会话。")
            st.rerun()

        notice = st.session_state.get("llm_connection_notice")
        if notice:
            getattr(st, notice[0])(notice[1])
        if current["source"] == "session":
            st.button(
                "恢复服务器配置", key="clear_runtime_llm", use_container_width=True,
                on_click=_clear_runtime_llm_config,
            )
        st.caption("Key 不会写入项目文件；刷新或会话过期后可能需要重新填写。")
    return _effective_llm_settings()


def _seed_verification(template: str) -> None:
    st.session_state["verification_seed_template"] = template
    st.session_state["main_nav"] = "量化策略验证"


def _render_home() -> None:
    st.markdown(
        """
        <div class="qv-hero">
          <div class="qv-eyebrow">QUANT IDEAS, EXPLAINED</div>
          <h1>把一句投资想法，变成一次看得懂的历史验证</h1>
          <p>量见会和你澄清规则、准备行情、执行回测与稳健性实验，最后用 AI 生成每条结论都有证据的报告。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action_col, guide_col, spacer = st.columns([1.15, 1.15, 2.7])
    action_col.button(
        "开始验证我的想法", type="primary", use_container_width=True,
        on_click=_navigate, args=("量化策略验证",),
    )
    guide_col.button(
        "先看使用方法", use_container_width=True,
        on_click=_navigate, args=("使用文档",),
    )

    st.markdown("### 你会得到什么")
    metric_cols = st.columns(4)
    for column, title, value, detail in zip(
        metric_cols,
        ("可执行策略", "验证实验", "行情处理", "结论证据"),
        ("3 种", "3 类", "自动", "逐条绑定"),
        (
            "趋势与均值回归", "参数、成本与时段", "缓存优先，缺失时获取",
            "图表、表格与原始指标",
        ),
    ):
        with column.container(border=True):
            st.caption(title)
            st.markdown(f"### {value}")
            st.caption(detail)

    st.markdown("### 一次验证怎样完成")
    for index, (title, body) in enumerate((
        ("说出想法", "不用写代码。描述想买什么、什么情况下买卖，条件不完整时 Agent 会继续追问。"),
        ("确认规则", "系统把想法整理成可编辑草案，买卖条件、区间、资金和成本都由你确认。"),
        ("运行实验", "固定本次行情数据，执行原策略回测、成本敏感性、参数邻域和分时段实验。"),
        ("阅读报告", "先看收益和风险，再看热力图、对比图与 AI 结论；每条结论都能展开证据。"),
    ), start=1):
        with st.container(border=True):
            step_col, text_col = st.columns([0.55, 5])
            step_col.markdown(f"<div class='qv-step'>步骤 {index}</div>", unsafe_allow_html=True)
            text_col.markdown(f"**{title}**  \n{body}")

    st.markdown("### 当前可直接验证的策略")
    strategy_col1, strategy_col2, strategy_col3 = st.columns(3)
    with strategy_col1.container(border=True):
        st.markdown("#### 双均线交叉")
        st.write("短期均线上穿长期均线时买入，下穿时卖出。适合用来理解趋势跟随和震荡损耗。")
        st.caption("需要选择：快线周期、慢线周期、交易标的和测试区间。")
        st.button(
            "用双均线开始", key="home_ma", use_container_width=True,
            on_click=_seed_verification, args=("ma_cross",),
        )
    with strategy_col2.container(border=True):
        st.markdown("#### 价格通道突破")
        st.write("突破过去一段时间的最高价时买入，跌破较短周期最低价时退出。")
        st.caption("需要选择：突破周期、退出周期、交易标的和测试区间。")
        st.button(
            "用价格突破开始", key="home_breakout", use_container_width=True,
            on_click=_seed_verification, args=("breakout",),
        )
    with strategy_col3.container(border=True):
        st.markdown("#### RSI 均值回归")
        st.write("RSI 进入超卖区时买入，进入超买区时卖出，用来测试价格向常态回归的想法。")
        st.caption("需要选择：RSI 周期、买入阈值、卖出阈值、标的和区间。")
        st.button(
            "用 RSI 开始", key="home_rsi", use_container_width=True,
            on_click=_seed_verification, args=("rsi_reversion",),
        )

    st.markdown("### 报告不会只给一个分数")
    sample_left, sample_right = st.columns([1.15, 1])
    with sample_left.container(border=True):
        st.caption("报告版式示意 · 非真实实验结果")
        st.markdown("#### 历史表现预览")
        st.write("原始策略的收益、最大回撤和相对基准表现会分开呈现。")
        sample_metrics = st.columns(3)
        sample_metrics[0].metric("策略收益", "12.4%")
        sample_metrics[1].metric("最大回撤", "-8.7%")
        sample_metrics[2].metric("超额收益", "+1.9%")
        st.caption("这些数字只用于展示报告布局，不来自任何策略或行情。")
    with sample_right.container(border=True):
        st.markdown("#### 每条结论附带")
        st.write("✓ 本次实验中的具体指标")
        st.write("✓ 对应热力图、对比图或数据表")
        st.write("✓ 数据范围和结论局限")
        st.write("✓ 可下载的完整证据包")

    st.info("当前不做实时选股、自动交易或新闻因果归因。历史回测用于理解策略规则和风险，不预测未来。")


def _render_docs() -> None:
    st.markdown("<div class='qv-eyebrow'>GUIDE</div>", unsafe_allow_html=True)
    st.title("使用文档")
    st.caption("从第一次输入到读懂报告，所有默认设置和修改影响都集中在这里。")
    st.button(
        "进入量化策略验证", type="primary",
        on_click=_navigate, args=("量化策略验证",),
    )

    quick_tab, strategy_tab, defaults_tab, report_tab, ai_tab = st.tabs([
        "快速入门", "策略规则", "默认值", "怎样看报告", "AI 回答方式",
    ])
    with quick_tab:
        st.markdown("### 从一句话开始")
        st.code("用沪深300ETF测试双均线，5日线上穿20日线买入，回测近两年。", language=None)
        st.markdown(
            """
1. 在“量化策略验证”中输入想法，也可以从侧栏典型策略开始。
2. Agent 会追问无法执行的缺失条件，并生成策略确认卡。
3. 检查买卖规则、日期、标的、资金与成本，然后点击确认。
4. 点击“开始验证并生成报告”，等待回测、实验和报告完成。
5. 在报告区查看表现、图表、结论和数据，也可以在对话中继续追问。
            """
        )
        st.warning("确认卡里的规则就是实际执行规则。只修改聊天文字不会绕过确认。")
    with strategy_tab:
        st.markdown("### 双均线交叉")
        st.write("快线反映较近的平均价格，慢线反映较长趋势。快线上穿慢线买入，下穿卖出。")
        st.dataframe(pd.DataFrame([
            {"参数": "快线周期", "含义": "短期平均价格使用的交易日数", "要求": "正整数且小于慢线"},
            {"参数": "慢线周期", "含义": "长期平均价格使用的交易日数", "要求": "正整数且大于快线"},
        ]), hide_index=True, width="stretch")
        st.markdown("### 价格通道突破")
        st.write("收盘价超过此前突破周期内的最高价时买入，低于此前退出周期内的最低价时卖出。")
        st.dataframe(pd.DataFrame([
            {"参数": "突破周期", "含义": "判断向上突破所看的历史长度", "要求": "大于退出周期"},
            {"参数": "退出周期", "含义": "判断趋势转弱所看的历史长度", "要求": "正整数"},
        ]), hide_index=True, width="stretch")
        st.markdown("### RSI 均值回归")
        st.write("RSI 进入超卖区域时买入，进入超买区域时卖出，尝试捕捉价格向常态回归的过程。")
        st.dataframe(pd.DataFrame([
            {"参数": "RSI 周期", "含义": "计算近期上涨和下跌强弱所用的交易日数", "要求": "正整数"},
            {"参数": "买入阈值", "含义": "RSI 不高于该值时形成买入信号", "要求": "大于零且小于卖出阈值"},
            {"参数": "卖出阈值", "含义": "RSI 不低于该值时形成卖出信号", "要求": "大于买入阈值且小于一百"},
        ]), hide_index=True, width="stretch")
        st.info("三种策略都在当日收盘后形成信号，于下一交易日开盘执行，避免使用尚未出现的信息。")
    with defaults_tab:
        st.markdown("### 当前执行默认值")
        st.dataframe(pd.DataFrame([
            {"设置": "初始资金", "默认值": "100,000 元", "修改后影响": "持仓规模和最低佣金影响"},
            {"设置": "成交时点", "默认值": "信号次日开盘", "修改后影响": "当前固定，不可修改"},
            {"设置": "复权口径", "默认值": "前复权", "修改后影响": "当前由数据层固定"},
            {"设置": "仓位", "默认值": "单标的满仓、仅多头", "修改后影响": "当前固定，不可修改"},
            {"设置": "单边佣金", "默认值": "0.01%", "修改后影响": "改变每次成交成本"},
            {"设置": "最低佣金", "默认值": "0 元", "修改后影响": "小额交易受影响更明显"},
            {"设置": "单边滑点", "默认值": "1 bps", "修改后影响": "成交越频繁，累计影响通常越大"},
            {"设置": "比较基准", "默认值": "同一标的买入持有", "修改后影响": "改变超额收益的比较对象"},
        ]), hide_index=True, width="stretch")
        st.caption("资金、佣金、最低佣金、滑点和比较基准可在确认卡的折叠设置中修改。")
    with report_tab:
        st.markdown("### 建议阅读顺序")
        st.markdown(
            """
1. **总收益和基准收益**：策略是否赚钱，以及是否优于简单买入持有。
2. **最大回撤**：历史上从高点跌到低点最严重的一段。
3. **参数热力图**：结果是否只依赖一个特殊参数组合。
4. **成本敏感性**：滑点提高后，历史结果是否明显恶化。
5. **分时段比较**：不同历史阶段的结果是否一致。
6. **结论附件**：展开表格和图表，确认 AI 的解释来自本次实验。
            """
        )
        st.error("正收益比例不是未来成功概率；历史表现、参数邻域和分时段实验都不构成统计证明。")
    with ai_tab:
        st.markdown("### 模型连接")
        st.write("在任意页面展开侧栏的“模型连接”，填写 API Key、OpenAI 兼容接口 URL 和模型名。报告模型可以单独填写；留空时使用同一个模型。")
        st.markdown(
            "- **保存**：立即用于自然语言解析、金融问答、报告生成和报告追问，不额外发起测试请求。\n"
            "- **测试并保存**：先发送一个最小 JSON 请求，连接成功后再启用。\n"
            "- **会话范围**：配置只存在当前浏览器会话中，不写入 `.env`、仓库、报告或下载文件。"
        )
        st.caption("如果部署者已经通过环境变量配置模型，侧栏会显示“服务器环境配置”；用户填写的会话配置优先。")
        st.markdown("### 回答方式")
        st.write("侧栏可以随时切换，设置会用于 AI 金融问答；实验事实和证据不会随表达方式变化。")
        st.dataframe(pd.DataFrame([
            {"模式": "新手讲解", "适合": "第一次接触量化", "表达": "解释术语并给简短例子"},
            {"模式": "简洁摘要", "适合": "快速查看答案", "表达": "只保留关键结论和限制"},
            {"模式": "技术细节", "适合": "希望了解计算口径", "表达": "增加公式、统计和实现细节"},
        ]), hide_index=True, width="stretch")
        st.info("AI 金融问答目前只讲基础概念和常见策略，不查询实时价格或新闻，也不提供个性化投资建议。")


def _render_finance_qa(answer_mode: str, *, llm_available: bool, llm_complete) -> None:
    st.markdown("<div class='qv-eyebrow'>LEARN</div>", unsafe_allow_html=True)
    st.title("AI 金融问答")
    st.caption("先理解概念，再决定是否把一个受支持的策略带入历史验证。")
    if not llm_available:
        st.info("模型尚未配置，当前使用本地基础知识回答常见概念。")

    st.markdown("### 可以这样问")
    quick_cols = st.columns(3)
    quick_prompts = (
        "什么是量化投资？",
        "最大回撤是什么意思？",
        "双均线策略有什么局限？",
        "价格突破策略怎样运行？",
        "RSI 超买超卖是什么意思？",
    )
    selected_question = None
    for index, prompt in enumerate(quick_prompts):
        if quick_cols[index % 3].button(prompt, key=f"finance_quick_{index}", use_container_width=True):
            selected_question = prompt

    typed_question = st.chat_input("问一个量化或回测相关问题……", key="finance_chat_input")
    question = (typed_question or selected_question or "").strip()
    history = st.session_state.setdefault("finance_chat", [])
    if question:
        history.append({"role": "user", "content": question})
        with st.spinner("正在整理这个概念……"):
            answer = answer_finance_question(
                question, mode=answer_mode, history=history,
                use_llm=llm_available, complete=llm_complete,
            )
        history.append({"role": "assistant", **answer})
        st.rerun()

    if not history:
        with st.chat_message("assistant"):
            st.write("你好，我可以解释量化、回测、收益与风险指标，以及常见策略怎样运行。")
            st.caption("涉及当前报告的具体问题，请回到“量化策略验证”，在报告完成后的对话里提问。")
    for index, message in enumerate(history):
        with st.chat_message(message["role"]):
            st.write(message.get("content") or message.get("answer"))
            if message["role"] == "assistant":
                concepts = message.get("related_concepts") or []
                if concepts:
                    st.caption("相关概念：" + " · ".join(concepts))
                st.caption("AI 解释" if message.get("source") == "ai" else "本地基础知识")
                suggested = message.get("suggested_strategy")
                if suggested:
                    label = {
                        "ma_cross": "用双均线创建验证草案",
                        "breakout": "用价格突破创建验证草案",
                        "rsi_reversion": "用 RSI 创建验证草案",
                    }[suggested]
                    st.button(
                        label, key=f"finance_seed_{index}", type="primary",
                        on_click=_seed_verification, args=(suggested,),
                    )
    st.caption(DISCLAIMER)


# 四个模块共用同一套导航和回答偏好；验证工作台在下方继续渲染。
st.sidebar.markdown(f"## {APP_NAME}")
st.sidebar.caption(f"{APP_TAGLINE} · {APP_VERSION}")
page = _sidebar_cards(
    "功能模块", _PAGES, state_key="main_nav",
    icons=("⌂", "▤", "◆", "✦"),
)
st.sidebar.divider()
answer_mode = _sidebar_cards(
    "AI 回答方式", _ANSWER_MODES, state_key="answer_mode",
    icons=("○", "◇", "□"),
)
st.sidebar.caption("回答方式不会修改回测结果。")
st.sidebar.divider()
llm_settings = _render_llm_connection()
llm_available = bool(
    llm_settings["api_key"] and llm_settings["base_url"] and llm_settings["model"]
)
llm_complete = _llm_complete(llm_settings)
llm_report_complete = _llm_complete(llm_settings, report=True)

if page == "首页":
    _render_home()
    st.caption(DISCLAIMER)
    st.stop()
if page == "使用文档":
    _render_docs()
    st.caption(DISCLAIMER)
    st.stop()
if page == "AI 金融问答":
    _render_finance_qa(
        answer_mode, llm_available=llm_available, llm_complete=llm_complete,
    )
    st.stop()


# 侧栏在验证模块中继续显示实验环境与快捷入口。
st.sidebar.divider()
st.sidebar.markdown("### 验证工作台")
st.sidebar.header("验证环境")
snapshots = list_snapshots()
if not snapshots:
    st.error("本地没有可用快照。请先通过数据模块创建真实或模拟快照。")
    st.stop()

snapshot_by_id = {meta.snapshot_id: meta for meta in snapshots}
snapshot_options = sorted(
    snapshot_by_id,
    key=lambda sid: (
        snapshot_by_id[sid].is_synthetic,
        -pd.Timestamp(snapshot_by_id[sid].actual_end).value,
        sid,
    ),
)
pending_snapshot_id = st.session_state.pop("_pending_snapshot_id", None)
if pending_snapshot_id in snapshot_by_id:
    st.session_state["selected_snapshot_id"] = pending_snapshot_id
elif st.session_state.get("selected_snapshot_id") not in snapshot_by_id:
    st.session_state.pop("selected_snapshot_id", None)
snapshot_id = st.sidebar.selectbox(
    "当前实验数据",
    options=snapshot_options,
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
    st.session_state["strategy_input_history"] = []
    st.session_state["report_qa_history"] = []
    for key in list(st.session_state):
        if key.startswith("quick_"):
            st.session_state.pop(key, None)

seed_template = st.session_state.pop("verification_seed_template", None)
if seed_template in {"ma_cross", "breakout", "rsi_reversion"}:
    seeded = _default_draft(snapshot_meta, prices)
    seeded.template = seed_template
    seeded.origin = "preset"
    seeded.specified_fields = ["推荐策略模板"]
    st.session_state["strategy_input_history"] = []
    _install_draft(seeded, snapshot_id)
    strategy_name = {
        "ma_cross": "双均线",
        "breakout": "价格通道突破",
        "rsi_reversion": "RSI 均值回归",
    }[seed_template]
    _append_chat(
        "assistant",
        f"我已从上一页带入{strategy_name}草案。请在确认卡中检查参数、标的和测试区间。",
    )

if st.sidebar.button("新建对话", use_container_width=True):
    _clear_draft()
    st.session_state["strategy_chat"] = []
    st.session_state["strategy_input_history"] = []
    st.session_state["report_qa_history"] = []
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
        st.session_state["strategy_input_history"] = []
        _install_draft(preset, snapshot_id)
        _append_chat("assistant", f"我已载入 {label} 的典型双均线草案，请在对话中检查并确认。")
        st.rerun()

st.sidebar.markdown("**典型价格突破**")
for label, preset_entry, preset_exit in (
    ("20日突破 / 10日退出", 20, 10),
    ("55日突破 / 20日退出", 55, 20),
):
    if st.sidebar.button(
        label,
        key=f"preset_breakout_{preset_entry}_{preset_exit}",
        use_container_width=True,
    ):
        preset = _default_draft(snapshot_meta, prices)
        preset.template = "breakout"
        preset.entry_window = preset_entry
        preset.exit_window = preset_exit
        preset.origin = "preset"
        preset.specified_fields = ["典型策略模板"]
        st.session_state["strategy_input_history"] = []
        _install_draft(preset, snapshot_id)
        _append_chat("assistant", f"我已载入 {label} 的价格突破草案，请在对话中检查并确认。")
        st.rerun()

st.sidebar.markdown("**典型 RSI 均值回归**")
for label, preset_period, preset_buy, preset_sell in (
    ("14日 RSI / 30买入 / 70卖出", 14, 30, 70),
    ("10日 RSI / 25买入 / 75卖出", 10, 25, 75),
):
    if st.sidebar.button(
        label,
        key=f"preset_rsi_{preset_period}_{preset_buy}_{preset_sell}",
        use_container_width=True,
    ):
        preset = _default_draft(snapshot_meta, prices)
        preset.template = "rsi_reversion"
        preset.rsi_period = preset_period
        preset.buy_threshold = preset_buy
        preset.sell_threshold = preset_sell
        preset.origin = "preset"
        preset.specified_fields = ["典型策略模板"]
        st.session_state["strategy_input_history"] = []
        _install_draft(preset, snapshot_id)
        _append_chat("assistant", f"我已载入 {label} 的 RSI 草案，请在对话中检查并确认。")
        st.rerun()

with st.sidebar.form("quick_strategy_form"):
    st.markdown("**按选项创建双均线**")
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
            st.session_state["strategy_input_history"] = []
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
    st.write("**突破周期**：用过去一段时间的最高价判断是否向上突破。")
    st.write("**退出周期**：用较短区间的最低价判断何时退出持仓。")
    st.write("**RSI 周期**：计算近期上涨和下跌强弱时使用的交易日数。")
    st.write("**超卖/超买阈值**：RSI 到达买入或卖出区域的边界。")
    st.write("**比较基准**：用来比较策略是否优于直接买入并持有。")


# chat_input 固定显示在页面底部，但先处理输入，确保新草案在本轮渲染前同步到编辑控件。
idea_text = st.chat_input(
    "询问这份报告，或明确告诉我如何修改策略……"
    if _current_report_ready(snapshot_id)
    else "告诉我你的策略想法，也可以继续补充或修改……",
    key="agent_chat_input",
    disabled=quality_report.has_error,
)
if idea_text and idea_text.strip():
    report_ready = _current_report_ready(snapshot_id)
    intent = route_agent_message(idea_text, report_ready=report_ready)
    if intent == "report_question":
        _append_chat("user", idea_text.strip())
        evidence_stored = st.session_state["evidence_result"]
        pack = evidence_stored["pack"]
        report = evidence_stored["analysis"]
        history = st.session_state.setdefault("report_qa_history", [])
        with st.spinner("正在查阅本次实验的指标、图表和证据……"):
            answer = answer_report_question(
                idea_text.strip(), pack, report,
                history=history, answer_mode=answer_mode,
                use_llm=llm_available, complete=llm_report_complete,
            )
        history.append({
            "question": idea_text.strip(),
            "answer": answer["answer"],
            "evidence_refs": answer["evidence_refs"],
        })
        _append_chat(
            "assistant", format_report_answer(answer, pack),
            evidence_refs=answer["evidence_refs"],
        )
        st.rerun()
    _append_chat("user", idea_text.strip())
    strategy_inputs = st.session_state.setdefault("strategy_input_history", [])
    strategy_inputs.append(idea_text.strip())
    combined_text = "\n".join(strategy_inputs)
    with st.spinner("正在理解你的想法……"):
        parsed = parse_user_input(
            combined_text,
            reference_end=date.today().isoformat(),
            use_llm=llm_available,
            complete=llm_complete,
        )
    if parsed.needs_clarification:
        reply = "为了帮你测试，想再确认：\n\n" + "\n".join(f"- {question}" for question in parsed.questions)
        if parsed.degraded:
            reply += "\n\n暂时只能识别简单描述，也可以在侧栏选择示例开始。"
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
            use_selected_snapshot = _snapshot_covers_request(
                prices, parsed_draft.symbol, parsed_draft.start, parsed_draft.end,
            )
            with st.spinner("正在检查本地行情，缺失时将从外部接口获取……"):
                if use_selected_snapshot:
                    resolved_meta, resolved_quality, cache_hit = (
                        snapshot_meta, quality_report, True
                    )
                else:
                    resolved_meta, resolved_quality, cache_hit = ensure_market_snapshot(
                        [parsed_draft.symbol], parsed_draft.start, parsed_draft.end,
                    )
            if resolved_quality.has_error:
                raise RuntimeError("获取到的行情未通过数据质量检查")
            resolved_prices, resolved_meta, resolved_quality = load_snapshot(
                resolved_meta.snapshot_id,
            )
            if use_selected_snapshot:
                data_reply = f"当前实验数据 `{resolved_meta.snapshot_id}` 已覆盖该区间。"
            else:
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


st.title("量化策略验证")
st.caption("说说你的投资想法，用历史行情试一试，看看过去会怎样。")
if snapshot_meta.is_synthetic:
    st.warning("当前使用模拟数据，只能验证系统流程，不能用于判断策略有效性。")
if quality_report.has_error:
    st.error("当前行情存在错误级质量问题，请在侧栏更换数据。")
if not llm_available:
    st.info("还没连接 AI，暂时只能识别简单描述。可以在侧栏选择示例，或打开“模型连接”完成设置。")

with st.chat_message("assistant"):
    st.write(
        "你好，我是量见。说说你想买什么、什么时候买卖。不用懂专业术语，我会帮你一步步说清楚。"
        "你确认后，我们再用历史行情试一试。"
    )
    st.caption(
        "比如：“我想买沪深300基金，跌得多时买，涨回来就卖。”规则还没想清楚也可以先聊。"
    )

for message_index, message in enumerate(st.session_state.setdefault("strategy_chat", [])):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        _render_chat_evidence(message, message_index)


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
            st.markdown("**策略类型：" + {
                "ma_cross": "双均线交叉",
                "breakout": "价格通道突破",
                "rsi_reversion": "RSI 均值回归",
            }[draft.template] + "**")
            fast_value, slow_value = draft.fast, draft.slow
            entry_value, exit_value = draft.entry_window, draft.exit_window
            rsi_period_value = draft.rsi_period
            buy_threshold_value = draft.buy_threshold
            sell_threshold_value = draft.sell_threshold
            if draft.template == "ma_cross":
                param_col1, param_col2 = st.columns(2)
                fast_value = int(param_col1.number_input(
                    "快线周期（日）", min_value=1, step=1, key=_DRAFT_WIDGET_KEYS["fast"],
                    help="较短时间的平均价格，对近期变化更敏感。",
                ))
                slow_value = int(param_col2.number_input(
                    "慢线周期（日）", min_value=2, step=1, key=_DRAFT_WIDGET_KEYS["slow"],
                    help="较长时间的平均价格，用来表示长期趋势。",
                ))
            elif draft.template == "breakout":
                param_col1, param_col2 = st.columns(2)
                entry_value = int(param_col1.number_input(
                    "突破周期（日）", min_value=2, step=1,
                    key=_DRAFT_WIDGET_KEYS["entry_window"],
                    help="收盘价超过此前这段时间的最高价时形成买入信号。",
                ))
                exit_value = int(param_col2.number_input(
                    "退出周期（日）", min_value=1, step=1,
                    key=_DRAFT_WIDGET_KEYS["exit_window"],
                    help="收盘价低于此前这段时间的最低价时形成卖出信号。",
                ))
            else:
                param_col1, param_col2, param_col3 = st.columns(3)
                rsi_period_value = int(param_col1.number_input(
                    "RSI 周期（日）", min_value=1, step=1,
                    key=_DRAFT_WIDGET_KEYS["rsi_period"],
                    help="计算近期上涨和下跌强弱时使用的交易日数。",
                ))
                buy_threshold_value = float(param_col2.number_input(
                    "超卖买入阈值", min_value=1.0, max_value=99.0, step=1.0,
                    key=_DRAFT_WIDGET_KEYS["buy_threshold"],
                    help="RSI 不高于这个值时形成买入信号。",
                ))
                sell_threshold_value = float(param_col3.number_input(
                    "超买卖出阈值", min_value=1.0, max_value=99.0, step=1.0,
                    key=_DRAFT_WIDGET_KEYS["sell_threshold"],
                    help="RSI 不低于这个值时形成卖出信号。",
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
            entry_window=entry_value,
            exit_window=exit_value,
            rsi_period=rsi_period_value,
            buy_threshold=buy_threshold_value,
            sell_threshold=sell_threshold_value,
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
        if draft.template == "ma_cross":
            st.write(
                f"当 **{draft.fast} 日均线向上穿过 {draft.slow} 日均线** 时买入 "
                f"**{_symbol_label(draft.symbol)}**；向下穿过时卖出。"
            )
        elif draft.template == "breakout":
            st.write(
                f"当收盘价突破此前 **{draft.entry_window} 日最高价** 时买入 "
                f"**{_symbol_label(draft.symbol)}**；跌破此前 "
                f"**{draft.exit_window} 日最低价** 时卖出。"
            )
        else:
            st.write(
                f"计算 **{draft.rsi_period} 日 RSI**；当 RSI 不高于 "
                f"**{draft.buy_threshold:g}** 时买入 **{_symbol_label(draft.symbol)}**，"
                f"不低于 **{draft.sell_threshold:g}** 时卖出。"
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
            execution_config = draft.to_config(snapshot_id)
            benchmark_symbol = draft.benchmark_symbol or draft.symbol
            try:
                with st.status("Agent 正在验证策略……", expanded=True) as status:
                    st.write("正在读取固定行情快照并运行原始策略回测……")
                    trades, daily, metrics = run_backtest_from_snapshot(
                        execution_config, benchmark_symbol=draft.benchmark_symbol,
                    )
                    st.session_state["backtest_result"] = {
                        "key": result_key, "config": execution_config, "trades": trades,
                        "daily": daily, "metrics": metrics,
                    }
                    st.write("正在测试交易成本、邻近参数和不同时段……")
                    validation_report = run_validation_suite(
                        execution_config, benchmark_symbol=benchmark_symbol,
                    )
                    st.session_state["validation_result"] = {
                        "key": result_key, "report": validation_report,
                    }
                    st.write("正在整理热力图、对比图和可追溯证据……")
                    pack = build_evidence(validation_report)
                    if llm_available:
                        st.write("正在让 AI 阅读结构化实验数据并撰写结论……")
                    else:
                        st.write("模型未配置，正在生成程序事实摘要……")
                    analysis = generate_report(
                        pack, use_llm=llm_available, complete=llm_report_complete,
                    )
                    bundle = export_bundle(pack, analysis)
                    st.session_state["evidence_result"] = {
                        "key": result_key, "pack": pack, "analysis": analysis, "bundle": bundle,
                    }
                    st.session_state["report_qa_history"] = []
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
