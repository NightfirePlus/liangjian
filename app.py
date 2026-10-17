"""量见 - 最小可跑骨架。

当前用假数据占位回测/数据模块，目的是先打通界面流程、验证部署。
组员把 core/data 下的函数实现后，这里改成调用真实快照即可。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import APP_NAME, APP_TAGLINE, DISCLAIMER, LLM_AVAILABLE

st.set_page_config(page_title=APP_NAME, layout="wide")


def fake_equity_curve(start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """占位：在 core/backtest 实现前，用随机游走模拟净值曲线。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    strat_ret = rng.normal(0.0004, 0.01, n)
    bench_ret = rng.normal(0.0003, 0.009, n)
    strat_equity = 100_000 * (1 + strat_ret).cumprod()
    bench_equity = 100_000 * (1 + bench_ret).cumprod()
    return pd.DataFrame({"date": dates, "strategy": strat_equity, "benchmark": bench_equity})


st.title(APP_NAME)
st.caption(APP_TAGLINE)
st.warning(DISCLAIMER)

if not LLM_AVAILABLE:
    st.info("当前未配置大模型 API，AI 解释环节不可用，仅展示基于规则的计算结果。")

st.header("1. 输入策略想法")
idea = st.text_area(
    "用自然语言描述你的策略和想验证的观点",
    placeholder="例如：沪深300ETF，5日均线上穿20日均线买入，下穿卖出；我认为这个策略对交易成本不敏感。",
)

st.header("2. 策略确认卡（占位）")
st.caption("自然语言解析模块完成后，这里展示提取出的规则、默认假设与待补充条件，供用户确认。")
with st.expander("示例确认卡", expanded=False):
    st.json({
        "template": "ma_cross",
        "symbols": ["510300"],
        "params": {"fast": 5, "slow": 20},
        "assumptions": ["收盘形成信号，次日开盘成交", "前复权价格", "仅多头，无杠杆"],
    })

st.header("3. 回测结果（占位数据，非真实回测）")
col1, col2 = st.columns(2)
start = col1.date_input("开始日期", value=pd.Timestamp("2023-01-01"))
end = col2.date_input("结束日期", value=pd.Timestamp("2024-12-31"))

if st.button("运行占位回测"):
    df = fake_equity_curve(str(start), str(end))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["strategy"], name="策略"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["benchmark"], name="基准（买入持有）"))
    fig.update_layout(title="净值曲线（占位数据）", xaxis_title="日期", yaxis_title="净值")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("以上为随机生成的占位数据，不代表任何真实策略表现。")

st.header("4. 验证实验（占位）")
st.caption("成本敏感性 / 参数邻域 / 分时段表现，core/experiments 实现后接入。")

st.header("5. 证据报告（占位）")
st.caption("程序诊断 + 受证据约束的 AI 解释，core/report 实现后接入。结论将可点击回看依据。")
