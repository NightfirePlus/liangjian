"""两人共享的数据契约。改动字段前先在群里同步，别单方面改。

表格类数据用 pandas.DataFrame，列名在下面的常量里固定。
结构类数据用 dataclass，可 asdict() 后存成 JSON。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------- 行情数据
# 行情快照为长表：每行是 (symbol, date) 的一根日线。
# 价格口径见 SnapshotMeta.adjust；日期为 datetime64[ns]，无时区，升序，(symbol,date) 唯一。
PRICE_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]

Severity = Literal["info", "warn", "error"]
IssueKind = Literal[
    "unsorted",          # 日期未升序（已整理，仅记录）
    "duplicate",         # 重复日期（已去重，记录保留哪条）
    "missing_days",      # 相对资产池交易日历缺失
    "price_anomaly",     # 单日涨跌幅异常，只标记不删除
    "nonpositive_price", # 价格 <= 0 或为空
    "ohlc_inconsistent", # high < low、close 不在 [low, high] 等
    "short_history",     # 历史长度不足以支撑请求区间/参数
    "misaligned",        # 不同资产起止日期不一致
]


@dataclass
class QualityIssue:
    symbol: str
    kind: IssueKind
    severity: Severity
    message: str                      # 人能读懂的一句话
    dates: list[str] = field(default_factory=list)   # 涉及日期 YYYY-MM-DD，可截断
    action: str = "flagged"           # flagged(仅标记) / dropped / filled


@dataclass
class QualityReport:
    snapshot_id: str
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def has_error(self) -> bool:
        return any(i.severity == "error" for i in self.issues)


@dataclass
class SnapshotMeta:
    snapshot_id: str                  # 例 "etf_20260117_a1b2c3"，内容哈希保证同数据同 id
    symbols: list[str]
    start: str                        # 请求区间 YYYY-MM-DD
    end: str
    actual_start: str                 # 实际数据区间（可能因上市较晚而更短）
    actual_end: str                   # 展示给用户的"数据截止日期"
    source: str                       # 例 "akshare/fund_etf_hist_em"
    fetched_at: str                   # ISO 时间；缓存回退时是原始获取时间
    adjust: Literal["qfq", "hfq", "none"]   # 复权口径
    is_synthetic: bool = False        # True 表示模拟数据，界面必须显著标注
    from_cache: bool = False          # 本次是否由缓存回退而来
    redistributable: bool | None = None  # 数据能否随公开仓库分发；None=未核实
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 策略配置
@dataclass
class CostConfig:
    commission_rate: float = 0.0001   # 单边佣金率
    min_commission: float = 0.0       # 单笔最低佣金（ETF 简化为 0，显式展示）
    slippage_bps: float = 1.0         # 单边滑点，基点


@dataclass
class StrategyConfig:
    template: Literal["ma_cross", "momentum_rotation"]
    symbols: list[str]
    start: str
    end: str
    initial_cash: float = 100_000.0
    params: dict = field(default_factory=dict)   # ma_cross: fast/slow；rotation: lookback/top_n/rebalance
    cost: CostConfig = field(default_factory=CostConfig)
    snapshot_id: str | None = None
    assumptions: list[str] = field(default_factory=list)  # 默认假设，确认卡逐条展示


# ---------------------------------------------------------------- 回测输出
# 逐笔交易
TRADE_COLUMNS = [
    "trade_id", "signal_date", "exec_date", "symbol", "side",   # side: buy/sell
    "shares", "price", "gross", "commission", "slippage_cost", "rule_trigger",
]
# 每日持仓与净值（长表：每个 date 每个持仓 symbol 一行 + 汇总行用 symbol="__TOTAL__"）
DAILY_COLUMNS = [
    "date", "symbol", "shares", "close", "market_value", "cash", "equity",
    "benchmark_equity", "daily_pnl",
]


@dataclass
class Metrics:
    total_return: float | None
    annual_return: float | None       # 口径：按 252 交易日复利年化；历史 < 60 日返回 None
    annual_vol: float | None
    max_drawdown: float | None
    max_dd_start: str | None
    max_dd_end: str | None
    n_trades: int = 0
    turnover: float | None = None     # 累计成交额 / 平均净值
    total_cost: float = 0.0
    benchmark_return: float | None = None
    excess_return: float | None = None
    notes: list[str] = field(default_factory=list)   # 边界情况说明，如"无交易""零波动"


# ---------------------------------------------------------------- 实验与结论
@dataclass
class ExperimentSpec:
    experiment_id: str
    kind: Literal["cost_sweep", "param_neighborhood", "period_split"]
    claim_text: str                   # 用户原话，如"交易成本影响小"
    metric: str                       # 如 "net_excess_return"
    grid: dict                        # 预先固定的扫描范围，如 {"slippage_bps":[0,1,2,5,10]}
    support_rule: str                 # 判断条件，如 "各档位 excess_return > 0"
    confirmed_by_user: bool = False


@dataclass
class ExperimentResult:
    experiment_id: str
    rows: list[dict]                  # 每个网格点的 metric 值
    verdict: Literal["supported", "not_supported", "mixed", "insufficient"]
    first_failure: dict | None = None  # 已测档位中首个不满足判断条件的点；不是全局边界
    config_hash: str = ""


@dataclass
class EvidenceRef:
    kind: Literal["metric", "chart", "config", "snapshot", "experiment", "trade", "day"]
    ref_id: str                       # 指向上面各对象的 id 或 date
    label: str = ""


@dataclass
class Finding:
    text: str                         # 报告中的一句结论
    level: Literal["fact", "diagnosis", "unverified"]  # 已计算事实 / 规则诊断 / 尚未验证
    evidence: list[EvidenceRef] = field(default_factory=list)
