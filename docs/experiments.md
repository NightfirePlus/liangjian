# 验证实验 v1

本模块为量化策略验证提供结构化事实，产品定位与新手交互规划见 [product.md](product.md)。
后续界面应先说明每类实验解决什么问题，将参数设置放在可展开选项中；当前 0.1 使用手动配置流程。

## 输入与执行边界

自然语言模块输出 `StrategyConfig` 和可选的 `ExperimentSpec`。NLP 只识别用户观点及实验类型；实验模块负责补齐固定网格、指标和判断规则。调用方必须：

1. 让用户确认策略与观点；
2. 给 `StrategyConfig.snapshot_id` 绑定一份固定快照；
3. 将 `ExperimentSpec.confirmed_by_user` 设为 `true`；
4. 单项实验调用 `run_experiment(config, spec, benchmark_symbol)`，完整验证调用
   `run_validation_suite(config, benchmark_symbol)`。

未指定基准时，实验默认用策略首个标的的买入持有净值计算超额收益。

## 默认实验

完整验证始终先运行用户原始策略，作为分析基准。其他扫描用于观察结果随条件变化的情况，
不能替代用户配置。

| 类型 | 改变的内容 | 保持不变的内容 | 默认扫描 | v1 判断规则 |
|---|---|---|---|---|
| `cost_sweep` | 滑点 | 策略参数、手续费、行情快照和区间 | 滑点 `0/1/2/5/10 bps`，并包含原始滑点 | `0~5 bps` 档位均有正超额收益 |
| `param_neighborhood` | 用户策略参数 | 行情快照和完整区间 | 快线 `基准±2`、慢线 `基准±5` 的合法组合 | 至少 70% 参数组合有正超额收益 |
| `period_split` | 使用的真实行情区间 | 用户原始策略参数 | 按自然日中点分为前后两段 | 至少一个时段有正超额收益 |

单项实验结果：达到规则为 `supported`；有正有负为 `mixed`；全部失败为
`not_supported`；指标缺失或某个点无法运行为 `insufficient`。`first_failure`
保存第一个不满足条件的扫描点。

以上判断字段保留用于兼容单项实验调用，不作为报告结论。完整实验输出原始数据及统计，
`ValidationReport.verdict` 固定为 `None`，`reasons` 为空。报告层读取事实绘图，再由模型
基于证据解释；不再使用固定比例或成交数阈值对策略综合定级，详见 [report.md](report.md)。

## 输出

`ExperimentResult.rows` 的每一行包含：

- 当前扫描维度，例如 `slippage_bps`、`fast/slow` 或时段；
- 完整 `Metrics` 字段；
- `metric`、`metric_value`、`passed` 和 `error`；
- 成本与参数实验的交易次数。

`config_hash` 是策略配置、完整实验规格与实际比较基准的 SHA-256，用于判断两次实验是否使用完全相同的计算输入。

`ValidationReport.event_analysis` 是时事归因预留接口，v1 固定为 `None`，不访问
外部事件数据，也不参与验证结论。

## 当前限制

- 当前回测引擎只实现 `ma_cross`，动量轮动实验会明确拒绝。
- 分时段实验把每段视为独立回测，每段开头重新进行均线预热。
- v1 只支持上述三类固定规则；“收益下降不超过 2 个百分点”等自定义阈值需要后续扩展结构化规则。
- NLP 能记录 `same_close`、仓位方式等覆盖要求，但当前回测引擎只执行次日开盘、单标的满仓规则。接入界面时必须展示这一限制，不能静默忽略用户覆盖要求。
