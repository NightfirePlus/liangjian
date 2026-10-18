"""成本敏感性 / 参数邻域 / 分时段验证实验。

与 core/nlp/ 的接口约定（给本模块负责人）：
---------------------------------------------------------------
core/nlp/ 解析用户输入时，如果识别到一句"关于策略表现的观点"（如"我认为这个
策略对交易成本不敏感"），会生成一个 core.schemas.ExperimentSpec，但只填：

    claim_text      用户原话，摘录不改写
    kind            粗略猜测的实验类型（cost_sweep / param_neighborhood / period_split），
                    基于 claim_text 里的关键词简单判断，不保证准确，可能需要你重新判断
    metric          一个初步猜测的指标名字符串（目前固定给 "net_excess_return"）
    experiment_id   留空字符串，由本模块生成时分配
    grid            留空 dict {}，本模块需要自行决定具体的扫描网格
                    （如 cost_sweep 测哪些滑点/佣金档位）
    support_rule    留空字符串，本模块需要自行定义"支持/不支持"的判定条件

即 core/nlp/ 只负责"是否存在一个用户想验证的观点，以及这个观点大致是哪一类"，
不涉及实验怎么跑、网格怎么定、如何判定支持与否——这些都是本模块的职责。

如果 claim_text 为 None（用户没表达任何可验证的观点），core/nlp/ 不会生成
ExperimentSpec，对应 ParseResult.experiment_spec 为 None，调用方（app.py）
应该跳过验证实验环节。
"""
