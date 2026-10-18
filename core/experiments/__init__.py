"""验证实验公共入口。

NLP 只提供观点原文、实验类型和初步指标；本模块确定默认网格与判定规则，
并在用户确认后复用同一份行情快照运行成本、参数或分时段扫描。
"""

from core.experiments.runner import (
    resolve_experiment_spec,
    run_experiment,
    run_validation_suite,
)

__all__ = ["resolve_experiment_spec", "run_experiment", "run_validation_suite"]
