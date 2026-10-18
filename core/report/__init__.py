"""结构化实验 → 证据图表 → 可引用报告。"""
from core.report.evidence import build_evidence, chart_for
from core.report.analysis import generate_report
from core.report.export import export_bundle

__all__ = ["build_evidence", "chart_for", "generate_report", "export_bundle"]
