"""结构化实验 → 证据图表 → 可引用报告。"""
from core.report.evidence import build_evidence, chart_for
from core.report.analysis import generate_report
from core.report.export import export_bundle
from core.report.qa import answer_report_question, format_report_answer

__all__ = [
    "answer_report_question", "build_evidence", "chart_for", "export_bundle",
    "format_report_answer", "generate_report",
]
