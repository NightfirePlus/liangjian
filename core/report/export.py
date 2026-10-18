"""导出报告与逐结论证据附件，图表在离线浏览器中可打开。"""
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from core.report.evidence import chart_for, dumps


def evidence_filename(eid, entry):
    suffix = eid.split(":", 1)[1].replace(":", "_")
    ext = "html" if entry["kind"] == "chart" else "csv" if entry["kind"] == "table" else "json"
    return f"evidence/{suffix}.{ext}"


def export_bundle(pack, report):
    if report["run_id"] != pack["run_id"] or report["evidence_hash"] != pack["evidence_hash"]:
        raise ValueError("报告与证据不属于同一次运行")
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("report.json", dumps(report))
        archive.writestr("evidence.json", dumps(pack))
        lines = ["# 量见实验报告", "", report["notice"], ""]
        context = pack["entries"][f"{pack['run_id']}:context"]["data"]
        if context.get("snapshot", {}).get("is_synthetic"):
            lines += ["**本报告使用模拟数据，仅用于流程演示。**", ""]
        lines += ["实验范围和限制：", "", *[f"- {s}" for s in context.get("limitations", [])], ""]
        for finding in report["findings"]:
            lines += ["## " + finding["title"], "", finding["statement"], ""]
            lines += [f"- 局限：{s}" for s in finding["limitations"]]
            for eid in finding["evidence_refs"]:
                entry = pack["entries"][eid]
                lines.append(f"- 证据：[{entry['title']}]({evidence_filename(eid, entry)})（{eid}）")
            lines.append("")
        archive.writestr("report.md", "\n".join(lines))
        for eid, entry in pack["entries"].items():
            if entry["kind"] == "chart":
                content = chart_for(pack, eid).to_html(include_plotlyjs=True, full_html=True)
            elif entry["kind"] == "table":
                content = pd.DataFrame(entry["data"]).to_csv(index=False).encode("utf-8-sig")
            else:
                content = dumps(entry)
            archive.writestr(evidence_filename(eid, entry), content)
    return buffer.getvalue()
