import re

from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage

FORMAT_PATTERNS = {
    "numbered_list": re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE),
    "bullet_points": re.compile(r"^\s*[-•*]\s+", re.MULTILINE),
    "headers":       re.compile(r"^#+\s+|^[A-Z][A-Za-z ]+:\s*$", re.MULTILINE),
    "qa_format":     re.compile(r"^(Q:|Question:|A:|Answer:)", re.MULTILINE | re.IGNORECASE),
}


class FormatComplianceMetric(BaseMetric):
    name         = "format_compliance"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.LIGHT
    task_types   = ["instructions"]
    category     = "structured_output"

    def compute(self, outputs, language, task):
        correct = total = 0
        for o in outputs:
            fmt = o.get("expected_constraints", {}).get("format", "")
            text = o.get("output", "")
            pattern = FORMAT_PATTERNS.get(fmt)
            if pattern is None:
                continue   # unknown format — skip entirely
            total += 1
            if pattern.search(text):
                correct += 1

        rate = round(correct / total, 4) if total > 0 else 0.0
        return MetricResult(
            metric_name="format_compliance", language=language, task=task,
            scores={"format_compliance": rate, "correct": correct, "total": total},
            sample_count=len(outputs),
        )
