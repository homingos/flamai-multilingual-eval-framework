import re

from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage
from src.metrics.number_verbalization import get_rule


class DigitLeakageMetric(BaseMetric):
    """Mean raw digit characters remaining in output. Skips digit-by-digit samples."""

    name         = "digit_leakage"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.LIGHT
    task_types   = ["instructions"]
    category     = "number_verbalization"

    def compute(self, outputs, language, task):
        leakage_counts = []
        for o in outputs:
            if get_rule(o) == "digit_by_digit":
                continue   # individual digits are expected there
            count = len(re.findall(r"\d", o.get("output", "")))
            leakage_counts.append(count)

        mean = round(sum(leakage_counts) / len(leakage_counts), 2) if leakage_counts else 0.0
        return MetricResult(
            metric_name="digit_leakage", language=language, task=task,
            scores={"mean_digit_leakage": mean},
            sample_count=len(leakage_counts),
        )
