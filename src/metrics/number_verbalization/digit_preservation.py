import re

from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage
from src.metrics.number_verbalization import get_numbers


class DigitPreservationMetric(BaseMetric):
    """All input digits must appear somewhere in the output (none dropped or hallucinated)."""

    name         = "digit_preservation"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.LIGHT
    task_types   = ["instructions"]
    category     = "number_verbalization"

    def compute(self, outputs, language, task):
        correct = total = 0
        for o in outputs:
            numbers = get_numbers(o)
            text = o.get("output", "")
            total += 1
            input_digits  = "".join(re.findall(r"\d", "".join(numbers)))
            output_digits = "".join(re.findall(r"\d", text))
            if all(d in output_digits for d in input_digits):
                correct += 1

        rate = round(correct / total, 4) if total > 0 else 0.0
        return MetricResult(
            metric_name="digit_preservation", language=language, task=task,
            scores={"digit_preservation_accuracy": rate},
            sample_count=total,
        )
