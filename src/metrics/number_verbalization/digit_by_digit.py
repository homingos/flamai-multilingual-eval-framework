from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage
from src.metrics.number_verbalization import get_numbers, should_be_digit_by_digit


class DigitByDigitMetric(BaseMetric):
    name         = "digit_by_digit"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.LIGHT
    task_types   = ["instructions"]
    category     = "number_verbalization"

    def compute(self, outputs, language, task):
        correct = total = 0
        for o in outputs:
            if not should_be_digit_by_digit(o):
                continue
            numbers = get_numbers(o)
            text = o.get("output", "")
            total += 1
            # Pass: raw number sequence does NOT appear unbroken in output
            if not any(num in text for num in numbers):
                correct += 1

        rate = round(correct / total, 4) if total > 0 else 0.0
        return MetricResult(
            metric_name="digit_by_digit", language=language, task=task,
            scores={"digit_by_digit_compliance": rate},
            sample_count=total,
        )
