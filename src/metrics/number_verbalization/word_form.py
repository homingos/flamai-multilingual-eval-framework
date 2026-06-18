import re

from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage
from src.metrics.number_verbalization import get_numbers, should_be_word_form


class WordFormMetric(BaseMetric):
    name         = "word_form"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.LIGHT
    task_types   = ["instructions"]
    category     = "number_verbalization"

    def compute(self, outputs, language, task):
        correct = total = 0
        for o in outputs:
            if not should_be_word_form(o):
                continue
            numbers = get_numbers(o)
            text = o.get("output", "")
            total += 1
            # Pass: no raw number string appears in output
            if not any(re.search(r"\d", num) and num in text for num in numbers):
                correct += 1

        rate = round(correct / total, 4) if total > 0 else 0.0
        return MetricResult(
            metric_name="word_form", language=language, task=task,
            scores={"word_form_compliance": rate},
            sample_count=total,
        )
