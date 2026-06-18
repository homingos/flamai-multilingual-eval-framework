from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage


class NumberLanguageMetric(BaseMetric):
    """Verbalized number words should be in the target language, not English."""

    name         = "number_language"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.LIGHT
    task_types   = ["instructions"]
    category     = "number_verbalization"

    ENGLISH_NUMBER_WORDS = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
        "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "hundred", "thousand", "million", "billion",
    }

    def compute(self, outputs, language, task):
        correct = total = 0
        for o in outputs:
            text = o.get("output", "").lower()
            total += 1
            if not set(text.split()).intersection(self.ENGLISH_NUMBER_WORDS):
                correct += 1

        rate = round(correct / total, 4) if total > 0 else 0.0
        return MetricResult(
            metric_name="number_language", language=language, task=task,
            scores={"language_of_number_words": rate},
            sample_count=total,
        )
