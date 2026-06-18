from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage
from src.metrics.number_verbalization import get_numbers, get_rule


class MixedSentenceMetric(BaseMetric):
    """For rule='mixed': all numbers must be handled correctly (no raw digits in output)."""

    name         = "mixed_sentence"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.LIGHT
    task_types   = ["instructions"]
    category     = "number_verbalization"

    def compute(self, outputs, language, task):
        mixed = [o for o in outputs if get_rule(o) == "mixed"]
        if not mixed:
            return MetricResult(
                metric_name="mixed_sentence", language=language, task=task,
                scores={"mixed_sentence_consistency": None}, sample_count=0,
                notes="No mixed-type samples in this batch",
            )

        correct = sum(
            1 for o in mixed
            if not any(num in o.get("output", "") for num in get_numbers(o))
        )
        rate = round(correct / len(mixed), 4)
        return MetricResult(
            metric_name="mixed_sentence", language=language, task=task,
            scores={"mixed_sentence_consistency": rate},
            sample_count=len(mixed),
        )
