import re

from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage


class LengthAccuracyMetric(BaseMetric):
    name         = "length_accuracy"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.LIGHT
    task_types   = ["instructions"]
    category     = "length_constraint"

    def compute(self, outputs, language, task):
        correct = total = 0
        for o in outputs:
            constraints = o.get("expected_constraints", {})
            text = o.get("output", "")
            total += 1
            ok = True

            max_sentences = constraints.get("max_sentences")
            if max_sentences is not None:
                sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
                if len(sentences) > max_sentences:
                    ok = False

            max_words = constraints.get("max_words")
            if max_words is not None:
                if len(text.split()) > max_words:
                    ok = False

            if ok:
                correct += 1

        rate = round(correct / total, 4) if total > 0 else 0.0
        return MetricResult(
            metric_name="length_accuracy", language=language, task=task,
            scores={"length_accuracy": rate, "correct": correct, "total": total},
            sample_count=len(outputs),
        )
