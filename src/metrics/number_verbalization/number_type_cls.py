from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage
from src.metrics.number_verbalization import get_numbers, get_rule


class NumberTypeClassificationMetric(BaseMetric):
    """Composite of N1+N2: correct rule applied for the declared number type."""

    name         = "number_type_cls"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.LIGHT
    task_types   = ["instructions"]
    category     = "number_verbalization"

    def compute(self, outputs, language, task):
        correct = total = 0
        for o in outputs:
            rule        = get_rule(o)
            numbers     = get_numbers(o)
            text        = o.get("output", "")
            constraints = o.get("expected_constraints", {})
            total += 1

            if rule in ("digit_by_digit", "word_form"):
                # Pass: raw number sequence does not appear in output
                if not any(num in text for num in numbers):
                    correct += 1
            elif rule == "mixed":
                dbd_types = set(constraints.get("digit_by_digit_types", []))
                wf_types  = set(constraints.get("word_form_types", []))
                num_type  = constraints.get("number_type", "")
                # Pass if number is handled per its declared rule
                dbd_ok = num_type in dbd_types and not any(n in text for n in numbers)
                wf_ok  = num_type in wf_types  and not any(n in text for n in numbers)
                if dbd_ok or wf_ok:
                    correct += 1

        rate = round(correct / total, 4) if total > 0 else 0.0
        return MetricResult(
            metric_name="number_type_cls", language=language, task=task,
            scores={"number_type_classification_accuracy": rate},
            sample_count=total,
        )
