from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage


class ChrFMetric(BaseMetric):
    name         = "chrf"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.LIGHT
    task_types   = ["translation"]
    category     = None

    def compute(self, outputs, language, task):
        import sacrebleu

        en_to_tgt = [o for o in outputs if o.get("direction") == "en→target"]
        tgt_to_en = [o for o in outputs if o.get("direction") == "target→en"]

        scores = {}
        errors = []

        for direction, subset in [("en_to_target", en_to_tgt), ("target_to_en", tgt_to_en)]:
            if not subset:
                continue
            try:
                hypotheses = [o["output"] for o in subset]
                references = [[o["reference"] for o in subset]]
                result = sacrebleu.corpus_chrf(hypotheses, references, word_order=2)
                scores[f"chrf_{direction}"] = round(result.score, 2)
            except Exception as exc:
                errors.append(f"{direction}: {exc}")

        return MetricResult(
            metric_name="chrf", language=language, task=task,
            scores=scores, sample_count=len(outputs), errors=errors,
            notes="Primary metric for Indic/Arabic/script-heavy languages (chrF++)",
        )
