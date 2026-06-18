"""
src/metrics/translation/bertscore.py
======================================
BERTScore translation quality metric.
compute_tier = MODEL — runs on GPU in ModelMetricWorker.
Requires: bert-score  (installed in model_metrics_image, not in the dev environment)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage

if TYPE_CHECKING:
    pass  # bert_score is a GPU-only dep; imported lazily at runtime below


class BERTScoreMetric(BaseMetric):
    name         = "bertscore"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.MODEL
    task_types   = ["translation"]
    category     = None

    def compute(self, outputs, language, task):
        try:
            from bert_score import score as bert_score  # type: ignore[import-untyped]
        except ImportError:
            return MetricResult(
                metric_name="bertscore", language=language, task=task,
                scores={}, sample_count=0,
                errors=["bert-score not installed"],
            )

        en_to_tgt = [o for o in outputs if o.get("direction") == "en→target"]
        tgt_to_en = [o for o in outputs if o.get("direction") == "target→en"]

        scores = {}
        errors = []

        for direction_key, subset in [("en_to_target", en_to_tgt), ("target_to_en", tgt_to_en)]:
            if not subset:
                continue
            try:
                hyps = [o["output"] or "" for o in subset]
                refs = [o["reference"] for o in subset]
                # Use multilingual model; rescale_with_baseline=True normalises scores
                P, R, F1 = bert_score(
                    hyps, refs,
                    model_type="microsoft/mdeberta-v3-base",
                    lang=None,
                    rescale_with_baseline=False,
                    device="cuda",
                    verbose=False,
                )
                scores[f"bertscore_f1_{direction_key}"] = round(float(F1.mean()), 4)
                scores[f"bertscore_p_{direction_key}"]  = round(float(P.mean()),  4)
                scores[f"bertscore_r_{direction_key}"]  = round(float(R.mean()),  4)
            except Exception as exc:
                errors.append(f"{direction_key}: {exc}")

        return MetricResult(
            metric_name="bertscore", language=language, task=task,
            scores=scores, sample_count=len(outputs), errors=errors,
        )