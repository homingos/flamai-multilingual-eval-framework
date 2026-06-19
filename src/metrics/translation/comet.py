"""
src/metrics/translation/comet.py
===================================
T3 — COMET translation quality metric.

compute_tier = MODEL — runs on GPU in ModelMetricWorker, same as
BERTScore and back-translation.

Unlike T5 (back-translation), COMET needs no per-language model
selection — Unbabel/wmt22-comet-da is a single multilingual checkpoint
built on XLM-R that scores any (source, hypothesis, reference) triplet
regardless of language, so there's no coverage gap to document here.

Requires: unbabel-comet (installed in model_metrics_image, not the dev
environment — imported lazily so this file still loads without it).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage

if TYPE_CHECKING:
    pass  # comet is a GPU-only dep; imported lazily at runtime below

_COMET_MODEL_ID = "Unbabel/wmt22-comet-da"

# Module-level cache — the checkpoint is several hundred MB; load once per
# worker process, not once per compute() call. Mirrors the pattern in
# back_translation.py's _model_cache.
_model_cache = None


def _get_comet_model():
    global _model_cache
    if _model_cache is None:
        from comet import download_model, load_from_checkpoint  # type: ignore[import-untyped]
        model_path = download_model(_COMET_MODEL_ID)
        _model_cache = load_from_checkpoint(model_path)
    return _model_cache


class COMETMetric(BaseMetric):
    name         = "comet"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.MODEL
    task_types   = ["translation"]
    category     = None

    def compute(self, outputs, language, task):
        try:
            model = _get_comet_model()
        except ImportError:
            return MetricResult(
                metric_name="comet", language=language, task=task,
                scores={}, sample_count=0,
                errors=["unbabel-comet not installed"],
            )
        except Exception as exc:
            return MetricResult(
                metric_name="comet", language=language, task=task,
                scores={}, sample_count=0,
                errors=[f"Failed to load {_COMET_MODEL_ID}: {exc}"],
            )

        en_to_tgt = [o for o in outputs if o.get("direction") == "en→target"]
        tgt_to_en = [o for o in outputs if o.get("direction") == "target→en"]

        scores = {}
        errors = []

        for direction_key, subset in [("en_to_target", en_to_tgt), ("target_to_en", tgt_to_en)]:
            if not subset:
                continue

            # COMET needs all three of source, hypothesis, reference —
            # skip any sample missing one rather than letting predict()
            # choke on a None partway through a batch.
            data = []
            skipped = 0
            for o in subset:
                src = o.get("source")
                mt  = o.get("output")
                ref = o.get("reference")
                if not src or not mt or not ref:
                    skipped += 1
                    continue
                data.append({"src": src, "mt": mt, "ref": ref})

            if not data:
                errors.append(f"{direction_key}: no samples with source+output+reference")
                continue

            try:
                result = model.predict(data, batch_size=8, gpus=1)
                # result.scores is per-segment; result.system_score is the
                # corpus-level mean — same shape sacrebleu's corpus_bleu gives us.
                scores[f"comet_{direction_key}"] = round(float(result.system_score), 4)
                if skipped:
                    errors.append(f"{direction_key}: skipped {skipped} samples missing src/mt/ref")
            except Exception as exc:
                errors.append(f"{direction_key}: {exc}")

        return MetricResult(
            metric_name="comet", language=language, task=task,
            scores=scores, sample_count=len(outputs), errors=errors,
        )