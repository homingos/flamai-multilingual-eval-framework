"""
src/metrics/instruction/topic_boundary.py
============================================
I4 — Topic Boundary Respect.

Applies to: topic_boundary category.
Checks whether the model stays on-topic relative to the user's prompt,
using embedding cosine similarity. A response that wanders into
speculation or unrelated content will have low similarity to the
original question.

Library: sentence-transformers, paraphrase-multilingual-MiniLM-L12-v2
         (multilingual — works across all 17 evaluation languages
         without needing a per-language model).
Threshold: cosine similarity < 0.3 flags a potential boundary violation
           (per docs/evaluation-metrics.md §2.4).

LIGHT tier: this model is small (~118M params) and runs on CPU in a
few hundred ms/sample, well within LightMetricWorker's CPU container —
no GPU needed despite being a model-based check.
"""
from __future__ import annotations

from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage

SIMILARITY_THRESHOLD = 0.3
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Module-level cache — the model is ~470MB; load once per worker process,
# not once per compute() call. LightMetricWorker reuses one process across
# many score() calls in the same container, so this actually pays off.
_model_cache = None


def _get_model():
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer(_MODEL_NAME)
    return _model_cache


class TopicBoundaryMetric(BaseMetric):
    name         = "topic_boundary"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.LIGHT
    task_types   = ["instructions"]
    category     = "topic_boundary"

    def compute(self, outputs, language, task):
        try:
            model = _get_model()
        except ImportError:
            return MetricResult(
                metric_name="topic_boundary", language=language, task=task,
                scores={}, sample_count=0,
                errors=["sentence-transformers not installed"],
            )

        prompts, responses, valid_indices = [], [], []
        for i, o in enumerate(outputs):
            prompt = o.get("user_prompt", "")
            response = o.get("output", "")
            if not prompt.strip() or not response.strip():
                continue
            prompts.append(prompt)
            responses.append(response)
            valid_indices.append(i)

        if not prompts:
            return MetricResult(
                metric_name="topic_boundary", language=language, task=task,
                scores={"topic_boundary_compliance": 0.0}, sample_count=0,
                notes="No samples with both a non-empty prompt and response",
            )

        try:
            import numpy as np

            prompt_embs   = model.encode(prompts,   convert_to_numpy=True, show_progress_bar=False)
            response_embs = model.encode(responses, convert_to_numpy=True, show_progress_bar=False)

            # Cosine similarity per pair: (a·b) / (|a||b|)
            dot    = (prompt_embs * response_embs).sum(axis=1)
            norm_p = np.linalg.norm(prompt_embs,   axis=1)
            norm_r = np.linalg.norm(response_embs, axis=1)
            sims   = dot / (norm_p * norm_r + 1e-10)

        except Exception as exc:
            return MetricResult(
                metric_name="topic_boundary", language=language, task=task,
                scores={}, sample_count=0,
                errors=[f"Embedding computation failed: {exc}"],
            )

        on_topic_count = int((sims >= SIMILARITY_THRESHOLD).sum())
        total = len(sims)
        avg_similarity = float(sims.mean())

        return MetricResult(
            metric_name="topic_boundary", language=language, task=task,
            scores={
                "topic_boundary_compliance": round(on_topic_count / total, 4),
                "avg_cosine_similarity":     round(avg_similarity, 4),
                "on_topic_count":            on_topic_count,
                "total":                     total,
            },
            sample_count=len(outputs),
            notes=(
                f"{len(outputs) - total} samples skipped (empty prompt or response)"
                if total < len(outputs) else None
            ),
        )