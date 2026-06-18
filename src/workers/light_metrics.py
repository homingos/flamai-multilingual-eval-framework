"""
src/workers/light_metrics.py
=============================
LightMetricWorker — CPU-only automated metrics.
No GPU. No model loading. No Modal imports.

Modal registration: modal_app.py → LightMetricWorkerModal
(score is wrapped with @modal.method() there)
"""
from __future__ import annotations

import importlib
import json
import pkgutil
from typing import Dict, List, Optional

from src.metrics.base import BaseMetric, ComputeTier, MetricResult
from src.pipeline.run import append_output, metric_path


def _discover_light_metrics() -> List[BaseMetric]:
    """
    Auto-discovers all LIGHT-tier BaseMetric subclasses by scanning src/metrics/.
    Adding a new file to any subpackage makes it available without touching this file.
    """
    import src.metrics.translation
    import src.metrics.instruction
    import src.metrics.number_verbalization

    metrics: List[BaseMetric] = []
    seen_names: set = set()

    for package in (
        src.metrics.translation,
        src.metrics.instruction,
        src.metrics.number_verbalization,
    ):
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package.__name__}.{module_name}")
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseMetric)
                    and obj is not BaseMetric
                    and getattr(obj, "compute_tier", None) == ComputeTier.LIGHT
                    and getattr(obj, "name", "")
                    and obj.name not in seen_names
                ):
                    metrics.append(obj())
                    seen_names.add(obj.name)

    return metrics


def _load_jsonl(path: str) -> List[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def _write_result(run_id: str, metric_name: str, slug: str, result: MetricResult) -> None:
    path = metric_path(run_id, metric_name, slug)
    append_output(path, {
        "metric":       metric_name,
        "language":     result.language,
        "scores":       result.scores,
        "sample_count": result.sample_count,
        "errors":       result.errors,
        "notes":        result.notes,
    })


class LightMetricWorker:
    """
    Runs all LIGHT-tier metrics for one (model_id, task) pair.

    model_id="gemma-4-26b" → reads from gemma_output_path
    anything else          → reads from regional_output_path using slug
    """

    def _get_metrics(self) -> List[BaseMetric]:
        if not hasattr(self, "_metrics_cache"):
            self._metrics_cache = _discover_light_metrics()
        return self._metrics_cache

    def score(
        self,
        run_id: str,
        slug: str,
        task: str,
        language: str,
        model_id: str = "",
    ) -> Dict[str, dict]:
        """
        Score outputs for one model + task pair.

        Returns:
            {metric_name: scores_dict} for all applicable metrics.
        """
        from src.pipeline.run import gemma_output_path, regional_output_path

        if model_id == "gemma-4-26b":
            out_path = gemma_output_path(run_id, task)
        else:
            out_path = regional_output_path(run_id, slug, task)

        outputs = _load_jsonl(out_path)
        if not outputs:
            return {}

        results: Dict[str, MetricResult] = {}
        for metric in self._get_metrics():
            if not metric.applies_to(task):
                continue

            # For category-scoped instruction metrics, filter to that category
            if task == "instructions" and metric.category is not None:
                subset = [o for o in outputs if o.get("category") == metric.category]
            else:
                subset = outputs

            if not subset:
                continue

            result = metric.compute(subset, language=language, task=task)
            results[metric.name] = result
            _write_result(run_id, metric.name, slug, result)

        return {k: v.scores for k, v in results.items()}
