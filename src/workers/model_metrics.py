"""
src/workers/model_metrics.py
=============================
ModelMetricWorker — GPU-tier metric worker.

Runs MODEL-tier metrics (COMET, BERTScore) on inference outputs.
One instance is shared across all languages; the slug + model_id
identify which output file to load.

Modal registration: modal_app.py → ModelMetricWorkerModal
"""
from __future__ import annotations

import importlib
import json
import pkgutil
from typing import Dict, List

from src.metrics.base import BaseMetric, ComputeTier, MetricResult
from src.pipeline.run import append_output, metric_path


def _discover_model_metrics() -> List[BaseMetric]:
    """Auto-discovers MODEL-tier BaseMetric subclasses from src/metrics/."""
    import src.metrics.translation

    metrics: List[BaseMetric] = []
    seen: set = set()

    for package in (src.metrics.translation,):
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package.__name__}.{module_name}")
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseMetric)
                    and obj is not BaseMetric
                    and getattr(obj, "compute_tier", None) == ComputeTier.MODEL
                    and getattr(obj, "name", "")
                    and obj.name not in seen
                ):
                    metrics.append(obj())
                    seen.add(obj.name)

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


class ModelMetricWorker:
    """
    Runs MODEL-tier metrics (COMET, BERTScore) for one (model_id, task) pair.
    Requires GPU — registered with L4 GPU in modal_app.py.
    """

    def _get_metrics(self) -> List[BaseMetric]:
        if not hasattr(self, "_metrics_cache"):
            self._metrics_cache = _discover_model_metrics()
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
        Score outputs for one model + task pair using MODEL-tier metrics.
        Returns {metric_name: scores_dict}.
        """
        from src.pipeline.run import gemma_output_path, regional_output_path

        if model_id == "gemma-4-26b":
            out_path = gemma_output_path(run_id, task)
        else:
            out_path = regional_output_path(run_id, slug, task)

        outputs = _load_jsonl(out_path)
        if not outputs:
            print(f"[ModelMetricWorker] No outputs at {out_path}")
            return {}

        print(f"[ModelMetricWorker] Scoring {len(outputs)} outputs "
              f"for {model_id or 'gemma-4-26b'} / {task} / {language}")

        results: Dict[str, MetricResult] = {}
        for metric in self._get_metrics():
            if not metric.applies_to(task):
                continue
            print(f"[ModelMetricWorker] Running {metric.name}...")
            try:
                result = metric.compute(outputs, language=language, task=task)
                results[metric.name] = result
                _write_result(run_id, metric.name, slug, result)
                print(f"[ModelMetricWorker] {metric.name} → {result.scores}")
                if result.errors:
                    print(f"[ModelMetricWorker] {metric.name} errors: {result.errors}")
            except Exception as exc:
                print(f"[ModelMetricWorker] {metric.name} failed: {exc}")

        return {k: v.scores for k, v in results.items()}