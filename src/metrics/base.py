"""
src/metrics/base.py
===================
Abstract base class for all Phase 2A evaluation metrics.
No Modal imports. No network calls. Pure Python computation.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ComputeTier(str, Enum):
    LIGHT = "light"   # CPU-only, rule-based — runs in LightMetricWorker
    MODEL = "model"   # Requires GPU model — runs in ModelMetricWorker (Phase 4)


class MetricStage(str, Enum):
    POST_INFERENCE = "post_inference"  # fires after all outputs are written
    POST_JUDGE     = "post_judge"      # fires after judge verdicts are written


@dataclass
class MetricResult:
    metric_name: str
    language: str
    task: str
    scores: Dict[str, Any]         # e.g. {"bleu_en_to_target": 32.4}
    sample_count: int
    notes: Optional[str] = None
    errors: List[str] = field(default_factory=list)


class BaseMetric(abc.ABC):
    """
    Subclasses must set these class attributes:
        name:         str            — unique metric name e.g. "bleu"
        stage:        MetricStage   — when this metric fires
        compute_tier: ComputeTier   — LIGHT or MODEL
        task_types:   list[str]     — ["translation"] / ["instructions"] / ["all"]
        category:     str | None    — instruction sub-category filter or None
    """

    name: str = ""
    stage: MetricStage = MetricStage.POST_INFERENCE
    compute_tier: ComputeTier = ComputeTier.LIGHT
    task_types: List[str] = []
    category: Optional[str] = None

    @abc.abstractmethod
    def compute(
        self,
        outputs: List[dict],
        language: str,
        task: str,
    ) -> MetricResult:
        """
        Compute metric over a list of output record dicts.

        Translation records have: prompt_id, output, source, reference, direction.
        Instruction records have: prompt_id, output, category, system_instruction,
                                  user_prompt, expected_constraints.

        Must not raise — catch exceptions internally and append to MetricResult.errors.
        """

    def applies_to(self, task: str, category: Optional[str] = None) -> bool:
        """
        Returns True if this metric should run for the given task.

        category is optional and used two different ways depending on caller:
          - Pass a specific category to check exact match against self.category
            (e.g. "does this metric apply to THIS sample's category?").
          - Omit it (default None) to check task-level applicability only —
            this is what LightMetricWorker does, since it performs its own
            category filtering on the output list afterward rather than
            asking applies_to to do it. Without this distinction, every
            category-scoped metric (format_compliance, currency_unit,
            topic_boundary, tone_register, ...) would be silently skipped
            by any caller that doesn't pass category — which is exactly
            what happened before this fix.
        """
        task_ok = "all" in self.task_types or task in self.task_types
        if not task_ok:
            return False
        if category is not None and self.category is not None and category != self.category:
            return False
        return True