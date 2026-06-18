from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MetricStage(str, Enum):
    POST_INFERENCE = "post_inference"
    POST_JUDGE     = "post_judge"


class ComputeTier(str, Enum):
    MODEL = "model"    # runs on L4 GPU
    LIGHT = "light"    # runs on CPU


@dataclass
class MetricConfig:
    name: str                          # unique identifier e.g. "bleu"
    stage: MetricStage
    compute_tier: ComputeTier
    task_types: list                   # ["translation"] / ["instructions"] / ["all"]
    category: Optional[str]            # instruction sub-category filter e.g. "number_verbalization"
    enabled: bool = True
    weight: float = 1.0                # report weight
    description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "stage": self.stage.value,
            "compute_tier": self.compute_tier.value,
            "task_types": self.task_types,
            "category": self.category,
            "enabled": self.enabled,
            "weight": self.weight,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MetricConfig":
        return cls(
            name=d["name"],
            stage=MetricStage(d["stage"]),
            compute_tier=ComputeTier(d["compute_tier"]),
            task_types=d["task_types"],
            category=d.get("category"),
            enabled=d.get("enabled", True),
            weight=d.get("weight", 1.0),
            description=d.get("description"),
        )
