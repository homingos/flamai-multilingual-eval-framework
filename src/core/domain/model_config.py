from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelState(str, Enum):
    ACTIVE     = "active"
    DISABLED   = "disabled"
    DEPRECATED = "deprecated"


@dataclass
class ModelConfig:
    id: str                          # slug e.g. "tamil-mistral-7b"
    name: str                        # display name e.g. "Tamil-Mistral-7B"
    hf_model_id: str                 # HuggingFace repo e.g. "abhinand/tamil-mistral-7b"
    language: str                    # e.g. "Tamil"
    slug: str                        # dataset folder name e.g. "tamil"
    region: str                      # e.g. "Indic"
    gpu_preset: str                  # key into GPU_PRESETS e.g. "l4"
    params_billions: float           # e.g. 7.0
    state: ModelState = ModelState.ACTIVE
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.88
    max_model_len: int = 2048
    notes: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "hf_model_id": self.hf_model_id,
            "language": self.language,
            "slug": self.slug,
            "region": self.region,
            "gpu_preset": self.gpu_preset,
            "params_billions": self.params_billions,
            "state": self.state.value,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(
            id=d["id"],
            name=d["name"],
            hf_model_id=d["hf_model_id"],
            language=d["language"],
            slug=d["slug"],
            region=d["region"],
            gpu_preset=d["gpu_preset"],
            params_billions=d["params_billions"],
            state=ModelState(d.get("state", "active")),
            dtype=d.get("dtype", "bfloat16"),
            gpu_memory_utilization=d.get("gpu_memory_utilization", 0.88),
            max_model_len=d.get("max_model_len", 2048),
            notes=d.get("notes"),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
        )
