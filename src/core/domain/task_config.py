from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskConfig:
    name: str                          # e.g. "translation"
    active: bool = True
    samples_per_language: int = 0
    description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "active": self.active,
            "samples_per_language": self.samples_per_language,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskConfig":
        return cls(
            name=d["name"],
            active=d.get("active", True),
            samples_per_language=d.get("samples_per_language", 0),
            description=d.get("description"),
        )
