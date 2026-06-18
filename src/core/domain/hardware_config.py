from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HardwareConfig:
    id: str               # e.g. "l4"
    gpu: str              # Modal GPU string e.g. "L4"
    vram_gb: int          # e.g. 24
    cost_per_hr: float    # e.g. 0.80
    use_for: str          # description of intended use

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "gpu": self.gpu,
            "vram_gb": self.vram_gb,
            "cost_per_hr": self.cost_per_hr,
            "use_for": self.use_for,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HardwareConfig":
        return cls(
            id=d["id"],
            gpu=d["gpu"],
            vram_gb=d["vram_gb"],
            cost_per_hr=d["cost_per_hr"],
            use_for=d["use_for"],
        )
