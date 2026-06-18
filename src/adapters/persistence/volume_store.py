"""
Production backend — reads/writes to Modal volume at /data/registry/.

All writes are atomic (temp-file then rename). All store methods are async
to satisfy the RegistryStore Protocol; they do synchronous file I/O internally
(Modal containers are single-threaded, so this is fine).
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Optional

from src.core.domain.hardware_config import HardwareConfig
from src.core.domain.metric_config import MetricConfig
from src.core.domain.model_config import ModelConfig, ModelState
from src.core.domain.task_config import TaskConfig

REGISTRY_ROOT = "/data/registry"

_DEFAULT_TASKS = [
    {"name": "translation",   "active": True,  "samples_per_language": 2024,
     "description": "English ↔ target language translation (FLORES-200)"},
    {"name": "instructions",  "active": True,  "samples_per_language": 1200,
     "description": "Talking Avatar instruction following across 6 categories"},
    {"name": "summarization", "active": False, "samples_per_language": 0,
     "description": "Summarization task (not yet active)"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _path(filename: str) -> str:
    return os.path.join(REGISTRY_ROOT, filename)


def _atomic_write(path: str, data: dict | list) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.rename(tmp, path)


def _read_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# VolumeRegistryStore
# ---------------------------------------------------------------------------

class VolumeRegistryStore:
    def __init__(self) -> None:
        os.makedirs(REGISTRY_ROOT, exist_ok=True)
        self._init_tasks()

    def _init_tasks(self) -> None:
        path = _path("tasks.json")
        if not os.path.exists(path):
            _atomic_write(path, _DEFAULT_TASKS)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    async def get_model(self, model_id: str) -> Optional[ModelConfig]:
        models = _read_json(_path("models.json"), [])
        for d in models:
            if d["id"] == model_id:
                return ModelConfig.from_dict(d)
        return None

    async def list_models(self, state: Optional[ModelState] = None) -> list[ModelConfig]:
        models = _read_json(_path("models.json"), [])
        result = [ModelConfig.from_dict(d) for d in models]
        if state is not None:
            result = [m for m in result if m.state == state]
        return result

    async def create_model(self, model: ModelConfig) -> ModelConfig:
        models = _read_json(_path("models.json"), [])
        models.append(model.to_dict())
        _atomic_write(_path("models.json"), models)
        return model

    async def update_model(self, model: ModelConfig) -> Optional[ModelConfig]:
        models = _read_json(_path("models.json"), [])
        for i, d in enumerate(models):
            if d["id"] == model.id:
                models[i] = model.to_dict()
                _atomic_write(_path("models.json"), models)
                return model
        return None

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------

    async def get_hardware(self, hardware_id: str) -> Optional[HardwareConfig]:
        items = _read_json(_path("hardware.json"), [])
        for d in items:
            if d["id"] == hardware_id:
                return HardwareConfig.from_dict(d)
        return None

    async def list_hardware(self) -> list[HardwareConfig]:
        items = _read_json(_path("hardware.json"), [])
        return [HardwareConfig.from_dict(d) for d in items]

    async def create_hardware(self, hardware: HardwareConfig) -> HardwareConfig:
        items = _read_json(_path("hardware.json"), [])
        items.append(hardware.to_dict())
        _atomic_write(_path("hardware.json"), items)
        return hardware

    async def update_hardware(self, hardware: HardwareConfig) -> Optional[HardwareConfig]:
        items = _read_json(_path("hardware.json"), [])
        for i, d in enumerate(items):
            if d["id"] == hardware.id:
                items[i] = hardware.to_dict()
                _atomic_write(_path("hardware.json"), items)
                return hardware
        return None

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def get_metric(self, name: str) -> Optional[MetricConfig]:
        items = _read_json(_path("metrics.json"), [])
        for d in items:
            if d["name"] == name:
                return MetricConfig.from_dict(d)
        return None

    async def list_metrics(self, enabled_only: bool = False) -> list[MetricConfig]:
        items = _read_json(_path("metrics.json"), [])
        result = [MetricConfig.from_dict(d) for d in items]
        if enabled_only:
            result = [m for m in result if m.enabled]
        return result

    async def create_metric(self, metric: MetricConfig) -> MetricConfig:
        items = _read_json(_path("metrics.json"), [])
        items.append(metric.to_dict())
        _atomic_write(_path("metrics.json"), items)
        return metric

    async def update_metric(self, metric: MetricConfig) -> Optional[MetricConfig]:
        items = _read_json(_path("metrics.json"), [])
        for i, d in enumerate(items):
            if d["name"] == metric.name:
                items[i] = metric.to_dict()
                _atomic_write(_path("metrics.json"), items)
                return metric
        return None

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    async def get_task(self, name: str) -> Optional[TaskConfig]:
        items = _read_json(_path("tasks.json"), [])
        for d in items:
            if d["name"] == name:
                return TaskConfig.from_dict(d)
        return None

    async def list_tasks(self, active_only: bool = False) -> list[TaskConfig]:
        items = _read_json(_path("tasks.json"), [])
        result = [TaskConfig.from_dict(d) for d in items]
        if active_only:
            result = [t for t in result if t.active]
        return result

    async def update_task(self, task: TaskConfig) -> Optional[TaskConfig]:
        items = _read_json(_path("tasks.json"), [])
        for i, d in enumerate(items):
            if d["name"] == task.name:
                items[i] = task.to_dict()
                _atomic_write(_path("tasks.json"), items)
                return task
        return None

    # ------------------------------------------------------------------
    # Runs index
    # ------------------------------------------------------------------

    async def append_run(self, run_id: str, status: str) -> None:
        runs = _read_json(_path("runs_index.json"), [])
        runs.append({"run_id": run_id, "status": status, "created_at": _now_iso()})
        _atomic_write(_path("runs_index.json"), runs)

    async def list_runs(self) -> list[dict]:
        return _read_json(_path("runs_index.json"), [])

    async def update_run_status(self, run_id: str, status: str) -> None:
        runs = _read_json(_path("runs_index.json"), [])
        for entry in runs:
            if entry["run_id"] == run_id:
                entry["status"] = status
                break
        _atomic_write(_path("runs_index.json"), runs)
