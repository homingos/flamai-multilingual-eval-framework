"""
In-memory backend — identical interface to VolumeRegistryStore, used in tests.
Enabled via PERSISTENCE_BACKEND=memory.
"""
from __future__ import annotations

import copy
import datetime
from typing import Optional

from src.core.domain.hardware_config import HardwareConfig
from src.core.domain.metric_config import MetricConfig
from src.core.domain.model_config import ModelConfig, ModelState
from src.core.domain.task_config import TaskConfig

_DEFAULT_TASKS = [
    TaskConfig(name="translation",   active=True,  samples_per_language=2024,
               description="English ↔ target language translation (FLORES-200)"),
    TaskConfig(name="instructions",  active=True,  samples_per_language=1200,
               description="Talking Avatar instruction following across 6 categories"),
    TaskConfig(name="summarization", active=False, samples_per_language=0,
               description="Summarization task (not yet active)"),
]


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class InMemoryRegistryStore:
    def __init__(self) -> None:
        self._models: dict[str, ModelConfig] = {}
        self._hardware: dict[str, HardwareConfig] = {}
        self._metrics: dict[str, MetricConfig] = {}
        self._tasks: dict[str, TaskConfig] = {t.name: copy.deepcopy(t) for t in _DEFAULT_TASKS}
        self._runs: list[dict] = []

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    async def get_model(self, model_id: str) -> Optional[ModelConfig]:
        return copy.deepcopy(self._models.get(model_id))

    async def list_models(self, state: Optional[ModelState] = None) -> list[ModelConfig]:
        models = list(self._models.values())
        if state is not None:
            models = [m for m in models if m.state == state]
        return copy.deepcopy(models)

    async def create_model(self, model: ModelConfig) -> ModelConfig:
        self._models[model.id] = copy.deepcopy(model)
        return copy.deepcopy(model)

    async def update_model(self, model: ModelConfig) -> Optional[ModelConfig]:
        if model.id not in self._models:
            return None
        self._models[model.id] = copy.deepcopy(model)
        return copy.deepcopy(model)

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------

    async def get_hardware(self, hardware_id: str) -> Optional[HardwareConfig]:
        return copy.deepcopy(self._hardware.get(hardware_id))

    async def list_hardware(self) -> list[HardwareConfig]:
        return copy.deepcopy(list(self._hardware.values()))

    async def create_hardware(self, hardware: HardwareConfig) -> HardwareConfig:
        self._hardware[hardware.id] = copy.deepcopy(hardware)
        return copy.deepcopy(hardware)

    async def update_hardware(self, hardware: HardwareConfig) -> Optional[HardwareConfig]:
        if hardware.id not in self._hardware:
            return None
        self._hardware[hardware.id] = copy.deepcopy(hardware)
        return copy.deepcopy(hardware)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def get_metric(self, name: str) -> Optional[MetricConfig]:
        return copy.deepcopy(self._metrics.get(name))

    async def list_metrics(self, enabled_only: bool = False) -> list[MetricConfig]:
        metrics = list(self._metrics.values())
        if enabled_only:
            metrics = [m for m in metrics if m.enabled]
        return copy.deepcopy(metrics)

    async def create_metric(self, metric: MetricConfig) -> MetricConfig:
        self._metrics[metric.name] = copy.deepcopy(metric)
        return copy.deepcopy(metric)

    async def update_metric(self, metric: MetricConfig) -> Optional[MetricConfig]:
        if metric.name not in self._metrics:
            return None
        self._metrics[metric.name] = copy.deepcopy(metric)
        return copy.deepcopy(metric)

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    async def get_task(self, name: str) -> Optional[TaskConfig]:
        return copy.deepcopy(self._tasks.get(name))

    async def list_tasks(self, active_only: bool = False) -> list[TaskConfig]:
        tasks = list(self._tasks.values())
        if active_only:
            tasks = [t for t in tasks if t.active]
        return copy.deepcopy(tasks)

    async def update_task(self, task: TaskConfig) -> Optional[TaskConfig]:
        if task.name not in self._tasks:
            return None
        self._tasks[task.name] = copy.deepcopy(task)
        return copy.deepcopy(task)

    # ------------------------------------------------------------------
    # Runs index
    # ------------------------------------------------------------------

    async def append_run(self, run_id: str, status: str) -> None:
        self._runs.append({"run_id": run_id, "status": status, "created_at": _now_iso()})

    async def list_runs(self) -> list[dict]:
        return copy.deepcopy(self._runs)

    async def update_run_status(self, run_id: str, status: str) -> None:
        for entry in self._runs:
            if entry["run_id"] == run_id:
                entry["status"] = status
                return
