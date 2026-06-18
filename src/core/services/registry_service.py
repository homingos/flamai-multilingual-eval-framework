"""
Business logic layer — depends only on RegistryStore port.
No HTTP, no Modal, no file I/O.
"""
from __future__ import annotations

import datetime
import random
import string
from typing import Optional

from src.core.domain.hardware_config import HardwareConfig
from src.core.domain.metric_config import MetricConfig, MetricStage, ComputeTier
from src.core.domain.model_config import ModelConfig, ModelState
from src.core.domain.task_config import TaskConfig
from src.core.ports.registry_store import RegistryStore
from src.core.services.state_machine import validate_transition


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _generate_run_id() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp = now.strftime("%Y-%m-%d_%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"{stamp}_{suffix}"


class RegistryService:
    def __init__(self, store: RegistryStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    async def create_model(self, data: dict) -> ModelConfig:
        existing = await self._store.get_model(data["id"])
        if existing is not None:
            raise ValueError(f"Model '{data['id']}' already exists.")
        now = _now_iso()
        model = ModelConfig(
            id=data["id"],
            name=data["name"],
            hf_model_id=data["hf_model_id"],
            language=data["language"],
            slug=data["slug"],
            region=data["region"],
            gpu_preset=data["gpu_preset"],
            params_billions=data["params_billions"],
            state=ModelState.ACTIVE,
            dtype=data.get("dtype", "bfloat16"),
            gpu_memory_utilization=data.get("gpu_memory_utilization", 0.88),
            max_model_len=data.get("max_model_len", 2048),
            notes=data.get("notes"),
            created_at=now,
            updated_at=now,
        )
        return await self._store.create_model(model)

    async def get_model(self, model_id: str) -> Optional[ModelConfig]:
        return await self._store.get_model(model_id)

    async def list_models(self, state: Optional[str] = None) -> list[ModelConfig]:
        if state == "all":
            return await self._store.list_models(state=None)
        if state is not None:
            return await self._store.list_models(state=ModelState(state))
        # default: active only
        return await self._store.list_models(state=ModelState.ACTIVE)

    async def update_model(self, model_id: str, updates: dict) -> Optional[ModelConfig]:
        model = await self._store.get_model(model_id)
        if model is None:
            return None
        allowed = {"gpu_preset", "dtype", "gpu_memory_utilization", "max_model_len", "notes"}
        for key, value in updates.items():
            if key in allowed and value is not None:
                setattr(model, key, value)
        model.updated_at = _now_iso()
        return await self._store.update_model(model)

    async def disable_model(self, model_id: str) -> Optional[ModelConfig]:
        return await self._transition_model(model_id, ModelState.DISABLED)

    async def enable_model(self, model_id: str) -> Optional[ModelConfig]:
        return await self._transition_model(model_id, ModelState.ACTIVE)

    async def deprecate_model(self, model_id: str) -> Optional[ModelConfig]:
        return await self._transition_model(model_id, ModelState.DEPRECATED)

    async def _transition_model(
        self, model_id: str, target: ModelState
    ) -> Optional[ModelConfig]:
        model = await self._store.get_model(model_id)
        if model is None:
            return None
        validate_transition(model.state, target)  # raises ValueError on invalid
        model.state = target
        model.updated_at = _now_iso()
        return await self._store.update_model(model)

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------

    async def create_hardware(self, data: dict) -> HardwareConfig:
        existing = await self._store.get_hardware(data["id"])
        if existing is not None:
            raise ValueError(f"Hardware '{data['id']}' already exists.")
        hw = HardwareConfig(
            id=data["id"],
            gpu=data["gpu"],
            vram_gb=data["vram_gb"],
            cost_per_hr=data["cost_per_hr"],
            use_for=data["use_for"],
        )
        return await self._store.create_hardware(hw)

    async def list_hardware(self) -> list[HardwareConfig]:
        return await self._store.list_hardware()

    async def get_hardware(self, hardware_id: str) -> Optional[HardwareConfig]:
        return await self._store.get_hardware(hardware_id)

    async def update_hardware(
        self, hardware_id: str, updates: dict
    ) -> Optional[HardwareConfig]:
        hw = await self._store.get_hardware(hardware_id)
        if hw is None:
            return None
        allowed = {"gpu", "vram_gb", "cost_per_hr", "use_for"}
        for key, value in updates.items():
            if key in allowed and value is not None:
                setattr(hw, key, value)
        return await self._store.update_hardware(hw)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def list_metrics(self, enabled_only: bool = False) -> list[MetricConfig]:
        return await self._store.list_metrics(enabled_only=enabled_only)

    async def get_metric(self, name: str) -> Optional[MetricConfig]:
        return await self._store.get_metric(name)

    async def create_metric(self, data: dict) -> MetricConfig:
        existing = await self._store.get_metric(data["name"])
        if existing is not None:
            raise ValueError(f"Metric '{data['name']}' already exists.")
        metric = MetricConfig(
            name=data["name"],
            stage=MetricStage(data["stage"]),
            compute_tier=ComputeTier(data["compute_tier"]),
            task_types=data["task_types"],
            category=data.get("category"),
            enabled=data.get("enabled", True),
            weight=data.get("weight", 1.0),
            description=data.get("description"),
        )
        return await self._store.create_metric(metric)

    async def update_metric(
        self, name: str, updates: dict
    ) -> Optional[MetricConfig]:
        metric = await self._store.get_metric(name)
        if metric is None:
            return None
        allowed = {"enabled", "weight", "description"}
        for key, value in updates.items():
            if key in allowed and value is not None:
                setattr(metric, key, value)
        return await self._store.update_metric(metric)

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    async def list_tasks(self, active_only: bool = False) -> list[TaskConfig]:
        return await self._store.list_tasks(active_only=active_only)

    async def get_task(self, name: str) -> Optional[TaskConfig]:
        return await self._store.get_task(name)

    async def update_task(self, name: str, updates: dict) -> Optional[TaskConfig]:
        task = await self._store.get_task(name)
        if task is None:
            return None
        allowed = {"active", "description"}
        for key, value in updates.items():
            if key in allowed and value is not None:
                setattr(task, key, value)
        return await self._store.update_task(task)

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    async def create_run(self, run_config: dict) -> dict:
        run_id = _generate_run_id()
        now = _now_iso()
        manifest = {
            "run_id": run_id,
            "created_at": now,
            "task_scope": run_config.get("task_scope", []),
            "prompt_count": 0,
            "translation_samples": 0,
            "instruction_samples": 0,
            "benchmark_hash": None,
            "models": {},
            "inference_config": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512},
            "judge": {
                "model": run_config.get("judge_model", "claude-haiku-4-5"),
                "swap_runs": 2,
            },
            "metrics_enabled": [],
            "notes": run_config.get("notes"),
            "status": "started",
        }
        await self._store.append_run(run_id, "started")
        return manifest

    async def list_runs(self) -> list[dict]:
        return await self._store.list_runs()

    async def update_run_status(self, run_id: str, status: str) -> None:
        await self._store.update_run_status(run_id, status)
