"""
Registry API routes.

All write endpoints require registry:write scope.
All read endpoints require a valid JWT (get_current_user).
No DELETE endpoints — models are disabled or deprecated, never deleted.
State transitions validated in the service layer; ValueError → 409.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.auth import get_current_user, require_scope
from src.api.schemas import (
    CreateHardwareRequest,
    CreateMetricRequest,
    CreateModelRequest,
    CreateRunRequest,
    FetchChatTemplateResponse,
    HealthCheckResponse,
    HealthStatus,
    HardwareListResponse,
    HardwareResponse,
    MetricListResponse,
    MetricResponse,
    ModelListResponse,
    ModelResponse,
    RunListResponse,
    RunResponse,
    TaskListResponse,
    TaskResponse,
    UpdateHardwareRequest,
    UpdateMetricRequest,
    UpdateModelRequest,
    UpdateTaskRequest,
)
from src.core.services.registry_service import RegistryService
from src.deps import get_registry_service

router = APIRouter()

_WRITE = Depends(require_scope("registry:write"))
_READ  = Depends(get_current_user)


# ---------------------------------------------------------------------------
# Health (no auth)
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthCheckResponse)
async def health() -> HealthCheckResponse:
    return HealthCheckResponse(status=HealthStatus.OK)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@router.get("/models", response_model=ModelListResponse, dependencies=[_READ])
async def list_models(
    state: str = Query("active", description="active|disabled|deprecated|all"),
    svc: RegistryService = Depends(get_registry_service),
) -> ModelListResponse:
    models = await svc.list_models(state=state)
    return ModelListResponse(
        models=[m.to_dict() for m in models],
        total=len(models),
        state_filter=state,
    )


@router.post("/models", response_model=ModelResponse, status_code=201, dependencies=[_WRITE])
async def create_model(
    body: CreateModelRequest,
    svc: RegistryService = Depends(get_registry_service),
) -> ModelResponse:
    try:
        model = await svc.create_model(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return ModelResponse(status="created", data=model.to_dict())


@router.get("/models/{model_id}", response_model=ModelResponse, dependencies=[_READ])
async def get_model(
    model_id: str,
    svc: RegistryService = Depends(get_registry_service),
) -> ModelResponse:
    model = await svc.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return ModelResponse(status="ok", data=model.to_dict())


@router.patch("/models/{model_id}", response_model=ModelResponse, dependencies=[_WRITE])
async def update_model(
    model_id: str,
    body: UpdateModelRequest,
    svc: RegistryService = Depends(get_registry_service),
) -> ModelResponse:
    model = await svc.update_model(model_id, body.model_dump(exclude_unset=True))
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return ModelResponse(status="updated", data=model.to_dict())


@router.patch("/models/{model_id}/disable", response_model=ModelResponse, dependencies=[_WRITE])
async def disable_model(
    model_id: str,
    svc: RegistryService = Depends(get_registry_service),
) -> ModelResponse:
    try:
        model = await svc.disable_model(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return ModelResponse(status="disabled", data=model.to_dict())


@router.patch("/models/{model_id}/enable", response_model=ModelResponse, dependencies=[_WRITE])
async def enable_model(
    model_id: str,
    svc: RegistryService = Depends(get_registry_service),
) -> ModelResponse:
    try:
        model = await svc.enable_model(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return ModelResponse(status="enabled", data=model.to_dict())


@router.patch("/models/{model_id}/deprecate", response_model=ModelResponse, dependencies=[_WRITE])
async def deprecate_model(
    model_id: str,
    svc: RegistryService = Depends(get_registry_service),
) -> ModelResponse:
    try:
        model = await svc.deprecate_model(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return ModelResponse(status="deprecated", data=model.to_dict())


@router.post(
    "/models/{model_id}/fetch-chat-template",
    response_model=FetchChatTemplateResponse,
    dependencies=[_WRITE],
    summary="Fetch and store chat template from HuggingFace",
    description=(
        "Downloads the tokenizer for the model's hf_model_id from HuggingFace, "
        "extracts the Jinja2 chat_template string, and stores it in the registry. "
        "Idempotent — safe to call multiple times. If the tokenizer has no chat "
        "template, stores None (vLLM will fall back to plain-text prompts). "
        "Requires HF_TOKEN env var to be set on the registry container for gated repos."
    ),
)
async def fetch_chat_template(
    model_id: str,
    svc: RegistryService = Depends(get_registry_service),
) -> FetchChatTemplateResponse:
    import os
    hf_token = os.environ.get("HF_TOKEN")
    try:
        result = await svc.fetch_and_store_chat_template(model_id, hf_token=hf_token)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Template fetch failed: {exc}")
    return FetchChatTemplateResponse(**result)


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

@router.get("/hardware", response_model=HardwareListResponse, dependencies=[_READ])
async def list_hardware(
    svc: RegistryService = Depends(get_registry_service),
) -> HardwareListResponse:
    items = await svc.list_hardware()
    return HardwareListResponse(hardware=[h.to_dict() for h in items], total=len(items))


@router.post("/hardware", response_model=HardwareResponse, status_code=201, dependencies=[_WRITE])
async def create_hardware(
    body: CreateHardwareRequest,
    svc: RegistryService = Depends(get_registry_service),
) -> HardwareResponse:
    try:
        hw = await svc.create_hardware(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return HardwareResponse(status="created", data=hw.to_dict())


@router.get("/hardware/{hardware_id}", response_model=HardwareResponse, dependencies=[_READ])
async def get_hardware(
    hardware_id: str,
    svc: RegistryService = Depends(get_registry_service),
) -> HardwareResponse:
    hw = await svc.get_hardware(hardware_id)
    if hw is None:
        raise HTTPException(status_code=404, detail=f"Hardware '{hardware_id}' not found.")
    return HardwareResponse(status="ok", data=hw.to_dict())


@router.patch("/hardware/{hardware_id}", response_model=HardwareResponse, dependencies=[_WRITE])
async def update_hardware(
    hardware_id: str,
    body: UpdateHardwareRequest,
    svc: RegistryService = Depends(get_registry_service),
) -> HardwareResponse:
    hw = await svc.update_hardware(hardware_id, body.model_dump(exclude_unset=True))
    if hw is None:
        raise HTTPException(status_code=404, detail=f"Hardware '{hardware_id}' not found.")
    return HardwareResponse(status="updated", data=hw.to_dict())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@router.get("/metrics", response_model=MetricListResponse, dependencies=[_READ])
async def list_metrics(
    enabled_only: bool = Query(False),
    svc: RegistryService = Depends(get_registry_service),
) -> MetricListResponse:
    metrics = await svc.list_metrics(enabled_only=enabled_only)
    return MetricListResponse(metrics=[m.to_dict() for m in metrics], total=len(metrics))


@router.post("/metrics", response_model=MetricResponse, status_code=201, dependencies=[_WRITE])
async def create_metric(
    body: CreateMetricRequest,
    svc: RegistryService = Depends(get_registry_service),
) -> MetricResponse:
    try:
        metric = await svc.create_metric(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return MetricResponse(status="created", data=metric.to_dict())


@router.get("/metrics/{name}", response_model=MetricResponse, dependencies=[_READ])
async def get_metric(
    name: str,
    svc: RegistryService = Depends(get_registry_service),
) -> MetricResponse:
    metric = await svc.get_metric(name)
    if metric is None:
        raise HTTPException(status_code=404, detail=f"Metric '{name}' not found.")
    return MetricResponse(status="ok", data=metric.to_dict())


@router.patch("/metrics/{name}", response_model=MetricResponse, dependencies=[_WRITE])
async def update_metric(
    name: str,
    body: UpdateMetricRequest,
    svc: RegistryService = Depends(get_registry_service),
) -> MetricResponse:
    metric = await svc.update_metric(name, body.model_dump(exclude_unset=True))
    if metric is None:
        raise HTTPException(status_code=404, detail=f"Metric '{name}' not found.")
    return MetricResponse(status="updated", data=metric.to_dict())


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@router.get("/tasks", response_model=TaskListResponse, dependencies=[_READ])
async def list_tasks(
    active_only: bool = Query(False),
    svc: RegistryService = Depends(get_registry_service),
) -> TaskListResponse:
    tasks = await svc.list_tasks(active_only=active_only)
    return TaskListResponse(tasks=[t.to_dict() for t in tasks], total=len(tasks))


@router.get("/tasks/{name}", response_model=TaskResponse, dependencies=[_READ])
async def get_task(
    name: str,
    svc: RegistryService = Depends(get_registry_service),
) -> TaskResponse:
    task = await svc.get_task(name)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{name}' not found.")
    return TaskResponse(status="ok", data=task.to_dict())


@router.patch("/tasks/{name}", response_model=TaskResponse, dependencies=[_WRITE])
async def update_task(
    name: str,
    body: UpdateTaskRequest,
    svc: RegistryService = Depends(get_registry_service),
) -> TaskResponse:
    task = await svc.update_task(name, body.model_dump(exclude_unset=True))
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{name}' not found.")
    return TaskResponse(status="updated", data=task.to_dict())


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

@router.post("/runs", response_model=RunResponse, status_code=201, dependencies=[_WRITE])
async def create_run(
    body: CreateRunRequest,
    svc: RegistryService = Depends(get_registry_service),
) -> RunResponse:
    manifest = await svc.create_run(body.model_dump())
    return RunResponse(status="started", run_id=manifest["run_id"], data=manifest)


@router.get("/runs", response_model=RunListResponse, dependencies=[_READ])
async def list_runs(
    svc: RegistryService = Depends(get_registry_service),
) -> RunListResponse:
    runs = await svc.list_runs()
    return RunListResponse(runs=runs, total=len(runs))


@router.get("/runs/{run_id}", response_model=RunResponse, dependencies=[_READ])
async def get_run(
    run_id: str,
    svc: RegistryService = Depends(get_registry_service),
) -> RunResponse:
    runs = await svc.list_runs()
    for entry in runs:
        if entry["run_id"] == run_id:
            return RunResponse(status="ok", run_id=run_id, data=entry)
    raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")