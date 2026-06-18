"""
Pydantic request/response schemas for the Phase 2A registry API.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / template schemas (keep from template)
# ---------------------------------------------------------------------------

class HealthStatus(str, Enum):
    OK      = "ok"
    DEGRADED = "degraded"


class HealthCheckResponse(BaseModel):
    status: HealthStatus
    service: str = "phase2a-registry"
    version: str = "0.1.0"


class BaseResponse(BaseModel):
    pass


class ErrorDetail(BaseModel):
    code: str
    message: str


class TokenPayload(BaseModel):
    sub: str = "anonymous"
    scopes: list[str] = Field(default_factory=list)
    exp: Optional[int] = None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ModelStateEnum(str, Enum):
    ACTIVE     = "active"
    DISABLED   = "disabled"
    DEPRECATED = "deprecated"


class CreateModelRequest(BaseModel):
    id: str = Field(..., min_length=1, description="Unique slug e.g. 'tamil-mistral-7b'")
    name: str
    hf_model_id: str
    language: str
    slug: str = Field(..., description="Dataset folder name e.g. 'tamil'")
    region: str
    gpu_preset: str
    params_billions: float
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = Field(0.88, ge=0.1, le=1.0)
    max_model_len: int = Field(2048, ge=128)
    notes: Optional[str] = None
    chat_template: Optional[str] = None


class UpdateModelRequest(BaseModel):
    name: Optional[str] = None
    hf_model_id: Optional[str] = None
    gpu_preset: Optional[str] = None
    dtype: Optional[str] = None
    gpu_memory_utilization: Optional[float] = Field(None, ge=0.1, le=1.0)
    max_model_len: Optional[int] = Field(None, ge=128)
    notes: Optional[str] = None
    chat_template: Optional[str] = None


class FetchChatTemplateResponse(BaseModel):
    model_id: str
    hf_model_id: str
    template_found: bool
    template_length: int
    stored: bool


class ModelResponse(BaseModel):
    status: str
    data: Optional[dict] = None


class ModelListResponse(BaseModel):
    models: list[dict]
    total: int
    state_filter: Optional[str] = None


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

class CreateHardwareRequest(BaseModel):
    id: str
    gpu: str
    vram_gb: int
    cost_per_hr: float
    use_for: str


class UpdateHardwareRequest(BaseModel):
    gpu: Optional[str] = None
    vram_gb: Optional[int] = None
    cost_per_hr: Optional[float] = None
    use_for: Optional[str] = None


class HardwareResponse(BaseModel):
    status: str
    data: Optional[dict] = None


class HardwareListResponse(BaseModel):
    hardware: list[dict]
    total: int


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class CreateMetricRequest(BaseModel):
    name: str
    stage: str = Field(..., description="post_inference or post_judge")
    compute_tier: str = Field(..., description="model or light")
    task_types: list[str]
    category: Optional[str] = None
    enabled: bool = True
    weight: float = Field(1.0, ge=0.0)
    description: Optional[str] = None


class UpdateMetricRequest(BaseModel):
    enabled: Optional[bool] = None
    weight: Optional[float] = Field(None, ge=0.0)
    description: Optional[str] = None


class MetricResponse(BaseModel):
    status: str
    data: Optional[dict] = None


class MetricListResponse(BaseModel):
    metrics: list[dict]
    total: int


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class UpdateTaskRequest(BaseModel):
    active: Optional[bool] = None
    description: Optional[str] = None


class TaskResponse(BaseModel):
    status: str
    data: Optional[dict] = None


class TaskListResponse(BaseModel):
    tasks: list[dict]
    total: int


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class CreateRunRequest(BaseModel):
    task_scope: list[str] = Field(..., description="e.g. ['translation', 'instructions']")
    judge_model: str = Field("claude-haiku-4-5", description="API model string for LLM judge")
    notes: Optional[str] = None


class RunResponse(BaseModel):
    status: str
    run_id: str
    data: Optional[dict] = None


class RunListResponse(BaseModel):
    runs: list[dict]
    total: int