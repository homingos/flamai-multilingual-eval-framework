"""Pydantic schema validation tests (kept from template pattern)."""
import pytest
from pydantic import ValidationError

from src.api.schemas import CreateModelRequest, UpdateModelRequest


def test_create_model_valid():
    req = CreateModelRequest(
        id="tamil-mistral-7b",
        name="Tamil-Mistral-7B",
        hf_model_id="abhinand/tamil-mistral-7b",
        language="Tamil",
        slug="tamil",
        region="Indic",
        gpu_preset="l4",
        params_billions=7.0,
    )
    assert req.id == "tamil-mistral-7b"
    assert req.dtype == "bfloat16"          # default
    assert req.gpu_memory_utilization == 0.88


def test_create_model_empty_id_rejected():
    with pytest.raises(ValidationError):
        CreateModelRequest(
            id="",
            name="x", hf_model_id="x", language="x", slug="x",
            region="x", gpu_preset="l4", params_billions=1.0,
        )


def test_create_model_gpu_utilization_bounds():
    with pytest.raises(ValidationError):
        CreateModelRequest(
            id="m", name="m", hf_model_id="o/m", language="x", slug="x",
            region="x", gpu_preset="l4", params_billions=1.0,
            gpu_memory_utilization=1.5,  # > 1.0
        )

    with pytest.raises(ValidationError):
        CreateModelRequest(
            id="m", name="m", hf_model_id="o/m", language="x", slug="x",
            region="x", gpu_preset="l4", params_billions=1.0,
            gpu_memory_utilization=0.0,  # < 0.1
        )


def test_create_model_max_model_len_min():
    with pytest.raises(ValidationError):
        CreateModelRequest(
            id="m", name="m", hf_model_id="o/m", language="x", slug="x",
            region="x", gpu_preset="l4", params_billions=1.0,
            max_model_len=64,  # < 128
        )


def test_update_model_all_optional():
    req = UpdateModelRequest()
    assert req.gpu_preset is None
    assert req.dtype is None


def test_update_model_weight_bound():
    with pytest.raises(ValidationError):
        UpdateModelRequest(gpu_memory_utilization=0.0)  # below 0.1
