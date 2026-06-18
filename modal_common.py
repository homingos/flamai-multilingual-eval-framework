"""
Phase 2A — shared Modal resources: volumes, images, GPU presets, EnvConfig.

No src/ imports here. modal_app.py and pipeline/ workers import from this module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
REGISTRY_VOLUME_NAME   = "phase2a-registry"
BENCHMARKS_VOLUME_NAME = "phase2a-benchmarks"
WEIGHTS_VOLUME_NAME    = "phase2a-weights"
OUTPUTS_VOLUME_NAME    = "phase2a-outputs"

registry_volume   = modal.Volume.from_name(REGISTRY_VOLUME_NAME,   create_if_missing=True)
benchmarks_volume = modal.Volume.from_name(BENCHMARKS_VOLUME_NAME, create_if_missing=True)
weights_volume    = modal.Volume.from_name(WEIGHTS_VOLUME_NAME,    create_if_missing=True)
outputs_volume    = modal.Volume.from_name(OUTPUTS_VOLUME_NAME,    create_if_missing=True)

VOLUME_MOUNTS: dict[str, modal.Volume] = {
    "/data/registry":   registry_volume,
    "/data/benchmarks": benchmarks_volume,
    "/data/weights":    weights_volume,
    "/data/outputs":    outputs_volume,
}

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

# Registry service — CPU only, FastAPI + JWT auth
registry_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "jq")
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "pydantic>=2.0",
        "PyJWT",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("modal_app.py",    remote_path="/root/modal_app.py")
    .add_local_file("modal_common.py", remote_path="/root/modal_common.py")
)

# vLLM inference image — GPU, Phase 2
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install(
        "vllm>=0.4.0",
        "torch",
        "transformers",
        "huggingface_hub",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("modal_app.py",    remote_path="/root/modal_app.py")
    .add_local_file("modal_common.py", remote_path="/root/modal_common.py")
)

# ---------------------------------------------------------------------------
# GPU Presets  (Modal live prices, June 2026)
# ---------------------------------------------------------------------------
GPU_PRESETS: dict[str, dict] = {
    "t4":           {"gpu": "T4",           "vram_gb": 16, "cost_per_hr": 0.59},
    "l4":           {"gpu": "L4",           "vram_gb": 24, "cost_per_hr": 0.80},
    "a10":          {"gpu": "A10G",         "vram_gb": 24, "cost_per_hr": 1.10},
    "l40s":         {"gpu": "L40S",         "vram_gb": 48, "cost_per_hr": 1.95},
    "a100_40gb":    {"gpu": "A100-40GB",    "vram_gb": 40, "cost_per_hr": 2.10},
    "a100_80gb":    {"gpu": "A100-80GB",    "vram_gb": 80, "cost_per_hr": 2.50},
    "rtx_pro_6000": {"gpu": "RTX_PRO_6000", "vram_gb": 96, "cost_per_hr": 3.03},
    "h100":         {"gpu": "H100",         "vram_gb": 80, "cost_per_hr": 3.95},
}

# Model → GPU assignment for Phase 2A inference workers
MODEL_GPU_MAP: dict[str, str] = {
    "Goldfish-mri-39M":  "t4",
    "Goldfish-tpi-125M": "t4",
    "Tucano-2b4":        "l4",
    "MaLLaM-5B":         "l4",
    "Tamil-Mistral-7B":  "l4",
    "Gujju-Llama-7B":    "l4",
    "Ambari-7B":         "l4",
    "MahaMarathi-7B":    "l4",
    "DictaLM-2.0-7B":    "l4",
    "Lucie-7B":          "l4",
    "Viking-7B":         "l4",
    "CSMPT-7B":          "l4",
    "Meltemi-7B":        "l4",
    "Swahili-Gemma-7B":  "l4",
    "Walia-LLM-7B":      "l4",
    "Jais-2-8B":         "l4",
    "Polyglot-Ko-12B":   "l40s",
    "Gemma-4":           "a100_80gb",
}

# ---------------------------------------------------------------------------
# EnvConfig
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    env_name: str
    max_concurrent_requests: int
    timeout: int = 300
    memory: int = 4096
    gpu_type: Optional[str] = None
    gpu_count: int = 1
    vram_gb: Optional[int] = None

_ENV_PRESETS: dict[str, EnvConfig] = {
    "feat":     EnvConfig("feat",     max_concurrent_requests=4,  timeout=120,  memory=2048),
    "dev":      EnvConfig("dev",      max_concurrent_requests=8,  timeout=300,  memory=4096),
    "prod":     EnvConfig("prod",     max_concurrent_requests=32, timeout=600,  memory=8192),
    "registry": EnvConfig("registry", max_concurrent_requests=16, timeout=300,  memory=4096),
}


def get_env_config(env_name: Optional[str] = None) -> EnvConfig:
    name = env_name or os.environ.get("DEPLOY_ENV", "dev")
    return _ENV_PRESETS.get(name, _ENV_PRESETS["dev"])


def configure_env_vars(config: EnvConfig) -> dict[str, str]:
    return {
        "DEPLOY_ENV": config.env_name,
        "PERSISTENCE_BACKEND": os.environ.get("PERSISTENCE_BACKEND", "volume"),
        "JWT_SECRET": os.environ.get("JWT_SECRET", "dev-secret-change-in-prod"),
    }


# ---------------------------------------------------------------------------
# Modal config builders
# ---------------------------------------------------------------------------

def build_fastapi_config(env: EnvConfig) -> dict:
    """Legacy alias — kept for template compatibility."""
    return build_registry_config(env)


def build_registry_config(env: EnvConfig) -> dict:
    """Config dict for @app.cls — registry FastAPI service (CPU only)."""
    return {
        "image": registry_image,
        "volumes": VOLUME_MOUNTS,
        "secrets": [modal.Secret.from_name("phase2a-auth-secrets")],
        "timeout": env.timeout,
        "memory": env.memory,
        "enable_memory_snapshot": True,
    }


def build_inference_config(model_name: str, env: EnvConfig) -> dict:
    """Config dict for @app.cls — vLLM inference worker (GPU)."""
    preset_key = MODEL_GPU_MAP.get(model_name, "l4")
    preset = GPU_PRESETS[preset_key]
    return {
        "image": vllm_image,
        "gpu": preset["gpu"],
        "volumes": VOLUME_MOUNTS,
        "timeout": 1800,
        "memory": 32768,
    }


# Module-level singleton used by modal_app.py
env_config: EnvConfig = get_env_config()
