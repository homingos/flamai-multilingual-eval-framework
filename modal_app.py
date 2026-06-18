"""
Phase 2A — Modal application entrypoint.

Defines:
  RegistryService  — FastAPI registry (CPU, Phase 0/1)
  VLLMWorkerT4     — inference worker for T4 GPU  (tiny models <500M)
  VLLMWorkerL4     — inference worker for L4 GPU  (7B/8B models + metric workers)
  VLLMWorkerL40S   — inference worker for L40S GPU (12B models)
  VLLMWorkerA100   — inference worker for A100-80GB (Gemma-4 26B baseline)

Usage:
    modal deploy modal_app.py                     # deploy all services
    modal run modal_app.py::pilot                 # Phase 2 pilot run (runs on Modal)
    modal run modal_app.py::pilot --model-id tamil-mistral-7b --limit 200
    modal run modal_app.py::pilot --run-id 2026-06-17_143022_a3f9b1  # resume
"""
import modal
from modal_common import (
    build_registry_config,
    env_config,
    registry_image,
    vllm_image,
    VOLUME_MOUNTS,
)
from src.workers.inference import VLLMWorker as _VLLMWorker

APP_NAME = f"flamai-Multilingual-Evaluation-Pipeline-{env_config.env_name}"
app = modal.App(APP_NAME)

_registry_secret = modal.Secret.from_name("phase2a-registry-url")


# ---------------------------------------------------------------------------
# Registry Service (Phase 0/1 — unchanged)
# ---------------------------------------------------------------------------

@app.cls(**build_registry_config(env_config))
@modal.concurrent(max_inputs=env_config.max_concurrent_requests)
class RegistryService:
    """FastAPI registry — serves /models, /hardware, /metrics, /tasks, /runs."""

    @modal.enter(snap=True)
    def preload(self) -> None:
        import src.main  # noqa: F401

    @modal.enter(snap=False)
    def startup(self) -> None:
        pass

    @modal.asgi_app(label="phase2a-registry")
    def fastapi_app(self):
        from src.main import app as _app
        return _app


# ---------------------------------------------------------------------------
# vLLM inference workers — one class per GPU tier
# ---------------------------------------------------------------------------

@app.cls(
    image=vllm_image,
    gpu="T4",
    timeout=3600,
    secrets=[_registry_secret],
    volumes=VOLUME_MOUNTS,
)
class VLLMWorkerT4(_VLLMWorker):
    """T4 GPU — tiny models <500M (Goldfish series)."""
    pass


@app.cls(
    image=vllm_image,
    gpu="L4",
    timeout=3600,
    secrets=[_registry_secret],
    volumes=VOLUME_MOUNTS,
)
class VLLMWorkerL4(_VLLMWorker):
    """L4 GPU — 7B/8B models + metric workers (most regional models)."""
    pass


@app.cls(
    image=vllm_image,
    gpu="L40S",
    timeout=3600,
    secrets=[_registry_secret],
    volumes=VOLUME_MOUNTS,
)
class VLLMWorkerL40S(_VLLMWorker):
    """L40S GPU — 12B models (Polyglot-Ko-12B)."""
    pass


@app.cls(
    image=vllm_image,
    gpu="A100-80GB",
    timeout=3600,
    secrets=[_registry_secret],
    volumes=VOLUME_MOUNTS,
)
class VLLMWorkerA100(_VLLMWorker):
    """A100-80GB — Gemma-4 26B baseline."""
    pass


# ---------------------------------------------------------------------------
# GPU_WORKER_MAP — orchestrator looks up the right worker class by preset key
# ---------------------------------------------------------------------------

GPU_WORKER_MAP = {
    "t4":        VLLMWorkerT4,
    "l4":        VLLMWorkerL4,
    "l40s":      VLLMWorkerL40S,
    "a100_80gb": VLLMWorkerA100,
}


def get_worker_class(gpu_preset: str):
    """Returns the Modal worker class for a given gpu_preset key."""
    if gpu_preset not in GPU_WORKER_MAP:
        raise ValueError(
            f"No worker class for gpu_preset='{gpu_preset}'. "
            f"Valid: {list(GPU_WORKER_MAP)}"
        )
    return GPU_WORKER_MAP[gpu_preset]


# ---------------------------------------------------------------------------
# Pilot orchestrator — runs entirely on Modal (Mac-compatible)
# ---------------------------------------------------------------------------

@app.function(
    image=registry_image,
    volumes=VOLUME_MOUNTS,
    secrets=[_registry_secret],
    timeout=7200,
)
def _run_pilot(model_id: str, task: str, limit: int, run_id: str) -> dict:
    """
    Pilot orchestrator that runs in a Modal container.
    Has full access to benchmarks + outputs volumes.
    Dispatches to the correct vLLM worker class and waits for results.
    """
    import json
    import os
    import urllib.request

    from src.pipeline.loader import load_samples
    from src.pipeline.run import (
        append_run_to_index,
        generate_run_id,
        regional_output_path,
        update_manifest_status,
        write_manifest,
    )

    if not run_id:
        run_id = generate_run_id()

    print(f"Run ID: {run_id}")
    print(f"Model:  {model_id}")
    print(f"Task:   {task}")
    print(f"Limit:  {limit}")

    # Fetch model config from registry (REGISTRY_URL + JWT_TOKEN from secret)
    registry_url = os.environ["REGISTRY_URL"]
    jwt_token    = os.environ["JWT_TOKEN"]
    req = urllib.request.Request(
        f"{registry_url}/models/{model_id}",
        headers={"Authorization": f"Bearer {jwt_token}"},
    )
    with urllib.request.urlopen(req) as resp:
        model_data = json.loads(resp.read())["data"]

    slug       = model_data["slug"]
    gpu_preset = model_data["gpu_preset"]

    # Load samples from /data/benchmarks volume
    samples = load_samples(task, slug, limit=limit)
    print(f"Loaded {len(samples)} samples from /data/benchmarks/{task}/{slug}/")

    # Write manifest + runs index to /data/outputs volume
    manifest = {
        "run_id":       run_id,
        "task_scope":   [task],
        "slug":         slug,
        "model_id":     model_id,
        "model_name":   model_data["name"],
        "hf_model_id":  model_data["hf_model_id"],
        "prompt_count": len(samples),
        "inference_config": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512},
        "status": "started",
    }
    write_manifest(run_id, manifest)
    append_run_to_index(run_id, status="started")

    # Dispatch to the correct GPU worker and wait for results
    WorkerCls = get_worker_class(gpu_preset)
    worker = WorkerCls(model_id=model_id, run_id=run_id, task=task)
    results = worker.generate.remote(samples)

    out_path = regional_output_path(run_id, slug, task)
    print(f"\nDone.")
    print(f"  Outputs: {len(results)}")
    print(f"  Path:    {out_path}")
    print(f"  Run ID:  {run_id}")

    update_manifest_status(run_id, "completed")
    return {"run_id": run_id, "total_outputs": len(results), "output_path": out_path}


@app.local_entrypoint()
def pilot(
    model_id: str = "tamil-mistral-7b",
    task: str = "translation",
    limit: int = 200,
    run_id: str = "",
):
    """
    Fires the pilot on Modal. All work runs remotely — Mac-compatible.

    Usage:
        modal run modal_app.py::pilot
        modal run modal_app.py::pilot --model-id tamil-mistral-7b --limit 200
        modal run modal_app.py::pilot --run-id <id>   # resume existing run
    """
    result = _run_pilot.remote(
        model_id=model_id, task=task, limit=limit, run_id=run_id
    )
    print(f"\nPilot complete:")
    print(f"  Run ID:  {result['run_id']}")
    print(f"  Outputs: {result['total_outputs']}")
    print(f"  Path:    {result['output_path']}")
