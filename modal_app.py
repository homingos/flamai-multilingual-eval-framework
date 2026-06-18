"""
Phase 2A — Modal application entrypoint.

Defines:
  RegistryService            — FastAPI registry (CPU, Phase 0/1)
  VLLMWorkerT4/L4/L40S/A100  — inference workers per GPU tier (Phase 2/3)
  LightMetricWorkerModal     — CPU metric worker (Phase 3)
  ModelMetricWorkerModal     — GPU metric worker — COMET, BERTScore (Phase 4)
  JudgeWorkerModal           — LLM-as-judge via Anthropic or Gemini API (Phase 4)

Usage:
    modal deploy modal_app.py                         # deploy all services
    modal run modal_app.py::compare                   # Phase 3 pilot: Tamil vs Gemma-4
    modal run modal_app.py::compare --task instructions --limit 100
    modal run modal_app.py::compare --run-id <id>     # resume
    modal run modal_app.py::phase4                    # Phase 4: COMET + judge + report
    modal run modal_app.py::phase4 --slug greek --language Greek --regional-model-id meltemi-7b
"""
import modal
from modal_common import (
    build_registry_config,
    env_config,
    judge_image,
    model_metrics_image,
    registry_image,
    vllm_image,
    VOLUME_MOUNTS,
)
from src.workers.inference import VLLMWorker as _VLLMWorker
from src.workers.judge import JudgeWorker as _JudgeWorker
from src.workers.light_metrics import LightMetricWorker as _LightMetricWorker
from src.workers.model_metrics import ModelMetricWorker as _ModelMetricWorker
from src.workers.reporter import ReportGenerator as _ReportGenerator

APP_NAME = f"flamai-Multilingual-Evaluation-Pipeline-{env_config.env_name}"
app = modal.App(APP_NAME)

_registry_secret = modal.Secret.from_name("phase2a-registry-url")
_auth_secret     = modal.Secret.from_name("phase2a-auth-secrets")  # JWT_SECRET + HF_TOKEN

_inference_secrets = [_registry_secret, _auth_secret]

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

metrics_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("sacrebleu", "langdetect")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("modal_app.py",    remote_path="/root/modal_app.py")
    .add_local_file("modal_common.py", remote_path="/root/modal_common.py")
)

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
    secrets=_inference_secrets,
    volumes=VOLUME_MOUNTS,
    enable_memory_snapshot=True,
)
class VLLMWorkerT4(_VLLMWorker):
    model_id: str = modal.parameter()
    run_id:   str = modal.parameter()
    task:     str = modal.parameter()

    @modal.enter(snap=False)
    def post_restore(self) -> None:
        # Runs after snapshot restore. CUDA context is rebuilt by vLLM automatically.
        pass


@app.cls(
    image=vllm_image,
    gpu="L4",
    timeout=3600,
    secrets=_inference_secrets,
    volumes=VOLUME_MOUNTS,
    enable_memory_snapshot=True,
)
class VLLMWorkerL4(_VLLMWorker):
    model_id: str = modal.parameter()
    run_id:   str = modal.parameter()
    task:     str = modal.parameter()

    @modal.enter(snap=False)
    def post_restore(self) -> None:
        pass


@app.cls(
    image=vllm_image,
    gpu="L40S",
    timeout=3600,
    secrets=_inference_secrets,
    volumes=VOLUME_MOUNTS,
    enable_memory_snapshot=True,
)
class VLLMWorkerL40S(_VLLMWorker):
    model_id: str = modal.parameter()
    run_id:   str = modal.parameter()
    task:     str = modal.parameter()

    @modal.enter(snap=False)
    def post_restore(self) -> None:
        pass


@app.cls(
    image=vllm_image,
    gpu="A100-80GB",
    timeout=3600,
    secrets=_inference_secrets,
    volumes=VOLUME_MOUNTS,
    enable_memory_snapshot=True,
)
class VLLMWorkerA100(_VLLMWorker):
    model_id: str = modal.parameter()
    run_id:   str = modal.parameter()
    task:     str = modal.parameter()

    @modal.enter(snap=False)
    def post_restore(self) -> None:
        pass


GPU_WORKER_MAP = {
    "t4":        VLLMWorkerT4,
    "l4":        VLLMWorkerL4,
    "l40s":      VLLMWorkerL40S,
    "a100_80gb": VLLMWorkerA100,
}


def get_worker_class(gpu_preset: str):
    if gpu_preset not in GPU_WORKER_MAP:
        raise ValueError(f"No worker for gpu_preset='{gpu_preset}'. Valid: {list(GPU_WORKER_MAP)}")
    return GPU_WORKER_MAP[gpu_preset]


# ---------------------------------------------------------------------------
# Light metric worker (Phase 3)
# ---------------------------------------------------------------------------

@app.cls(
    image=metrics_image,
    cpu=2,
    memory=4096,
    timeout=600,
    secrets=[_registry_secret],
    volumes=VOLUME_MOUNTS,
)
class LightMetricWorkerModal(_LightMetricWorker):
    """CPU-only metric worker — sacrebleu, langdetect, rule-based checks."""

    @modal.method()
    def score(self, run_id, slug, task, language, model_id=""):
        return super().score(run_id, slug, task, language, model_id)


# ---------------------------------------------------------------------------
# Phase 4 — Model metric worker (COMET, BERTScore)
# ---------------------------------------------------------------------------

_judge_secret = modal.Secret.from_name("phase2a-judge")  # GEMINI_API_KEY and/or ANTHROPIC_API_KEY

@app.cls(
    image=model_metrics_image,
    gpu="L4",
    cpu=2,
    memory=16384,
    timeout=3600,
    secrets=[_registry_secret],
    volumes=VOLUME_MOUNTS,
)
class ModelMetricWorkerModal(_ModelMetricWorker):
    """GPU metric worker — COMET, BERTScore. L4 is sufficient for inference-only scoring."""

    @modal.method()
    def score(self, run_id, slug, task, language, model_id=""):
        return super().score(run_id, slug, task, language, model_id)


# ---------------------------------------------------------------------------
# Phase 4 — LLM judge worker (Anthropic or Gemini)
# ---------------------------------------------------------------------------

@app.cls(
    image=judge_image,
    cpu=2,
    memory=2048,
    timeout=7200,
    secrets=[_judge_secret],
    volumes=VOLUME_MOUNTS,
)
class JudgeWorkerModal(_JudgeWorker):
    """
    LLM-as-judge via Anthropic or Gemini API.
    Provider is selected automatically by the judge_model string prefix:
      "gemini-*"  → Google Gemini  (requires GEMINI_API_KEY in phase2a-judge secret)
      "claude-*"  → Anthropic      (requires ANTHROPIC_API_KEY in phase2a-judge secret)
    CPU-only — all calls are remote API calls, no local model loading.
    """

    @modal.method()
    def judge(self, run_id, slug, task, language, regional_model_id,
              judge_model="gemini-2.0-flash", swap_runs=2, limit=None):
        return super().judge(
            run_id=run_id, slug=slug, task=task, language=language,
            regional_model_id=regional_model_id, judge_model=judge_model,
            swap_runs=swap_runs, limit=limit,
        )


# ---------------------------------------------------------------------------
# Phase 3 — compare orchestrator
# ---------------------------------------------------------------------------

@app.function(
    image=metrics_image,
    cpu=2,
    memory=4096,
    timeout=7200,
    secrets=[_registry_secret],
    volumes=VOLUME_MOUNTS,
)
def _run_compare(
    run_id: str = "",
    slug: str = "tamil",
    language: str = "Tamil",
    regional_model_id: str = "tamil-mistral-7b",
    regional_gpu_preset: str = "l4",
    limit: int = 200,
    task: str = "translation",
) -> str:
    """Runs in a Modal container with full volume access. Returns run_id."""
    import os

    from src.pipeline.loader import load_samples
    from src.pipeline.run import (
        append_run_to_index,
        gemma_output_path,
        generate_run_id,
        regional_output_path,
        update_manifest_status,
        write_manifest,
    )

    # ── Step 1: Run ID ──────────────────────────────────────────────────────
    if not run_id:
        run_id = generate_run_id()
    print(f"\nRun ID:   {run_id}")
    print(f"Language: {language}  ({slug})")
    print(f"Task:     {task}")
    print(f"Limit:    {limit} prompts\n")

    # ── Step 2: Load samples ─────────────────────────────────────────────────
    samples = load_samples(task, slug, limit=limit)
    print(f"Loaded {len(samples)} samples from /data/benchmarks/{task}/{slug}/")

    # ── Step 3: Write manifest ───────────────────────────────────────────────
    manifest = {
        "run_id":       run_id,
        "task_scope":   [task],
        "slug":         slug,
        "language":     language,
        "models":       [regional_model_id, "gemma-4-26b"],
        "prompt_count": len(samples),
        "inference_config": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512},
        "status": "started",
    }
    write_manifest(run_id, manifest)
    append_run_to_index(run_id, status="started")

    # ── Step 4: Launch inference (skip if outputs already complete) ─────────────
    reg_path   = regional_output_path(run_id, slug, task)
    gemma_path = gemma_output_path(run_id, task)

    reg_done   = os.path.exists(reg_path)   and os.path.getsize(reg_path)   > 0
    gemma_done = os.path.exists(gemma_path) and os.path.getsize(gemma_path) > 0

    if reg_done and gemma_done:
        print("Both output files already exist — skipping inference, going straight to metrics.")
    else:
        regional_cls = GPU_WORKER_MAP[regional_gpu_preset]
        gemma_cls    = GPU_WORKER_MAP["a100_80gb"]

        regional_worker = regional_cls(model_id=regional_model_id, run_id=run_id, task=task)
        gemma_worker    = gemma_cls(   model_id="gemma-4-26b",      run_id=run_id, task=task)

        print("Launching regional model and Gemma-4 simultaneously...")
        regional_handle = regional_worker.generate.spawn(samples)
        gemma_handle    = gemma_worker.generate.spawn(samples)
        regional_out    = regional_handle.get()
        gemma_out       = gemma_handle.get()
        print(f"Inference complete — regional: {len(regional_out)}, Gemma-4: {len(gemma_out)} new outputs")

        # Reload volume so this container sees files written by worker containers
        import modal as _modal
        _modal.Volume.from_name("phase2a-outputs").reload()

    # ── Step 5: Gate — confirm both output files exist ───────────────────────
    update_manifest_status(run_id, "inference_complete")

    for path, label in [(reg_path, regional_model_id), (gemma_path, "Gemma-4")]:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"WARNING: {label} output file missing or empty: {path}")
            update_manifest_status(run_id, "inference_partial")
            return run_id

    print("Gate passed: both output files confirmed\n")

    # ── Step 6: Light metrics on both models (in parallel) ───────────────────
    print("Computing light metrics...")
    metric_worker = LightMetricWorkerModal()

    regional_score_handle = metric_worker.score.spawn(
        run_id=run_id, slug=slug, task=task,
        language=language, model_id=regional_model_id,
    )
    gemma_score_handle = metric_worker.score.spawn(
        run_id=run_id, slug="baseline", task=task,
        language="Baseline", model_id="gemma-4-26b",
    )
    regional_scores = regional_score_handle.get()
    gemma_scores    = gemma_score_handle.get()

    # ── Step 7: Side-by-side comparison table ────────────────────────────────
    regional_label = regional_model_id.split("-")[0].capitalize() + "-7B"
    width = 65

    print(f"\n{'=' * width}")
    print(f"  PHASE 3 COMPARISON — {language} ({task})")
    print(f"{'=' * width}")
    print(f"  {'Metric':<32} {regional_label:>14}  {'Gemma-4':>10}")
    print(f"  {'-'*32} {'-'*14}  {'-'*10}")

    all_keys = sorted(set(list(regional_scores.keys()) + list(gemma_scores.keys())))
    for metric_name in all_keys:
        reg_vals = regional_scores.get(metric_name, {})
        gem_vals = gemma_scores.get(metric_name, {})
        for score_key in sorted(set(list(reg_vals.keys()) + list(gem_vals.keys()))):
            r_val = reg_vals.get(score_key)
            g_val = gem_vals.get(score_key)
            r_str = f"{r_val:.4f}" if isinstance(r_val, float) else str(r_val or "—")
            g_str = f"{g_val:.4f}" if isinstance(g_val, float) else str(g_val or "—")
            winner = " ↑" if isinstance(r_val, float) and isinstance(g_val, float) and r_val > g_val else ""
            print(f"  {score_key:<32} {r_str + winner:>14}  {g_str:>10}")

    print(f"{'=' * width}")
    print(f"\nOutputs:  {reg_path}")
    print(f"          {gemma_path}")
    print(f"Metrics:  runs/{run_id}/metrics/")
    print(f"Run ID:   {run_id}\n")

    update_manifest_status(run_id, "completed")
    return run_id


@app.local_entrypoint()
def compare(
    slug: str = "tamil",
    language: str = "Tamil",
    regional_model_id: str = "tamil-mistral-7b",
    regional_gpu_preset: str = "l4",
    limit: int = 200,
    task: str = "translation",
    run_id: str = "",
):
    """
    Phase 3 pilot: run one regional model against Gemma-4 and compare.
    All work runs on Modal — Mac-compatible.

    Usage:
        modal run modal_app.py::compare
        modal run modal_app.py::compare --task instructions --limit 100
        modal run modal_app.py::compare --run-id <id>   # resume
    """
    result = _run_compare.remote(
        run_id=run_id,
        slug=slug,
        language=language,
        regional_model_id=regional_model_id,
        regional_gpu_preset=regional_gpu_preset,
        limit=limit,
        task=task,
    )
    if result:
        print(f"Run complete: {result}")


# ---------------------------------------------------------------------------
# Phase 4 — full evaluation: COMET + judge + report
# ---------------------------------------------------------------------------

@app.function(
    image=metrics_image,
    cpu=2,
    memory=4096,
    timeout=14400,  # 4h — judge can be slow for large prompt sets
    secrets=[_registry_secret],
    volumes=VOLUME_MOUNTS,
)
def _run_phase4(
    run_id: str,
    slug: str,
    language: str,
    regional_model_id: str,
    task: str = "translation",
    judge_model: str = "gemini-2.0-flash",
    swap_runs: int = 2,
    judge_limit: int = 0,
    skip_model_metrics: bool = False,
    skip_judge: bool = False,
) -> dict:
    """
    Phase 4 orchestrator. Assumes Phase 3 inference outputs already exist.

    Steps:
      1. MODEL-tier metrics  (COMET + BERTScore) for regional + Gemma-4  [parallel]
      2. LLM judge           (pairwise comparison on all/limited prompts)
      3. Report generation   (classifies model A–E, writes final_report.json)

    Returns the final report dict.
    """
    import modal as _modal

    from src.pipeline.run import (
        gemma_output_path,
        regional_output_path,
        update_manifest_status,
    )

    print(f"\n{'='*60}")
    print(f"  PHASE 4 — {language} ({slug}) / {task}")
    print(f"  Run ID:         {run_id}")
    print(f"  Regional model: {regional_model_id}")
    print(f"  Judge model:    {judge_model}")
    print(f"  Swap runs:      {swap_runs}")
    print(f"  Judge limit:    {judge_limit or 'all'}")
    print(f"{'='*60}\n")

    # ── Verify inference outputs exist ───────────────────────────────────────
    import os
    reg_path   = regional_output_path(run_id, slug, task)
    gemma_path = gemma_output_path(run_id, task)

    for path, label in [(reg_path, regional_model_id), (gemma_path, "Gemma-4")]:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise RuntimeError(
                f"Phase 4 requires Phase 3 inference outputs.\n"
                f"Missing: {label} → {path}\n"
                f"Run `modal run modal_app.py::compare --run-id {run_id}` first."
            )

    # ── Step 1: MODEL-tier metrics (COMET + BERTScore) ──────────────────────
    if not skip_model_metrics:
        print("Step 1: MODEL-tier metrics (COMET + BERTScore)...")
        model_metric_worker = ModelMetricWorkerModal()

        regional_mm_handle = model_metric_worker.score.spawn(
            run_id=run_id, slug=slug, task=task,
            language=language, model_id=regional_model_id,
        )
        gemma_mm_handle = model_metric_worker.score.spawn(
            run_id=run_id, slug="baseline", task=task,
            language="Baseline", model_id="gemma-4-26b",
        )
        regional_mm = regional_mm_handle.get()
        gemma_mm    = gemma_mm_handle.get()

        _modal.Volume.from_name("phase2a-outputs").reload()
        print(f"  Regional model metrics: {regional_mm}")
        print(f"  Gemma-4 model metrics:  {gemma_mm}")
    else:
        print("Step 1: MODEL-tier metrics — SKIPPED")

    # ── Step 2: LLM judge ────────────────────────────────────────────────────
    if not skip_judge:
        print(f"\nStep 2: LLM judge ({judge_model}, {swap_runs} swap runs)...")
        judge_worker = JudgeWorkerModal()
        verdicts = judge_worker.judge.remote(
            run_id=run_id,
            slug=slug,
            task=task,
            language=language,
            regional_model_id=regional_model_id,
            judge_model=judge_model,
            swap_runs=swap_runs,
            limit=judge_limit if judge_limit > 0 else None,
        )
        _modal.Volume.from_name("phase2a-outputs").reload()
        print(f"  Judge complete — {len(verdicts)} verdicts written")
    else:
        print("Step 2: LLM judge — SKIPPED")

    # ── Step 3: Report ───────────────────────────────────────────────────────
    print("\nStep 3: Generating report...")
    reporter = _ReportGenerator()
    report = reporter.generate(
        run_id=run_id,
        slug=slug,
        task=task,
        language=language,
        regional_model_id=regional_model_id,
    )

    update_manifest_status(run_id, "completed")

    cls  = report["languages"][slug]["classification"]
    rat  = report["languages"][slug]["classification_rationale"]
    print(f"\n{'='*60}")
    print(f"  PHASE 4 COMPLETE — {language}")
    print(f"  Classification: {cls}")
    print(f"  Rationale:      {rat}")
    print(f"{'='*60}\n")

    return report


@app.local_entrypoint()
def phase4(
    run_id: str = "",
    slug: str = "tamil",
    language: str = "Tamil",
    regional_model_id: str = "tamil-mistral-7b",
    task: str = "translation",
    judge_model: str = "gemini-2.0-flash",
    swap_runs: int = 2,
    judge_limit: int = 50,       # default to 50 for calibration pilot
    skip_model_metrics: bool = False,
    skip_judge: bool = False,
):
    """
    Phase 4: COMET + BERTScore + LLM judge + final report.
    Requires Phase 3 inference outputs to already exist.

    Judge model is selected by prefix — set via --judge-model:
      gemini-2.0-flash         (default — requires GEMINI_API_KEY)
      gemini-1.5-pro           (higher quality, slower)
      claude-haiku-4-5         (requires ANTHROPIC_API_KEY)

    Usage:
        # Calibration pilot — 50 prompts judged with Gemini
        modal run modal_app.py::phase4 --run-id <id>

        # Full Greek/Meltemi run with Gemini
        modal run modal_app.py::phase4 \\
          --run-id <id> \\
          --slug greek \\
          --language Greek \\
          --regional-model-id meltemi-7b \\
          --judge-limit 0

        # Use Gemini Pro instead
        modal run modal_app.py::phase4 --run-id <id> --judge-model gemini-1.5-pro

        # Skip model metrics (COMET/BERTScore) — judge only
        modal run modal_app.py::phase4 --run-id <id> --skip-model-metrics

        # Skip judge — COMET/BERTScore only
        modal run modal_app.py::phase4 --run-id <id> --skip-judge
    """
    if not run_id:
        print("ERROR: --run-id is required for Phase 4.")
        print("Find your run ID with: modal run modal_app.py::compare (it prints at the top)")
        return

    report = _run_phase4.remote(
        run_id=run_id,
        slug=slug,
        language=language,
        regional_model_id=regional_model_id,
        task=task,
        judge_model=judge_model,
        swap_runs=swap_runs,
        judge_limit=judge_limit,
        skip_model_metrics=skip_model_metrics,
        skip_judge=skip_judge,
    )

    if report:
        slug_report = report.get("languages", {}).get(slug, {})
        cls = slug_report.get("classification", "?")
        rat = slug_report.get("classification_rationale", "")
        print(f"\nPhase 4 complete.")
        print(f"  Classification: {cls}")
        print(f"  Rationale:      {rat}")
        print(f"  Run ID:         {run_id}")
        print(f"  Report:         /data/outputs/runs/{run_id}/reports/final_report.json")