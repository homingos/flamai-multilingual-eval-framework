"""
Phase 2A — Modal application entrypoint.

Worker registrations + one entrypoint. All pipeline logic lives in:
  src/pipeline/state_machine.py  — the state machine (PENDING→...→DONE/FAILED)
  src/pipeline/entrypoints.py    — run_pipeline()
  src/pipeline/report_render.py  — markdown summary generator

There is one command: run_pipeline. "Phase 3/4/5" don't exist as separate
commands anymore — they were always the same state machine with a
different stop point and a different number of languages:
  --slug tamil --stop-at light_metrics   (old "compare")
  --slug tamil --stop-at report          (old "phase4")
  --slug all   --stop-at report          (old "phase5")

Usage:
    modal deploy modal_app.py
    modal run modal_app.py::run_pipeline --slug greek
    modal run modal_app.py::run_pipeline --slug greek --stop-at light_metrics
    modal run modal_app.py::run_pipeline --slug arabic,hebrew,amharic
    modal run modal_app.py::run_pipeline --slug all
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
from src.workers.inference     import VLLMWorker     as _VLLMWorker
from src.workers.judge         import JudgeWorker    as _JudgeWorker
from src.workers.light_metrics import LightMetricWorker as _LightMetricWorker
from src.workers.model_metrics import ModelMetricWorker as _ModelMetricWorker
from src.workers.reporter      import ReportGenerator as _ReportGenerator
from src.pipeline.state_machine import LanguageSpec, State, WorkerHandles

APP_NAME = f"flamai-Multilingual-Evaluation-Pipeline-{env_config.env_name}"
app      = modal.App(APP_NAME)

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

_registry_secret   = modal.Secret.from_name("phase2a-registry-url")
_auth_secret       = modal.Secret.from_name("phase2a-auth-secrets")
_judge_secret      = modal.Secret.from_name("phase2a-judge")
_inference_secrets = [_registry_secret, _auth_secret]

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

metrics_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("sacrebleu", "langdetect")
    .add_local_dir("src",              remote_path="/root/src")
    .add_local_file("modal_app.py",    remote_path="/root/modal_app.py")
    .add_local_file("modal_common.py", remote_path="/root/modal_common.py")
)

# ---------------------------------------------------------------------------
# Registry (Phase 0/1)
# ---------------------------------------------------------------------------

@app.cls(**build_registry_config(env_config))
@modal.concurrent(max_inputs=env_config.max_concurrent_requests)
class RegistryService:
    """FastAPI registry — /models, /hardware, /metrics, /tasks, /runs."""

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

def _vllm_cls(gpu: str):
    return dict(image=vllm_image, gpu=gpu, timeout=3600,
                secrets=_inference_secrets, volumes=VOLUME_MOUNTS,
                enable_memory_snapshot=True)


@app.cls(**_vllm_cls("T4"))
class VLLMWorkerT4(_VLLMWorker):
    model_id: str = modal.parameter()
    run_id:   str = modal.parameter()
    task:     str = modal.parameter()

    @modal.enter(snap=False)
    def post_restore(self) -> None:
        pass


@app.cls(**_vllm_cls("L4"))
class VLLMWorkerL4(_VLLMWorker):
    model_id: str = modal.parameter()
    run_id:   str = modal.parameter()
    task:     str = modal.parameter()

    @modal.enter(snap=False)
    def post_restore(self) -> None:
        pass


@app.cls(**_vllm_cls("L40S"))
class VLLMWorkerL40S(_VLLMWorker):
    model_id: str = modal.parameter()
    run_id:   str = modal.parameter()
    task:     str = modal.parameter()

    @modal.enter(snap=False)
    def post_restore(self) -> None:
        pass


@app.cls(**_vllm_cls("A100-80GB"))
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


# ---------------------------------------------------------------------------
# Metric + judge workers
# ---------------------------------------------------------------------------

@app.cls(image=metrics_image, cpu=2, memory=4096, timeout=600,
         secrets=[_registry_secret], volumes=VOLUME_MOUNTS)
class LightMetricWorkerModal(_LightMetricWorker):
    @modal.method()
    def score(self, run_id, slug, task, language, model_id=""):
        return super().score(run_id, slug, task, language, model_id)


@app.cls(image=model_metrics_image, gpu="L4", cpu=2, memory=16384, timeout=3600,
         secrets=[_registry_secret], volumes=VOLUME_MOUNTS)
class ModelMetricWorkerModal(_ModelMetricWorker):
    @modal.method()
    def score(self, run_id, slug, task, language, model_id=""):
        return super().score(run_id, slug, task, language, model_id)


@app.cls(image=judge_image, cpu=2, memory=2048, timeout=7200,
         secrets=[_judge_secret], volumes=VOLUME_MOUNTS)
class JudgeWorkerModal(_JudgeWorker):
    @modal.method()
    def judge(self, run_id, slug, task, language, regional_model_id,
              judge_model="gemini-3.5-flash", swap_runs=2, limit=None):
        return super().judge(
            run_id=run_id, slug=slug, task=task, language=language,
            regional_model_id=regional_model_id, judge_model=judge_model,
            swap_runs=swap_runs, limit=limit,
        )


def _handles() -> WorkerHandles:
    """Bundles the worker classes for injection into the state machine."""
    return WorkerHandles(
        gpu_worker_map=GPU_WORKER_MAP,
        LightMetricWorker=LightMetricWorkerModal,
        ModelMetricWorker=ModelMetricWorkerModal,
        JudgeWorker=JudgeWorkerModal,
        ReportGenerator=_ReportGenerator,
    )


# ---------------------------------------------------------------------------
# Model registry — the 17 regional models
# ---------------------------------------------------------------------------

ALL_MODELS = [
    LanguageSpec("tamil-mistral-7b",  "Tamil",                "tamil",                "l4"),
    LanguageSpec("mahamarathi-7b",    "Marathi",              "marathi",              "l4"),
    LanguageSpec("ambari-7b",         "Kannada",              "kannada",              "l4"),
    LanguageSpec("gujju-llama-7b",    "Gujarati",             "gujarati",             "l4"),
    LanguageSpec("jais-2-8b",         "Arabic",               "arabic",               "l4"),
    LanguageSpec("dictalm-2-7b",      "Hebrew",               "hebrew",               "l4"),
    LanguageSpec("polyglot-ko-12b",   "Korean",               "korean",               "l40s"),
    LanguageSpec("mallam-5b",         "Malay",                "malay",                "l4"),
    LanguageSpec("swahili-gemma-7b",  "Swahili",              "swahili",              "l4"),
    LanguageSpec("walia-llm-7b",      "Amharic",              "amharic",              "l4"),
    LanguageSpec("lucie-7b",          "French",               "french",               "l4"),
    LanguageSpec("viking-7b",         "Swedish",              "swedish",              "l4"),
    LanguageSpec("csmpt-7b",          "Czech",                "czech",                "l4"),
    LanguageSpec("meltemi-7b",        "Greek",                "greek",                "l4"),
    LanguageSpec("tucano-2b4",        "Brazilian Portuguese", "brazilian_portuguese", "l4"),
    LanguageSpec("goldfish-mri-39m",  "Māori",                "maori",                "t4"),
    LanguageSpec("goldfish-tpi-125m", "Tok Pisin",            "tok_pisin",            "t4"),
]

BASE_MODELS = {  # no instruct variant — plain-text prompts only
    "mahamarathi-7b", "ambari-7b", "gujju-llama-7b",
    "viking-7b", "csmpt-7b", "polyglot-ko-12b",
    "goldfish-mri-39m", "goldfish-tpi-125m",
}


# ---------------------------------------------------------------------------
# run_pipeline — the one command. Replaces compare / phase4 / phase5.
#
# What used to be three separate commands is now one command with two
# knobs: which language(s), and how far to run them.
#   --slug tamil                     → one language
#   --slug greek,arabic,hebrew       → a few languages, run in parallel
#   --slug all                       → all 17, run in parallel
#   --stop-at light_metrics          → inference + light metrics only (old "compare")
#   --stop-at report                 → full pipeline incl. judge (old "phase4"/"phase5")
# ---------------------------------------------------------------------------

@app.function(image=vllm_image, gpu="A100-80GB", timeout=3600,
              secrets=_inference_secrets, volumes=VOLUME_MOUNTS)
def _run_gemma4_baseline(run_id: str, task: str, limit: int, any_slug: str) -> None:
    """Thin Modal wrapper so ensure_gemma4_baseline can run as a real container."""
    from src.pipeline.state_machine import RunContext, ensure_gemma4_baseline
    ctx = RunContext(run_id=run_id, task=task, limit=limit, judge_model="",
                      judge_limit=0, swap_runs=0, skip_model_metrics=True, handles=_handles())
    ensure_gemma4_baseline(ctx, VLLMWorkerA100, any_slug=any_slug)


@app.function(image=metrics_image, cpu=2, memory=4096, timeout=86400,
              secrets=[_registry_secret, _judge_secret], volumes=VOLUME_MOUNTS)
def _run_one_language(run_id: str, model_id: str, language: str, slug: str,
                      gpu_preset: str, stop_at_value: str, task: str, limit: int,
                      judge_model: str, judge_limit: int, swap_runs: int,
                      skip_model_metrics: bool) -> dict:
    """The container spawned once per language when running more than one."""
    from src.pipeline.entrypoints import run_single_language_to_state
    spec = LanguageSpec(model_id, language, slug, gpu_preset)
    return run_single_language_to_state(
        run_id=run_id, spec=spec, stop_at_value=stop_at_value, task=task, limit=limit,
        judge_model=judge_model, judge_limit=judge_limit, swap_runs=swap_runs,
        skip_model_metrics=skip_model_metrics, handles=_handles(),
    )


@app.function(image=metrics_image, cpu=2, memory=4096, timeout=86400,
              secrets=[_registry_secret, _judge_secret], volumes=VOLUME_MOUNTS)
def _run_pipeline(run_id: str, slugs: list, stop_at_value: str, task: str, limit: int,
                  judge_model: str, judge_limit: int, swap_runs: int,
                  skip_model_metrics: bool) -> dict:
    from src.pipeline.entrypoints import run_pipeline

    specs = [s for s in ALL_MODELS if s.slug in slugs] if slugs else ALL_MODELS

    def _spawn(spec, stop_at):
        return _run_one_language.spawn(
            run_id=run_id, model_id=spec.model_id, language=spec.language,
            slug=spec.slug, gpu_preset=spec.gpu_preset, stop_at_value=stop_at.value,
            task=task, limit=limit, judge_model=judge_model, judge_limit=judge_limit,
            swap_runs=swap_runs, skip_model_metrics=skip_model_metrics,
        )

    return run_pipeline(
        run_id=run_id, specs=specs, stop_at=State(stop_at_value), task=task, limit=limit,
        handles=_handles(), judge_model=judge_model, judge_limit=judge_limit,
        swap_runs=swap_runs, skip_model_metrics=skip_model_metrics,
        VLLMWorkerA100=VLLMWorkerA100, advance_remote_fn=_spawn,
    )


@app.local_entrypoint()
def run_pipeline(slug="tamil", stop_at="report", task="translation", limit=200,
                 judge_model="gemini-3.5-flash", judge_limit=50, swap_runs=2,
                 skip_model_metrics=True, run_id=""):
    """
    Run the evaluation pipeline for one language, a few, or all 17 — to
    whatever stop point you need. Replaces compare / phase4 / phase5.

    --slug          one slug, comma-separated list, or "all"
    --stop-at       "light_metrics" (inference + BLEU/chrF only)
                    "report"        (full pipeline incl. judge — default)

    Examples:
        # One language, full pipeline (old "phase4", run fresh)
        modal run modal_app.py::run_pipeline --slug greek

        # One language, inference + light metrics only (old "compare")
        modal run modal_app.py::run_pipeline --slug greek --stop-at light_metrics

        # A few languages in parallel
        modal run modal_app.py::run_pipeline --slug arabic,hebrew,amharic

        # All 17 (old "phase5") — prints cost estimate, confirms before launch
        modal run modal_app.py::run_pipeline --slug all

        # Resume an existing run — already-completed languages are skipped
        modal run modal_app.py::run_pipeline --slug all --run-id <id>
    """
    from src.pipeline.entrypoints import estimate_cost
    from src.pipeline.run import generate_run_id

    slugs = [] if slug == "all" else [s.strip() for s in slug.split(",") if s.strip()]
    specs = [s for s in ALL_MODELS if not slugs or s.slug in slugs]
    if not specs:
        print(f"ERROR: no matching slug(s) for '{slug}'. Valid slugs: "
              f"{', '.join(s.slug for s in ALL_MODELS)}")
        return

    run_id = run_id or generate_run_id()
    stop_state = State(stop_at)

    cost = estimate_cost(specs, judge_limit, swap_runs) if stop_state == State.REPORT else None

    print(f"\n{'='*65}\n  RUN PIPELINE — PRE-FLIGHT\n{'='*65}")
    print(f"  Run ID:        {run_id}")
    print(f"  Languages:     {len(specs)}")
    print(f"  Stop at:       {stop_state.value}")
    print(f"  Task:          {task}  |  Prompts/model: {limit}")
    if cost:
        print(f"  Judge:         {judge_model}  |  Limit: {judge_limit} prompts/model")
        print(f"\n  ESTIMATED COST:")
        print(f"    Regional inference: ${cost['regional_inference_usd']:.2f}")
        print(f"    Gemma-4 inference:  ${cost['gemma4_inference_usd']:.2f}")
        print(f"    Judge API:          ${cost['judge_usd']:.2f}  ({cost['total_judge_calls']:,} calls)")
        print(f"    TOTAL:              ~${cost['total_usd']:.2f}")
    print(f"{'='*65}\n  Languages:")
    for s in specs:
        base = " ⚠ base" if s.model_id in BASE_MODELS else ""
        print(f"    {s.language:<26} {s.model_id:<22} gpu={s.gpu_preset}{base}")

    if len(specs) > 1:
        print(f"\n  Press Enter to launch, Ctrl-C to abort...")
        input()

    summary = _run_pipeline.remote(
        run_id=run_id, slugs=slugs, stop_at_value=stop_state.value, task=task, limit=limit,
        judge_model=judge_model, judge_limit=judge_limit, swap_runs=swap_runs,
        skip_model_metrics=skip_model_metrics,
    )

    c = summary.get("classification_counts", {})
    print(f"\nRun complete — Run ID: {run_id}")
    if any(c.values()):
        print(f"  A={c.get('A',0)} B={c.get('B',0)} C={c.get('C',0)} D={c.get('D',0)} E={c.get('E',0)}")
    if summary.get("failed"):
        print(f"  Failed: {summary['failed']}")