"""
Phase 2A — Modal application entrypoint (v2: pointwise evaluation).

Worker registrations + one entrypoint. All pipeline logic lives in:
  src/pipeline/state_machine.py  — state machine (PENDING→SAMPLE→INFERENCE→JUDGE→REPORT→DONE)
  src/pipeline/entrypoints.py    — run_pipeline()

v2 changes:
  - No Gemma-4 baseline — regional LLM evaluated standalone (pointwise).
  - --n-samples replaces --limit (stratified random sampling, default 200).
  - LightMetricWorker and ModelMetricWorker removed from pipeline.
  - JudgeWorker now pointwise (score 0–1 per dimension, no swap_runs).

Usage:
    modal deploy modal_app.py
    modal run modal_app.py::run_pipeline --slug german
    modal run modal_app.py::run_pipeline --slug german --task translation --n-samples 200
    modal run modal_app.py::run_pipeline --slug german,dutch,polish
    modal run modal_app.py::run_pipeline --slug all
"""
import modal
from modal_common import (
    build_registry_config,
    env_config,
    judge_image,
    registry_image,
    vllm_image,
    VOLUME_MOUNTS,
)
from src.workers.inference import VLLMWorker     as _VLLMWorker
from src.workers.judge     import JudgeWorker    as _JudgeWorker
from src.workers.reporter  import ReportGenerator as _ReportGenerator
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
                secrets=_inference_secrets, volumes=VOLUME_MOUNTS)


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


@app.cls(**_vllm_cls("A100-80GB"), max_containers=1)
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
# Judge worker (pointwise, no swap_runs)
# ---------------------------------------------------------------------------

@app.cls(image=judge_image, cpu=2, memory=2048, timeout=7200,
         secrets=[_judge_secret], volumes=VOLUME_MOUNTS)
class JudgeWorkerModal(_JudgeWorker):
    @modal.method()
    def judge(self, run_id, slug, task, language, regional_model_id,
              judge_model="gemini-3.5-flash", limit=None):
        return super().judge(
            run_id=run_id, slug=slug, task=task, language=language,
            regional_model_id=regional_model_id, judge_model=judge_model,
            limit=limit,
        )


def _handles() -> WorkerHandles:
    return WorkerHandles(
        gpu_worker_map=GPU_WORKER_MAP,
        JudgeWorker=JudgeWorkerModal,
        ReportGenerator=_ReportGenerator,
    )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

ALL_MODELS = [
    LanguageSpec("sarvam-m-tamil",           "Tamil",                "tamil",                "a100_80gb"),
    LanguageSpec("sarvam-m-marathi",         "Marathi",              "marathi",              "a100_80gb"),
    LanguageSpec("sarvam-m-kannada",         "Kannada",              "kannada",              "a100_80gb"),
    LanguageSpec("sarvam-m-gujarati",        "Gujarati",             "gujarati",             "a100_80gb"),
    LanguageSpec("jais-2-8b",                "Arabic",               "arabic",               "l4"),
    LanguageSpec("dictalm-3-nemotron-12b",   "Hebrew",               "hebrew",               "l40s"),
    LanguageSpec("exaone-3-5-32b",           "Korean",               "korean",               "a100_80gb"),
    LanguageSpec("mallam-5b",                "Malay",                "malay",                "l4"),
    LanguageSpec("swahili-gemma-7b",         "Swahili",              "swahili",              "l4"),
    LanguageSpec("walia-llm-7b",             "Amharic",              "amharic",              "l4"),
    LanguageSpec("lucie-7b",                 "French",               "french",               "l4"),
    LanguageSpec("viking-7b",                "Swedish",              "swedish",              "l4"),
    LanguageSpec("csmpt-7b",                 "Czech",                "czech",                "l4"),
    LanguageSpec("krikri-8b",                "Greek",                "greek",                "l4"),
    LanguageSpec("tucano-2b4",               "Brazilian Portuguese", "brazilian_portuguese", "l4"),
    LanguageSpec("goldfish-mri-39m",         "Māori",                "maori",                "t4"),
    LanguageSpec("goldfish-tpi-125m",        "Tok Pisin",            "tok_pisin",            "t4"),
    # EuroLLM-22B — EU multilingual expansion (24 languages)
    LanguageSpec("eurollm-22b", "German",     "german",     "a100_80gb"),
    LanguageSpec("eurollm-22b", "Italian",    "italian",    "a100_80gb"),
    LanguageSpec("eurollm-22b", "Portuguese", "portuguese", "a100_80gb"),
    LanguageSpec("eurollm-22b", "Dutch",      "dutch",      "a100_80gb"),
    LanguageSpec("eurollm-22b", "Polish",     "polish",     "a100_80gb"),
    LanguageSpec("eurollm-22b", "Romanian",   "romanian",   "a100_80gb"),
    LanguageSpec("eurollm-22b", "Ukrainian",  "ukrainian",  "a100_80gb"),
    LanguageSpec("eurollm-22b", "Russian",    "russian",    "a100_80gb"),
    LanguageSpec("eurollm-22b", "Danish",     "danish",     "a100_80gb"),
    LanguageSpec("eurollm-22b", "Finnish",    "finnish",    "a100_80gb"),
    LanguageSpec("eurollm-22b", "Hungarian",  "hungarian",  "a100_80gb"),
    LanguageSpec("eurollm-22b", "Croatian",   "croatian",   "a100_80gb"),
    LanguageSpec("eurollm-22b", "Slovak",     "slovak",     "a100_80gb"),
    LanguageSpec("eurollm-22b", "Slovenian",  "slovenian",  "a100_80gb"),
    LanguageSpec("eurollm-22b", "Bulgarian",  "bulgarian",  "a100_80gb"),
    LanguageSpec("eurollm-22b", "Lithuanian", "lithuanian", "a100_80gb"),
    LanguageSpec("eurollm-22b", "Latvian",    "latvian",    "a100_80gb"),
    LanguageSpec("eurollm-22b", "Estonian",   "estonian",   "a100_80gb"),
    LanguageSpec("eurollm-22b", "Irish",      "irish",      "a100_80gb"),
    LanguageSpec("eurollm-22b", "Norwegian",  "norwegian",  "a100_80gb"),
    LanguageSpec("eurollm-22b", "Maltese",    "maltese",    "a100_80gb"),
    LanguageSpec("eurollm-22b", "Swedish",    "swedish",    "a100_80gb"),
    LanguageSpec("eurollm-22b", "Czech",      "czech",      "a100_80gb"),
    LanguageSpec("eurollm-22b", "Greek",      "greek",      "a100_80gb"),
]

BASE_MODELS = {"viking-7b", "csmpt-7b", "goldfish-mri-39m", "goldfish-tpi-125m"}


# ---------------------------------------------------------------------------
# Internal Modal functions
# ---------------------------------------------------------------------------

@app.function(image=vllm_image, cpu=2, memory=4096, timeout=86400,
              secrets=[_registry_secret, _judge_secret], volumes=VOLUME_MOUNTS)
def _run_one_language(run_id: str, model_id: str, language: str, slug: str,
                      gpu_preset: str, stop_at_value: str, task: str,
                      n_samples: int, seed: int, judge_model: str) -> dict:
    """Container spawned once per language during fan-out."""
    from src.pipeline.entrypoints import run_single_language_to_state
    spec = LanguageSpec(model_id, language, slug, gpu_preset)
    return run_single_language_to_state(
        run_id=run_id, spec=spec, stop_at_value=stop_at_value, task=task,
        n_samples=n_samples, seed=seed, judge_model=judge_model, handles=_handles(),
    )


@app.function(image=vllm_image, cpu=2, memory=4096, timeout=86400,
              secrets=[_registry_secret, _judge_secret], volumes=VOLUME_MOUNTS)
def _run_pipeline(run_id: str, slugs: list, stop_at_value: str, task: str,
                  n_samples: int, seed: int, judge_model: str) -> dict:
    from src.pipeline.entrypoints import run_pipeline

    specs = [s for s in ALL_MODELS if s.slug in slugs] if slugs else ALL_MODELS

    def _spawn(spec, stop_at):
        return _run_one_language.spawn(
            run_id=run_id, model_id=spec.model_id, language=spec.language,
            slug=spec.slug, gpu_preset=spec.gpu_preset, stop_at_value=stop_at.value,
            task=task, n_samples=n_samples, seed=seed, judge_model=judge_model,
        )

    return run_pipeline(
        run_id=run_id, specs=specs, stop_at=State(stop_at_value), task=task,
        n_samples=n_samples, seed=seed, handles=_handles(), judge_model=judge_model,
        advance_remote_fn=_spawn,
    )


# ---------------------------------------------------------------------------
# run_pipeline — the one command
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_pipeline(slug="all", model_id="", stop_at="report", task="translation",
                 n_samples=200, seed=42, judge_model="gemini-3.5-flash", run_id=""):
    """
    Run the v2 pointwise evaluation pipeline for one language, a few, or all.

    --slug          one slug, comma-separated list, or "all"
    --task          "translation" or "instructions"
    --n-samples     number of stratified samples per run (default 200)
    --stop-at       "sample" | "inference" | "judge" | "report" (default)

    Examples:
        # One language, full pipeline
        modal run modal_app.py::run_pipeline --slug german

        # Just sample + inference (no judge)
        modal run modal_app.py::run_pipeline --slug german --stop-at inference

        # A few languages in parallel
        modal run modal_app.py::run_pipeline --slug dutch,polish,romanian

        # All models
        modal run modal_app.py::run_pipeline --slug all

        # Resume an existing run
        modal run modal_app.py::run_pipeline --slug all --run-id <id>
    """
    from src.pipeline.entrypoints import estimate_cost
    from src.pipeline.run import generate_run_id

    n_samples = int(n_samples)
    seed      = int(seed)

    slugs     = [] if slug == "all" else [s.strip() for s in slug.split(",") if s.strip()]
    model_ids = [m.strip() for m in model_id.split(",") if m.strip()] if model_id else []
    specs = [s for s in ALL_MODELS
             if (not slugs or s.slug in slugs)
             and (not model_ids or s.model_id in model_ids)]
    if not specs:
        print(f"ERROR: no matching slug(s) for '{slug}'. Valid slugs: "
              f"{', '.join(s.slug for s in ALL_MODELS)}")
        return

    run_id     = run_id or generate_run_id()
    stop_state = State(stop_at)

    cost = estimate_cost(specs, n_samples) if stop_state == State.DONE else None

    print(f"\n{'='*65}\n  RUN PIPELINE — PRE-FLIGHT\n{'='*65}")
    print(f"  Run ID:        {run_id}")
    print(f"  Languages:     {len(specs)}")
    print(f"  Task:          {task}  |  Samples: {n_samples} (stratified, seed={seed})")
    print(f"  Stop at:       {stop_state.value}")
    print(f"  Judge:         {judge_model}")
    if cost:
        print(f"\n  ESTIMATED COST:")
        print(f"    Regional inference: ${cost['regional_inference_usd']:.2f}")
        print(f"    Judge API:          ${cost['judge_usd']:.2f}  ({cost['total_judge_calls']:,} calls)")
        print(f"    TOTAL:              ~${cost['total_usd']:.2f}")
    print(f"{'='*65}\n  Languages:")
    for s in specs:
        base = " ⚠ base" if s.model_id in BASE_MODELS else ""
        print(f"    {s.language:<26} {s.model_id:<22} gpu={s.gpu_preset}{base}")

    if len(specs) > 1:
        print(f"\n  Press Enter to launch, Ctrl-C to abort...")
        input()

    print(f"\n  Run ID: {run_id}")
    print(f"  (Use --detach to disconnect after launch)\n")

    _run_pipeline.remote(
        run_id=run_id, slugs=slugs, stop_at_value=stop_state.value, task=task,
        n_samples=n_samples, seed=seed, judge_model=judge_model,
    )

    print(f"\n  Run complete — Run ID: {run_id}")
    print(f"  Check results: modal run modal_app.py::check_run --run-id {run_id}")


# ---------------------------------------------------------------------------
# check_run + utilities
# ---------------------------------------------------------------------------

@app.function(image=vllm_image, cpu=1, memory=512, timeout=60,
              secrets=[_registry_secret], volumes=VOLUME_MOUNTS)
def _check_run_impl(run_id: str, target_slugs: list) -> str:
    import json, os
    from src.pipeline.run import manifest_path
    from src.workers.reporter import load_report

    mpath = manifest_path(run_id)
    if not os.path.exists(mpath):
        return f"Run not found: {run_id}"

    with open(mpath) as f:
        manifest = json.load(f)

    report = load_report(run_id)
    langs  = report.get("languages", {})

    lines = [
        f"\nRun ID : {run_id}",
        f"Task   : {manifest.get('task', '?')}",
        f"Status : {manifest.get('status', '?')}",
        f"Stop at: {manifest.get('stop_at', '?')}",
        f"Samples: {manifest.get('n_samples', '?')}",
        f"\nLanguages ({len(langs)}/{len(ALL_MODELS)} completed):",
    ]

    for spec in ALL_MODELS:
        if target_slugs and spec.slug not in target_slugs:
            continue
        entry = langs.get(spec.language, {})
        if entry:
            grade = entry.get("classification", "?")
            score = entry.get("avg_score")
            score_s = f"  avg={score:.2f}" if score is not None else ""
            lines.append(f"  ✓  {spec.language:<26} Grade {grade}{score_s}")
        else:
            lines.append(f"  …  {spec.language:<26} (running)")

    counts = {g: sum(1 for e in langs.values() if e.get("classification") == g)
              for g in ["A", "B", "C", "D"]}
    if any(counts.values()):
        lines.append(f"\nGrades: A={counts['A']} B={counts['B']} C={counts['C']} D={counts['D']}")
    return "\n".join(lines)


@app.function(image=vllm_image, volumes=VOLUME_MOUNTS, timeout=120)
def _clear_weights_cache(path: str) -> str:
    import shutil, os
    full = f"/data/weights/{path}"
    if os.path.exists(full):
        shutil.rmtree(full)
        return f"Deleted {full}"
    return f"Not found: {full}"


@app.local_entrypoint()
def clear_csmpt_cache():
    """Delete the stale flash_attn_triton.py cache for BUT-FIT/csmpt7b."""
    result = _clear_weights_cache.remote("modules/transformers_modules/BUT_hyphen_FIT/csmpt7b")
    print(result)


@app.local_entrypoint()
def check_run(run_id: str = "", slug: str = "all"):
    """
    Poll the status of a pipeline run from the Modal volume.

    Examples:
        modal run modal_app.py::check_run --run-id 2026-06-23_120000_abc123
        modal run modal_app.py::check_run --slug german
    """
    target_slugs = (
        [s.strip() for s in slug.split(",") if s.strip()]
        if slug != "all" else []
    )
    print(_check_run_impl.remote(run_id, target_slugs))
