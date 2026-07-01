"""
src/pipeline/state_machine.py
===============================
State machine for the v2 pointwise evaluation pipeline.

State graph (per language):

    PENDING ─► SAMPLE ─► INFERENCE ─► JUDGE ─► REPORT ─► DONE
       │           │           │          │        │
       └───────────┴───────────┴──────────┴────────┴──► FAILED

Key changes from v1:
- No Gemma-4 baseline: regional LLM is evaluated standalone.
- SAMPLE state: stratified random sampler runs before inference (200 samples, configurable).
- LIGHT_METRICS and MODEL_METRICS removed: metrics derived from judge scores only.
- JUDGE is pointwise: Gemini scores each output 0–1 per rubric dimension.

Each state is a pure function: (ctx, lr) -> bool (success/failure).
State functions read/write the outputs volume via src.pipeline.run helpers
and call into Modal worker classes passed via WorkerHandles.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class State(str, Enum):
    PENDING   = "pending"
    SAMPLE    = "sample"
    INFERENCE = "inference"
    JUDGE     = "judge"
    REPORT    = "report"
    DONE      = "done"
    FAILED    = "failed"

    @property
    def order(self) -> int:
        return list(State).index(self)


_FORWARD_PATH = [
    State.PENDING, State.SAMPLE, State.INFERENCE,
    State.JUDGE, State.REPORT, State.DONE,
]


def _next_state(current: State) -> State:
    idx = _FORWARD_PATH.index(current)
    return _FORWARD_PATH[idx + 1]


# ---------------------------------------------------------------------------
# Worker handles — injected, not imported, to avoid circularity with modal_app
# ---------------------------------------------------------------------------

@dataclass
class WorkerHandles:
    """Modal worker classes, passed in by the @app.function wrapper in modal_app.py."""
    gpu_worker_map:  dict   # {"l4": VLLMWorkerL4, ...}
    JudgeWorker:     Any
    ReportGenerator: Any


# ---------------------------------------------------------------------------
# Per-language run config + live state
# ---------------------------------------------------------------------------

@dataclass
class LanguageSpec:
    """Static config for one language's run — never mutated after creation."""
    model_id:   str
    language:   str
    slug:       str
    gpu_preset: str


@dataclass
class LanguageRun:
    """Mutable state for one language as it moves through the pipeline."""
    spec:        LanguageSpec
    state:       State = State.PENDING
    error:       Optional[str] = None
    started_at:  Optional[datetime] = None
    finished_at: Optional[datetime] = None
    report:      Optional[dict] = None

    @property
    def slug(self) -> str:
        return self.spec.slug


# ---------------------------------------------------------------------------
# Run-level context — shared, read-only config for a single run
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    run_id:      str
    task:        str
    n_samples:   int    # stratified sample size (default 200)
    seed:        int    # random seed for sampler (default 42)
    judge_model: str
    handles:     WorkerHandles


# ---------------------------------------------------------------------------
# State transition functions
# ---------------------------------------------------------------------------

def _do_sample(ctx: RunContext, lr: LanguageRun) -> bool:
    """
    Runs the stratified random sampler and writes sampled IDs to the volume.
    Idempotent: skips if sampled_ids file already exists.
    """
    from src.pipeline.run import sampled_ids_path
    from src.pipeline.sampler import sample_stratified

    out_path = sampled_ids_path(ctx.run_id, lr.slug, ctx.task)
    if os.path.exists(out_path):
        print(f"[{lr.slug}] SAMPLE — already done, skipping")
        return True

    samples = sample_stratified(ctx.task, lr.slug, n=ctx.n_samples, seed=ctx.seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump([s["id"] for s in samples], f)

    # Stash full sample records so inference can reuse without re-reading
    lr._sampled_records = samples  # type: ignore[attr-defined]
    print(f"[{lr.slug}] SAMPLE — {len(samples)} samples selected")
    return True


def _do_inference(ctx: RunContext, lr: LanguageRun) -> bool:
    """
    Runs the regional LLM on the sampled subset.
    Loads sampled IDs from volume, filters full dataset, runs inference.
    """
    import modal as _modal
    from src.pipeline.run import sampled_ids_path, regional_output_path
    from src.pipeline.loader import load_samples

    spec     = lr.spec
    reg_path = regional_output_path(ctx.run_id, spec.slug, ctx.task)

    if os.path.exists(reg_path) and os.path.getsize(reg_path) > 0:
        print(f"[{spec.slug}] INFERENCE — outputs already exist, skipping")
        return True

    # Load sampled IDs
    ids_path = sampled_ids_path(ctx.run_id, spec.slug, ctx.task)
    if not os.path.exists(ids_path):
        lr.error = f"Sampled IDs file missing at {ids_path} — SAMPLE state must run first"
        return False

    with open(ids_path) as f:
        sampled_ids = set(json.load(f))

    all_samples = load_samples(ctx.task, spec.slug)
    samples     = [s for s in all_samples if s["id"] in sampled_ids]

    if not samples:
        lr.error = f"No samples matched sampled IDs for {spec.slug}/{ctx.task}"
        return False

    worker_cls = ctx.handles.gpu_worker_map[spec.gpu_preset]
    worker     = worker_cls(model_id=spec.model_id, run_id=ctx.run_id, task=ctx.task)

    print(f"[{spec.slug}] INFERENCE — {len(samples)} samples on {spec.gpu_preset}")
    outputs = worker.generate.remote(samples)
    print(f"[{spec.slug}] INFERENCE — {len(outputs)} outputs written")

    _modal.Volume.from_name("phase2a-outputs").reload()
    return True


def _do_judge(ctx: RunContext, lr: LanguageRun) -> bool:
    """
    Pointwise judge: Gemini scores each output 0–1 per rubric dimension.
    No Gemma-4 comparison — regional output is evaluated standalone.
    """
    import modal as _modal

    spec     = lr.spec
    verdicts = ctx.handles.JudgeWorker().judge.remote(
        run_id=ctx.run_id,
        slug=spec.slug,
        task=ctx.task,
        language=spec.language,
        regional_model_id=spec.model_id,
        judge_model=ctx.judge_model,
    )
    print(f"[{spec.slug}] JUDGE — {len(verdicts)} verdicts written")

    _modal.Volume.from_name("phase2a-outputs").reload()
    return True


def _do_report(ctx: RunContext, lr: LanguageRun) -> bool:
    spec   = lr.spec
    report = ctx.handles.ReportGenerator().generate(
        run_id=ctx.run_id,
        slug=spec.slug,
        task=ctx.task,
        language=spec.language,
        regional_model_id=spec.model_id,
    )
    lr.report = report

    cls = report["languages"][spec.slug]["classification"]
    avg = report["languages"][spec.slug].get("avg_score", "?")
    print(f"[{spec.slug}] REPORT — {cls} | avg_score={avg}")
    return True


_TRANSITIONS: dict[State, Callable[[RunContext, LanguageRun], bool]] = {
    State.SAMPLE:     _do_sample,
    State.INFERENCE:  _do_inference,
    State.JUDGE:      _do_judge,
    State.REPORT:     _do_report,
}


# ---------------------------------------------------------------------------
# Single-language driver
# ---------------------------------------------------------------------------

def advance(ctx: RunContext, lr: LanguageRun, stop_at: State) -> LanguageRun:
    """
    Drives lr forward from its current state to stop_at (inclusive),
    or to FAILED if any transition raises / returns False.
    """
    lr.started_at = lr.started_at or datetime.now(timezone.utc)

    while lr.state != stop_at and lr.state != State.DONE:
        target        = _next_state(lr.state)
        transition_fn = _TRANSITIONS.get(target)

        if transition_fn is None:
            lr.state = target
            continue

        try:
            ok = transition_fn(ctx, lr)
        except Exception as exc:
            ok = False
            lr.error = f"{target.value} raised: {exc}"

        if not ok:
            lr.state      = State.FAILED
            lr.finished_at = datetime.now(timezone.utc)
            print(f"[{lr.slug}] FAILED at {target.value}: {lr.error}")
            return lr

        lr.state = target

    lr.finished_at = datetime.now(timezone.utc)
    return lr


# ---------------------------------------------------------------------------
# Multi-language fan-out
# ---------------------------------------------------------------------------

@dataclass
class FanOutResult:
    run_id:    str
    runs:      dict[str, LanguageRun]
    elapsed_s: float


def run_fanout(
    ctx: RunContext,
    specs: list[LanguageSpec],
    stop_at: State,
    advance_remote_fn: Callable[..., Any],
    poll_interval_s: int = 60,
    max_wait_s: int = 6 * 3600,
    resume_completed: Optional[set] = None,
    resume_report: Optional[dict] = None,
) -> FanOutResult:
    """
    Runs advance() for every spec in parallel, one Modal container each.
    advance_remote_fn(spec, stop_at) must return a Modal FunctionCall handle.
    """
    resume_completed = resume_completed or set()
    resume_report    = resume_report or {}
    start = time.time()

    runs:    dict[str, LanguageRun] = {}
    pending: dict[str, tuple] = {}

    for spec in specs:
        if spec.slug in resume_completed:
            lr = LanguageRun(
                spec=spec,
                state=State.DONE,
                report=resume_report if spec.slug in resume_report.get("languages", {}) else None,
                finished_at=datetime.now(timezone.utc),
            )
            runs[spec.slug] = lr
            continue

        handle = advance_remote_fn(spec=spec, stop_at=stop_at)
        pending[spec.slug] = (spec, handle)
        runs[spec.slug]    = LanguageRun(spec=spec, started_at=datetime.now(timezone.utc))
        print(f"  ↗  {spec.language:<24} {spec.model_id}")

    print(f"\n[fanout] {len(pending)} containers spawned, {len(resume_completed)} resumed. Polling...\n")

    deadline = time.time() + max_wait_s
    while pending and time.time() < deadline:
        done_slugs = []
        for slug, (spec, handle) in list(pending.items()):
            try:
                result_dict = handle.get(timeout=0)
                runs[slug]  = _dict_to_language_run(spec, result_dict)
                done_slugs.append(slug)
            except TimeoutError:
                pass
            except Exception as exc:
                lr = runs[slug]
                lr.state      = State.FAILED
                lr.error      = str(exc)[:200]
                lr.finished_at = datetime.now(timezone.utc)
                done_slugs.append(slug)

        for slug in done_slugs:
            del pending[slug]

        _print_fanout_status(runs, pending)

        if pending:
            time.sleep(poll_interval_s)

    for slug, (spec, handle) in pending.items():
        try:
            result_dict = handle.get(timeout=30)
            runs[slug]  = _dict_to_language_run(spec, result_dict)
        except Exception as exc:
            lr = runs[slug]
            lr.state      = State.FAILED
            lr.error      = f"timeout: {exc}"
            lr.finished_at = datetime.now(timezone.utc)

    return FanOutResult(run_id=ctx.run_id, runs=runs, elapsed_s=time.time() - start)


def _dict_to_language_run(spec: LanguageSpec, d: dict) -> LanguageRun:
    return LanguageRun(
        spec=spec,
        state=State(d.get("state", State.FAILED.value)),
        error=d.get("error"),
        report=d.get("report"),
        finished_at=datetime.now(timezone.utc),
    )


def language_run_to_dict(lr: LanguageRun) -> dict:
    return {"state": lr.state.value, "error": lr.error, "report": lr.report}


def _print_fanout_status(runs: dict[str, LanguageRun], pending: dict) -> None:
    now        = datetime.now(timezone.utc)
    done_count = sum(1 for lr in runs.values() if lr.state in (State.DONE, State.FAILED))
    print(f"\n[{now.strftime('%H:%M')}] {done_count}/{len(runs)} complete, {len(pending)} running")

    for slug, lr in sorted(runs.items()):
        if lr.state in (State.DONE, State.FAILED):
            icon     = "✓" if lr.state == State.DONE else "✗"
            cls      = ""
            if lr.report:
                cls = lr.report.get("languages", {}).get(slug, {}).get("classification", "")
            note = lr.error or cls
            print(f"  {icon} {lr.spec.language:<24} {lr.spec.model_id:<22} {lr.state.value:<14} {note[:50]}")

    for slug, (spec, _) in sorted(pending.items()):
        lr      = runs[slug]
        elapsed = int((now - lr.started_at).total_seconds() / 60) if lr.started_at else 0
        print(f"  ⏳ {spec.language:<24} {spec.model_id:<22} running ~{elapsed}min")
