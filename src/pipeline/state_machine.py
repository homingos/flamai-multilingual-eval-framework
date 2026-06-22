"""
src/pipeline/state_machine.py
===============================
A single state machine drives every evaluation run, for one language
or seventeen. There is no "Phase 3 orchestrator" and "Phase 5
orchestrator" — there is one pipeline with a stop point.

State graph (per language):

    PENDING ─► INFERENCE ─► LIGHT_METRICS ─► MODEL_METRICS ─► JUDGE ─► REPORT ─► DONE
       │            │              │               │            │        │
       └────────────┴──────────────┴───────────────┴────────────┴────────┴──► FAILED

Each state is a pure function: (ctx) -> next_state. State functions
read/write the outputs volume directly via src.pipeline.run helpers
and call into Modal worker classes passed in via WorkerHandles. No
state function imports modal_app — that dependency only exists in
the @app.function wrappers in modal_app.py.

"Phase 3" = run the state machine for one language, stop_at=LIGHT_METRICS.
"Phase 4" = run the state machine for one language, stop_at=REPORT,
            on a language that already reached LIGHT_METRICS.
"Phase 5" = run the state machine for N languages in parallel, stop_at=REPORT,
            with a shared Gemma-4 precondition step run once up front.

There's one pipeline. The "phases" are just different (stop_at, fan_out)
parameters on the same run.
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
    PENDING        = "pending"
    INFERENCE      = "inference"
    LIGHT_METRICS  = "light_metrics"
    MODEL_METRICS  = "model_metrics"
    JUDGE          = "judge"
    REPORT         = "report"
    DONE           = "done"
    FAILED         = "failed"

    @property
    def order(self) -> int:
        return list(State).index(self)


# Canonical forward path. FAILED can be reached from anywhere; not part of this list.
_FORWARD_PATH = [
    State.PENDING, State.INFERENCE, State.LIGHT_METRICS,
    State.MODEL_METRICS, State.JUDGE, State.REPORT, State.DONE,
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
    gpu_worker_map:     dict       # {"l4": VLLMWorkerL4, ...}
    LightMetricWorker:  Any
    ModelMetricWorker:  Any
    JudgeWorker:        Any
    ReportGenerator:    Any


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
    spec:          LanguageSpec
    state:         State = State.PENDING
    error:         Optional[str] = None
    started_at:    Optional[datetime] = None
    finished_at:   Optional[datetime] = None
    report:        Optional[dict] = None

    @property
    def slug(self) -> str:
        return self.spec.slug


# ---------------------------------------------------------------------------
# Run-level context — shared, read-only config for a single state-machine run
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    run_id:       str
    task:         str
    limit:        int
    judge_model:  str
    judge_limit:  int
    swap_runs:    int
    skip_model_metrics: bool
    handles:      WorkerHandles


# ---------------------------------------------------------------------------
# State transition functions — one per edge in the graph
# ---------------------------------------------------------------------------
# Each function does the work for entering `lr.state` and returns True on
# success (caller advances lr.state) or False on failure (caller sets FAILED).
# Skipping is handled inside each function by checking existing volume state.

def _do_inference(ctx: RunContext, lr: LanguageRun) -> bool:
    import modal as _modal
    from src.pipeline.loader import load_samples
    from src.pipeline.run import gemma_output_path, regional_output_path

    spec = lr.spec
    reg_path   = regional_output_path(ctx.run_id, spec.slug, ctx.task)
    gemma_path = gemma_output_path(ctx.run_id, ctx.task)

    if not os.path.exists(gemma_path) or os.path.getsize(gemma_path) == 0:
        lr.error = f"Gemma-4 outputs missing at {gemma_path} — run gemma4_precondition first"
        return False

    if os.path.exists(reg_path) and os.path.getsize(reg_path) > 0:
        print(f"[{spec.slug}] INFERENCE — outputs already exist, skipping")
        return True

    samples = load_samples(ctx.task, spec.slug, limit=ctx.limit)
    worker_cls = ctx.handles.gpu_worker_map[spec.gpu_preset]
    worker     = worker_cls(model_id=spec.model_id, run_id=ctx.run_id, task=ctx.task)

    print(f"[{spec.slug}] INFERENCE — {len(samples)} samples on {spec.gpu_preset}")
    outputs = worker.generate.remote(samples)
    print(f"[{spec.slug}] INFERENCE — {len(outputs)} outputs written")

    _modal.Volume.from_name("phase2a-outputs").reload()
    return True


def _do_light_metrics(ctx: RunContext, lr: LanguageRun) -> bool:
    import modal as _modal

    spec = lr.spec
    mw = ctx.handles.LightMetricWorker()
    rh = mw.score.spawn(run_id=ctx.run_id, slug=spec.slug,   task=ctx.task,
                        language=spec.language, model_id=spec.model_id)
    gh = mw.score.spawn(run_id=ctx.run_id, slug="baseline",  task=ctx.task,
                        language=spec.language, model_id="gemma-4-26b")

    reg_scores = rh.get()
    gh.get()
    print(f"[{spec.slug}] LIGHT_METRICS — {reg_scores}")

    _modal.Volume.from_name("phase2a-outputs").reload()
    return True


def _do_model_metrics(ctx: RunContext, lr: LanguageRun) -> bool:
    import modal as _modal

    if ctx.skip_model_metrics:
        print(f"[{lr.slug}] MODEL_METRICS — skipped by config")
        return True

    spec = lr.spec
    mm = ctx.handles.ModelMetricWorker()
    rh = mm.score.spawn(run_id=ctx.run_id, slug=spec.slug,   task=ctx.task,
                        language=spec.language, model_id=spec.model_id)
    gh = mm.score.spawn(run_id=ctx.run_id, slug="baseline",  task=ctx.task,
                        language=spec.language, model_id="gemma-4-26b")

    print(f"[{spec.slug}] MODEL_METRICS — regional: {rh.get()}  gemma4: {gh.get()}")
    _modal.Volume.from_name("phase2a-outputs").reload()
    return True


def _do_judge(ctx: RunContext, lr: LanguageRun) -> bool:
    import modal as _modal

    spec     = lr.spec
    verdicts = ctx.handles.JudgeWorker().judge.remote(
        run_id=ctx.run_id, slug=spec.slug, task=ctx.task, language=spec.language,
        regional_model_id=spec.model_id, judge_model=ctx.judge_model,
        swap_runs=ctx.swap_runs, limit=ctx.judge_limit if ctx.judge_limit > 0 else None,
    )
    print(f"[{spec.slug}] JUDGE — {len(verdicts)} verdicts written")

    _modal.Volume.from_name("phase2a-outputs").reload()
    return True


def _do_report(ctx: RunContext, lr: LanguageRun) -> bool:
    spec   = lr.spec
    report = ctx.handles.ReportGenerator().generate(
        run_id=ctx.run_id, slug=spec.slug, task=ctx.task,
        language=spec.language, regional_model_id=spec.model_id,
    )
    lr.report = report

    cls = report["languages"][spec.slug]["classification"]
    rat = report["languages"][spec.slug]["classification_rationale"]
    print(f"[{spec.slug}] REPORT — {cls} | {rat}")
    return True


_TRANSITIONS: dict[State, Callable[[RunContext, LanguageRun], bool]] = {
    State.INFERENCE:     _do_inference,
    State.LIGHT_METRICS: _do_light_metrics,
    State.MODEL_METRICS: _do_model_metrics,
    State.JUDGE:         _do_judge,
    State.REPORT:        _do_report,
}


# ---------------------------------------------------------------------------
# Single-language driver — advances one LanguageRun until stop_at or FAILED
# ---------------------------------------------------------------------------

def advance(ctx: RunContext, lr: LanguageRun, stop_at: State) -> LanguageRun:
    """
    Drives lr forward from its current state to stop_at (inclusive),
    or to FAILED if any transition raises / returns False.

    This is the entire "orchestrator" — Phase 3, 4, and 5 are all just
    calls to advance() with a different stop_at and a different number
    of LanguageRun instances driven in parallel.
    """
    lr.started_at = lr.started_at or datetime.now(timezone.utc)

    while lr.state != stop_at and lr.state != State.DONE:
        target = _next_state(lr.state)
        transition_fn = _TRANSITIONS.get(target)

        if transition_fn is None:
            # DONE has no transition function — just arriving there is enough
            lr.state = target
            continue

        try:
            ok = transition_fn(ctx, lr)
        except Exception as exc:
            ok = False
            lr.error = f"{target.value} raised: {exc}"

        if not ok:
            lr.state = State.FAILED
            lr.finished_at = datetime.now(timezone.utc)
            print(f"[{lr.slug}] FAILED at {target.value}: {lr.error}")
            return lr

        lr.state = target

    lr.finished_at = datetime.now(timezone.utc)
    return lr


# ---------------------------------------------------------------------------
# Gemma-4 precondition — not a per-language state, a shared run-level gate
# ---------------------------------------------------------------------------

def ensure_gemma4_baseline(ctx: RunContext, VLLMWorkerA100: Any, any_slug: str) -> None:
    """
    Generates Gemma-4 outputs once for the whole run, if not already present.
    Must be called before any LanguageRun reaches INFERENCE.
    Idempotent — safe to call even if outputs already exist.
    """
    import modal as _modal
    from src.pipeline.loader import load_samples
    from src.pipeline.run import gemma_output_path

    out_path = gemma_output_path(ctx.run_id, ctx.task)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print("[gemma4] Baseline outputs already exist — skipping")
        return

    samples = load_samples(ctx.task, any_slug, limit=ctx.limit)
    print(f"[gemma4] Generating baseline — {len(samples)} samples")
    worker  = VLLMWorkerA100(model_id="gemma-4-26b", run_id=ctx.run_id, task=ctx.task)
    outputs = worker.generate.remote(samples)
    print(f"[gemma4] Baseline ready — {len(outputs)} outputs")

    _modal.Volume.from_name("phase2a-outputs").reload()


# ---------------------------------------------------------------------------
# Multi-language fan-out — this is what "Phase 5" reduces to
# ---------------------------------------------------------------------------

@dataclass
class FanOutResult:
    run_id:  str
    runs:    dict[str, LanguageRun]   # keyed by slug
    elapsed_s: float


def run_fanout(
    ctx: RunContext,
    specs: list[LanguageSpec],
    stop_at: State,
    advance_remote_fn: Callable[..., Any],   # spawns advance() in its own container
    poll_interval_s: int = 60,
    max_wait_s: int = 6 * 3600,
    resume_completed: Optional[set] = None,
    resume_report: Optional[dict] = None,
) -> FanOutResult:
    """
    Runs `advance()` for every spec in parallel, one Modal container each,
    polling until all reach stop_at/FAILED or max_wait_s elapses.

    advance_remote_fn(spec, stop_at) -> a Modal FunctionCall handle (the
    result of .spawn()) whose .get() returns a LanguageRun-shaped dict.
    The caller (modal_app.py) supplies this so this module never imports
    modal_app or references @app.function directly.

    resume_report, if given, is the existing final_report.json dict —
    used to populate LanguageRun.report for slugs in resume_completed so
    their classification survives into the final summary instead of being
    silently dropped.
    """
    resume_completed = resume_completed or set()
    resume_report     = resume_report or {}
    start = time.time()

    runs:    dict[str, LanguageRun] = {}
    pending: dict[str, tuple] = {}

    for spec in specs:
        if spec.slug in resume_completed:
            # Carry the already-computed report forward so classification
            # counts and the markdown summary reflect the real result,
            # not a placeholder.
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
        runs[spec.slug] = LanguageRun(spec=spec, started_at=datetime.now(timezone.utc))
        print(f"  ↗  {spec.language:<24} {spec.model_id}")

    print(f"\n[fanout] {len(pending)} containers spawned, {len(resume_completed)} resumed. Polling...\n")

    deadline = time.time() + max_wait_s
    while pending and time.time() < deadline:
        done_slugs = []
        for slug, (spec, handle) in list(pending.items()):
            try:
                result_dict = handle.get(timeout=0)
                runs[slug] = _dict_to_language_run(spec, result_dict)
                done_slugs.append(slug)
            except TimeoutError:
                pass
            except Exception as exc:
                lr = runs[slug]
                lr.state = State.FAILED
                lr.error = str(exc)[:200]
                lr.finished_at = datetime.now(timezone.utc)
                done_slugs.append(slug)

        for slug in done_slugs:
            del pending[slug]

        _print_fanout_status(runs, pending)

        if pending:
            time.sleep(poll_interval_s)

    # Timeout fallback — block briefly to collect whatever's left
    for slug, (spec, handle) in pending.items():
        try:
            result_dict = handle.get(timeout=30)
            runs[slug] = _dict_to_language_run(spec, result_dict)
        except Exception as exc:
            lr = runs[slug]
            lr.state = State.FAILED
            lr.error = f"timeout: {exc}"
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
    """Serializes a LanguageRun for crossing the Modal RPC boundary."""
    return {"state": lr.state.value, "error": lr.error, "report": lr.report}


def _print_fanout_status(runs: dict[str, LanguageRun], pending: dict) -> None:
    now = datetime.now(timezone.utc)
    done_count = sum(1 for lr in runs.values() if lr.state in (State.DONE, State.FAILED))
    print(f"\n[{now.strftime('%H:%M')}] {done_count}/{len(runs)} complete, {len(pending)} running")

    for slug, lr in sorted(runs.items()):
        if lr.state in (State.DONE, State.FAILED):
            icon = "✓" if lr.state == State.DONE else "✗"
            cls  = ""
            if lr.report:
                lang_data = lr.report.get("languages", {}).get(slug, {})
                cls = lang_data.get("classification", "")
            note = lr.error or cls
            print(f"  {icon} {lr.spec.language:<24} {lr.spec.model_id:<22} {lr.state.value:<14} {note[:50]}")

    for slug, (spec, _) in sorted(pending.items()):
        lr      = runs[slug]
        elapsed = int((now - lr.started_at).total_seconds() / 60) if lr.started_at else 0
        print(f"  ⏳ {spec.language:<24} {spec.model_id:<22} running ~{elapsed}min")