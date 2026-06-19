"""
src/pipeline/entrypoints.py
=============================
One function drives every evaluation run: run_pipeline().

There is no compare_one / evaluate_one / evaluate_many anymore. Those
existed only because someone needed to (a) write a manifest, (b) run
1-or-N specs through the state machine, (c) render a report — and all
three of those are the same code regardless of how many specs you pass
or where you tell it to stop.

"Phase 3"  = run_pipeline(specs=[one], stop_at=LIGHT_METRICS)
"Phase 4"  = run_pipeline(specs=[one], stop_at=REPORT, resume_from_existing=True)
"Phase 5"  = run_pipeline(specs=[...17], stop_at=REPORT)

The only thing that actually branches on len(specs) is whether the
Gemma-4 baseline and the per-language work happen inline (1 spec, no
Modal spawn needed) or fanned out to parallel containers (N specs).
That branch lives inside run_pipeline, not as a separate function.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

from src.pipeline.state_machine import (
    LanguageRun,
    LanguageSpec,
    RunContext,
    State,
    WorkerHandles,
    advance,
    ensure_gemma4_baseline,
    language_run_to_dict,
    run_fanout,
)


# ---------------------------------------------------------------------------
# Cost estimation — run-level, not pipeline-state concern
# ---------------------------------------------------------------------------

_GPU_COST_PER_HR = {"t4": 0.59, "l4": 0.80, "l40s": 1.95, "a100_80gb": 2.50}
_INFER_HOURS     = {"t4": 0.08, "l4": 0.33, "l40s": 0.40, "a100_80gb": 0.42}


def estimate_cost(specs: list[LanguageSpec], judge_limit: int, swap_runs: int = 2, dimensions: int = 3) -> dict:
    regional = sum(_GPU_COST_PER_HR[s.gpu_preset] * _INFER_HOURS[s.gpu_preset] for s in specs)
    gemma    = _GPU_COST_PER_HR["a100_80gb"] * _INFER_HOURS["a100_80gb"]
    calls    = len(specs) * judge_limit * swap_runs * dimensions
    judge    = calls * 600 / 1_000_000 * 1.50  # ~600 tokens/call, $1.50/1M input
    return {
        "regional_inference_usd": round(regional, 2),
        "gemma4_inference_usd":   round(gemma, 2),
        "judge_usd":              round(judge, 2),
        "total_usd":              round(regional + gemma + judge, 2),
        "total_judge_calls":      calls,
    }


# ---------------------------------------------------------------------------
# run_pipeline — the only entrypoint
# ---------------------------------------------------------------------------

def run_pipeline(
    *,
    run_id: str,
    specs: list[LanguageSpec],
    stop_at: State,
    task: str,
    limit: int,
    handles: WorkerHandles,
    judge_model: str = "",
    judge_limit: int = 0,
    swap_runs: int = 0,
    skip_model_metrics: bool = True,
    VLLMWorkerA100: Optional[Any] = None,
    advance_remote_fn: Optional[Any] = None,
    poll_interval_s: int = 60,
) -> dict:
    """
    Runs every language in `specs` through the state machine up to `stop_at`,
    then writes a manifest, a JSON summary, and a markdown report.

    len(specs) == 1   → runs inline, in this process. advance_remote_fn unused.
    len(specs) > 1    → fans out to parallel containers via advance_remote_fn,
                         which the caller (modal_app.py) supplies as a Modal
                         .spawn() wrapper. VLLMWorkerA100 is required in this
                         case so the Gemma-4 baseline can be generated once,
                         up front, before any per-language container starts.

    If a manifest/report already exists for run_id, languages already
    present in it are skipped (resume) regardless of whether you're running
    1 or N specs — this is what "Phase 4 assumes Phase 3 already ran" and
    "Phase 5 resume" both reduce to.

    Returns a summary dict: {run_id, state_per_slug, results, classification_counts, ...}.
    """
    from src.pipeline.run import (
        append_run_to_index,
        report_path,
        run_dir,
        update_manifest_status,
        write_manifest,
    )
    from src.workers.reporter import load_report, summarize_run
    from src.pipeline.report_render import render_markdown, write_markdown_report

    if not run_id:
        from src.pipeline.run import generate_run_id
        run_id = generate_run_id()

    write_manifest(run_id, {
        "run_id":  run_id, "task": task,
        "models":  [s.model_id for s in specs] + ["gemma-4-26b"],
        "limit":   limit, "stop_at": stop_at.value, "status": "started",
    })
    append_run_to_index(run_id, status="started")

    # ── Resume detection — same check whether 1 spec or 17 ──────────────────
    existing_report = load_report(run_id)
    completed = set(existing_report.get("languages", {}).keys())
    if completed:
        print(f"[run_pipeline] Resuming — already done: {sorted(completed)}")

    ctx = RunContext(
        run_id=run_id, task=task, limit=limit,
        judge_model=judge_model, judge_limit=judge_limit, swap_runs=swap_runs,
        skip_model_metrics=skip_model_metrics, handles=handles,
    )

    # ── Gemma-4 baseline precondition ───────────────────────────────────────
    # Needed once before INFERENCE runs for anyone — whether that's 1 spec
    # running inline or 17 specs about to fan out. Skip entirely if every
    # spec is already past INFERENCE (resume case), since the baseline
    # would already exist on disk and ensure_gemma4_baseline is a no-op
    # then anyway — but for a single un-started spec we still need
    # VLLMWorkerA100 from somewhere.
    needs_baseline = any(s.slug not in completed for s in specs) and stop_at != State.PENDING
    if needs_baseline:
        a100 = VLLMWorkerA100 or handles.gpu_worker_map.get("a100_80gb")
        ensure_gemma4_baseline(ctx, a100, any_slug=specs[0].slug)

    # ── Run every spec to stop_at — inline if 1, fanned out if N ────────────
    runs: dict[str, LanguageRun] = {}
    elapsed_s = 0.0

    pending_specs = [s for s in specs if s.slug not in completed]

    if not pending_specs:
        pass  # everything already done — fall straight through to reporting

    elif len(specs) == 1 or advance_remote_fn is None:
        # Inline path — no Modal spawn needed. Used by single-language runs
        # ("Phase 3"/"Phase 4") and as a fallback if no fan-out fn is given.
        for spec in pending_specs:
            lr = advance(ctx, LanguageRun(spec=spec), stop_at=stop_at)
            runs[spec.slug] = lr
            if lr.state == State.FAILED:
                print(f"[run_pipeline] FAILED — {spec.language}: {lr.error}")

    else:
        # Fan-out path — N specs in parallel containers ("Phase 5").
        result = run_fanout(
            ctx, specs, stop_at=stop_at,
            advance_remote_fn=advance_remote_fn,
            poll_interval_s=poll_interval_s,
            resume_completed=completed,
            resume_report=existing_report,
        )
        runs = result.runs
        elapsed_s = result.elapsed_s

    failed = [slug for slug, lr in runs.items() if lr.state == State.FAILED]

    # ── Manifest status ──────────────────────────────────────────────────────
    if failed and len(failed) == len(specs):
        update_manifest_status(run_id, "failed")
    elif failed:
        update_manifest_status(run_id, "partial")
    else:
        update_manifest_status(run_id, "completed")

    # ── Reporting — identical for 1 spec or N ───────────────────────────────
    # final_report.json is the single source of truth: _do_report (inside
    # advance()) already wrote every successful language's classification
    # there. summarize_run reads it once; failed languages never reached
    # _do_report so they're patched in here with their error.
    all_slugs = [s.slug for s in specs]
    agg = summarize_run(run_id, all_slugs, task)

    for slug in failed:
        lr = runs[slug]
        agg["results"][slug] = {
            "language": lr.spec.language, "regional_model": lr.spec.model_id,
            "bleu_regional": None, "bleu_gemma4": None, "judge_win_rate": None,
            "classification": "?",
            "classification_rationale": lr.error or "unknown failure",
        }
    agg["classification_counts"] = {
        g: sum(1 for r in agg["results"].values() if r["classification"] == g)
        for g in ["A", "B", "C", "D", "E", "?"]
    }

    generated_at = datetime.now(timezone.utc).isoformat()
    cost = estimate_cost(specs, judge_limit, swap_runs) if judge_limit else None

    summary = {
        "run_id": run_id, "task": task, "stop_at": stop_at.value,
        "generated_at": generated_at,
        "models_run": len(specs), "failed": failed,
        "results": agg["results"],
        "classification_counts": agg["classification_counts"],
        "cost_estimate": cost,
        "elapsed_minutes": round(elapsed_s / 60, 1) if elapsed_s else None,
    }

    # One spec gets a per-language filename; many get the run-level name.
    # Same write, same renderer — only the filename and title differ.
    is_single = len(specs) == 1
    summary_filename = f"{specs[0].slug}_summary.json" if is_single else "phase5_summary.json"
    md_filename       = f"{specs[0].slug}_summary.md"   if is_single else "phase5_summary.md"
    title = f"Evaluation report — {specs[0].language}" if is_single else "Phase 5 evaluation report"

    summary_path = os.path.join(run_dir(run_id), "reports", summary_filename)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    md = render_markdown(
        run_id=run_id, task=task, generated_at=generated_at,
        results=agg["results"], classification_counts=agg["classification_counts"],
        title=title, cost=cost, failed=failed,
    )
    write_markdown_report(run_id, md, filename=md_filename)

    # ── Console summary ──────────────────────────────────────────────────────
    counts = summary["classification_counts"]
    print(f"\n{'='*65}\n  PIPELINE COMPLETE — {len(specs) - len(failed)}/{len(specs)} succeeded")
    print(f"{'='*65}")
    for g in ["A", "B", "C", "D", "E"]:
        print(f"  {g}: {counts.get(g, 0):2d}  {'█' * counts.get(g, 0)}")
    if failed:
        print(f"\n  Failed: {failed}")
    print(f"\n  Summary: {summary_path}")
    print(f"{'='*65}\n")

    return summary


# ---------------------------------------------------------------------------
# Single-language remote runner — what advance_remote_fn spawns per language
# ---------------------------------------------------------------------------

def run_single_language_to_state(
    *,
    run_id: str,
    spec: LanguageSpec,
    stop_at_value: str,
    task: str,
    limit: int,
    judge_model: str,
    judge_limit: int,
    swap_runs: int,
    skip_model_metrics: bool,
    handles: WorkerHandles,
) -> dict:
    """
    The function body that runs INSIDE one per-language Modal container
    during a fan-out. Drives one LanguageRun from PENDING through stop_at
    and returns a JSON-serializable dict — this is the unit advance_remote_fn
    spawns, not a separate orchestration layer.
    """
    ctx = RunContext(
        run_id=run_id, task=task, limit=limit,
        judge_model=judge_model, judge_limit=judge_limit, swap_runs=swap_runs,
        skip_model_metrics=skip_model_metrics, handles=handles,
    )
    lr = advance(ctx, LanguageRun(spec=spec), stop_at=State(stop_at_value))
    return language_run_to_dict(lr)