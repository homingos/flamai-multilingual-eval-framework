"""
src/pipeline/entrypoints.py
=============================
Composes the state machine (state_machine.py) into the three call shapes
modal_app.py needs. There is no separate "Phase 3/4/5 orchestrator" —
these are all the same advance()/run_fanout() calls with different
stop_at and fan-out size.

    compare_one(...)   — 1 language, stop_at=LIGHT_METRICS   ("Phase 3")
    evaluate_one(...)  — 1 language, stop_at=REPORT           ("Phase 4")
    evaluate_many(...) — N languages, stop_at=REPORT          ("Phase 5")

Manifest writing, run-level summaries, and cost estimation live here
because they're run-level concerns, not pipeline-state concerns.
"""
from __future__ import annotations

import json
import os
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

_GPU_COST_PER_HR  = {"t4": 0.59, "l4": 0.80, "l40s": 1.95, "a100_80gb": 2.50}
_INFER_HOURS      = {"t4": 0.08, "l4": 0.33, "l40s": 0.40, "a100_80gb": 0.42}


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
# "Phase 3" — one language, stop after light metrics
# ---------------------------------------------------------------------------

def compare_one(
    *,
    run_id: str,
    spec: LanguageSpec,
    task: str,
    limit: int,
    handles: WorkerHandles,
) -> dict:
    """
    Runs inference + light metrics for one language vs the Gemma-4 baseline.
    Generates the Gemma-4 baseline itself if it doesn't exist yet (single-
    language convenience — Phase 5 generates it once up front instead).
    Returns a result dict with state, scores, and paths.
    """
    from src.pipeline.run import (
        append_run_to_index,
        gemma_output_path,
        generate_run_id,
        regional_output_path,
        update_manifest_status,
        write_manifest,
    )

    run_id = run_id or generate_run_id()
    print(f"\nRun ID:   {run_id}\nLanguage: {spec.language} ({spec.slug})\nTask:     {task}\nLimit:    {limit}\n")

    write_manifest(run_id, {
        "run_id": run_id, "task_scope": [task], "slug": spec.slug,
        "language": spec.language, "models": [spec.model_id, "gemma-4-26b"],
        "inference_config": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512},
        "status": "started",
    })
    append_run_to_index(run_id, status="started")

    ctx = RunContext(
        run_id=run_id, task=task, limit=limit,
        judge_model="", judge_limit=0, swap_runs=0,
        skip_model_metrics=True, handles=handles,
    )

    # Phase 3 is a single-language convenience path — ensure the baseline
    # exists inline rather than requiring a separate precondition call.
    ensure_gemma4_baseline(ctx, handles.gpu_worker_map["a100_80gb"], any_slug=spec.slug)

    lr = LanguageRun(spec=spec)
    lr = advance(ctx, lr, stop_at=State.LIGHT_METRICS)

    if lr.state == State.FAILED:
        update_manifest_status(run_id, "inference_partial")
        print(f"FAILED: {lr.error}")
        return {"run_id": run_id, "state": lr.state.value, "error": lr.error}

    update_manifest_status(run_id, "completed")

    reg_path   = regional_output_path(run_id, spec.slug, task)
    gemma_path = gemma_output_path(run_id, task)
    print(f"\nOutputs:  {reg_path}\n          {gemma_path}")
    print(f"Metrics:  runs/{run_id}/metrics/\nRun ID:   {run_id}\n")

    return {"run_id": run_id, "state": lr.state.value}


# ---------------------------------------------------------------------------
# "Phase 4" — one language, stop after report (assumes light metrics done)
# ---------------------------------------------------------------------------

def evaluate_one(
    *,
    run_id: str,
    spec: LanguageSpec,
    task: str,
    judge_model: str,
    judge_limit: int,
    swap_runs: int,
    skip_model_metrics: bool,
    handles: WorkerHandles,
) -> dict:
    """
    Runs model metrics + judge + report for one language.
    Requires Phase-3-equivalent outputs (regional + Gemma-4 inference) to
    already exist — does not run inference itself.
    """
    from src.pipeline.run import (
        gemma_output_path,
        regional_output_path,
        update_manifest_status,
    )

    reg_path   = regional_output_path(run_id, spec.slug, task)
    gemma_path = gemma_output_path(run_id, task)
    for path, label in [(reg_path, spec.model_id), (gemma_path, "Gemma-4")]:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise RuntimeError(
                f"evaluate_one requires existing inference outputs.\n"
                f"Missing: {label} → {path}\n"
                f"Run compare_one (Phase 3) first."
            )

    print(f"\n{'='*60}\n  EVALUATE — {spec.language} ({spec.slug}) / {task}")
    print(f"  Run ID: {run_id}  Model: {spec.model_id}  Judge: {judge_model}")
    print(f"{'='*60}\n")

    ctx = RunContext(
        run_id=run_id, task=task, limit=0,
        judge_model=judge_model, judge_limit=judge_limit, swap_runs=swap_runs,
        skip_model_metrics=skip_model_metrics, handles=handles,
    )

    # Already past INFERENCE/LIGHT_METRICS — start from there.
    lr = LanguageRun(spec=spec, state=State.LIGHT_METRICS)
    lr = advance(ctx, lr, stop_at=State.REPORT)

    if lr.state == State.FAILED:
        update_manifest_status(run_id, "failed")
        print(f"FAILED: {lr.error}")
        return {"run_id": run_id, "state": lr.state.value, "error": lr.error}

    update_manifest_status(run_id, "completed")

    cls = lr.report["languages"][spec.slug]["classification"]
    rat = lr.report["languages"][spec.slug]["classification_rationale"]
    print(f"\n{'='*60}\n  EVALUATE COMPLETE — {spec.language}")
    print(f"  Classification: {cls}\n  Rationale:      {rat}\n{'='*60}\n")

    # Same renderer evaluate_many uses — one row instead of seventeen.
    from datetime import datetime, timezone
    from src.workers.reporter import summarize_run
    from src.pipeline.report_render import render_markdown, write_markdown_report

    agg = summarize_run(run_id, [spec.slug], task)
    md  = render_markdown(
        run_id=run_id, task=task, generated_at=datetime.now(timezone.utc).isoformat(),
        results=agg["results"], classification_counts=agg["classification_counts"],
        title=f"Evaluation Report — {spec.language}",
    )
    write_markdown_report(run_id, md, filename=f"{spec.slug}_summary.md")

    return {"run_id": run_id, "state": lr.state.value, "report": lr.report}


# ---------------------------------------------------------------------------
# "Phase 5" — N languages in parallel, stop after report
# ---------------------------------------------------------------------------

def evaluate_many(
    *,
    run_id: str,
    specs: list[LanguageSpec],
    task: str,
    limit: int,
    judge_model: str,
    judge_limit: int,
    swap_runs: int,
    skip_model_metrics: bool,
    handles: WorkerHandles,
    VLLMWorkerA100: Any,
    advance_remote_fn: Any,
    poll_interval_s: int = 60,
) -> dict:
    """
    Runs the full pipeline for every spec in parallel:
      1. Generates the Gemma-4 baseline once (shared precondition)
      2. Fans out advance() to one container per language
      3. Polls until all reach REPORT or FAILED
      4. Writes a consolidated summary + markdown report

    advance_remote_fn is supplied by modal_app.py — it spawns the
    @app.function wrapper around advance() and returns its handle.
    """
    from src.pipeline.run import (
        append_run_to_index,
        report_path,
        update_manifest_status,
        write_manifest,
    )
    from src.workers.reporter import summarize_run
    from src.pipeline.report_render import render_markdown, write_markdown_report

    write_manifest(run_id, {
        "run_id": run_id, "phase": 5, "task": task,
        "models": [s.model_id for s in specs] + ["gemma-4-26b"],
        "limit": limit, "status": "started",
    })
    append_run_to_index(run_id, status="started")

    ctx = RunContext(
        run_id=run_id, task=task, limit=limit,
        judge_model=judge_model, judge_limit=judge_limit, swap_runs=swap_runs,
        skip_model_metrics=skip_model_metrics, handles=handles,
    )

    print("\n[evaluate_many] Generating shared Gemma-4 baseline...")
    ensure_gemma4_baseline(ctx, VLLMWorkerA100, any_slug=specs[0].slug)
    print("[evaluate_many] Baseline ready.\n")

    # Resume detection — read final_report.json for already-completed slugs
    from src.workers.reporter import load_report
    existing_report = load_report(run_id)
    completed = set(existing_report.get("languages", {}).keys())
    if completed:
        print(f"[evaluate_many] Resuming — already done: {sorted(completed)}")

    result = run_fanout(
        ctx, specs, stop_at=State.REPORT,
        advance_remote_fn=advance_remote_fn,
        poll_interval_s=poll_interval_s,
        resume_completed=completed,
        resume_report=existing_report,
    )

    failed = [slug for slug, lr in result.runs.items() if lr.state == State.FAILED]

    # ── Single source of truth for results + classification counts ──────────
    # Every LanguageRun, successful or failed, writes its outcome into
    # final_report.json via _do_report (or never gets that far if it failed
    # before REPORT). summarize_run reads that file once — no separate
    # result-building logic here, no re-derivation in the renderer.
    slugs = [s.slug for s in specs]
    agg   = summarize_run(run_id, slugs, task)

    # Failed languages never reach _do_report, so they're absent from
    # final_report.json. Patch them in explicitly with their error.
    for slug in failed:
        lr = result.runs[slug]
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

    from datetime import datetime, timezone
    generated_at = datetime.now(timezone.utc).isoformat()
    cost = estimate_cost(specs, judge_limit, swap_runs)

    summary = {
        "run_id": run_id, "phase": 5, "task": task,
        "generated_at": generated_at,
        "models_run": len(specs), "failed": failed,
        "results": agg["results"],
        "classification_counts": agg["classification_counts"],
        "cost_estimate": cost,
        "elapsed_minutes": round(result.elapsed_s / 60, 1),
    }

    from src.pipeline.run import run_dir
    import os as _os
    summary_path = _os.path.join(run_dir(run_id), "reports", "phase5_summary.json")
    _os.makedirs(_os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    md = render_markdown(
        run_id=run_id, task=task, generated_at=generated_at,
        results=agg["results"], classification_counts=agg["classification_counts"],
        title="Phase 5 Evaluation Report", cost=cost, failed=failed,
    )
    write_markdown_report(run_id, md, filename="phase5_summary.md")

    update_manifest_status(run_id, "failed" if failed else "completed")

    counts = summary["classification_counts"]
    print(f"\n{'='*65}\n  EVALUATE_MANY COMPLETE — {len(specs) - len(failed)}/{len(specs)} succeeded")
    print(f"{'='*65}")
    for g in ["A", "B", "C", "D", "E"]:
        print(f"  {g}: {counts.get(g, 0):2d}  {'█' * counts.get(g, 0)}")
    if failed:
        print(f"\n  Failed: {failed}")
    print(f"\n  Summary: {summary_path}")
    print(f"{'='*65}\n")

    return summary


# ---------------------------------------------------------------------------
# Single-language remote runner — what advance_remote_fn actually spawns
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
    The function body that runs INSIDE the per-language Modal container
    spawned by evaluate_many. Drives one LanguageRun from LIGHT_METRICS
    (inference already done by the time this is called — see note below)
    through stop_at, and returns a JSON-serializable dict.

    Note: in the Phase-5 fan-out, INFERENCE still runs inside this same
    container — only the Gemma-4 baseline is precomputed outside it. So
    this actually starts at PENDING, not LIGHT_METRICS.
    """
    ctx = RunContext(
        run_id=run_id, task=task, limit=limit,
        judge_model=judge_model, judge_limit=judge_limit, swap_runs=swap_runs,
        skip_model_metrics=skip_model_metrics, handles=handles,
    )
    lr = LanguageRun(spec=spec)
    lr = advance(ctx, lr, stop_at=State(stop_at_value))
    return language_run_to_dict(lr)