"""
src/pipeline/entrypoints.py
=============================
One function drives every evaluation run: run_pipeline().

v2 pipeline flow: PENDING → SAMPLE → INFERENCE → JUDGE → REPORT → DONE
- No Gemma-4 baseline — regional LLM evaluated standalone (pointwise).
- 200 stratified samples per run (configurable via n_samples).
- Grade derived from avg_score across rubric dimensions, not win rate.

"Run one language"   = run_pipeline(specs=[one], stop_at=REPORT)
"Run multiple"       = run_pipeline(specs=[...], stop_at=REPORT, advance_remote_fn=...)
"Resume"             = run_pipeline(specs=[...], stop_at=REPORT, run_id=existing_id)
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
    language_run_to_dict,
    run_fanout,
)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

_GPU_COST_PER_HR = {"t4": 0.59, "l4": 0.80, "l40s": 1.95, "a100_80gb": 2.50}
_INFER_HOURS     = {"t4": 0.08, "l4": 0.33, "l40s": 0.40, "a100_80gb": 0.42}


def estimate_cost(specs: list[LanguageSpec], n_samples: int, dimensions: int = 3) -> dict:
    """Estimates cost for a v2 pointwise run (no Gemma-4)."""
    scale    = n_samples / 1000.0
    regional = sum(_GPU_COST_PER_HR[s.gpu_preset] * _INFER_HOURS[s.gpu_preset] * scale for s in specs)
    calls    = len(specs) * n_samples * dimensions  # 1 call per (sample, dimension), no swap
    judge    = calls * 600 / 1_000_000 * 1.50       # ~600 tokens/call, $1.50/1M input
    return {
        "regional_inference_usd": round(regional, 2),
        "judge_usd":              round(judge, 2),
        "total_usd":              round(regional + judge, 2),
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
    n_samples: int = 200,
    seed: int = 42,
    handles: WorkerHandles,
    judge_model: str = "",
    advance_remote_fn: Optional[Any] = None,
    poll_interval_s: int = 60,
) -> dict:
    """
    Drives every language in `specs` through the state machine to `stop_at`.
    Writes a manifest, a JSON summary, and a markdown report.

    len(specs) == 1   → runs inline, in this process.
    len(specs) > 1    → fans out via advance_remote_fn (Modal .spawn() wrapper).

    Existing completed languages are skipped (resume support).
    Returns a summary dict: {run_id, task, results, classification_counts, ...}.
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
        "run_id":    run_id, "task": task,
        "models":    [s.model_id for s in specs],
        "n_samples": n_samples, "stop_at": stop_at.value, "status": "started",
    })
    append_run_to_index(run_id, status="started")

    # ── Resume detection ─────────────────────────────────────────────────────
    existing_report = load_report(run_id)
    completed       = set(existing_report.get("languages", {}).keys())
    if completed:
        print(f"[run_pipeline] Resuming — already done: {sorted(completed)}")

    ctx = RunContext(
        run_id=run_id, task=task, n_samples=n_samples, seed=seed,
        judge_model=judge_model, handles=handles,
    )

    # ── Drive every spec to stop_at ─────────────────────────────────────────
    runs: dict[str, LanguageRun] = {}
    elapsed_s = 0.0

    pending_specs = [s for s in specs if s.slug not in completed]

    if not pending_specs:
        pass  # everything already done

    elif len(specs) == 1 or advance_remote_fn is None:
        for spec in pending_specs:
            lr = advance(ctx, LanguageRun(spec=spec), stop_at=stop_at)
            runs[spec.slug] = lr
            if lr.state == State.FAILED:
                print(f"[run_pipeline] FAILED — {spec.language}: {lr.error}")

    else:
        result = run_fanout(
            ctx, specs, stop_at=stop_at,
            advance_remote_fn=advance_remote_fn,
            poll_interval_s=poll_interval_s,
            resume_completed=completed,
            resume_report=existing_report,
        )
        runs      = result.runs
        elapsed_s = result.elapsed_s

    failed = [slug for slug, lr in runs.items() if lr.state == State.FAILED]

    # ── Manifest status ──────────────────────────────────────────────────────
    if failed and len(failed) == len(specs):
        update_manifest_status(run_id, "failed")
    elif failed:
        update_manifest_status(run_id, "partial")
    else:
        update_manifest_status(run_id, "completed")

    # ── Reporting ────────────────────────────────────────────────────────────
    all_slugs = [s.slug for s in specs]
    agg       = summarize_run(run_id, all_slugs, task)

    for slug in failed:
        lr = runs[slug]
        agg["results"][slug] = {
            "language":       lr.spec.language,
            "regional_model": lr.spec.model_id,
            "avg_score":      None,
            "classification": "?",
            "error":          lr.error,
        }

    classification_counts = {
        g: sum(1 for r in agg["results"].values() if r["classification"] == g)
        for g in ["A", "B", "C", "D", "?"]
    }

    generated_at = datetime.now(timezone.utc).isoformat()
    cost         = estimate_cost(specs, n_samples) if stop_at == State.DONE else None

    summary = {
        "run_id": run_id, "task": task, "stop_at": stop_at.value,
        "generated_at": generated_at,
        "models_run": len(specs), "failed": failed,
        "results": agg["results"],
    }

    is_single       = len(specs) == 1
    summary_filename = f"{specs[0].slug}_summary.json" if is_single else "run_summary.json"
    md_filename      = f"{specs[0].slug}_summary.md"   if is_single else "run_summary.md"
    title = f"Evaluation report — {specs[0].language}" if is_single else "Evaluation report"

    summary_path = os.path.join(run_dir(run_id), "reports", summary_filename)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    md = render_markdown(
        run_id=run_id, task=task, generated_at=generated_at,
        results=agg["results"], classification_counts=classification_counts,
        title=title, cost=cost, failed=failed,
    )
    write_markdown_report(run_id, md, filename=md_filename)

    counts = classification_counts
    print(f"\n{'='*65}\n  PIPELINE COMPLETE — {len(specs) - len(failed)}/{len(specs)} succeeded")
    print(f"{'='*65}")
    for g in ["A", "B", "C", "D"]:
        print(f"  {g}: {counts.get(g, 0):2d}  {'█' * counts.get(g, 0)}")
    if failed:
        print(f"\n  Failed: {failed}")
    print(f"\n  Summary: {summary_path}\n{'='*65}\n")

    return summary


# ---------------------------------------------------------------------------
# Single-language remote runner — spawned once per language in fan-out
# ---------------------------------------------------------------------------

def run_single_language_to_state(
    *,
    run_id: str,
    spec: LanguageSpec,
    stop_at_value: str,
    task: str,
    n_samples: int,
    seed: int,
    judge_model: str,
    handles: WorkerHandles,
) -> dict:
    """
    Runs inside one per-language Modal container during a fan-out.
    Drives one LanguageRun from PENDING through stop_at and returns
    a JSON-serializable dict.
    """
    ctx = RunContext(
        run_id=run_id, task=task, n_samples=n_samples, seed=seed,
        judge_model=judge_model, handles=handles,
    )
    lr = advance(ctx, LanguageRun(spec=spec), stop_at=State(stop_at_value))
    return language_run_to_dict(lr)
