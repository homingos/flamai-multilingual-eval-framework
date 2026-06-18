"""
src/pipeline/phase5_report.py
==============================
Generates a human-readable markdown summary report from phase5_summary.json.

No Modal imports. Pure Python. Writes to /data/outputs/.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

from src.pipeline.run import report_path, run_dir

# Models with no instruct variant — plain-text prompts only.
# Instruction-following results for these are not meaningful.
BASE_MODELS = {
    "mahamarathi-7b", "ambari-7b", "gujju-llama-7b",
    "viking-7b", "csmpt-7b", "polyglot-ko-12b",
    "goldfish-mri-39m", "goldfish-tpi-125m",
}

GRADE_LABEL = {
    "A": "Regional Superior   (wins >60%)",
    "B": "Regional Preferred  (wins 50–60%)",
    "C": "Comparable          (40–60%)",
    "D": "Gemma-4 Preferred   (wins 50–60%)",
    "E": "Gemma-4 Superior    (wins >60%)",
    "?": "Unknown / Failed",
}


def generate_phase5_report(run_id: str, summary: dict) -> str:
    """
    Generates a markdown summary table from phase5_summary data.
    Reads per-model metrics from final_report.json if available.
    Writes to /data/outputs/runs/{run_id}/reports/phase5_summary.md.
    Returns the markdown string.
    """
    results    = summary.get("results", {})
    task       = summary.get("task", "translation")
    generated  = summary.get("generated_at", datetime.now(timezone.utc).isoformat())
    total      = summary.get("models_run", len(results))
    failed     = summary.get("failed", [])
    counts     = summary.get("classification_counts", {})
    cost       = summary.get("cost_estimate", {})

    # Load per-model light metrics from final_report.json
    per_model_metrics = _load_per_model_metrics(run_id, results)

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        f"# Phase 5 Evaluation Report",
        f"",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Run ID | `{run_id}` |",
        f"| Task | {task} |",
        f"| Models evaluated | {total} |",
        f"| Generated | {generated[:19].replace('T', ' ')} UTC |",
    ]
    if cost:
        lines.append(f"| Estimated cost | ~${cost.get('total_usd', 0):.2f} |")
    if failed:
        lines.append(f"| Failed | {', '.join(failed)} |")
    lines.append("")

    # ── Results table ─────────────────────────────────────────────────────────
    lines += [
        f"## Results",
        f"",
        f"| Language | Model | BLEU (regional) | BLEU (Gemma-4) | Judge win rate | Grade | Notes |",
        f"|----------|-------|:-:|:-:|:-:|:-:|-------|",
    ]

    for slug, r in sorted(results.items(), key=lambda x: x[1].get("language", "")):
        lang   = r.get("language", slug)
        model  = r.get("regional_model", "")
        grade  = r.get("classification", "?")
        m      = per_model_metrics.get(slug, {})

        bleu_reg  = f"{m.get('bleu_regional', 0):.2f}"  if m.get("bleu_regional")  is not None else "—"
        bleu_gem  = f"{m.get('bleu_gemma4', 0):.2f}"    if m.get("bleu_gemma4")    is not None else "—"
        win_rate  = f"{m.get('judge_win_rate', 0):.0%}" if m.get("judge_win_rate") is not None else "—"

        notes = []
        if model in BASE_MODELS:
            notes.append("⚠ base model")
        if r.get("classification") == "?":
            notes.append("failed")

        lines.append(
            f"| {lang} | `{model}` | {bleu_reg} | {bleu_gem} | {win_rate} | **{grade}** | {', '.join(notes)} |"
        )

    lines.append("")

    # ── Classification distribution ───────────────────────────────────────────
    lines += ["## Classification distribution", ""]
    lines.append("```")
    for grade in ["A", "B", "C", "D", "E"]:
        n   = counts.get(grade, 0)
        bar = "█" * n
        lines.append(f"  {grade}  {GRADE_LABEL.get(grade, grade):<38}  {n:2d}  {bar}")
    lines.append("```")
    lines.append("")

    # ── Key findings ──────────────────────────────────────────────────────────
    lines += ["## Key findings", ""]

    winners = [r for r in results.values() if r.get("classification") in ("A", "B")]
    comparable = [r for r in results.values() if r.get("classification") == "C"]
    losers  = [r for r in results.values() if r.get("classification") in ("D", "E")]

    if winners:
        names = ", ".join(f"{r['language']} ({r['regional_model']})" for r in winners)
        lines.append(f"- **Regional models that outperform Gemma-4 (A/B):** {names}")
    if comparable:
        names = ", ".join(f"{r['language']} ({r['regional_model']})" for r in comparable)
        lines.append(f"- **Comparable to Gemma-4 (C):** {names}")
    if losers:
        names = ", ".join(f"{r['language']} ({r['regional_model']})" for r in losers)
        lines.append(f"- **Gemma-4 outperforms regional model (D/E):** {names}")

    base_model_slugs = [
        r["language"] for r in results.values() if r.get("regional_model") in BASE_MODELS
    ]
    if base_model_slugs:
        lines += [
            f"",
            f"> ⚠ **Base model note:** {', '.join(base_model_slugs)} used base (non-instruct) models.",
            f"> Translation quality may reflect continued text rather than instruction-following.",
            f"> These results should be interpreted with caution.",
        ]

    lines.append("")

    # ── Cost summary ──────────────────────────────────────────────────────────
    if cost:
        lines += [
            "## Cost summary",
            "",
            f"| Component | Cost |",
            f"|-----------|------|",
            f"| Regional model inference | ${cost.get('regional_inference_usd', 0):.2f} |",
            f"| Gemma-4 inference (shared) | ${cost.get('gemma4_inference_usd', 0):.2f} |",
            f"| Judge API ({cost.get('total_judge_calls', 0):,} calls) | ${cost.get('judge_usd', 0):.2f} |",
            f"| **Total** | **~${cost.get('total_usd', 0):.2f}** |",
            "",
        ]

    md = "\n".join(lines)

    # Write to disk
    out_path = os.path.join(run_dir(run_id), "reports", "phase5_summary.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[phase5_report] Markdown report written to {out_path}")
    return md


def _load_per_model_metrics(run_id: str, results: dict) -> dict:
    """
    Reads final_report.json and extracts BLEU and judge win rate per slug.
    Returns {slug: {bleu_regional, bleu_gemma4, judge_win_rate}}.
    """
    rpath = report_path(run_id)
    if not os.path.exists(rpath):
        return {}

    try:
        with open(rpath) as f:
            report = json.load(f)
    except Exception:
        return {}

    out = {}
    for slug in results:
        lang_data = report.get("languages", {}).get(slug, {})
        tasks     = lang_data.get("tasks", {})

        # Try translation first, then any task
        task_data = tasks.get("translation") or (next(iter(tasks.values()), {}) if tasks else {})

        light   = task_data.get("light_metrics", {})
        judge   = task_data.get("judge", {})

        bleu_reg = (light.get("bleu") or {}).get("bleu_en_to_target")
        win_rate = judge.get("regional_win_rate")

        # Gemma-4 BLEU is stored under the baseline slug in the metrics dir
        # If not in this report, leave as None
        bleu_gem = None
        baseline_data = report.get("languages", {}).get("baseline", {})
        if baseline_data:
            bl_tasks = baseline_data.get("tasks", {})
            bl_task  = bl_tasks.get("translation") or (next(iter(bl_tasks.values()), {}) if bl_tasks else {})
            bleu_gem = (bl_task.get("light_metrics", {}).get("bleu") or {}).get("bleu_en_to_target")

        out[slug] = {
            "bleu_regional":  bleu_reg,
            "bleu_gemma4":    bleu_gem,
            "judge_win_rate": win_rate,
        }

    return out