"""
src/pipeline/report_render.py
================================
Pure presentation layer. Takes already-computed summary data (from
src.workers.reporter.summarize_run) and renders it as markdown.

No file reading of source data (metrics, verdicts). No classification
logic. No re-derivation of fields. If a number is wrong, the bug is in
reporter.py, not here — this module only formats what it's given.

Used for both single-language summaries (after compare/phase4) and
multi-language summaries (after phase5). The renderer is the same;
only the run-level metadata (cost, failed list) differs, and that's
optional input, not a separate code path.
"""
from __future__ import annotations

from typing import Optional

# Models with no instruct variant — plain-text prompts only.
# Surfaced as a caveat in the report, not used for any scoring decision.
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


def render_markdown(
    *,
    run_id: str,
    task: str,
    generated_at: str,
    results: dict,                 # {slug: {language, regional_model, bleu_regional, bleu_gemma4, judge_win_rate, classification, classification_rationale}}
    classification_counts: dict,   # {"A": n, "B": n, ...}
    title: str = "Evaluation Report",
    cost: Optional[dict] = None,
    failed: Optional[list] = None,
) -> str:
    """
    Renders a results dict into markdown. Works identically for one
    language or seventeen — the table just has one row or seventeen.
    """
    failed = failed or []
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    lines += [
        f"# {title}",
        f"",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Run ID | `{run_id}` |",
        f"| Task | {task} |",
        f"| Models evaluated | {len(results)} |",
        f"| Generated | {generated_at[:19].replace('T', ' ')} UTC |",
    ]
    if cost:
        lines.append(f"| Estimated cost | ~${cost.get('total_usd', 0):.2f} |")
    if failed:
        lines.append(f"| Failed | {', '.join(failed)} |")
    lines.append("")

    # ── Results table ─────────────────────────────────────────────────────
    lines += [
        "## Results",
        "",
        "| Language | Model | BLEU (regional) | BLEU (Gemma-4) | Judge win rate | Grade | Notes |",
        "|----------|-------|:-:|:-:|:-:|:-:|-------|",
    ]
    for slug, r in sorted(results.items(), key=lambda kv: kv[1].get("language", "")):
        lines.append(_render_result_row(r))
    lines.append("")

    # ── Classification distribution ─────────────────────────────────────────
    lines += ["## Classification distribution", "", "```"]
    for grade in ["A", "B", "C", "D", "E"]:
        n = classification_counts.get(grade, 0)
        lines.append(f"  {grade}  {GRADE_LABEL.get(grade, grade):<38}  {n:2d}  {'█' * n}")
    lines += ["```", ""]

    # ── Key findings ──────────────────────────────────────────────────────
    lines += _render_key_findings(results)

    # ── Cost summary ─────────────────────────────────────────────────────
    if cost:
        lines += _render_cost_summary(cost)

    return "\n".join(lines)


def _render_result_row(r: dict) -> str:
    lang  = r.get("language", "")
    model = r.get("regional_model", "")
    grade = r.get("classification", "?")

    bleu_reg = f"{r['bleu_regional']:.2f}"  if r.get("bleu_regional")  is not None else "—"
    bleu_gem = f"{r['bleu_gemma4']:.2f}"    if r.get("bleu_gemma4")    is not None else "—"
    win_rate = f"{r['judge_win_rate']:.0%}" if r.get("judge_win_rate") is not None else "—"

    notes = []
    if model in BASE_MODELS:
        notes.append("⚠ base model")
    if grade == "?":
        notes.append("failed")

    return f"| {lang} | `{model}` | {bleu_reg} | {bleu_gem} | {win_rate} | **{grade}** | {', '.join(notes)} |"


def _render_key_findings(results: dict) -> list[str]:
    lines = ["## Key findings", ""]

    winners    = [r for r in results.values() if r.get("classification") in ("A", "B")]
    comparable = [r for r in results.values() if r.get("classification") == "C"]
    losers     = [r for r in results.values() if r.get("classification") in ("D", "E")]

    if winners:
        names = ", ".join(f"{r['language']} ({r['regional_model']})" for r in winners)
        lines.append(f"- **Regional models that outperform Gemma-4 (A/B):** {names}")
    if comparable:
        names = ", ".join(f"{r['language']} ({r['regional_model']})" for r in comparable)
        lines.append(f"- **Comparable to Gemma-4 (C):** {names}")
    if losers:
        names = ", ".join(f"{r['language']} ({r['regional_model']})" for r in losers)
        lines.append(f"- **Gemma-4 outperforms regional model (D/E):** {names}")

    base_model_langs = [
        r["language"] for r in results.values() if r.get("regional_model") in BASE_MODELS
    ]
    if base_model_langs:
        lines += [
            "",
            f"> ⚠ **Base model note:** {', '.join(base_model_langs)} used base (non-instruct) models.",
            "> Translation quality may reflect continued text rather than instruction-following.",
            "> These results should be interpreted with caution.",
        ]

    lines.append("")
    return lines


def _render_cost_summary(cost: dict) -> list[str]:
    return [
        "## Cost summary",
        "",
        "| Component | Cost |",
        "|-----------|------|",
        f"| Regional model inference | ${cost.get('regional_inference_usd', 0):.2f} |",
        f"| Gemma-4 inference (shared) | ${cost.get('gemma4_inference_usd', 0):.2f} |",
        f"| Judge API ({cost.get('total_judge_calls', 0):,} calls) | ${cost.get('judge_usd', 0):.2f} |",
        f"| **Total** | **~${cost.get('total_usd', 0):.2f}** |",
        "",
    ]


def write_markdown_report(run_id: str, md: str, filename: str = "summary.md") -> str:
    """Writes md to /data/outputs/runs/{run_id}/reports/{filename}. Returns the path."""
    import os
    from src.pipeline.run import run_dir

    out_path = os.path.join(run_dir(run_id), "reports", filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[report_render] Markdown written to {out_path}")
    return out_path