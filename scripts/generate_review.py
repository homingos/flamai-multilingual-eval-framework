#!/usr/bin/env python3
"""
Generate an HTML review of model outputs with pointwise judge scores.

v2: standalone pointwise evaluation — no Gemma-4 comparison column.
Each sample shows the regional output with per-dimension score bars (0–1).

Usage:
    python scripts/generate_review.py \\
        --run-id 2026-06-25_081543_85cde3 \\
        --slug german \\
        --task translation

    # Use already-downloaded local files (skips Modal download):
    python scripts/generate_review.py \\
        --run-id 2026-06-25_081543_85cde3 \\
        --slug german \\
        --task translation \\
        --local-regional /path/to/regional.jsonl \\
        --local-verdicts /path/to/verdicts.jsonl

Output: data/review_static/{slug}_{task}_{run_id}/review.html
"""
import argparse
import html as html_mod
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parent.parent
MODAL_VOLUME = "phase2a-outputs"


# ---------------------------------------------------------------------------
# Modal download
# ---------------------------------------------------------------------------

def modal_get(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["modal", "volume", "get", MODAL_VOLUME, remote, str(local)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"[warn] modal volume get {remote}: {r.stderr.strip()}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list:
    if not path or not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def aggregate_verdicts(verdicts: list) -> dict:
    """
    Returns: prompt_id → dimension → {score, confidence, reasoning}
    v2: single call per (prompt_id, dimension) — no swap aggregation needed.
    """
    result: dict = defaultdict(dict)
    for v in verdicts:
        if v.get("error"):
            continue
        pid = v["prompt_id"]
        dim = v["dimension"]
        result[pid][dim] = {
            "score":      float(v.get("score", 0.0)),
            "confidence": v.get("confidence", ""),
            "reasoning":  v.get("reasoning", ""),
        }
    return dict(result)


def overall_score(dim_results: dict) -> float:
    """Average score across all dimensions in a verdict dict."""
    scores = [v["score"] for v in dim_results.values() if v.get("score") is not None]
    return sum(scores) / len(scores) if scores else 0.0


def score_to_grade(avg: float) -> str:
    if avg >= 0.75: return "A"
    if avg >= 0.50: return "B"
    if avg >= 0.25: return "C"
    return "D"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

GRADE_CLASS = {"A": "grade-a", "B": "grade-b", "C": "grade-c", "D": "grade-d"}

DIM_ORDER_TRANSLATION  = ["fluency", "adequacy", "overall"]
DIM_ORDER_INSTRUCTIONS = ["language_compliance", "instruction_following", "helpfulness", "overall"]


def e(s) -> str:
    return html_mod.escape(str(s)) if s is not None else ""


def score_bar(score: float) -> str:
    pct = int(score * 100)
    cls = ("bar-a" if score >= 0.75 else
           "bar-b" if score >= 0.50 else
           "bar-c" if score >= 0.25 else "bar-d")
    return (f'<div class="score-bar-wrap">'
            f'<div class="score-bar {cls}" style="width:{pct}%"></div>'
            f'<span class="score-val">{score:.2f}</span></div>')


def generate_card(
    pid: str,
    regional: dict,
    verdicts: dict,
    task: str,
    model_display: str,
) -> str:
    avg   = overall_score(verdicts)
    grade = score_to_grade(avg) if verdicts else "?"
    gclass = GRADE_CLASS.get(grade, "")

    category = regional.get("category", "")
    direction = regional.get("direction", "")
    source    = regional.get("source", "")
    reference = regional.get("reference", "")
    system_instruction = regional.get("system_instruction", "")
    user_prompt        = regional.get("user_prompt", "")
    constraints        = regional.get("expected_constraints", {})
    output = regional.get("output", "")

    # Meta fields
    meta_rows = []
    if task == "translation":
        if direction:
            meta_rows.append(f'<div class="field"><label>Direction</label><div class="fval">{e(direction)}</div></div>')
        if source:
            meta_rows.append(f'<div class="field"><label>Source</label><div class="fval source-text">{e(source)}</div></div>')
        if reference:
            meta_rows.append(f'<div class="field"><label>Reference</label><div class="fval muted-text">{e(reference)}</div></div>')
    else:
        if system_instruction:
            meta_rows.append(f'<div class="field"><label>System</label><div class="fval sys-text">{e(system_instruction)}</div></div>')
        if user_prompt:
            meta_rows.append(f'<div class="field"><label>Prompt</label><div class="fval user-text">{e(user_prompt)}</div></div>')
        if constraints:
            meta_rows.append(f'<div class="field"><label>Constraints</label><code class="constraints">{e(json.dumps(constraints, ensure_ascii=False))}</code></div>')
    meta_html = "\n".join(meta_rows)

    # Verdict rows
    dim_order = DIM_ORDER_TRANSLATION if task == "translation" else DIM_ORDER_INSTRUCTIONS
    verdict_rows = []
    for dim in dim_order:
        if dim not in verdicts:
            continue
        v       = verdicts[dim]
        score   = v["score"]
        conf    = v.get("confidence", "")
        reason  = v.get("reasoning", "")
        verdict_rows.append(f"""
          <tr class="vrow">
            <td class="dim">{e(dim.replace("_", " ").title())}</td>
            <td class="vscore">{score_bar(score)}</td>
            <td class="vconf">{e(conf)}</td>
            <td class="vreason">{e(reason)}</td>
          </tr>""")

    verdicts_html = ""
    if verdict_rows:
        verdicts_html = f"""<table class="vtable">
          <thead><tr><th>Dimension</th><th>Score</th><th>Confidence</th><th>Reasoning</th></tr></thead>
          <tbody>{"".join(verdict_rows)}</tbody>
        </table>"""
    else:
        verdicts_html = '<p class="no-verdict">No judge verdicts recorded.</p>'

    return f"""<div class="card" data-category="{e(category)}" data-grade="{grade}" data-avg="{avg:.2f}">
  <div class="card-header">
    <span class="pid">{e(pid)}</span>
    {"<span class='cat'>" + e(category) + "</span>" if category else ""}
    <span class="grade-badge {gclass} ml-auto">{grade} <span class="avg-score">{avg:.2f}</span></span>
  </div>
  {meta_html}
  <div class="output-section">
    <div class="out-label">Output · {e(model_display)}</div>
    <div class="out-text">{"<em class='missing'>No output</em>" if not output else e(output)}</div>
  </div>
  {verdicts_html}
</div>"""


def generate_html(
    run_id: str,
    slug: str,
    task: str,
    model_display: str,
    regional_records: list,
    aggregated: dict,
) -> str:
    regional_by_id = {r["prompt_id"]: r for r in regional_records}
    all_ids        = sorted(set(regional_by_id) | set(aggregated))
    total          = len(all_ids)

    avgs    = {pid: overall_score(aggregated.get(pid, {})) for pid in all_ids}
    grades  = {pid: score_to_grade(avgs[pid]) if aggregated.get(pid) else "?" for pid in all_ids}
    overall_avg = sum(avgs.values()) / len(avgs) if avgs else 0.0
    overall_grade = score_to_grade(overall_avg)
    grade_counts = {g: sum(1 for v in grades.values() if v == g) for g in ["A", "B", "C", "D", "?"]}

    categories = sorted({(r.get("category") or "") for r in regional_records} - {""})
    cat_options = "\n".join(f'<option value="{e(c)}">{e(c)}</option>' for c in categories)

    cards_html    = "\n".join(
        generate_card(pid, regional_by_id.get(pid, {}), aggregated.get(pid, {}), task, model_display)
        for pid in all_ids
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Review: {e(slug)} · {e(task)} — {e(run_id)}</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg: #0d1117; --s1: #161b22; --s2: #21262d; --bdr: #30363d;
  --text: #e6edf3; --muted: #8b949e; --accent: #8957e5;
  --grade-a: #3fb950; --grade-b: #58a6ff; --grade-c: #d29922; --grade-d: #f85149;
  --radius: 8px;
}}
body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 13px; line-height: 1.5; }}

/* ── Header ── */
#hdr {{ position: sticky; top: 0; z-index: 200; background: var(--s1); border-bottom: 1px solid var(--bdr); padding: 10px 20px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
#hdr h1 {{ font-size: 15px; font-weight: 600; white-space: nowrap; }}
.run-meta {{ font-size: 11px; color: var(--muted); white-space: nowrap; }}
.stats {{ display: flex; gap: 8px; margin-left: auto; flex-wrap: wrap; align-items: center; }}
.stat {{ font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 20px; }}
.stat.overall-badge {{ background: rgba(137,87,229,.18); color: var(--accent); font-size: 13px; }}
.stat.ga {{ background: rgba(63,185,80,.15);  color: var(--grade-a); }}
.stat.gb {{ background: rgba(88,166,255,.15); color: var(--grade-b); }}
.stat.gc {{ background: rgba(210,153,34,.15); color: var(--grade-c); }}
.stat.gd {{ background: rgba(248,81,73,.15);  color: var(--grade-d); }}

/* ── Filters ── */
#flt {{ display: flex; align-items: center; gap: 8px; padding: 8px 20px; background: var(--s1); border-bottom: 1px solid var(--bdr); flex-wrap: wrap; }}
#flt select, #flt input {{ background: var(--s2); border: 1px solid var(--bdr); color: var(--text); padding: 5px 9px; border-radius: 6px; font-size: 12px; }}
#flt select:focus, #flt input:focus {{ outline: none; border-color: var(--accent); }}
#cnt {{ font-size: 11px; color: var(--muted); margin-left: auto; }}

/* ── Cards ── */
#cards {{ padding: 16px 20px; display: flex; flex-direction: column; gap: 14px; max-width: 1200px; margin: 0 auto; }}
.card {{ background: var(--s1); border: 1px solid var(--bdr); border-radius: var(--radius); overflow: hidden; }}
.card.hidden {{ display: none; }}

/* Card header */
.card-header {{ display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: var(--s2); border-bottom: 1px solid var(--bdr); flex-wrap: wrap; }}
.pid {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 11px; color: var(--muted); }}
.cat {{ font-size: 10px; padding: 2px 7px; border-radius: 4px; background: rgba(137,87,229,.18); color: var(--accent); text-transform: uppercase; letter-spacing: .04em; }}
.ml-auto {{ margin-left: auto; }}

/* Grade badges */
.grade-badge {{ display: inline-flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 700; padding: 3px 10px; border-radius: 12px; white-space: nowrap; }}
.grade-badge.grade-a {{ background: rgba(63,185,80,.15);  color: var(--grade-a); }}
.grade-badge.grade-b {{ background: rgba(88,166,255,.15); color: var(--grade-b); }}
.grade-badge.grade-c {{ background: rgba(210,153,34,.15); color: var(--grade-c); }}
.grade-badge.grade-d {{ background: rgba(248,81,73,.15);  color: var(--grade-d); }}
.avg-score {{ font-size: 10px; opacity: .8; }}

/* Meta fields */
.field {{ display: flex; gap: 10px; padding: 7px 12px; border-bottom: 1px solid var(--bdr); align-items: flex-start; }}
.field label {{ color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .05em; min-width: 80px; padding-top: 2px; flex-shrink: 0; }}
.fval {{ white-space: pre-wrap; word-break: break-word; line-height: 1.5; }}
.sys-text   {{ font-size: 12px; color: var(--muted); }}
.user-text  {{ font-size: 14px; font-weight: 500; }}
.source-text {{ font-size: 14px; font-weight: 500; }}
.muted-text {{ font-size: 12px; color: var(--muted); }}
.constraints {{ font-family: monospace; font-size: 11px; color: var(--accent); background: var(--s2); padding: 2px 6px; border-radius: 4px; }}

/* Output section */
.output-section {{ padding: 10px 12px; border-bottom: 1px solid var(--bdr); }}
.out-label {{ font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid var(--bdr); }}
.out-text {{ white-space: pre-wrap; word-break: break-word; font-size: 13px; line-height: 1.65; }}
.missing {{ color: var(--muted); font-style: italic; }}

/* Score bars */
.score-bar-wrap {{ display: flex; align-items: center; gap: 8px; min-width: 120px; }}
.score-bar {{ height: 8px; border-radius: 4px; }}
.score-val {{ font-size: 11px; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--text); }}
.bar-a {{ background: var(--grade-a); }}
.bar-b {{ background: var(--grade-b); }}
.bar-c {{ background: var(--grade-c); }}
.bar-d {{ background: var(--grade-d); }}

/* Verdict table */
.vtable {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.vtable th {{ background: var(--s2); color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .05em; padding: 6px 12px; text-align: left; border-bottom: 1px solid var(--bdr); }}
.vtable td {{ padding: 7px 12px; vertical-align: middle; border-bottom: 1px solid var(--bdr); }}
.vtable tr:last-child td {{ border-bottom: none; }}
.dim {{ font-weight: 600; white-space: nowrap; font-size: 11px; }}
.vconf {{ font-size: 10px; color: var(--muted); white-space: nowrap; }}
.vreason {{ line-height: 1.5; color: #cdd9e5; }}
.no-verdict {{ padding: 10px 12px; color: var(--muted); font-style: italic; font-size: 12px; }}
</style>
</head>
<body>

<div id="hdr">
  <h1>Review: {e(slug)} · {e(task)}</h1>
  <div class="run-meta">Run: {e(run_id)} &nbsp;|&nbsp; Model: {e(model_display)} &nbsp;|&nbsp; Generated: {generated_at}</div>
  <div class="stats">
    <div class="stat overall-badge">Overall: {overall_grade} ({overall_avg:.2f})</div>
    <div class="stat ga">A: {grade_counts.get("A", 0)}</div>
    <div class="stat gb">B: {grade_counts.get("B", 0)}</div>
    <div class="stat gc">C: {grade_counts.get("C", 0)}</div>
    <div class="stat gd">D: {grade_counts.get("D", 0)}</div>
  </div>
</div>

<div id="flt">
  <select id="cat-sel" onchange="applyFilters()">
    <option value="">All categories</option>
    {cat_options}
  </select>
  <select id="grade-sel" onchange="applyFilters()">
    <option value="">All grades</option>
    <option value="A">Grade A (≥0.75)</option>
    <option value="B">Grade B (≥0.50)</option>
    <option value="C">Grade C (≥0.25)</option>
    <option value="D">Grade D (&lt;0.25)</option>
  </select>
  <input id="srch" type="search" placeholder="Search text…" oninput="applyFilters()">
  <span id="cnt">{total} samples</span>
</div>

<div id="cards">
{cards_html}
</div>

<script>
(function() {{
  var total = {total};
  function applyFilters() {{
    var cat   = document.getElementById('cat-sel').value;
    var grade = document.getElementById('grade-sel').value;
    var srch  = document.getElementById('srch').value.toLowerCase();
    var cards = document.querySelectorAll('.card');
    var vis = 0;
    cards.forEach(function(c) {{
      var ok = (!cat   || c.dataset.category === cat)
             && (!grade || c.dataset.grade   === grade)
             && (!srch  || c.textContent.toLowerCase().includes(srch));
      c.classList.toggle('hidden', !ok);
      if (ok) vis++;
    }});
    document.getElementById('cnt').textContent = vis + ' / ' + total + ' samples';
  }}
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id",   required=True)
    ap.add_argument("--slug",     required=True)
    ap.add_argument("--task",     required=True, choices=["instructions", "translation"])
    ap.add_argument("--model-display", default=None)
    ap.add_argument("--output-dir",    default=None)
    ap.add_argument("--local-regional", default=None)
    ap.add_argument("--local-verdicts", default=None)
    args = ap.parse_args()

    model_display = args.model_display or args.slug

    out_dir  = (Path(args.output_dir) if args.output_dir
                else REPO_ROOT / "data" / "review_static" / f"{args.slug}_{args.task}_{args.run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "review.html"

    with tempfile.TemporaryDirectory(prefix="review_") as tmp:
        tmp = Path(tmp)

        if args.local_regional:
            regional_path = Path(args.local_regional)
        else:
            regional_path = tmp / "regional.jsonl"
            remote = f"runs/{args.run_id}/regional/{args.slug}_{args.task}_outputs.jsonl"
            print(f"Downloading {remote} …")
            modal_get(remote, regional_path)

        if args.local_verdicts:
            verdicts_path = Path(args.local_verdicts)
        else:
            verdicts_path = tmp / "verdicts.jsonl"
            remote = f"runs/{args.run_id}/judge/{args.slug}_{args.task}_verdicts.jsonl"
            print(f"Downloading {remote} …")
            modal_get(remote, verdicts_path)

        regional_records = load_jsonl(regional_path)
        verdict_records  = load_jsonl(verdicts_path)

    print(f"Loaded: {len(regional_records)} regional outputs, {len(verdict_records)} verdicts")

    aggregated   = aggregate_verdicts(verdict_records)
    html_content = generate_html(
        run_id=args.run_id, slug=args.slug, task=args.task,
        model_display=model_display, regional_records=regional_records,
        aggregated=aggregated,
    )

    out_file.write_text(html_content, encoding="utf-8")
    print(f"Generated: {out_file}  ({len(html_content) // 1024} KB)")


if __name__ == "__main__":
    main()
