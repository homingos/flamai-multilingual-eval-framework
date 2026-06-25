#!/usr/bin/env python3
"""
Generate a side-by-side HTML review of model outputs vs Gemma-4 baseline with judge verdicts.

Usage:
    python scripts/generate_review.py \\
        --run-id 2026-06-25_081543_85cde3 \\
        --slug tamil \\
        --task instructions

    # Use already-downloaded local files (skips Modal download):
    python scripts/generate_review.py \\
        --run-id 2026-06-25_081543_85cde3 \\
        --slug tamil \\
        --task instructions \\
        --local-gemma4  /path/to/gemma4.jsonl \\
        --local-regional /path/to/regional.jsonl \\
        --local-verdicts /path/to/verdicts.jsonl

Output: data/review/{slug}_{task}/review.html
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

REPO_ROOT = Path(__file__).resolve().parent.parent
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
    records = []
    if not path or not path.exists():
        return records
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
    Returns: prompt_id -> dimension -> {winner_model, confidence, reasonings, raw}
    Majority-votes across swap runs. Genuine split → 'tie'.
    """
    by_prompt_dim: dict = defaultdict(lambda: defaultdict(list))
    for v in verdicts:
        if v.get("error"):
            continue
        by_prompt_dim[v["prompt_id"]][v["dimension"]].append(v)

    result = {}
    for pid, dims in by_prompt_dim.items():
        result[pid] = {}
        for dim, votes in dims.items():
            counts: dict = defaultdict(int)
            for v in votes:
                counts[v.get("winner_model", "tie")] += 1
            # If regional and gemma4 tied in count → call it tie
            r_count = counts.get("regional", 0)
            g_count = counts.get("gemma4", 0)
            if r_count > g_count:
                winner = "regional"
            elif g_count > r_count:
                winner = "gemma4"
            else:
                winner = "tie"
            result[pid][dim] = {
                "winner_model": winner,
                "confidence": votes[-1].get("confidence", ""),
                "reasonings": [v.get("reasoning", "") for v in votes if v.get("reasoning")],
                "raw": votes,
            }
    return result


def overall_winner(dim_results: dict) -> str:
    counts: dict = defaultdict(int)
    for res in dim_results.values():
        counts[res["winner_model"]] += 1
    if not counts:
        return "unknown"
    r = counts.get("regional", 0)
    g = counts.get("gemma4", 0)
    if r > g:
        return "regional"
    if g > r:
        return "gemma4"
    return "tie"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

WINNER_LABEL = {
    "regional": "✓ Regional",
    "gemma4": "✗ Gemma-4",
    "tie": "~ Tie",
    "unknown": "? No verdict",
}

DIM_ORDER = [
    "overall",
    "instruction_following",
    "language_quality",
    "translation_quality",
    "fluency",
    "adequacy",
    "topic_boundary",
]


def e(s) -> str:
    return html_mod.escape(str(s)) if s is not None else ""


def generate_card(
    pid: str,
    regional: dict,
    gemma4: dict,
    verdicts: dict,
    task: str,
    model_display: str,
) -> str:
    ow = overall_winner(verdicts)
    category = regional.get("category") or gemma4.get("category", "")
    system_instruction = regional.get("system_instruction") or gemma4.get("system_instruction", "")
    user_prompt = regional.get("user_prompt") or gemma4.get("user_prompt", "")
    constraints = regional.get("expected_constraints") or gemma4.get("expected_constraints", {})
    source = regional.get("source") or gemma4.get("source", "")
    reference = regional.get("reference") or gemma4.get("reference", "")
    direction = regional.get("direction") or gemma4.get("direction", "")

    regional_output = regional.get("output", "")
    gemma4_output = gemma4.get("output", "")

    # Build metadata fields
    meta_rows = []
    if task == "translation":
        if direction:
            meta_rows.append(f'<div class="field"><label>Direction</label><div class="fval">{e(direction)}</div></div>')
        if source:
            meta_rows.append(f'<div class="field"><label>Source</label><div class="fval source-text">{e(source)}</div></div>')
        if reference:
            meta_rows.append(f'<div class="field"><label>Reference</label><div class="fval">{e(reference)}</div></div>')
    else:
        if system_instruction:
            meta_rows.append(f'<div class="field"><label>System</label><div class="fval sys-text">{e(system_instruction)}</div></div>')
        if user_prompt:
            meta_rows.append(f'<div class="field"><label>Prompt</label><div class="fval user-text">{e(user_prompt)}</div></div>')
        if constraints:
            meta_rows.append(f'<div class="field"><label>Constraints</label><code class="constraints">{e(json.dumps(constraints, ensure_ascii=False))}</code></div>')

    meta_html = "\n".join(meta_rows)

    # Verdict rows
    verdict_rows = []
    for dim in DIM_ORDER:
        if dim not in verdicts:
            continue
        v = verdicts[dim]
        wm = v["winner_model"]
        label = {"regional": model_display, "gemma4": "Gemma-4", "tie": "Tie"}.get(wm, wm)
        conf = v.get("confidence", "")
        reasonings = v.get("reasonings", [])
        # Show each swap run reasoning separately
        reasoning_parts = []
        for i, r in enumerate(reasonings):
            if len(reasonings) > 1:
                reasoning_parts.append(f'<span class="swap-label">Run {i}</span> {e(r)}')
            else:
                reasoning_parts.append(e(r))
        reasoning_html = "<br><br>".join(reasoning_parts)
        verdict_rows.append(f"""
          <tr class="vrow {wm}">
            <td class="dim">{e(dim.replace("_", " ").title())}</td>
            <td class="vwinner"><span class="badge {wm}">{e(label)}</span> <span class="conf">{e(conf)}</span></td>
            <td class="vreason">{reasoning_html}</td>
          </tr>""")

    verdicts_html = ""
    if verdict_rows:
        verdicts_html = f"""<table class="vtable">
          <thead><tr><th>Dimension</th><th>Winner</th><th>Reasoning</th></tr></thead>
          <tbody>{"".join(verdict_rows)}</tbody>
        </table>"""
    else:
        verdicts_html = '<p class="no-verdict">No judge verdicts recorded.</p>'

    regional_missing = not regional_output
    gemma4_missing = not gemma4_output

    return f"""<div class="card" data-category="{e(category)}" data-winner="{ow}">
  <div class="card-header">
    <span class="pid">{e(pid)}</span>
    {"<span class='cat'>" + e(category) + "</span>" if category else ""}
    <span class="badge overall {ow} ml-auto">{WINNER_LABEL.get(ow, ow)}</span>
  </div>
  {meta_html}
  <div class="outputs">
    <div class="out-col regional-col">
      <div class="out-label regional">Regional · {e(model_display)}</div>
      <div class="out-text">{"<em class='missing'>No output</em>" if regional_missing else e(regional_output)}</div>
    </div>
    <div class="out-col gemma4-col">
      <div class="out-label gemma4">Baseline · Gemma-4-26B</div>
      <div class="out-text">{"<em class='missing'>No output</em>" if gemma4_missing else e(gemma4_output)}</div>
    </div>
  </div>
  {verdicts_html}
</div>"""


def generate_html(
    run_id: str,
    slug: str,
    task: str,
    model_display: str,
    regional_records: list,
    gemma4_records: list,
    aggregated: dict,
) -> str:
    regional_by_id = {r["prompt_id"]: r for r in regional_records}
    gemma4_by_id = {r["prompt_id"]: r for r in gemma4_records}

    all_ids = sorted(set(regional_by_id) | set(gemma4_by_id) | set(aggregated))
    total = len(all_ids)

    outcomes = {pid: overall_winner(aggregated.get(pid, {})) for pid in all_ids}
    regional_wins = sum(1 for v in outcomes.values() if v == "regional")
    gemma4_wins   = sum(1 for v in outcomes.values() if v == "gemma4")
    ties          = sum(1 for v in outcomes.values() if v == "tie")
    no_verdict    = sum(1 for v in outcomes.values() if v == "unknown")

    categories = sorted({
        (r.get("category") or "") for r in regional_records + gemma4_records
    } - {""})

    cat_options = "\n".join(
        f'<option value="{e(c)}">{e(c)}</option>' for c in categories
    )

    cards_html = "\n".join(
        generate_card(
            pid,
            regional_by_id.get(pid, {}),
            gemma4_by_id.get(pid, {}),
            aggregated.get(pid, {}),
            task,
            model_display,
        )
        for pid in all_ids
    )

    pct = lambda n: f"{n * 100 // total if total else 0}%"
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
  --text: #e6edf3; --muted: #8b949e; --link: #58a6ff;
  --regional: #3fb950; --gemma4: #f85149; --tie: #d29922; --unknown: #8b949e;
  --accent: #8957e5;
  --radius: 8px;
}}
body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 13px; line-height: 1.5; }}

/* ── Header ── */
#hdr {{ position: sticky; top: 0; z-index: 200; background: var(--s1); border-bottom: 1px solid var(--bdr); padding: 10px 20px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
#hdr h1 {{ font-size: 15px; font-weight: 600; white-space: nowrap; }}
.run-meta {{ font-size: 11px; color: var(--muted); white-space: nowrap; }}
.stats {{ display: flex; gap: 8px; margin-left: auto; flex-wrap: wrap; }}
.stat {{ font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 20px; }}
.stat.regional {{ background: rgba(63,185,80,.15); color: var(--regional); }}
.stat.gemma4   {{ background: rgba(248,81,73,.15);  color: var(--gemma4); }}
.stat.tie      {{ background: rgba(210,153,34,.15);  color: var(--tie); }}
.stat.unknown  {{ background: rgba(139,148,158,.12); color: var(--unknown); }}

/* ── Filters ── */
#flt {{ display: flex; align-items: center; gap: 8px; padding: 8px 20px; background: var(--s1); border-bottom: 1px solid var(--bdr); flex-wrap: wrap; }}
#flt select, #flt input {{ background: var(--s2); border: 1px solid var(--bdr); color: var(--text); padding: 5px 9px; border-radius: 6px; font-size: 12px; }}
#flt select:focus, #flt input:focus {{ outline: none; border-color: var(--accent); }}
#cnt {{ font-size: 11px; color: var(--muted); margin-left: auto; }}

/* ── Cards ── */
#cards {{ padding: 16px 20px; display: flex; flex-direction: column; gap: 14px; max-width: 1600px; margin: 0 auto; }}
.card {{ background: var(--s1); border: 1px solid var(--bdr); border-radius: var(--radius); overflow: hidden; }}
.card.hidden {{ display: none; }}

/* Card header */
.card-header {{ display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: var(--s2); border-bottom: 1px solid var(--bdr); flex-wrap: wrap; }}
.pid {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 11px; color: var(--muted); }}
.cat {{ font-size: 10px; padding: 2px 7px; border-radius: 4px; background: rgba(137,87,229,.18); color: var(--accent); text-transform: uppercase; letter-spacing: .04em; }}
.ml-auto {{ margin-left: auto; }}

/* Badges */
.badge {{ display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 10px; white-space: nowrap; }}
.badge.regional, .badge.overall.regional {{ background: rgba(63,185,80,.15); color: var(--regional); }}
.badge.gemma4,   .badge.overall.gemma4   {{ background: rgba(248,81,73,.15);  color: var(--gemma4); }}
.badge.tie,      .badge.overall.tie      {{ background: rgba(210,153,34,.15);  color: var(--tie); }}
.badge.unknown                           {{ background: rgba(139,148,158,.12); color: var(--unknown); }}

/* Meta fields */
.field {{ display: flex; gap: 10px; padding: 7px 12px; border-bottom: 1px solid var(--bdr); align-items: flex-start; }}
.field label {{ color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .05em; min-width: 80px; padding-top: 2px; flex-shrink: 0; }}
.fval {{ white-space: pre-wrap; word-break: break-word; line-height: 1.5; }}
.sys-text  {{ font-size: 12px; color: var(--muted); }}
.user-text {{ font-size: 14px; font-weight: 500; }}
.source-text {{ font-size: 14px; font-weight: 500; }}
.constraints {{ font-family: monospace; font-size: 11px; color: var(--accent); background: var(--s2); padding: 2px 6px; border-radius: 4px; }}

/* Outputs */
.outputs {{ display: grid; grid-template-columns: 1fr 1fr; border-bottom: 1px solid var(--bdr); }}
.out-col {{ padding: 10px 12px; }}
.regional-col {{ border-right: 1px solid var(--bdr); }}
.out-label {{ font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid var(--bdr); }}
.out-label.regional {{ color: var(--regional); }}
.out-label.gemma4   {{ color: var(--gemma4); }}
.out-text {{ white-space: pre-wrap; word-break: break-word; font-size: 13px; line-height: 1.65; }}
.missing {{ color: var(--muted); font-style: italic; }}

/* Verdict table */
.vtable {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.vtable th {{ background: var(--s2); color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .05em; padding: 6px 12px; text-align: left; border-bottom: 1px solid var(--bdr); }}
.vtable td {{ padding: 7px 12px; vertical-align: top; border-bottom: 1px solid var(--bdr); }}
.vtable tr:last-child td {{ border-bottom: none; }}
.dim {{ font-weight: 600; white-space: nowrap; font-size: 11px; }}
.vwinner {{ white-space: nowrap; }}
.conf {{ font-size: 10px; color: var(--muted); }}
.vreason {{ line-height: 1.5; color: #cdd9e5; }}
.swap-label {{ font-size: 10px; color: var(--muted); font-weight: 600; }}
.no-verdict {{ padding: 10px 12px; color: var(--muted); font-style: italic; font-size: 12px; }}

.vrow.regional td {{ background: rgba(63,185,80,.04); }}
.vrow.gemma4   td {{ background: rgba(248,81,73,.04); }}

@media (max-width: 860px) {{
  .outputs {{ grid-template-columns: 1fr; }}
  .regional-col {{ border-right: none; border-bottom: 1px solid var(--bdr); }}
}}
</style>
</head>
<body>

<div id="hdr">
  <h1>Review: {e(slug)} · {e(task)}</h1>
  <div class="run-meta">Run: {e(run_id)} &nbsp;|&nbsp; Model: {e(model_display)} &nbsp;|&nbsp; Generated: {generated_at}</div>
  <div class="stats">
    <div class="stat regional">✓ Regional: {regional_wins}/{total} ({pct(regional_wins)})</div>
    <div class="stat gemma4">✗ Gemma-4: {gemma4_wins}/{total} ({pct(gemma4_wins)})</div>
    <div class="stat tie">~ Tie: {ties}/{total} ({pct(ties)})</div>
    <div class="stat unknown">? No verdict: {no_verdict}/{total} ({pct(no_verdict)})</div>
  </div>
</div>

<div id="flt">
  <select id="cat-sel" onchange="applyFilters()">
    <option value="">All categories</option>
    {cat_options}
  </select>
  <select id="win-sel" onchange="applyFilters()">
    <option value="">All outcomes</option>
    <option value="regional">Regional wins</option>
    <option value="gemma4">Gemma-4 wins</option>
    <option value="tie">Ties</option>
    <option value="unknown">No verdict</option>
  </select>
  <input id="srch" type="search" placeholder="Search text…" oninput="applyFilters()">
  <span id="cnt">{total} prompts</span>
</div>

<div id="cards">
{cards_html}
</div>

<script>
(function() {{
  var total = {total};
  function applyFilters() {{
    var cat  = document.getElementById('cat-sel').value;
    var win  = document.getElementById('win-sel').value;
    var srch = document.getElementById('srch').value.toLowerCase();
    var cards = document.querySelectorAll('.card');
    var vis = 0;
    cards.forEach(function(c) {{
      var ok = (!cat  || c.dataset.category === cat)
             && (!win  || c.dataset.winner   === win)
             && (!srch || c.textContent.toLowerCase().includes(srch));
      c.classList.toggle('hidden', !ok);
      if (ok) vis++;
    }});
    document.getElementById('cnt').textContent = vis + ' / ' + total + ' prompts';
  }}
  document.getElementById('cat-sel').addEventListener('change', applyFilters);
  document.getElementById('win-sel').addEventListener('change', applyFilters);
  document.getElementById('srch').addEventListener('input', applyFilters);
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id",   required=True, help="Run ID, e.g. 2026-06-25_081543_85cde3")
    ap.add_argument("--slug",     required=True, help="Pipeline slug, e.g. tamil")
    ap.add_argument("--task",     required=True, choices=["instructions", "translation"])
    ap.add_argument("--model-display", default=None, help="Human-readable model name (default: slug)")
    ap.add_argument("--output-dir", default=None, help="Output directory (default: data/review/{slug}_{task}/)")
    # Local file overrides — skip Modal download
    ap.add_argument("--local-gemma4",   default=None, help="Path to already-downloaded gemma4 output JSONL")
    ap.add_argument("--local-regional", default=None, help="Path to already-downloaded regional output JSONL")
    ap.add_argument("--local-verdicts", default=None, help="Path to already-downloaded verdicts JSONL")
    args = ap.parse_args()

    model_display = args.model_display or args.slug

    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "data" / "review" / f"{args.slug}_{args.task}_{args.run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "review.html"

    # Resolve local file paths or download from Modal
    with tempfile.TemporaryDirectory(prefix="review_") as tmp:
        tmp = Path(tmp)

        if args.local_gemma4:
            gemma4_path = Path(args.local_gemma4)
        else:
            gemma4_path = tmp / "gemma4.jsonl"
            remote = f"runs/{args.run_id}/gemma4/{args.task}_outputs.jsonl"
            print(f"Downloading {remote} …")
            modal_get(remote, gemma4_path)

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

        gemma4_records   = load_jsonl(gemma4_path)
        regional_records = load_jsonl(regional_path)
        verdict_records  = load_jsonl(verdicts_path)

    print(f"Loaded: {len(gemma4_records)} gemma4, {len(regional_records)} regional, {len(verdict_records)} verdicts")

    aggregated = aggregate_verdicts(verdict_records)

    html_content = generate_html(
        run_id          = args.run_id,
        slug            = args.slug,
        task            = args.task,
        model_display   = model_display,
        regional_records = regional_records,
        gemma4_records   = gemma4_records,
        aggregated       = aggregated,
    )

    out_file.write_text(html_content, encoding="utf-8")
    print(f"Generated: {out_file}  ({len(html_content) // 1024} KB)")


if __name__ == "__main__":
    main()
