# Run: python scripts/dashboard.py
# Then open: http://localhost:8765
# Requirements: pip install fastapi uvicorn
"""
Falcon Language Support — Local Dashboard
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
PORT = 8765
MODAL_VOLUME = "phase2a-outputs"
QUAL_JSON = REPO_ROOT / "data" / "qualitative_results.json"
REVIEW_DIR = REPO_ROOT / "data" / "review"
VIZ_DIR = REPO_ROOT / "docs" / "viz"

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print("Missing dependencies. Run: pip install fastapi uvicorn")
    sys.exit(1)

app = FastAPI(title="Falcon Dashboard")

REVIEW_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/viz", StaticFiles(directory=str(VIZ_DIR)), name="viz")
app.mount("/static/review", StaticFiles(directory=str(REVIEW_DIR)), name="review")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

CSS_VARS = """
:root {
  --bg:  #0D1117;
  --sf:  #161B22;
  --sf2: #21262D;
  --bd:  #30363D;
  --tx:  #C9D1D9;
  --mu:  #8B949E;
  --ac:  #58A6FF;
  --gr:  #3FB950;
  --rd:  #F85149;
  --or:  #F5A623;
}
"""

GLOBAL_CSS = CSS_VARS + """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--tx);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.5;
}
a { color: var(--ac); text-decoration: none; }
a:hover { text-decoration: underline; }
code { font-family: "SF Mono", "Fira Code", Menlo, monospace; }
"""

GRADE_COLORS = {
    "A": "#3FB950",
    "B": "#52B788",
    "C": "#F5A623",
    "D": "#E76F51",
    "E": "#F85149",
    "skipped": "#8B949E",
    "failed": "#8B949E",
    "pending": "#484F58",
}

GRADE_LABELS = {
    "A": "Grade A — Regional superior (>60% judge win rate)",
    "B": "Grade B — Regional preferred (50–60%)",
    "C": "Grade C — Comparable (40–50%)",
    "D": "Grade D — Gemma-4 preferred (20–40%)",
    "E": "Grade E — Gemma-4 strongly preferred (<20%)",
    "skipped": "Skipped — run failed or crashed mid-inference",
    "failed": "Failed — infrastructure/pipeline error",
    "pending": "Pending — not yet evaluated",
}

REGION_ORDER = ["Indic", "Middle East", "East Asia", "SEA", "Africa", "Europe", "Americas", "Oceania"]

# ---------------------------------------------------------------------------
# Nav bar
# ---------------------------------------------------------------------------

def nav_html(active: str) -> str:
    tabs = [
        ("Tokenizer", "/tokenizer", "tokenizer"),
        ("Qualitative Analysis", "/qualitative", "qualitative"),
        ("Manual Review (QA)", "/review", "review"),
    ]

    def tab_link(label, href, key):
        if key == active:
            style = "color:var(--tx);border-bottom:2px solid var(--ac);padding-bottom:2px;font-weight:600;"
        else:
            style = "color:var(--mu);font-weight:500;"
        return f'<a href="{href}" style="text-decoration:none;font-size:13px;{style}">{label}</a>'

    links = "\n".join(tab_link(l, h, k) for l, h, k in tabs)
    return f"""<nav style="
        height:46px;display:flex;align-items:center;gap:28px;
        padding:0 24px;background:var(--sf);border-bottom:1px solid var(--bd);
        position:sticky;top:0;z-index:100;flex-shrink:0;">
      <span style="font-size:15px;font-weight:700;color:var(--tx);white-space:nowrap;letter-spacing:-.3px;">🌍 Falcon</span>
      <span style="color:var(--bd);">/</span>
      {links}
    </nav>"""

# ---------------------------------------------------------------------------
# Grade badge
# ---------------------------------------------------------------------------

def grade_badge(grade: str, size: str = "normal") -> str:
    color = GRADE_COLORS.get(grade, "#484F58")
    tooltip = GRADE_LABELS.get(grade, grade)
    bg = color + "22"
    fs = "13px" if size == "large" else "11px"
    pad = "4px 12px" if size == "large" else "2px 9px"
    return (
        f'<span title="{tooltip}" style="'
        f'display:inline-block;font-size:{fs};font-weight:700;padding:{pad};'
        f'border-radius:12px;background:{bg};color:{color};border:1px solid {color}55;'
        f'cursor:default;white-space:nowrap;letter-spacing:.02em;">{grade.upper()}</span>'
    )

# ---------------------------------------------------------------------------
# Metric bar helpers
# ---------------------------------------------------------------------------

def metric_bar(label: str, val_r, val_g, unit: str = "", scale: float = 100.0, hint: str = "↑ higher is better") -> str:
    """Side-by-side horizontal bars for regional vs Gemma-4."""
    def pct(v):
        if v is None:
            return 0.0
        return min(100.0, float(v) / scale * 100)

    def fmt(v):
        if v is None:
            return "—"
        if unit == "%":
            return f"{v:.0f}%"
        return f"{v:.2f}"

    better = val_r is not None and val_g is not None and float(val_r) >= float(val_g)
    r_color = "var(--gr)" if better else "var(--ac)"
    r_w = pct(val_r)
    g_w = pct(val_g)
    r_str = fmt(val_r)
    g_str = fmt(val_g)

    return f"""<div class="metric-block">
  <div class="metric-hdr">{label} <span class="metric-hint">{hint}</span></div>
  <div class="bar-group">
    <div class="bar-row">
      <span class="blabel">Regional</span>
      <div class="btrack"><div class="bfill" style="width:{r_w:.1f}%;background:{r_color};"></div></div>
      <span class="bval" style="color:{r_color if val_r is not None else 'var(--mu)'};">{r_str}</span>
    </div>
    <div class="bar-row">
      <span class="blabel">Gemma-4</span>
      <div class="btrack"><div class="bfill" style="width:{g_w:.1f}%;background:var(--mu);opacity:.6;"></div></div>
      <span class="bval" style="color:var(--mu);">{g_str}</span>
    </div>
  </div>
</div>"""


def win_rate_bar(win_r, win_g) -> str:
    """Prominent win rate comparison bar."""
    def pct(v):
        if v is None:
            return 0.0
        return min(100.0, float(v))

    def fmt(v):
        return f"{v:.1f}%" if v is not None else "—"

    r_w = pct(win_r)
    g_w = pct(win_g)
    r_better = win_r is not None and win_g is not None and win_r >= win_g
    r_color = "var(--gr)" if r_better else "var(--rd)"

    return f"""<div class="metric-block" style="margin-bottom:12px;">
  <div class="metric-hdr">Judge Win Rate <span class="metric-hint">↑ % of 50 judged prompts where regional model was preferred</span></div>
  <div class="bar-group">
    <div class="bar-row">
      <span class="blabel">Regional</span>
      <div class="btrack" style="height:10px;"><div class="bfill" style="width:{r_w:.1f}%;height:10px;background:{r_color};"></div></div>
      <span class="bval" style="color:{r_color};font-weight:700;">{fmt(win_r)}</span>
    </div>
    <div class="bar-row">
      <span class="blabel">Gemma-4</span>
      <div class="btrack" style="height:10px;"><div class="bfill" style="width:{g_w:.1f}%;height:10px;background:var(--mu);opacity:.5;"></div></div>
      <span class="bval" style="color:var(--mu);">{fmt(win_g)}</span>
    </div>
  </div>
</div>"""

# ---------------------------------------------------------------------------
# Qualitative page CSS
# ---------------------------------------------------------------------------

QUAL_PAGE_CSS = """
/* Layout */
.qual-page { max-width:1200px; margin:0 auto; padding:24px 20px; }
/* Stats bar */
.stats-bar {
  display:flex; flex-wrap:wrap; gap:10px; margin-bottom:20px;
  padding:14px 18px; background:var(--sf); border:1px solid var(--bd);
  border-radius:10px; align-items:center;
}
.stat-chip {
  display:flex; align-items:center; gap:6px; font-size:12px;
  padding:4px 10px; border-radius:20px; border:1px solid var(--bd);
  background:var(--sf2); white-space:nowrap;
}
.stat-chip .sc-val { font-weight:700; }
/* Filter row */
.filter-row {
  display:flex; flex-wrap:wrap; gap:6px; margin-bottom:22px;
  align-items:center;
}
.pill {
  font-size:12px; font-weight:500; padding:4px 14px; border-radius:20px;
  border:1px solid var(--bd); background:var(--sf2); color:var(--mu);
  cursor:pointer; transition:all .15s; user-select:none;
}
.pill:hover { border-color:var(--ac); color:var(--tx); }
.pill.active { background:var(--ac); color:#0D1117; border-color:var(--ac); font-weight:700; }
.refresh-btn {
  margin-left:auto; font-size:12px; padding:5px 14px; border-radius:8px;
  border:1px solid var(--bd); background:var(--sf2); color:var(--tx); cursor:pointer;
  display:flex; align-items:center; gap:5px;
}
.refresh-btn:hover { border-color:var(--ac); color:var(--ac); }
.last-updated { font-size:11px; color:var(--mu); margin-bottom:18px; }
/* Legend */
.legend-toggle {
  font-size:11px; color:var(--ac); cursor:pointer; margin-bottom:14px;
  display:inline-flex; align-items:center; gap:4px; border:none;
  background:none; padding:0; font-family:inherit;
}
.legend-box {
  background:var(--sf); border:1px solid var(--bd); border-radius:8px;
  padding:14px 18px; margin-bottom:22px; font-size:12px; display:none;
}
.legend-box.open { display:block; }
.legend-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px 32px; margin-top:10px; }
.legend-section h4 { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--mu); margin-bottom:8px; }
.legend-row { display:flex; gap:8px; margin-bottom:5px; color:var(--tx); }
.legend-key { font-weight:600; color:var(--ac); min-width:150px; }
.legend-val { color:var(--mu); }
/* Region sections */
.region-section { margin-bottom:36px; }
.region-header {
  font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.1em;
  color:var(--mu); padding:6px 0; border-bottom:1px solid var(--bd); margin-bottom:14px;
}
/* Language group */
.lang-group { margin-bottom:20px; }
.lang-group-header {
  font-size:16px; font-weight:700; color:var(--tx); margin-bottom:10px;
  display:flex; align-items:center; gap:10px;
}
/* Language card */
.lang-card {
  background:var(--sf); border:1px solid var(--bd); border-radius:10px;
  margin-bottom:10px; overflow:hidden;
}
.card-header {
  display:flex; align-items:center; gap:10px; padding:10px 16px;
  background:var(--sf2); border-bottom:1px solid var(--bd);
}
.card-model-name { font-size:13px; font-weight:600; color:var(--tx); }
.card-grade-area { display:flex; align-items:center; gap:10px; margin-left:auto; }
.region-chip {
  font-size:10px; padding:2px 8px; border-radius:4px;
  background:rgba(88,166,255,.1); color:var(--ac); border:1px solid rgba(88,166,255,.25);
}
/* Two-column panels */
.card-panels {
  display:grid; grid-template-columns:1fr 1px 1fr;
}
.panel-divider { background:var(--bd); }
.task-panel { padding:16px; }
.task-label-hdr {
  font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.08em;
  color:var(--mu); margin-bottom:12px; display:flex; align-items:center; gap:8px;
}
/* Run ID */
.run-id-row {
  display:flex; align-items:center; gap:8px; margin-top:12px;
  padding:8px 10px; background:var(--bg); border:1px solid var(--bd); border-radius:6px;
}
.run-id-code {
  font-family:"SF Mono","Fira Code",Menlo,monospace; font-size:11px; font-weight:600;
  color:var(--tx); letter-spacing:.02em; flex:1;
}
.run-id-link {
  font-size:11px; color:var(--ac); white-space:nowrap; font-weight:500;
}
/* Notes */
.notes-block {
  margin-top:10px; padding:8px 10px;
  background:rgba(248,81,73,.06); border-left:3px solid rgba(248,81,73,.4);
  border-radius:0 4px 4px 0; font-size:11px; color:var(--mu); font-style:italic;
  line-height:1.5;
}
/* Skipped/failed/pending panel */
.status-panel {
  padding:16px; display:flex; align-items:center; gap:10px;
  color:var(--mu); font-size:12px;
}
/* Metric bars */
.metric-block { margin-bottom:10px; }
.metric-hdr { font-size:11px; font-weight:600; color:var(--tx); margin-bottom:5px; }
.metric-hint { font-size:10px; color:var(--mu); font-weight:400; }
.bar-group { display:flex; flex-direction:column; gap:3px; }
.bar-row { display:flex; align-items:center; gap:8px; }
.blabel { font-size:10px; color:var(--mu); width:56px; flex-shrink:0; }
.btrack {
  flex:1; height:7px; background:var(--bd); border-radius:4px; overflow:hidden;
}
.bfill { height:100%; border-radius:4px; transition:width .3s ease; }
.bval { font-size:11px; font-weight:600; min-width:42px; text-align:right; }
/* Light metrics compact grid */
.light-metrics { margin-top:8px; }
.lm-grid {
  display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px;
}
.lm-item { background:var(--bg); border:1px solid var(--bd); border-radius:6px; padding:8px 10px; }
.lm-name { font-size:10px; color:var(--mu); margin-bottom:4px; font-weight:500; }
.lm-vals { display:flex; gap:10px; font-size:11px; align-items:baseline; }
.lm-r { font-weight:700; }
.lm-g { color:var(--mu); }
.lm-sep { color:var(--bd); }
/* No-data panel */
.no-data { color:var(--mu); font-size:12px; font-style:italic; padding:16px; }
"""

# ---------------------------------------------------------------------------
# Render helpers — qualitative page
# ---------------------------------------------------------------------------

def _val_color(val_r, val_g):
    """Return CSS color for the regional value."""
    if val_r is None:
        return "var(--mu)"
    if val_g is not None and float(val_r) >= float(val_g):
        return "var(--gr)"
    return "var(--rd)"


def render_light_metrics(ev: dict) -> str:
    """Compact 2x2 grid for instruction light metrics."""
    metrics = [
        ("lang_adherence", "Language Adherence"),
        ("format_compliance", "Format Compliance"),
        ("length_accuracy", "Length Accuracy"),
        ("tone_register", "Tone Register"),
    ]
    items = []
    for key, label in metrics:
        r = ev.get(f"{key}_regional")
        g = ev.get(f"{key}_gemma4")
        if r is None and g is None:
            continue
        r_str = f"{r:.0f}%" if r is not None else "—"
        g_str = f"{g:.0f}%" if g is not None else "—"
        r_col = _val_color(r, g)
        items.append(
            f'<div class="lm-item">'
            f'<div class="lm-name">{label} <span style="font-size:9px;color:var(--mu);">↑</span></div>'
            f'<div class="lm-vals">'
            f'<span class="lm-r" style="color:{r_col};">{r_str}</span>'
            f'<span class="lm-sep">·</span>'
            f'<span class="lm-g">Gemma-4 {g_str}</span>'
            f'</div></div>'
        )
    if not items:
        return ""
    return '<div class="light-metrics"><div class="metric-hdr" style="font-size:10px;color:var(--mu);margin-bottom:4px;">LIGHT METRICS <span class="metric-hint">↑ higher is better · Regional · Gemma-4</span></div><div class="lm-grid">' + "".join(items) + "</div></div>"


def render_task_panel(ev: Optional[dict], task_label: str) -> str:
    if ev is None:
        return f'<div class="no-data"><em>No data for {task_label.lower()} task</em></div>'

    grade = ev.get("grade", "pending")
    task = ev.get("task", "")
    run_id = ev.get("run_id")
    notes = ev.get("notes", "")

    if grade in ("skipped", "failed", "pending") and not run_id:
        badge = grade_badge(grade)
        note_html = f'<div style="color:var(--mu);font-size:11px;margin-top:6px;">{notes}</div>' if notes else ""
        return f'<div class="status-panel">{badge}<span>{GRADE_LABELS.get(grade, grade)}</span>{note_html}</div>'

    parts = [f'<div class="task-panel">']
    parts.append(f'<div class="task-label-hdr">{task_label} {grade_badge(grade)}</div>')

    win_r = ev.get("judge_win_rate")
    win_g = ev.get("gemma4_win_rate")
    parts.append(win_rate_bar(win_r, win_g))

    if task == "translation":
        bleu_r = ev.get("bleu_regional")
        bleu_g = ev.get("bleu_gemma4")
        chrf_r = ev.get("chrf_regional")
        chrf_g = ev.get("chrf_gemma4")
        if bleu_r is not None or bleu_g is not None:
            scale = max(v for v in [bleu_r, bleu_g, 1] if v is not None)
            parts.append(metric_bar("BLEU", bleu_r, bleu_g, scale=scale, hint="↑ word n-gram overlap with reference (0–100)"))
        if chrf_r is not None or chrf_g is not None:
            scale = max(v for v in [chrf_r, chrf_g, 1] if v is not None)
            parts.append(metric_bar("chrF", chrf_r, chrf_g, scale=scale, hint="↑ character-level F-score (0–100)"))

    if task == "instructions":
        lm_html = render_light_metrics(ev)
        if lm_html:
            parts.append(lm_html)

    if run_id:
        review_url = f"/review?run_id={run_id}"
        parts.append(
            f'<div class="run-id-row">'
            f'<code class="run-id-code">{run_id}</code>'
            f'<a class="run-id-link" href="{review_url}">Review ↗</a>'
            f'</div>'
        )

    if notes:
        parts.append(f'<div class="notes-block">{notes}</div>')

    parts.append("</div>")
    return "".join(parts)


def render_language_card(model: str, region: str, instr: Optional[dict], trans: Optional[dict]) -> str:
    # Pick grade from whichever task has data (prefer translation for headline)
    headline_ev = trans or instr
    grade = headline_ev.get("grade", "pending") if headline_ev else "pending"
    badge = grade_badge(grade, size="normal")

    instr_panel = render_task_panel(instr, "INSTRUCTIONS")
    trans_panel = render_task_panel(trans, "TRANSLATION")

    return f"""<div class="lang-card">
  <div class="card-header">
    <span class="card-model-name">{model}</span>
    <div class="card-grade-area">
      {badge}
      <span class="region-chip">{region}</span>
    </div>
  </div>
  <div class="card-panels">
    {instr_panel}
    <div class="panel-divider"></div>
    {trans_panel}
  </div>
</div>"""


def compute_stats(evaluations: list) -> dict:
    grades = {}
    for ev in evaluations:
        if ev.get("task") == "translation":  # one grade per model, use translation
            g = ev.get("grade", "pending")
            grades[g] = grades.get(g, 0) + 1
    return grades


def render_qual_cards(evaluations: list) -> str:
    by_region: dict = {}
    for ev in evaluations:
        region = ev.get("region", "Unknown")
        lang = ev.get("language", "Unknown")
        model = ev.get("model", "Unknown")
        by_region.setdefault(region, {}).setdefault(lang, {}).setdefault(model, []).append(ev)

    parts = []
    regions_in_data = [r for r in REGION_ORDER if r in by_region] + [
        r for r in by_region if r not in REGION_ORDER
    ]

    for region in regions_in_data:
        parts.append(f'<div class="region-section" data-region="{region}">')
        parts.append(f'<div class="region-header">{region}</div>')

        for lang, model_map in by_region[region].items():
            parts.append(f'<div class="lang-group"><div class="lang-group-header">{lang}</div>')
            for model_name, evals in model_map.items():
                instr = next((e for e in evals if e.get("task") == "instructions"), None)
                trans = next((e for e in evals if e.get("task") == "translation"), None)
                parts.append(render_language_card(model_name, region, instr, trans))
            parts.append("</div>")

        parts.append("</div>")

    return "".join(parts)


def render_stats_bar(evaluations: list) -> str:
    total_langs = len({(e["language"], e["model"]) for e in evaluations})
    grade_counts = {}
    for ev in evaluations:
        if ev.get("task") == "translation":
            g = ev.get("grade", "pending")
            grade_counts[g] = grade_counts.get(g, 0) + 1

    def chip(label, val, color):
        return (
            f'<div class="stat-chip">'
            f'<span style="width:8px;height:8px;border-radius:50%;background:{color};display:inline-block;flex-shrink:0;"></span>'
            f'<span>{label}</span><span class="sc-val" style="color:{color};">{val}</span>'
            f'</div>'
        )

    chips = [f'<span style="font-size:12px;color:var(--mu);">{total_langs} model evaluations</span>']
    for g in ("A", "B", "C", "D", "E", "skipped", "failed", "pending"):
        cnt = grade_counts.get(g, 0)
        if cnt:
            chips.append(chip(f"Grade {g.upper()}", cnt, GRADE_COLORS[g]))

    return '<div class="stats-bar">' + "".join(chips) + "</div>"


LEGEND_HTML = """<button class="legend-toggle" onclick="toggleLegend()">
  <span id="legend-arrow">▶</span> Benchmark legend & scoring guide
</button>
<div class="legend-box" id="legend-box">
  <div style="font-size:12px;color:var(--mu);margin-bottom:4px;">All metrics: <strong style="color:var(--gr);">↑ higher is better</strong>. Bars scale to the higher of the two values.</div>
  <div class="legend-grid">
    <div class="legend-section">
      <h4>Translation metrics</h4>
      <div class="legend-row"><span class="legend-key">Judge Win Rate</span><span class="legend-val">% of 50 judged prompts where the regional model was preferred by Gemini (LLM-as-judge). Each prompt judged twice (positions swapped) across 3 dimensions → 300 verdicts total. This is the primary signal.</span></div>
      <div class="legend-row"><span class="legend-key">BLEU (0–100)</span><span class="legend-val">Word n-gram overlap with reference translation. Fast but sensitive to exact wording. Higher = more literal match.</span></div>
      <div class="legend-row"><span class="legend-key">chrF (0–100)</span><span class="legend-val">Character-level F-score. More robust than BLEU for morphologically rich languages (Tamil, Arabic, Hebrew). Higher = better character-level match.</span></div>
    </div>
    <div class="legend-section">
      <h4>Instruction following metrics</h4>
      <div class="legend-row"><span class="legend-key">Judge Win Rate</span><span class="legend-val">Same as translation — primary judge signal. % of judged prompts where regional model followed the Talking Avatar instruction better than Gemma-4.</span></div>
      <div class="legend-row"><span class="legend-key">Language Adherence</span><span class="legend-val">% of responses generated in the correct target language. A model that replies in English scores 0% here.</span></div>
      <div class="legend-row"><span class="legend-key">Format Compliance</span><span class="legend-val">% of responses following structural constraints in the prompt (e.g. "respond in bullet points", "output JSON").</span></div>
      <div class="legend-row"><span class="legend-key">Length Accuracy</span><span class="legend-val">% of responses within the expected length range specified by the instruction.</span></div>
      <div class="legend-row"><span class="legend-key">Tone Register</span><span class="legend-val">% of responses matching the required formality or style (e.g. professional, casual, empathetic).</span></div>
    </div>
  </div>
  <div style="margin-top:12px;font-size:11px;color:var(--mu);">
    <strong>Grade scale:</strong>
    A (&gt;60%) · B (50–60%) · C (40–50%) · D (20–40%) · E (&lt;20%) — based on judge win rate.
    Grade A/B = regional model competitive with Gemma-4 26B. Grade D/E = use Gemma-4 directly.
  </div>
</div>"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/tokenizer")


@app.get("/tokenizer", response_class=HTMLResponse)
def tokenizer_page():
    return f"""<!DOCTYPE html>
<html style="margin:0;height:100vh;display:flex;flex-direction:column;background:#0D1117">
<head><meta charset="UTF-8"><title>Falcon — Tokenizer</title>
<style>{GLOBAL_CSS}</style></head>
<body style="margin:0;height:100vh;display:flex;flex-direction:column;overflow:hidden;">
{nav_html("tokenizer")}
<iframe src="/static/viz/language-map.html" style="flex:1;border:none;width:100%;display:block;"></iframe>
</body></html>"""


@app.get("/qualitative", response_class=HTMLResponse)
def qualitative_page():
    data = json.loads(QUAL_JSON.read_text()) if QUAL_JSON.exists() else {"evaluations": [], "last_updated": None}
    evs = data.get("evaluations", [])
    nav = nav_html("qualitative")
    stats_bar = render_stats_bar(evs)
    cards_html = render_qual_cards(evs)
    last_updated = data.get("last_updated") or "unknown"

    pills = "\n".join(
        f'<span class="pill{" active" if r == "All" else ""}" onclick="filterRegion(this,\'{r}\')">{r}</span>'
        for r in ["All"] + REGION_ORDER
    )

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>Falcon — Qualitative Analysis</title>
<style>{GLOBAL_CSS}{QUAL_PAGE_CSS}</style></head>
<body>
{nav}
<div class="qual-page">
  {stats_bar}
  <div class="filter-row" id="pill-bar">
    {pills}
    <button class="refresh-btn" onclick="doRefresh()">↺ Refresh</button>
  </div>
  <div class="last-updated" id="last-updated">Last updated: {last_updated}</div>
  {LEGEND_HTML}
  <div id="cards-container">{cards_html}</div>
</div>
<script>
var activeRegion = "All";
function filterRegion(el, region) {{
  activeRegion = region;
  document.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
  el.classList.add("active");
  document.querySelectorAll(".region-section").forEach(function(sec) {{
    sec.style.display = (region === "All" || sec.dataset.region === region) ? "" : "none";
  }});
}}
function doRefresh() {{
  var btn = document.querySelector(".refresh-btn");
  btn.textContent = "↺ Refreshing…"; btn.disabled = true;
  fetch("/api/qualitative/refresh")
    .then(r => r.json())
    .then(function(data) {{
      document.getElementById("last-updated").textContent = "Last updated: " + (data.last_updated || "unknown");
      document.getElementById("cards-container").innerHTML = data.html;
      filterRegion(document.querySelector(".pill.active") || document.querySelector(".pill"), activeRegion);
    }})
    .catch(err => alert("Refresh failed: " + err))
    .finally(function() {{ btn.textContent = "↺ Refresh"; btn.disabled = false; }});
}}
function toggleLegend() {{
  var box = document.getElementById("legend-box");
  var arrow = document.getElementById("legend-arrow");
  var open = box.classList.toggle("open");
  arrow.textContent = open ? "▼" : "▶";
}}
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# /review — renamed to Manual Review (QA)
# ---------------------------------------------------------------------------

REVIEW_PAGE_CSS = """
.review-layout { display:flex; height:calc(100vh - 46px); overflow:hidden; }
.review-sidebar {
  width:320px; flex-shrink:0; background:var(--sf); border-right:1px solid var(--bd);
  overflow-y:auto; display:flex; flex-direction:column; padding:20px; gap:14px;
}
.review-main { flex:1; display:flex; flex-direction:column; overflow:hidden; }
.review-iframe { flex:1; border:none; width:100%; background:#0D1117; }
.sidebar-label {
  font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.07em;
  color:var(--mu); margin-bottom:4px;
}
.run-input {
  width:100%; background:var(--bg); border:1px solid var(--bd); color:var(--tx);
  font-size:12px; font-family:"SF Mono","Fira Code",Menlo,monospace;
  padding:9px 10px; border-radius:7px; outline:none; letter-spacing:.02em;
}
.run-input:focus { border-color:var(--ac); box-shadow:0 0 0 3px rgba(88,166,255,.12); }
.load-btn {
  width:100%; padding:9px; background:var(--ac); color:#0D1117; font-weight:700;
  font-size:13px; border:none; border-radius:7px; cursor:pointer; letter-spacing:.01em;
}
.load-btn:disabled { opacity:.5; cursor:not-allowed; }
.meta-block {
  background:var(--sf2); border:1px solid var(--bd); border-radius:8px;
  padding:14px; display:none;
}
.meta-row { font-size:12px; margin-bottom:6px; }
.meta-key { color:var(--mu); }
.meta-val { color:var(--tx); font-weight:600; }
.error-block {
  background:rgba(248,81,73,.08); border:1px solid rgba(248,81,73,.4); border-radius:7px;
  padding:10px 12px; font-size:12px; color:#F85149; display:none; line-height:1.5;
}
.loading-overlay {
  flex:1; display:none; align-items:center; justify-content:center;
  flex-direction:column; gap:14px; color:var(--mu); font-size:13px;
}
.spinner {
  width:28px; height:28px; border:3px solid var(--bd); border-top-color:var(--ac);
  border-radius:50%; animation:spin .7s linear infinite;
}
@keyframes spin { to { transform:rotate(360deg); } }
.placeholder {
  flex:1; display:flex; align-items:center; justify-content:center;
  color:var(--mu); font-size:13px; flex-direction:column; gap:8px;
}
.placeholder-icon { font-size:28px; }
"""


@app.get("/review", response_class=HTMLResponse)
def review_page(run_id: str = ""):
    nav = nav_html("review")
    prefill = run_id or ""
    auto_load = "true" if prefill else "false"

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>Falcon — Manual Review (QA)</title>
<style>{GLOBAL_CSS}{REVIEW_PAGE_CSS}</style></head>
<body>
{nav}
<div class="review-layout">
  <div class="review-sidebar">
    <div>
      <div class="sidebar-label">Run ID</div>
      <input id="run-id-input" class="run-input" type="text"
             placeholder="2026-06-25_094523_dace81" value="{prefill}">
    </div>
    <button class="load-btn" id="load-btn" onclick="loadReview()">Load Review</button>
    <div class="error-block" id="error-block"></div>
    <div class="meta-block" id="meta-block">
      <div class="sidebar-label" style="margin-bottom:10px;">Run Info</div>
      <div class="meta-row">
        <span class="meta-key">Run ID<br></span>
        <code style="font-size:11px;font-weight:600;color:var(--tx);letter-spacing:.02em;" id="m-run-id"></code>
      </div>
      <div class="meta-row"><span class="meta-key">Slug </span><span class="meta-val" id="m-slug"></span></div>
      <div class="meta-row"><span class="meta-key">Task </span><span class="meta-val" id="m-task"></span></div>
      <div class="meta-row" id="m-grade-row" style="display:none">
        <span class="meta-key">Grade </span><span id="m-grade"></span>
      </div>
      <div class="meta-row" id="m-wr-row" style="display:none">
        <span class="meta-key">Win rate </span><span class="meta-val" id="m-wr"></span>
      </div>
      <div style="margin-top:12px;">
        <a id="m-full-link" href="#" target="_blank" style="font-size:12px;color:var(--ac);">Open full page ↗</a>
      </div>
    </div>
  </div>

  <div class="review-main" id="review-main">
    <div class="placeholder" id="placeholder">
      <span class="placeholder-icon">🔍</span>
      <span>Enter a run ID and click Load Review</span>
    </div>
    <div class="loading-overlay" id="loading-overlay" style="display:none;">
      <div class="spinner"></div>
      <span>Fetching from Modal volume…</span>
    </div>
    <iframe class="review-iframe" id="review-iframe" src="" style="display:none;"></iframe>
  </div>
</div>

<script>
var autoLoad = {auto_load};
function gradeColor(g) {{
  return {{A:"#3FB950",B:"#52B788",C:"#F5A623",D:"#E76F51",E:"#F85149"}}[g.toUpperCase()] || "#8B949E";
}}
function loadReview() {{
  var runId = document.getElementById("run-id-input").value.trim();
  if (!runId) {{ alert("Enter a run ID first."); return; }}
  document.getElementById("placeholder").style.display = "none";
  document.getElementById("review-iframe").style.display = "none";
  document.getElementById("loading-overlay").style.display = "flex";
  document.getElementById("error-block").style.display = "none";
  document.getElementById("meta-block").style.display = "none";
  document.getElementById("load-btn").disabled = true;
  fetch("/api/review/load", {{
    method:"POST", headers:{{"Content-Type":"application/json"}},
    body: JSON.stringify({{run_id: runId}})
  }})
  .then(r => r.json())
  .then(function(data) {{
    document.getElementById("loading-overlay").style.display = "none";
    if (data.error) {{
      document.getElementById("error-block").textContent = data.error;
      document.getElementById("error-block").style.display = "block";
      document.getElementById("placeholder").style.display = "flex";
      return;
    }}
    var iframe = document.getElementById("review-iframe");
    iframe.src = data.url; iframe.style.display = "block";
    document.getElementById("m-run-id").textContent = data.run_id;
    document.getElementById("m-slug").textContent = data.slug || "—";
    document.getElementById("m-task").textContent = data.task || "—";
    if (data.grade) {{
      document.getElementById("m-grade").innerHTML =
        '<span style="font-weight:700;font-size:13px;color:' + gradeColor(data.grade) + '">' + data.grade.toUpperCase() + '</span>';
      document.getElementById("m-grade-row").style.display = "";
    }}
    if (data.win_rate != null) {{
      document.getElementById("m-wr").textContent = data.win_rate + "%";
      document.getElementById("m-wr-row").style.display = "";
    }}
    document.getElementById("m-full-link").href = data.url;
    document.getElementById("meta-block").style.display = "block";
  }})
  .catch(function(err) {{
    document.getElementById("loading-overlay").style.display = "none";
    document.getElementById("error-block").textContent = "Request failed: " + err;
    document.getElementById("error-block").style.display = "block";
    document.getElementById("placeholder").style.display = "flex";
  }})
  .finally(function() {{ document.getElementById("load-btn").disabled = false; }});
}}
if (autoLoad) window.addEventListener("DOMContentLoaded", loadReview);
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# API: POST /api/review/load
# ---------------------------------------------------------------------------

def modal_get_file(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["modal", "volume", "get", "--force", MODAL_VOLUME, remote, str(local)],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def modal_ls(path: str) -> str:
    r = subprocess.run(
        ["modal", "volume", "ls", MODAL_VOLUME, path],
        capture_output=True, text=True,
    )
    return r.stdout + r.stderr


def discover_slug_task(run_id: str) -> tuple[str, str]:
    output = modal_ls(f"runs/{run_id}/judge/")
    for line in output.splitlines():
        m = re.search(r'([^/\s]+)_(instructions|translation)_verdicts\.jsonl', line.strip())
        if m:
            return m.group(1), m.group(2)
    output = modal_ls(f"runs/{run_id}/regional/")
    for line in output.splitlines():
        m = re.search(r'([^/\s]+)_(instructions|translation)_outputs\.jsonl', line.strip())
        if m:
            return m.group(1), m.group(2)
    return "", ""


def load_generate_review():
    gr_path = REPO_ROOT / "scripts" / "generate_review.py"
    spec = importlib.util.spec_from_file_location("generate_review", gr_path)
    gr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gr)
    return gr


@app.post("/api/review/load")
async def load_review_api(request: Request):
    body = await request.json()
    run_id = (body.get("run_id") or "").strip()
    if not run_id:
        return JSONResponse({"error": "run_id is required"})

    slug, task = discover_slug_task(run_id)
    if not slug or not task:
        return JSONResponse({"error": "Could not discover slug/task from Modal volume. Is the run ID correct and does it exist in phase2a-outputs?"})

    out_dir = REVIEW_DIR / f"{slug}_{task}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "review.html"

    with tempfile.TemporaryDirectory(prefix="dash_review_") as tmp:
        tmp_path = Path(tmp)
        gemma4_path  = tmp_path / "gemma4.jsonl"
        regional_path = tmp_path / "regional.jsonl"
        verdicts_path = tmp_path / "verdicts.jsonl"

        ok1 = modal_get_file(f"runs/{run_id}/gemma4/{task}_outputs.jsonl", gemma4_path)
        ok2 = modal_get_file(f"runs/{run_id}/regional/{slug}_{task}_outputs.jsonl", regional_path)
        ok3 = modal_get_file(f"runs/{run_id}/judge/{slug}_{task}_verdicts.jsonl", verdicts_path)

        if not (ok1 and ok2 and ok3):
            missing = []
            if not ok1: missing.append("gemma4 outputs")
            if not ok2: missing.append("regional outputs")
            if not ok3: missing.append("judge verdicts")
            return JSONResponse({"error": f"Could not download from Modal volume: missing {', '.join(missing)}. Run ID: {run_id}"})

        try:
            gr = load_generate_review()
        except Exception as e:
            return JSONResponse({"error": f"Failed to load generate_review.py: {e}"})

        gemma4_records   = gr.load_jsonl(gemma4_path)
        regional_records = gr.load_jsonl(regional_path)
        verdict_records  = gr.load_jsonl(verdicts_path)
        aggregated       = gr.aggregate_verdicts(verdict_records)

        html_content = gr.generate_html(
            run_id=run_id, slug=slug, task=task, model_display=slug,
            regional_records=regional_records, gemma4_records=gemma4_records,
            aggregated=aggregated,
        )
        out_file.write_text(html_content, encoding="utf-8")

    grade = win_rate = None
    try:
        data = json.loads(QUAL_JSON.read_text())
        for ev in data.get("evaluations", []):
            if ev.get("run_id") == run_id:
                grade = ev.get("grade")
                wr = ev.get("judge_win_rate")
                win_rate = round(wr, 1) if wr is not None else None
                break
    except Exception:
        pass

    return JSONResponse({
        "url": f"/static/review/{slug}_{task}_{run_id}/review.html",
        "slug": slug, "task": task, "run_id": run_id,
        "grade": grade, "win_rate": win_rate,
    })


# ---------------------------------------------------------------------------
# API: GET /api/qualitative/refresh
# ---------------------------------------------------------------------------

def parse_report_file(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    run_id = data.get("run_id")
    task = data.get("task")
    if not run_id or not task:
        return None

    if "classification" in data:
        wr = data.get("judge_win_rate")
        win_rate = round(float(wr) * 100, 1) if wr is not None and float(wr) <= 1.0 else (round(float(wr), 1) if wr is not None else None)
        ev = {"run_id": run_id, "task": task, "grade": data.get("classification"), "judge_win_rate": win_rate}
        for k in ["bleu_regional", "bleu_gemma4", "chrf_regional", "chrf_gemma4"]:
            if k in data:
                ev[k] = data[k]
        return ev

    results = data.get("results", {})
    if not results:
        return None
    slug = next(iter(results))
    res = results[slug]
    wr_raw = res.get("judge_win_rate")
    win_rate = round(float(wr_raw) * 100, 1) if wr_raw is not None else None
    ev = {"run_id": run_id, "task": task, "grade": res.get("classification"), "judge_win_rate": win_rate}
    if task == "instructions":
        for k in ["lang_adherence_regional", "lang_adherence_gemma4", "format_compliance_regional",
                  "format_compliance_gemma4", "length_accuracy_regional", "length_accuracy_gemma4",
                  "tone_register_regional", "tone_register_gemma4"]:
            v = res.get(k)
            ev[k] = round(v * 100) if v is not None else None
    else:
        for k in ["bleu_regional", "bleu_gemma4", "chrf_regional", "chrf_gemma4"]:
            ev[k] = res.get(k)
    return ev


@app.get("/api/qualitative/refresh")
def qualitative_refresh():
    data = json.loads(QUAL_JSON.read_text()) if QUAL_JSON.exists() else {"evaluations": []}
    evaluations = data.get("evaluations", [])
    run_id_index = {ev.get("run_id"): ev for ev in evaluations if ev.get("run_id")}

    reports_dir = REPO_ROOT / "data" / "reports"
    if reports_dir.exists():
        for report_file in sorted(reports_dir.glob("*.json")):
            parsed = parse_report_file(report_file)
            if not parsed or not parsed.get("run_id"):
                continue
            existing = run_id_index.get(parsed["run_id"])
            if existing:
                for field in ["grade", "judge_win_rate", "gemma4_win_rate",
                              "bleu_regional", "bleu_gemma4", "chrf_regional", "chrf_gemma4",
                              "lang_adherence_regional", "lang_adherence_gemma4",
                              "format_compliance_regional", "format_compliance_gemma4",
                              "length_accuracy_regional", "length_accuracy_gemma4",
                              "tone_register_regional", "tone_register_gemma4"]:
                    if field in parsed and parsed[field] is not None:
                        existing[field] = parsed[field]
                existing["status"] = "complete"

    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    data["evaluations"] = evaluations
    QUAL_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return JSONResponse({
        "last_updated": data["last_updated"],
        "html": render_qual_cards(evaluations),
    })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n  Falcon Dashboard → http://localhost:{PORT}\n")
    uvicorn.run(
        "dashboard:app",
        host="0.0.0.0", port=PORT, reload=True,
        app_dir=str(REPO_ROOT / "scripts"),
    )
