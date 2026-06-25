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
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print("Missing dependencies. Run: pip install fastapi uvicorn")
    sys.exit(1)

app = FastAPI(title="Falcon Dashboard")

# Mount static dirs
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/viz", StaticFiles(directory=str(VIZ_DIR)), name="viz")
app.mount("/static/review", StaticFiles(directory=str(REVIEW_DIR)), name="review")

# ---------------------------------------------------------------------------
# CSS / design tokens (dark GitHub-like theme)
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
"""

GRADE_COLORS = {
    "A": "#3FB950",
    "B": "#52B788",
    "C": "#F5A623",
    "D": "#E76F51",
    "E": "#F85149",
    "skipped": "#8B949E",
    "failed": "#8B949E",
    "pending": "#30363D",
}

GRADE_LABELS = {
    "A": "A: Regional superior (>60% win)",
    "B": "B: Regional preferred (50–60%)",
    "C": "C: Comparable (40–50%)",
    "D": "D: Gemma-4 preferred (20–40%)",
    "E": "E: Gemma-4 strongly preferred (<20%)",
    "skipped": "Skipped — run failed or crashed",
    "failed": "Failed — infrastructure error",
    "pending": "Pending — not yet run",
}

REGION_ORDER = ["Indic", "Middle East", "East Asia", "SEA", "Africa", "Europe", "Americas", "Oceania"]

# ---------------------------------------------------------------------------
# Nav bar
# ---------------------------------------------------------------------------

def nav_html(active: str) -> str:
    def tab(label: str, href: str, key: str) -> str:
        style = (
            "color:var(--ac);border-bottom:2px solid var(--ac);padding-bottom:2px;"
            if key == active
            else "color:var(--mu);"
        )
        return f'<a href="{href}" style="text-decoration:none;font-size:13px;font-weight:500;{style}">{label}</a>'

    return f"""<nav style="
        height:46px;
        display:flex;
        align-items:center;
        gap:24px;
        padding:0 24px;
        background:var(--sf);
        border-bottom:1px solid var(--bd);
        position:sticky;top:0;z-index:100;
        flex-shrink:0;
    ">
        <span style="font-size:15px;font-weight:700;color:var(--tx);white-space:nowrap;">🌍 Falcon</span>
        <span style="color:var(--bd);">/</span>
        {tab("Tokenizer", "/tokenizer", "tokenizer")}
        {tab("Qualitative Analysis", "/qualitative", "qualitative")}
        {tab("Review", "/review", "review")}
    </nav>"""

# ---------------------------------------------------------------------------
# Grade badge
# ---------------------------------------------------------------------------

def grade_badge(grade: str) -> str:
    color = GRADE_COLORS.get(grade, "#8B949E")
    tooltip = GRADE_LABELS.get(grade, grade)
    text_color = "#0D1117" if grade in ("A", "B") else color
    bg = color + "26"  # 15% alpha
    if grade in ("pending",):
        text_color = "#8B949E"
    return (
        f'<span title="{tooltip}" style="'
        f'display:inline-block;font-size:11px;font-weight:700;padding:2px 9px;'
        f'border-radius:10px;background:{bg};color:{color};border:1px solid {color}44;'
        f'cursor:default;white-space:nowrap;">{grade.upper()}</span>'
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/tokenizer")


@app.get("/tokenizer", response_class=HTMLResponse)
def tokenizer_page():
    nav = nav_html("tokenizer")
    return f"""<!DOCTYPE html>
<html style="margin:0;height:100vh;display:flex;flex-direction:column;background:#0D1117">
<head>
<meta charset="UTF-8">
<title>Falcon — Tokenizer</title>
<style>{GLOBAL_CSS}</style>
</head>
<body style="margin:0;height:100vh;display:flex;flex-direction:column;overflow:hidden;">
{nav}
<iframe src="/static/viz/language-map.html"
        style="flex:1;border:none;width:100%;display:block;"></iframe>
</body>
</html>"""


# ---------------------------------------------------------------------------
# /qualitative — server-rendered qualitative analysis page
# ---------------------------------------------------------------------------

def load_qual_data() -> dict:
    if not QUAL_JSON.exists():
        return {"last_updated": None, "evaluations": []}
    return json.loads(QUAL_JSON.read_text())


def render_qual_cards(evaluations: list, active_region: str = "All") -> str:
    """Render the language cards section (injected via innerHTML on refresh)."""
    # Group by region → language → model (list of evals)
    by_region: dict = {}
    for ev in evaluations:
        region = ev.get("region", "Unknown")
        lang = ev.get("language", "Unknown")
        model = ev.get("model", "Unknown")
        by_region.setdefault(region, {}).setdefault(lang, {}).setdefault(model, []).append(ev)

    html_parts = []
    regions_in_data = [r for r in REGION_ORDER if r in by_region] + [
        r for r in by_region if r not in REGION_ORDER
    ]

    for region in regions_in_data:
        hidden = "" if active_region in ("All", region) else ' style="display:none"'
        html_parts.append(
            f'<div class="region-section" data-region="{region}"{hidden}>'
            f'<h2 class="region-header">{region}</h2>'
        )

        lang_map = by_region[region]
        for lang, model_map in lang_map.items():
            html_parts.append(f'<div class="lang-group"><h3 class="lang-header">{lang}</h3>')
            for model_name, evals in model_map.items():
                instr = next((e for e in evals if e.get("task") == "instructions"), None)
                trans = next((e for e in evals if e.get("task") == "translation"), None)
                html_parts.append(render_language_card(lang, region, model_name, instr, trans))
            html_parts.append("</div>")

        html_parts.append("</div>")

    return "\n".join(html_parts)


def render_task_panel(ev: Optional[dict], task_label: str) -> str:
    if ev is None:
        return f"""<div class="task-panel">
            <div class="task-label">{task_label}</div>
            <div style="color:var(--mu);font-size:12px;">No data</div>
        </div>"""

    grade = ev.get("grade", "pending")
    win_rate = ev.get("judge_win_rate")
    run_id = ev.get("run_id")
    notes = ev.get("notes", "")
    task = ev.get("task", "")

    badge = grade_badge(grade)
    win_str = f"{win_rate:.1f}%" if win_rate is not None else "—"

    lines = [
        f'<div class="task-label">{task_label}</div>',
        f'<div class="metric-row">Grade: {badge}</div>',
        f'<div class="metric-row">Win rate: <strong>{win_str}</strong></div>',
    ]

    if task == "translation":
        bleu_r = ev.get("bleu_regional")
        bleu_g = ev.get("bleu_gemma4")
        chrf_r = ev.get("chrf_regional")
        chrf_g = ev.get("chrf_gemma4")
        if bleu_r is not None or bleu_g is not None:
            br = f"{bleu_r:.2f}" if bleu_r is not None else "—"
            bg = f"{bleu_g:.2f}" if bleu_g is not None else "—"
            lines.append(f'<div class="metric-row">BLEU: <span class="mv">{br}</span> vs <span class="mv gemma4-val">{bg}</span></div>')
        if chrf_r is not None or chrf_g is not None:
            cr = f"{chrf_r:.2f}" if chrf_r is not None else "—"
            cg = f"{chrf_g:.2f}" if chrf_g is not None else "—"
            lines.append(f'<div class="metric-row">chrF: <span class="mv">{cr}</span> vs <span class="mv gemma4-val">{cg}</span></div>')

    if run_id:
        review_url = f"/review?run_id={run_id}"
        lines.append(f'<div class="metric-row run-id">{run_id}</div>')
        lines.append(f'<div class="metric-row"><a href="{review_url}" class="review-link">Open Review ↗</a></div>')

    if notes:
        lines.append(f'<div class="notes">{notes}</div>')

    return '<div class="task-panel">' + "\n".join(lines) + "</div>"


def render_language_card(lang: str, region: str, model: str, instr: Optional[dict], trans: Optional[dict]) -> str:
    instr_panel = render_task_panel(instr, "INSTRUCTIONS")
    trans_panel = render_task_panel(trans, "TRANSLATION")
    return f"""<div class="lang-card">
    <div class="card-top">
        <span class="card-model">{model}</span>
        <span class="region-tag">{region}</span>
    </div>
    <div class="card-panels">
        {instr_panel}
        <div class="panel-divider"></div>
        {trans_panel}
    </div>
</div>"""


QUAL_PAGE_CSS = """
.region-header {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--mu);
    padding: 8px 0 6px;
    border-bottom: 1px solid var(--bd);
    margin-bottom: 12px;
}
.lang-group { margin-bottom: 24px; }
.lang-header {
    font-size: 15px;
    font-weight: 600;
    color: var(--tx);
    margin-bottom: 10px;
}
.lang-card {
    background: var(--sf);
    border: 1px solid var(--bd);
    border-radius: 8px;
    margin-bottom: 10px;
    overflow: hidden;
}
.card-top {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    background: var(--sf2);
    border-bottom: 1px solid var(--bd);
}
.card-model { font-size: 12px; font-weight: 600; color: var(--tx); }
.region-tag {
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 4px;
    background: rgba(88,166,255,.12);
    color: var(--ac);
    margin-left: auto;
}
.card-panels {
    display: grid;
    grid-template-columns: 1fr 1px 1fr;
}
.task-panel { padding: 12px 14px; }
.panel-divider { background: var(--bd); }
.task-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: var(--mu);
    margin-bottom: 8px;
}
.metric-row { font-size: 12px; margin-bottom: 4px; color: var(--tx); }
.run-id { font-family: monospace; font-size: 10px; color: var(--mu); }
.mv { font-weight: 600; color: var(--gr); }
.gemma4-val { color: #F85149; }
.review-link {
    font-size: 11px;
    color: var(--ac);
    text-decoration: none;
    font-weight: 500;
}
.review-link:hover { text-decoration: underline; }
.notes {
    font-size: 11px;
    color: var(--mu);
    margin-top: 6px;
    font-style: italic;
    line-height: 1.4;
    border-top: 1px solid var(--bd);
    padding-top: 6px;
}
.region-section { margin-bottom: 32px; }
/* Filter pills */
.pill-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 20px;
    align-items: center;
}
.pill {
    font-size: 12px;
    font-weight: 500;
    padding: 4px 13px;
    border-radius: 20px;
    border: 1px solid var(--bd);
    background: var(--sf2);
    color: var(--mu);
    cursor: pointer;
    transition: all .15s;
    user-select: none;
}
.pill:hover { border-color: var(--ac); color: var(--tx); }
.pill.active { background: var(--ac); color: #0D1117; border-color: var(--ac); font-weight: 700; }
.refresh-btn {
    margin-left: auto;
    font-size: 12px;
    padding: 5px 13px;
    border-radius: 6px;
    border: 1px solid var(--bd);
    background: var(--sf2);
    color: var(--tx);
    cursor: pointer;
}
.refresh-btn:hover { border-color: var(--ac); }
.last-updated { font-size: 11px; color: var(--mu); margin-bottom: 12px; }
"""


@app.get("/qualitative", response_class=HTMLResponse)
def qualitative_page():
    data = load_qual_data()
    nav = nav_html("qualitative")
    cards_html = render_qual_cards(data.get("evaluations", []))
    last_updated = data.get("last_updated") or "unknown"

    regions_list = ["All"] + REGION_ORDER
    pills_html = "\n".join(
        f'<span class="pill{"  active" if r == "All" else ""}" onclick="filterRegion(this, \'{r}\')">{r}</span>'
        for r in regions_list
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Falcon — Qualitative Analysis</title>
<style>{GLOBAL_CSS}{QUAL_PAGE_CSS}</style>
</head>
<body>
{nav}
<div style="max-width:1100px;margin:0 auto;padding:24px 20px;">
  <div class="pill-bar" id="pill-bar">
    {pills_html}
    <button class="refresh-btn" onclick="doRefresh()">↺ Refresh</button>
  </div>
  <div class="last-updated" id="last-updated">Last updated: {last_updated}</div>
  <div id="cards-container">
    {cards_html}
  </div>
</div>

<script>
var activeRegion = "All";

function filterRegion(el, region) {{
  activeRegion = region;
  document.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
  el.classList.add("active");
  document.querySelectorAll(".region-section").forEach(function(sec) {{
    if (region === "All" || sec.dataset.region === region) {{
      sec.style.display = "";
    }} else {{
      sec.style.display = "none";
    }}
  }});
}}

function doRefresh() {{
  var btn = document.querySelector(".refresh-btn");
  btn.textContent = "↺ Refreshing…";
  btn.disabled = true;
  fetch("/api/qualitative/refresh")
    .then(function(r) {{ return r.json(); }})
    .then(function(data) {{
      document.getElementById("last-updated").textContent = "Last updated: " + (data.last_updated || "unknown");
      document.getElementById("cards-container").innerHTML = data.html;
      // Re-apply active region filter
      document.querySelectorAll(".region-section").forEach(function(sec) {{
        if (activeRegion === "All" || sec.dataset.region === activeRegion) {{
          sec.style.display = "";
        }} else {{
          sec.style.display = "none";
        }}
      }});
    }})
    .catch(function(err) {{ alert("Refresh failed: " + err); }})
    .finally(function() {{
      btn.textContent = "↺ Refresh";
      btn.disabled = false;
    }});
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# /review page
# ---------------------------------------------------------------------------

REVIEW_PAGE_CSS = """
.review-layout {
    display: flex;
    height: calc(100vh - 46px);
    overflow: hidden;
}
.review-sidebar {
    width: 350px;
    flex-shrink: 0;
    background: var(--sf);
    border-right: 1px solid var(--bd);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    padding: 20px;
    gap: 14px;
}
.review-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.review-iframe {
    flex: 1;
    border: none;
    width: 100%;
    background: #0D1117;
}
.sidebar-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: var(--mu);
    margin-bottom: 4px;
}
.run-input {
    width: 100%;
    background: var(--sf2);
    border: 1px solid var(--bd);
    color: var(--tx);
    font-size: 12px;
    font-family: monospace;
    padding: 8px 10px;
    border-radius: 6px;
    outline: none;
}
.run-input:focus { border-color: var(--ac); }
.load-btn {
    width: 100%;
    padding: 8px;
    background: var(--ac);
    color: #0D1117;
    font-weight: 700;
    font-size: 13px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
}
.load-btn:disabled { opacity: .5; cursor: not-allowed; }
.meta-block {
    background: var(--sf2);
    border: 1px solid var(--bd);
    border-radius: 8px;
    padding: 12px;
    display: none;
}
.meta-row { font-size: 12px; margin-bottom: 6px; }
.meta-key { color: var(--mu); }
.meta-val { color: var(--tx); font-weight: 600; }
.error-block {
    background: rgba(248,81,73,.1);
    border: 1px solid #F85149;
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 12px;
    color: #F85149;
    display: none;
}
.loading-overlay {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 12px;
    color: var(--mu);
    font-size: 13px;
    display: none;
}
.spinner {
    width: 28px; height: 28px;
    border: 3px solid var(--bd);
    border-top-color: var(--ac);
    border-radius: 50%;
    animation: spin .7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.placeholder {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--mu);
    font-size: 13px;
}
"""


@app.get("/review", response_class=HTMLResponse)
def review_page(run_id: str = ""):
    nav = nav_html("review")
    prefill = run_id or ""
    auto_load = "true" if prefill else "false"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Falcon — Review</title>
<style>{GLOBAL_CSS}{REVIEW_PAGE_CSS}</style>
</head>
<body>
{nav}
<div class="review-layout">
  <!-- Sidebar -->
  <div class="review-sidebar">
    <div>
      <div class="sidebar-label">Run ID</div>
      <input id="run-id-input" class="run-input" type="text"
             placeholder="2026-06-25_081543_85cde3"
             value="{prefill}">
    </div>
    <button class="load-btn" id="load-btn" onclick="loadReview()">Load Review</button>

    <div class="error-block" id="error-block"></div>

    <div class="meta-block" id="meta-block">
      <div class="sidebar-label" style="margin-bottom:10px;">Run Metadata</div>
      <div class="meta-row"><span class="meta-key">Run ID: </span><span class="meta-val" id="m-run-id" style="font-family:monospace;font-size:11px;"></span></div>
      <div class="meta-row"><span class="meta-key">Slug: </span><span class="meta-val" id="m-slug"></span></div>
      <div class="meta-row"><span class="meta-key">Task: </span><span class="meta-val" id="m-task"></span></div>
      <div class="meta-row" id="m-grade-row" style="display:none">
        <span class="meta-key">Grade: </span><span id="m-grade"></span>
      </div>
      <div class="meta-row" id="m-wr-row" style="display:none">
        <span class="meta-key">Win rate: </span><span class="meta-val" id="m-wr"></span>
      </div>
      <div style="margin-top:10px;">
        <a id="m-full-link" href="#" target="_blank"
           style="font-size:12px;color:var(--ac);">Open full page ↗</a>
      </div>
    </div>
  </div>

  <!-- Main panel -->
  <div class="review-main" id="review-main">
    <div class="placeholder" id="placeholder">
      Enter a run ID and click Load Review
    </div>
    <div class="loading-overlay" id="loading-overlay">
      <div class="spinner"></div>
      <span>Fetching from Modal volume…</span>
    </div>
    <iframe class="review-iframe" id="review-iframe" src="" style="display:none;"></iframe>
  </div>
</div>

<script>
var autoLoad = {auto_load};

function loadReview() {{
  var runId = document.getElementById("run-id-input").value.trim();
  if (!runId) {{ alert("Enter a run ID first."); return; }}

  // Show loading
  document.getElementById("placeholder").style.display = "none";
  document.getElementById("review-iframe").style.display = "none";
  document.getElementById("loading-overlay").style.display = "flex";
  document.getElementById("error-block").style.display = "none";
  document.getElementById("meta-block").style.display = "none";
  document.getElementById("load-btn").disabled = true;

  fetch("/api/review/load", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{run_id: runId}})
  }})
  .then(function(r) {{ return r.json(); }})
  .then(function(data) {{
    document.getElementById("loading-overlay").style.display = "none";
    if (data.error) {{
      document.getElementById("error-block").textContent = data.error;
      document.getElementById("error-block").style.display = "block";
      document.getElementById("placeholder").style.display = "flex";
      return;
    }}
    // Show iframe
    var iframe = document.getElementById("review-iframe");
    iframe.src = data.url;
    iframe.style.display = "block";

    // Populate metadata
    document.getElementById("m-run-id").textContent = data.run_id;
    document.getElementById("m-slug").textContent = data.slug || "—";
    document.getElementById("m-task").textContent = data.task || "—";
    if (data.grade) {{
      document.getElementById("m-grade").innerHTML = '<span style="font-weight:700;color:' + gradeColor(data.grade) + '">' + data.grade.toUpperCase() + '</span>';
      document.getElementById("m-grade-row").style.display = "";
    }}
    if (data.win_rate !== undefined && data.win_rate !== null) {{
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
  .finally(function() {{
    document.getElementById("load-btn").disabled = false;
  }});
}}

function gradeColor(g) {{
  var colors = {{A:"#3FB950",B:"#52B788",C:"#F5A623",D:"#E76F51",E:"#F85149"}};
  return colors[g.toUpperCase()] || "#8B949E";
}}

if (autoLoad) {{
  window.addEventListener("DOMContentLoaded", function() {{
    loadReview();
  }});
}}
</script>
</body>
</html>"""


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
    """Return (slug, task) by listing the judge/ dir first, then regional/."""
    # Try judge/ dir: files like "tamil_instructions_verdicts.jsonl"
    output = modal_ls(f"runs/{run_id}/judge/")
    for line in output.splitlines():
        line = line.strip()
        m = re.search(r'([^/\s]+)_(instructions|translation)_verdicts\.jsonl', line)
        if m:
            return m.group(1), m.group(2)

    # Fallback: regional/ dir
    output = modal_ls(f"runs/{run_id}/regional/")
    for line in output.splitlines():
        line = line.strip()
        m = re.search(r'([^/\s]+)_(instructions|translation)_outputs\.jsonl', line)
        if m:
            return m.group(1), m.group(2)

    return "", ""


def load_generate_review():
    """Import generate_review.py functions."""
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
        return JSONResponse({"error": "Could not discover slug/task from Modal volume. Is the run ID correct?"})

    out_dir = REVIEW_DIR / f"{slug}_{task}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "review.html"

    with tempfile.TemporaryDirectory(prefix="dash_review_") as tmp:
        tmp_path = Path(tmp)

        gemma4_path = tmp_path / "gemma4.jsonl"
        regional_path = tmp_path / "regional.jsonl"
        verdicts_path = tmp_path / "verdicts.jsonl"

        ok1 = modal_get_file(f"runs/{run_id}/gemma4/{task}_outputs.jsonl", gemma4_path)
        ok2 = modal_get_file(f"runs/{run_id}/regional/{slug}_{task}_outputs.jsonl", regional_path)
        ok3 = modal_get_file(f"runs/{run_id}/judge/{slug}_{task}_verdicts.jsonl", verdicts_path)

        if not (ok1 and ok2 and ok3):
            return JSONResponse({"error": "Could not download from Modal volume. Is the run ID correct?"})

        try:
            gr = load_generate_review()
        except Exception as e:
            return JSONResponse({"error": f"Failed to load generate_review.py: {e}"})

        gemma4_records = gr.load_jsonl(gemma4_path)
        regional_records = gr.load_jsonl(regional_path)
        verdict_records = gr.load_jsonl(verdicts_path)

        aggregated = gr.aggregate_verdicts(verdict_records)

        html_content = gr.generate_html(
            run_id=run_id,
            slug=slug,
            task=task,
            model_display=slug,
            regional_records=regional_records,
            gemma4_records=gemma4_records,
            aggregated=aggregated,
        )

        out_file.write_text(html_content, encoding="utf-8")

    # Try to read grade/win_rate from qualitative_results.json
    grade = None
    win_rate = None
    try:
        data = load_qual_data()
        for ev in data.get("evaluations", []):
            if ev.get("run_id") == run_id:
                grade = ev.get("grade")
                wr = ev.get("judge_win_rate")
                win_rate = round(wr, 1) if wr is not None else None
                break
    except Exception:
        pass

    url = f"/static/review/{slug}_{task}_{run_id}/review.html"
    return JSONResponse({
        "url": url,
        "slug": slug,
        "task": task,
        "run_id": run_id,
        "grade": grade,
        "win_rate": win_rate,
    })


# ---------------------------------------------------------------------------
# API: GET /api/qualitative/refresh
# ---------------------------------------------------------------------------

def parse_report_file(path: Path) -> Optional[dict]:
    """
    Parse a report JSON from data/reports/ and return a flat evaluation dict.
    Handles both old format (nested under results.{slug}) and new flat format.
    """
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    run_id = data.get("run_id")
    task = data.get("task")

    # New flat format (e.g. sarvam re-runs)
    if "slug" in data and "classification" in data:
        slug = data.get("slug", "")
        model = data.get("model", slug)
        grade = data.get("classification", "pending")
        win_rate = data.get("judge_win_rate")
        if win_rate is not None:
            win_rate = round(float(win_rate) * 100, 1) if win_rate <= 1.0 else round(float(win_rate), 1)

        ev = {
            "run_id": run_id,
            "task": task,
            "grade": grade,
            "judge_win_rate": win_rate,
        }
        if task == "translation":
            ev.update({
                "bleu_regional": data.get("bleu_regional"),
                "bleu_gemma4": data.get("bleu_gemma4"),
                "chrf_regional": data.get("chrf_regional"),
                "chrf_gemma4": data.get("chrf_gemma4"),
                "bertscore_f1_regional": data.get("bertscore_f1_regional"),
                "bertscore_f1_gemma4": data.get("bertscore_f1_gemma4"),
            })
        return ev

    # Old format: results nested under language slug
    results = data.get("results", {})
    if not results:
        return None

    slug = next(iter(results))
    res = results[slug]
    model = res.get("regional_model", slug)
    grade = res.get("classification", "pending")
    win_rate_raw = res.get("judge_win_rate")
    win_rate = None
    if win_rate_raw is not None:
        win_rate = round(float(win_rate_raw) * 100, 1)

    ev = {
        "run_id": run_id,
        "task": task,
        "model_key": model,
        "grade": grade,
        "judge_win_rate": win_rate,
        "gemma4_win_rate": round(res.get("gemma4_win_rate", 0) * 100, 1) if res.get("gemma4_win_rate") is not None else None,
    }

    if task == "instructions":
        for k in ["lang_adherence_regional", "lang_adherence_gemma4",
                  "format_compliance_regional", "format_compliance_gemma4",
                  "length_accuracy_regional", "length_accuracy_gemma4",
                  "tone_register_regional", "tone_register_gemma4"]:
            v = res.get(k)
            ev[k] = round(v * 100) if v is not None else None
    else:
        for k in ["bleu_regional", "bleu_gemma4", "chrf_regional", "chrf_gemma4",
                  "bertscore_f1_regional", "bertscore_f1_gemma4"]:
            ev[k] = res.get(k)

    return ev


@app.get("/api/qualitative/refresh")
def qualitative_refresh():
    data = load_qual_data()
    evaluations = data.get("evaluations", [])

    # Index existing by run_id for fast lookup
    existing_run_ids = {ev.get("run_id") for ev in evaluations if ev.get("run_id")}

    reports_dir = REPO_ROOT / "data" / "reports"
    if reports_dir.exists():
        for report_file in sorted(reports_dir.glob("*.json")):
            parsed = parse_report_file(report_file)
            if parsed and parsed.get("run_id") and parsed["run_id"] not in existing_run_ids:
                # New run found — we can't determine language/region/model from report alone
                # so just update matching evaluation by run_id if it exists
                # (for now, log and skip — the canonical data is in qualitative_results.json)
                pass
            elif parsed and parsed.get("run_id"):
                # Update metrics on existing entry
                run_id = parsed["run_id"]
                for ev in evaluations:
                    if ev.get("run_id") == run_id:
                        # Update grade and metrics from report
                        for field in ["grade", "judge_win_rate", "gemma4_win_rate",
                                      "bleu_regional", "bleu_gemma4", "chrf_regional", "chrf_gemma4",
                                      "bertscore_f1_regional", "bertscore_f1_gemma4",
                                      "lang_adherence_regional", "lang_adherence_gemma4",
                                      "format_compliance_regional", "format_compliance_gemma4",
                                      "length_accuracy_regional", "length_accuracy_gemma4",
                                      "tone_register_regional", "tone_register_gemma4"]:
                            if field in parsed and parsed[field] is not None:
                                ev[field] = parsed[field]
                        ev["status"] = "complete"
                        break

    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    data["evaluations"] = evaluations
    QUAL_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    cards_html = render_qual_cards(evaluations)
    return JSONResponse({
        "last_updated": data["last_updated"],
        "html": cards_html,
    })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Falcon Dashboard → http://localhost:{PORT}")
    uvicorn.run("dashboard:app", host="0.0.0.0", port=PORT, reload=True, app_dir=str(REPO_ROOT / "scripts"))
