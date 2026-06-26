# Run: python scripts/dashboard.py
# Then open: http://localhost:8765
# Requirements: pip install fastapi uvicorn
"""
Falcon Language Support — Local Dashboard
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT    = Path(__file__).resolve().parent.parent
PORT         = 8765
MODAL_VOLUME = "phase2a-outputs"
QUAL_JSON    = REPO_ROOT / "data" / "qualitative_results.json"
CACHE_DIR          = REPO_ROOT / "data" / "modal_cache"
VIZ_DIR            = REPO_ROOT / "docs" / "viz"
STATIC_REVIEW_DIR  = REPO_ROOT / "data" / "review_static"

# On Vercel: filesystem is read-only except /tmp; Modal CLI is unavailable.
IS_VERCEL  = bool(os.environ.get("VERCEL"))
REVIEW_DIR = Path("/tmp/review") if IS_VERCEL else REPO_ROOT / "data" / "review"

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
STATIC_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/viz", StaticFiles(directory=str(VIZ_DIR)), name="viz")
app.mount("/static/review", StaticFiles(directory=str(REVIEW_DIR), html=False), name="review")
app.mount("/static/review_static", StaticFiles(directory=str(STATIC_REVIEW_DIR), html=False), name="review_static")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

CSS_VARS = """
:root {
  --bg:   #0D1117;
  --sf:   #161B22;
  --sf2:  #21262D;
  --bd:   #30363D;
  --tx:   #C9D1D9;
  --mu:   #8B949E;
  --ac:   #58A6FF;
  --gr:   #3FB950;
  --rd:   #F85149;
  --or:   #F5A623;
  --pu:   #A371F7;
  /* task accent colours */
  --inst: #A371F7;   /* purple — instructions */
  --tran: #58A6FF;   /* blue   — translation  */
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
code { font-family: "SF Mono","Fira Code",Menlo,monospace; }

/* ── CSS tooltip ── */
.tip { position: relative; cursor: help; }
.tip::after {
  content: attr(data-tip);
  position: absolute;
  bottom: calc(100% + 7px);
  left: 50%;
  transform: translateX(-50%);
  background: #1C2128;
  color: var(--tx);
  padding: 7px 11px;
  border-radius: 7px;
  border: 1px solid var(--bd);
  font-size: 11px;
  line-height: 1.5;
  width: 240px;
  white-space: normal;
  z-index: 500;
  opacity: 0;
  pointer-events: none;
  transition: opacity .15s;
  box-shadow: 0 6px 18px rgba(0,0,0,.55);
  text-align: left;
  font-weight: 400;
  font-style: normal;
}
.tip:hover::after { opacity: 1; }
"""

GRADE_COLORS = {
    "A": "#3FB950", "B": "#52B788", "C": "#F5A623",
    "D": "#E76F51", "E": "#F85149",
    "skipped": "#8B949E", "failed": "#8B949E", "pending": "#484F58",
}
GRADE_LABELS = {
    "A": "Grade A — Regional superior (>60% judge win rate)",
    "B": "Grade B — Regional preferred (50–60%)",
    "C": "Grade C — Comparable (40–50%)",
    "D": "Grade D — Gemma-4 preferred (20–40%)",
    "E": "Grade E — Gemma-4 strongly preferred (<20%)",
    "skipped": "Skipped — run failed or crashed mid-inference",
    "failed":  "Failed — infrastructure/pipeline error",
    "pending": "Pending — not yet evaluated",
}

REGION_ORDER = ["Indic","Middle East","East Asia","SEA","Africa","Europe","Americas","Oceania"]

# slug → (language, region, model_display_name)
# Registry slug is the source of truth — updated when a model is swapped.
SLUG_META: dict[str, tuple[str, str, str]] = {
    # Indic
    "tamil":                    ("Tamil",                "Indic",        "Tamil-Mistral-7B"),
    "sarvam-m-tamil":           ("Tamil",                "Indic",        "Sarvam-M-24B"),
    "marathi":                  ("Marathi",              "Indic",        "MahaMarathi-7B"),
    "sarvam-m-marathi":         ("Marathi",              "Indic",        "Sarvam-M-24B"),
    "kannada":                  ("Kannada",              "Indic",        "Ambari-7B"),
    "sarvam-m-kannada":         ("Kannada",              "Indic",        "Sarvam-M-24B"),
    "gujarati":                 ("Gujarati",             "Indic",        "Gujju-Llama-7B"),
    "sarvam-m-gujarati":        ("Gujarati",             "Indic",        "Sarvam-M-24B"),
    # Middle East
    "arabic":                   ("Arabic",               "Middle East",  "Jais-2-8B"),
    "jais-70b":                 ("Arabic",               "Middle East",  "Jais-2-70B-Chat"),
    "hebrew":                   ("Hebrew",               "Middle East",  "DictaLM-3.0-Nemotron-12B"),
    # East Asia
    "korean":                   ("Korean",               "East Asia",    "Polyglot-Ko-12B"),
    "exaone-korean":            ("Korean",               "East Asia",    "EXAONE-3.5-32B-Instruct"),
    # SEA
    "malay":                    ("Malay",                "SEA",          "MaLLaM-5B"),
    # Africa
    "swahili":                  ("Swahili",              "Africa",       "Swahili-Gemma-7B"),
    "amharic":                  ("Amharic",              "Africa",       "Walia-LLM-7B"),
    # Europe
    "greek":                    ("Greek",                "Europe",       "Meltemi-7B"),
    "krikri-greek":             ("Greek",                "Europe",       "Krikri-8B-Instruct"),
    "french":                   ("French",               "Europe",       "Lucie-7B-Instruct-v1.1"),
    "swedish":                  ("Swedish",              "Europe",       "Viking-7B"),
    "czech":                    ("Czech",                "Europe",       "CSMPT-7B"),
    # Oceania
    "maori":                    ("Māori",                "Oceania",      "Goldfish-mri-39M"),
    "tok_pisin":                ("Tok Pisin",            "Oceania",      "Goldfish-tpi-125M"),
    # Americas
    "brazilian_portuguese":     ("Brazilian Portuguese", "Americas",     "Tucano-2b4"),
}

# Metric tooltips shown on hover
METRIC_TIPS = {
    "Judge Win Rate":       "Primary signal. % of 50 judged prompts where the regional model was preferred by Gemini (LLM-as-judge). Each prompt is judged twice (positions swapped) across 3 quality dimensions → 300 verdicts total.",
    "BLEU":                 "Word n-gram overlap with a reference translation (0–100). Higher = more literal word-level match with ground truth. Fast proxy but sensitive to exact phrasing.",
    "chrF":                 "Character-level F-score (0–100). Compares character n-grams instead of words — more robust for morphologically rich languages like Tamil, Arabic, and Hebrew.",
    "Language Adherence":   "% of responses generated in the correct target language. A model that replies in English or mixes languages scores 0%.",
    "Format Compliance":    "% of responses that follow structural constraints specified in the prompt (e.g. bullet points, JSON output, numbered list).",
    "Length Accuracy":      "% of responses within the expected length range defined by the instruction (e.g. 'respond in 2 sentences').",
    "Tone Register":        "% of responses matching the required formality or style (e.g. professional, casual, empathetic) as specified in the Talking Avatar prompt.",
}

# ---------------------------------------------------------------------------
# Nav bar
# ---------------------------------------------------------------------------

def nav_html(active: str) -> str:
    tabs = [
        ("Tokenizer",           "/tokenizer",    "tokenizer"),
        ("EU Expansion",        "/eu-tokenizer", "eu-tokenizer"),
        ("Qualitative Analysis","/qualitative",  "qualitative"),
        ("Manual Review (QA)",  "/review",       "review"),
    ]
    def tab_link(label, href, key):
        s = ("color:var(--tx);border-bottom:2px solid var(--ac);padding-bottom:2px;font-weight:600;"
             if key == active else "color:var(--mu);font-weight:500;")
        return f'<a href="{href}" style="text-decoration:none;font-size:13px;{s}">{label}</a>'
    links = "\n".join(tab_link(*t) for t in tabs)
    return f"""<nav style="height:46px;display:flex;align-items:center;gap:28px;
        padding:0 24px;background:var(--sf);border-bottom:1px solid var(--bd);
        position:sticky;top:0;z-index:100;flex-shrink:0;">
      <span style="font-size:15px;font-weight:700;color:var(--tx);letter-spacing:-.3px;">🌍 Falcon</span>
      <span style="color:var(--bd);">/</span>
      {links}
    </nav>"""

# ---------------------------------------------------------------------------
# Grade badge
# ---------------------------------------------------------------------------

def grade_badge(grade: str, size: str = "normal") -> str:
    color   = GRADE_COLORS.get(grade, "#484F58")
    tooltip = GRADE_LABELS.get(grade, grade)
    bg      = color + "22"
    fs, pad = ("13px","4px 12px") if size == "large" else ("11px","2px 9px")
    return (f'<span title="{tooltip}" style="display:inline-block;font-size:{fs};font-weight:700;'
            f'padding:{pad};border-radius:12px;background:{bg};color:{color};'
            f'border:1px solid {color}55;cursor:default;white-space:nowrap;">{grade.upper()}</span>')

# ---------------------------------------------------------------------------
# Metric bar — with hoverable label tooltip
# ---------------------------------------------------------------------------

def metric_bar(name: str, val_r, val_g, unit: str = "", scale: Optional[float] = None) -> str:
    tip = METRIC_TIPS.get(name, "")
    hint_span = (f'<span class="tip" data-tip="{tip}" style="font-size:10px;color:var(--mu);'
                 f'border-bottom:1px dotted var(--mu);cursor:help;">{name}</span>' if tip
                 else f'<span style="font-size:10px;color:var(--mu);">{name}</span>')

    def auto_scale():
        vals = [float(v) for v in [val_r, val_g] if v is not None]
        if not vals:
            return 100.0
        return max(vals) * 1.1 or 1.0

    sc = scale if scale is not None else auto_scale()

    def pct(v):
        if v is None: return 0.0
        return min(100.0, float(v) / sc * 100)

    def fmt(v):
        if v is None: return "—"
        if unit == "%": return f"{v:.0f}%"
        if float(v) == int(float(v)): return f"{int(v)}"
        return f"{v:.2f}"

    better = val_r is not None and val_g is not None and float(val_r) >= float(val_g)
    r_col = "var(--gr)" if better else "var(--or)"
    g_col = "var(--mu)"

    r_w = pct(val_r)
    g_w = pct(val_g)
    r_str = fmt(val_r)
    g_str = fmt(val_g)

    return f"""<div style="margin-bottom:9px;">
  <div style="margin-bottom:4px;">{hint_span}</div>
  <div style="display:flex;flex-direction:column;gap:3px;">
    <div style="display:flex;align-items:center;gap:7px;">
      <span style="font-size:10px;color:var(--mu);width:58px;flex-shrink:0;">Regional</span>
      <div style="flex:1;height:7px;background:var(--bd);border-radius:4px;overflow:hidden;">
        <div style="height:100%;width:{r_w:.1f}%;background:{r_col};border-radius:4px;"></div>
      </div>
      <span style="font-size:11px;font-weight:700;min-width:38px;text-align:right;color:{r_col};">{r_str}</span>
    </div>
    <div style="display:flex;align-items:center;gap:7px;">
      <span style="font-size:10px;color:var(--mu);width:58px;flex-shrink:0;">Gemma-4</span>
      <div style="flex:1;height:7px;background:var(--bd);border-radius:4px;overflow:hidden;">
        <div style="height:100%;width:{g_w:.1f}%;background:{g_col};opacity:.55;border-radius:4px;"></div>
      </div>
      <span style="font-size:11px;min-width:38px;text-align:right;color:var(--mu);">{g_str}</span>
    </div>
  </div>
</div>"""

# ---------------------------------------------------------------------------
# Task panel renderer
# ---------------------------------------------------------------------------

TASK_META = {
    "instructions": {
        "label": "INSTRUCTIONS",
        "color": "var(--inst)",
        "bg":    "rgba(163,113,247,.08)",
        "icon":  "✦",
    },
    "translation": {
        "label": "TRANSLATION",
        "color": "var(--tran)",
        "bg":    "rgba(88,166,255,.08)",
        "icon":  "⇄",
    },
}


def render_task_panel(ev: Optional[dict], task: str) -> str:
    m = TASK_META.get(task, TASK_META["translation"])
    color = m["color"]
    label = m["label"]
    icon  = m["icon"]

    if ev is None:
        return (f'<div style="padding:14px 16px;border-top:2px solid {color};">'
                f'<div style="font-size:10px;font-weight:700;color:{color};letter-spacing:.07em;margin-bottom:8px;">'
                f'{icon} {label}</div>'
                f'<div style="color:var(--mu);font-size:12px;font-style:italic;">No data</div></div>')

    grade = ev.get("grade", "pending")
    run_id = ev.get("run_id")
    notes  = ev.get("notes", "")

    if grade in ("skipped", "failed", "pending") and not run_id:
        badge = grade_badge(grade)
        nt = f'<div style="color:var(--mu);font-size:11px;margin-top:6px;font-style:italic;">{notes}</div>' if notes else ""
        return (f'<div style="padding:14px 16px;border-top:2px solid {color};">'
                f'<div style="font-size:10px;font-weight:700;color:{color};letter-spacing:.07em;margin-bottom:8px;">'
                f'{icon} {label}</div>'
                f'<div style="display:flex;align-items:center;gap:8px;">{badge}'
                f'<span style="font-size:12px;color:var(--mu);">{GRADE_LABELS.get(grade,"")}</span>'
                f'</div>{nt}</div>')

    win_r = ev.get("judge_win_rate")
    win_g = ev.get("gemma4_win_rate")

    parts = [
        f'<div style="padding:14px 16px;border-top:2px solid {color};">',
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">',
        f'<span style="font-size:10px;font-weight:700;color:{color};letter-spacing:.07em;">{icon} {label}</span>',
        grade_badge(grade),
        "</div>",
        metric_bar("Judge Win Rate", win_r, win_g, unit="%", scale=100.0),
    ]

    if task == "translation":
        bleu_r = ev.get("bleu_regional"); bleu_g = ev.get("bleu_gemma4")
        chrf_r = ev.get("chrf_regional"); chrf_g = ev.get("chrf_gemma4")
        if bleu_r is not None or bleu_g is not None:
            parts.append(metric_bar("BLEU", bleu_r, bleu_g))
        if chrf_r is not None or chrf_g is not None:
            parts.append(metric_bar("chrF", chrf_r, chrf_g))

    if task == "instructions":
        for name, key in [
            ("Language Adherence", "lang_adherence"),
            ("Format Compliance",  "format_compliance"),
            ("Length Accuracy",    "length_accuracy"),
            ("Tone Register",      "tone_register"),
        ]:
            r = ev.get(f"{key}_regional")
            g = ev.get(f"{key}_gemma4")
            if r is not None or g is not None:
                parts.append(metric_bar(name, r, g, unit="%", scale=100.0))

    if run_id:
        review_url = f"/review?run_id={run_id}"
        parts.append(
            f'<div style="display:flex;align-items:center;gap:8px;margin-top:12px;'
            f'padding:8px 10px;background:var(--bg);border:1px solid var(--bd);border-radius:7px;">'
            f'<code style="font-size:11px;font-weight:600;color:var(--tx);letter-spacing:.02em;flex:1;">{run_id}</code>'
            f'<a href="{review_url}" style="font-size:11px;color:{color};font-weight:600;white-space:nowrap;">Review ↗</a>'
            f'</div>'
        )

    if notes:
        parts.append(
            f'<div style="margin-top:9px;padding:7px 10px;background:rgba(248,81,73,.06);'
            f'border-left:3px solid rgba(248,81,73,.45);border-radius:0 5px 5px 0;'
            f'font-size:11px;color:var(--mu);font-style:italic;line-height:1.5;">{notes}</div>'
        )

    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Language card
# ---------------------------------------------------------------------------

def render_language_card(model: str, region: str, instr: Optional[dict], trans: Optional[dict]) -> str:
    ev = trans or instr
    grade = ev.get("grade", "pending") if ev else "pending"

    instr_html = render_task_panel(instr, "instructions")
    trans_html = render_task_panel(trans, "translation")

    return f"""<div style="background:var(--sf);border:1px solid var(--bd);border-radius:10px;margin-bottom:10px;overflow:hidden;">
  <div style="display:flex;align-items:center;gap:10px;padding:9px 14px;background:var(--sf2);border-bottom:1px solid var(--bd);">
    <span style="font-size:13px;font-weight:600;color:var(--tx);">{model}</span>
    <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
      {grade_badge(grade)}
      <span style="font-size:10px;padding:2px 8px;border-radius:4px;background:rgba(88,166,255,.1);color:var(--ac);border:1px solid rgba(88,166,255,.22);">{region}</span>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1px 1fr;">
    {instr_html}
    <div style="background:var(--bd);"></div>
    {trans_html}
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Stats bar + cards
# ---------------------------------------------------------------------------

def render_stats_bar(evaluations: list) -> str:
    seen = set()
    grade_counts: dict[str, int] = {}
    for ev in evaluations:
        if ev.get("task") == "translation":
            key = (ev.get("language"), ev.get("model"))
            if key not in seen:
                seen.add(key)
                g = ev.get("grade", "pending")
                grade_counts[g] = grade_counts.get(g, 0) + 1

    def chip(label, val, color):
        return (f'<div style="display:flex;align-items:center;gap:6px;font-size:12px;'
                f'padding:4px 10px;border-radius:20px;border:1px solid var(--bd);background:var(--sf2);">'
                f'<span style="width:8px;height:8px;border-radius:50%;background:{color};display:inline-block;"></span>'
                f'<span>{label}</span><span style="font-weight:700;color:{color};">{val}</span></div>')

    chips = [f'<span style="font-size:12px;color:var(--mu);">{len(seen)} model evaluations &nbsp;·&nbsp; {len(evaluations)} task runs</span>']
    for g in ("A","B","C","D","E","skipped","failed","pending"):
        cnt = grade_counts.get(g, 0)
        if cnt:
            chips.append(chip(f"Grade {g.upper()}", cnt, GRADE_COLORS[g]))

    return ('<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px;'
            'padding:14px 18px;background:var(--sf);border:1px solid var(--bd);border-radius:10px;align-items:center;">'
            + "".join(chips) + "</div>")


def render_qual_cards(evaluations: list) -> str:
    by_region: dict = {}
    for ev in evaluations:
        r = ev.get("region","Unknown"); l = ev.get("language","Unknown"); m = ev.get("model","Unknown")
        by_region.setdefault(r,{}).setdefault(l,{}).setdefault(m,[]).append(ev)

    parts = []
    ordered = [r for r in REGION_ORDER if r in by_region] + [r for r in by_region if r not in REGION_ORDER]
    for region in ordered:
        parts.append(f'<div class="region-section" data-region="{region}">')
        parts.append(f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
                     f'letter-spacing:.1em;color:var(--mu);padding:6px 0;'
                     f'border-bottom:1px solid var(--bd);margin-bottom:14px;">{region}</div>')
        for lang, model_map in by_region[region].items():
            parts.append(f'<div style="margin-bottom:20px;">'
                         f'<div style="font-size:16px;font-weight:700;color:var(--tx);margin-bottom:10px;">{lang}</div>')
            for model_name, evals in model_map.items():
                instr = next((e for e in evals if e.get("task") == "instructions"), None)
                trans = next((e for e in evals if e.get("task") == "translation"),  None)
                parts.append(render_language_card(model_name, region, instr, trans))
            parts.append("</div>")
        parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Qualitative page CSS (pills / filter row only — cards are inline styles)
# ---------------------------------------------------------------------------

QUAL_PILL_CSS = """
.region-section { margin-bottom:36px; }
.filter-row { display:flex;flex-wrap:wrap;gap:6px;margin-bottom:22px;align-items:center; }
.pill {
  font-size:12px;font-weight:500;padding:4px 14px;border-radius:20px;
  border:1px solid var(--bd);background:var(--sf2);color:var(--mu);
  cursor:pointer;transition:all .15s;user-select:none;
}
.pill:hover { border-color:var(--ac);color:var(--tx); }
.pill.active { background:var(--ac);color:#0D1117;border-color:var(--ac);font-weight:700; }
.refresh-btn {
  margin-left:auto;font-size:12px;padding:5px 14px;border-radius:8px;
  border:1px solid var(--bd);background:var(--sf2);color:var(--tx);cursor:pointer;
}
.refresh-btn:hover { border-color:var(--ac);color:var(--ac); }
"""


# ---------------------------------------------------------------------------
# Grade legend (always-visible, top of Qualitative Analysis page)
# ---------------------------------------------------------------------------

def _grade_chip(grade: str, label: str, sublabel: str) -> str:
    color = GRADE_COLORS[grade]
    bg    = color + "18"
    return (
        f'<div style="display:flex;align-items:center;gap:8px;padding:8px 14px;'
        f'border-right:1px solid var(--bd);flex-shrink:0;">'
        f'<span style="font-size:13px;font-weight:800;padding:3px 10px;border-radius:10px;'
        f'background:{bg};color:{color};border:1px solid {color}44;">{grade}</span>'
        f'<div style="line-height:1.3;">'
        f'<div style="font-size:12px;font-weight:600;color:var(--tx);">{label}</div>'
        f'<div style="font-size:10px;color:var(--mu);">{sublabel}</div>'
        f'</div></div>'
    )

_GRADE_CHIPS = "".join([
    _grade_chip("A", "Regional superior",   ">60% judge win rate"),
    _grade_chip("B", "Regional preferred",  "50–60% win rate"),
    _grade_chip("C", "Comparable",          "40–50% win rate"),
    _grade_chip("D", "Gemma-4 preferred",   "20–40% win rate"),
    _grade_chip("E", "Gemma-4 dominant",    "<20% win rate"),
])

GRADE_LEGEND_HTML = (
    '<div style="display:flex;align-items:stretch;background:var(--sf);border:1px solid var(--bd);'
    'border-radius:10px;overflow:hidden;margin-bottom:18px;flex-wrap:wrap;">'
    '<div style="padding:8px 14px;display:flex;align-items:center;border-right:1px solid var(--bd);flex-shrink:0;">'
    '<span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;'
    'color:var(--mu);white-space:nowrap;">Grade scale</span></div>'
    + _GRADE_CHIPS +
    '<div style="padding:8px 14px;display:flex;align-items:center;flex:1;min-width:180px;">'
    '<span style="font-size:11px;color:var(--mu);line-height:1.5;">'
    'Based on <strong style="color:var(--tx);">judge win rate</strong> — '
    'the % of head-to-head comparisons where a Gemini judge preferred the regional model '
    'over Gemma-4 26B. 50 prompts judged per run, each twice (positions swapped) across '
    '3 quality dimensions.</span></div></div>'
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/tokenizer")


@app.get("/eu-tokenizer", response_class=HTMLResponse)
def eu_tokenizer_page():
    EU_CSV  = REPO_ROOT / "data" / "european_results.csv"
    EU_JSON = REPO_ROOT / "data" / "european_summary.json"

    if not EU_CSV.exists():
        return HTMLResponse("<p style='color:#8B949E;padding:40px;font-family:sans-serif'>european_results.csv not found. Run experiments/european_tokenizer_test.py first.</p>")

    import csv as _csv
    rows = []
    with open(EU_CSV) as f:
        for r in _csv.DictReader(f):
            rows.append(r)

    g4 = {r["language"]: float(r["fertility"]) for r in rows if r["model"] == "Gemma-4"}
    MODELS = ["EuroLLM-22B", "Aya-Vision-32B", "Llama-3.3-70B", "SauerkrautLM-70B",
              "Mistral-Small-3.2", "Teuken-7B", "GEITje-7B", "TildeOpen-30B"]
    LANGS  = ["French","German","Spanish","Italian","Portuguese","Dutch","Polish","Romanian",
              "Ukrainian","Swedish","Czech","Greek","Russian","Danish","Finnish","Hungarian",
              "Turkish","Croatian","Slovak","Slovenian","Bulgarian","Lithuanian","Latvian",
              "Estonian","Irish","Norwegian","Maltese","Serbian","Icelandic","Albanian"]

    by_key = {(r["model"], r["language"]): r for r in rows if r["model"] != "Gemma-4"}

    def gate(model, lang):
        r = by_key.get((model, lang))
        if r is None: return None
        g4f = g4.get(lang)
        if g4f is None: return None
        return float(r["fertility"]) < g4f and float(r["vocab_coverage"]) >= 80 and float(r["roundtrip_pass_rate"]) >= 95

    def cell(model, lang):
        g = gate(model, lang)
        r = by_key.get((model, lang))
        if r is None:
            return '<td style="background:#161B22;color:#484f58;text-align:center;font-size:11px;">—</td>'
        g4f = g4.get(lang, "?")
        tip = f"fertility={r['fertility']} vs G4={g4f} | vcov={r['vocab_coverage']}% | rt={r['roundtrip_pass_rate']}%"
        if g is True:
            return f'<td title="{tip}" style="background:rgba(46,160,67,.18);color:#3fb950;text-align:center;font-size:12px;font-weight:700;">✓</td>'
        elif g is False:
            return f'<td title="{tip}" style="background:rgba(248,81,73,.08);color:#f85149;text-align:center;font-size:11px;">✗</td>'
        return f'<td title="{tip}" style="background:#161B22;color:#8b949e;text-align:center;font-size:11px;">?</td>'

    pass_counts = {m: sum(1 for l in LANGS if gate(m, l) is True) for m in MODELS}

    header_cells = "".join(
        f'<th style="padding:8px 10px;font-size:11px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.06em;color:#8b949e;white-space:nowrap;border-bottom:1px solid #30363d;">'
        f'{m}<br><span style="color:#3fb950;font-weight:800;font-size:13px;">{pass_counts[m]}</span>'
        f'<span style="color:#8b949e;font-size:10px;">/30</span></th>'
        for m in MODELS
    )

    body_rows = ""
    for lang in LANGS:
        g4f = g4.get(lang)
        g4_cell = f'<td style="padding:6px 10px;font-size:12px;color:#8b949e;text-align:right;">{g4f if g4f else "—"}</td>'
        data_cells = "".join(cell(m, lang) for m in MODELS)
        body_rows += (
            f'<tr><td style="padding:6px 12px;font-size:13px;font-weight:500;color:#c9d1d9;'
            f'white-space:nowrap;border-bottom:1px solid #21262d;">{lang}</td>'
            f'{g4_cell}{data_cells}</tr>'
        )

    summary_html = ""
    if EU_JSON.exists():
        summary = json.loads(EU_JSON.read_text())
        n_pairs = len(summary.get("qualitative_eval_queue", []))
        summary_html = (
            f'<div style="margin-bottom:20px;padding:14px 18px;background:#161B22;border:1px solid #30363d;'
            f'border-radius:8px;font-size:13px;color:#c9d1d9;">'
            f'<strong style="color:#3fb950">{n_pairs} (model × language) pairs</strong> proceed to qualitative eval '
            f'— EuroLLM-22B ({pass_counts.get("EuroLLM-22B",0)}) + Aya-Vision-32B ({pass_counts.get("Aya-Vision-32B",0)}) '
            f'+ 2 Czech-only pairs (Llama-3.3-70B, SauerkrautLM-70B).<br>'
            f'<span style="color:#8b949e;font-size:12px;">Hover a cell for fertility / vocab / roundtrip details. '
            f'Gemma-4 column shows baseline fertility per language.</span></div>'
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Falcon — EU Expansion</title>
<style>{GLOBAL_CSS}</style></head>
<body>
{nav_html("eu-tokenizer")}
<div style="max-width:1300px;margin:0 auto;padding:24px 20px;">
  <h2 style="font-size:18px;font-weight:700;color:#c9d1d9;margin:0 0 6px;">European Language Expansion — Tokenizer Gate</h2>
  <p style="font-size:13px;color:#8b949e;margin:0 0 20px;">
    8 challenger models × 30 European languages · FLORES-200 devtest ·
    Gate: <code style="color:#c9d1d9">fertility &lt; Gemma-4 AND vocab_coverage ≥ 80% AND roundtrip ≥ 95%</code>
  </p>
  {summary_html}
  <div style="overflow-x:auto;">
  <table style="border-collapse:collapse;width:100%;font-family:inherit;">
    <thead><tr>
      <th style="padding:8px 12px;font-size:11px;font-weight:700;text-align:left;color:#8b949e;border-bottom:1px solid #30363d;">Language</th>
      <th style="padding:8px 10px;font-size:11px;font-weight:700;color:#8b949e;border-bottom:1px solid #30363d;white-space:nowrap;">G4 Fertility</th>
      {header_cells}
    </tr></thead>
    <tbody>{body_rows}</tbody>
  </table>
  </div>
  <p style="margin-top:16px;font-size:11px;color:#484f58;">
    Eliminated: Mistral-Small-3.2 &amp; TildeOpen-30B (roundtrip 0% every language) ·
    Teuken-7B (vocab &lt;80% on 29/30) · GEITje-7B (fertility + vocab gaps).
    16 languages with no gate winner stay on Gemma-4.
  </p>
</div>
</body></html>"""


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
    data = json.loads(QUAL_JSON.read_text()) if QUAL_JSON.exists() else {"evaluations":[],"last_updated":None}
    evs  = data.get("evaluations", [])
    last = data.get("last_updated") or "unknown"

    pills = "\n".join(
        f'<span class="pill{"  active" if r=="All" else ""}" onclick="filterRegion(this,\'{r}\')">{r}</span>'
        for r in ["All"] + REGION_ORDER
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Falcon — Qualitative Analysis</title>
<style>{GLOBAL_CSS}{QUAL_PILL_CSS}</style></head>
<body>
{nav_html("qualitative")}
<div style="max-width:1200px;margin:0 auto;padding:24px 20px;">
  {render_stats_bar(evs)}
  {GRADE_LEGEND_HTML}
  <div class="filter-row">
    {pills}
    <button class="refresh-btn" id="refresh-btn" onclick="doRefresh()">↺ Refresh</button>
  </div>
  <div style="font-size:11px;color:var(--mu);margin-bottom:18px;" id="last-updated">Last updated: {last}</div>
  <div id="refresh-msg" style="font-size:11px;color:var(--ac);margin-bottom:10px;margin-top:-10px;display:none;"></div>
  <div id="cards-container">{render_qual_cards(evs)}</div>
</div>
<script>
var activeRegion="All";
function filterRegion(el,region){{
  activeRegion=region;
  document.querySelectorAll(".pill").forEach(p=>p.classList.remove("active"));
  el.classList.add("active");
  document.querySelectorAll(".region-section").forEach(function(s){{
    s.style.display=(region==="All"||s.dataset.region===region)?"":"none";
  }});
}}
function doRefresh(){{
  var btn=document.getElementById("refresh-btn");
  var msg=document.getElementById("refresh-msg");
  btn.textContent="↺ Scanning Modal…";btn.disabled=true;
  msg.style.display="none";
  fetch("/api/qualitative/refresh").then(r=>r.json()).then(function(d){{
    document.getElementById("last-updated").textContent="Last updated: "+(d.last_updated||"unknown");
    document.getElementById("cards-container").innerHTML=d.html;
    var active=document.querySelector(".pill.active");
    if(active)filterRegion(active,activeRegion);
    if(d.new_runs&&d.new_runs.length>0){{
      msg.textContent="✓ Added "+d.new_runs.length+" new run"+(d.new_runs.length>1?"s":"")+": "+d.new_runs.join(", ");
      msg.style.display="block";
      setTimeout(function(){{msg.style.display="none";}},8000);
    }}
  }}).catch(e=>alert("Refresh failed: "+e))
    .finally(()=>{{btn.textContent="↺ Refresh";btn.disabled=false;}});
}}
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# /review — Manual Review (QA) with loading progress bar
# ---------------------------------------------------------------------------

REVIEW_CSS = """
.review-layout { display:flex;height:calc(100vh - 46px);overflow:hidden; }
.review-sidebar {
  width:310px;flex-shrink:0;background:var(--sf);border-right:1px solid var(--bd);
  overflow-y:auto;display:flex;flex-direction:column;padding:20px;gap:14px;
}
.review-main { flex:1;display:flex;flex-direction:column;overflow:hidden;position:relative; }
.review-iframe { flex:1;border:none;width:100%;background:var(--bg);display:none; }
.slabel {
  font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;
  color:var(--mu);margin-bottom:4px;
}
.run-input {
  width:100%;background:var(--bg);border:1px solid var(--bd);color:var(--tx);
  font-size:12px;font-family:"SF Mono","Fira Code",Menlo,monospace;
  padding:9px 10px;border-radius:7px;outline:none;letter-spacing:.02em;
}
.run-input:focus{border-color:var(--ac);box-shadow:0 0 0 3px rgba(88,166,255,.12);}
.load-btn {
  width:100%;padding:9px;background:var(--ac);color:#0D1117;font-weight:700;
  font-size:13px;border:none;border-radius:7px;cursor:pointer;
}
.load-btn:disabled{opacity:.45;cursor:not-allowed;}
.meta-block{background:var(--sf2);border:1px solid var(--bd);border-radius:8px;padding:14px;display:none;}
.mrow{font-size:12px;margin-bottom:6px;}
.mkey{color:var(--mu);}
.mval{color:var(--tx);font-weight:600;}
.err-block{background:rgba(248,81,73,.08);border:1px solid rgba(248,81,73,.4);border-radius:7px;padding:10px 12px;font-size:12px;color:#F85149;display:none;line-height:1.5;}

/* loading overlay */
.loading-cover {
  position:absolute;inset:0;
  background:var(--bg);
  display:none;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  gap:18px;
  z-index:10;
}
.loading-cover.active { display:flex; }
.progress-track {
  width:420px;max-width:90%;
  height:6px;
  background:var(--bd);
  border-radius:6px;
  overflow:hidden;
}
.progress-fill {
  height:100%;
  width:0%;
  background:var(--ac);
  border-radius:6px;
  transition:width .35s ease, background .3s;
}
.progress-msg { font-size:13px;color:var(--mu);text-align:center;min-height:20px; }
.progress-pct { font-size:20px;font-weight:700;color:var(--tx);font-variant-numeric:tabular-nums; }

/* placeholder */
.placeholder {
  position:absolute;inset:0;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:10px;color:var(--mu);font-size:13px;
}
.placeholder.hidden{display:none;}
"""

REVIEW_JS = """
var progressTimer=null, progressVal=0;
var STAGES={0:"Connecting to Modal volume…",12:"Discovering run metadata…",28:"Downloading Gemma-4 baseline outputs…",52:"Downloading regional model outputs…",70:"Downloading judge verdicts…",85:"Generating review HTML…",93:"Almost ready…"};

function startProgress(){
  progressVal=0;
  setProgress(0);
  progressTimer=setInterval(function(){
    var step=Math.max(0.35,(95-progressVal)*0.045);
    progressVal=Math.min(95,progressVal+step);
    setProgress(progressVal);
  },350);
}

function finishProgress(ok){
  clearInterval(progressTimer);
  progressVal=100;
  var fill=document.getElementById("prog-fill");
  fill.style.background=ok?"var(--gr)":"var(--rd)";
  setProgress(100);
}

function setProgress(pct){
  document.getElementById("prog-fill").style.width=pct.toFixed(1)+"%";
  document.getElementById("prog-pct").textContent=Math.round(pct)+"%";
  var keys=Object.keys(STAGES).map(Number).sort((a,b)=>a-b);
  var msg="Loading…";
  for(var i=keys.length-1;i>=0;i--){if(pct>=keys[i]){msg=STAGES[keys[i]];break;}}
  document.getElementById("prog-msg").textContent=msg;
}

function gradeColor(g){
  return({A:"#3FB950",B:"#52B788",C:"#F5A623",D:"#E76F51",E:"#F85149"})[g.toUpperCase()]||"#8B949E";
}

function loadReview(){
  var runId=document.getElementById("run-id-input").value.trim();
  if(!runId){alert("Enter a run ID first.");return;}

  document.querySelector(".placeholder").classList.add("hidden");
  document.getElementById("review-iframe").style.display="none";
  document.getElementById("err-block").style.display="none";
  document.getElementById("meta-block").style.display="none";
  document.getElementById("m-grade-row").style.display="none";
  document.getElementById("m-wr-row").style.display="none";

  var cover=document.getElementById("loading-cover");
  cover.classList.add("active");
  document.getElementById("prog-fill").style.background="var(--ac)";
  document.getElementById("load-btn").disabled=true;
  startProgress();

  fetch("/api/review/load",{
    method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({run_id:runId})
  })
  .then(r=>r.json())
  .then(function(data){
    if(data.error){
      finishProgress(false);
      setTimeout(function(){
        cover.classList.remove("active");
        document.getElementById("err-block").textContent=data.error;
        document.getElementById("err-block").style.display="block";
        document.querySelector(".placeholder").classList.remove("hidden");
      },400);
      return;
    }
    finishProgress(true);
    setTimeout(function(){
      cover.classList.remove("active");
      var iframe=document.getElementById("review-iframe");
      iframe.src=data.url;
      iframe.style.display="block";
      document.getElementById("m-run-id").textContent=data.run_id;
      document.getElementById("m-slug").textContent=data.slug||"—";
      document.getElementById("m-task").textContent=data.task||"—";
      if(data.grade){
        document.getElementById("m-grade").innerHTML='<span style="font-weight:700;font-size:13px;color:'+gradeColor(data.grade)+'">'+data.grade.toUpperCase()+'</span>';
        document.getElementById("m-grade-row").style.display="";
      }
      if(data.win_rate!=null){
        document.getElementById("m-wr").textContent=data.win_rate+"%";
        document.getElementById("m-wr-row").style.display="";
      }
      document.getElementById("m-full-link").href=data.url;
      document.getElementById("meta-block").style.display="block";
    },500);
  })
  .catch(function(err){
    finishProgress(false);
    setTimeout(function(){
      cover.classList.remove("active");
      document.getElementById("err-block").textContent="Request failed: "+err;
      document.getElementById("err-block").style.display="block";
      document.querySelector(".placeholder").classList.remove("hidden");
    },400);
  })
  .finally(function(){document.getElementById("load-btn").disabled=false;});
}
"""


@app.get("/review", response_class=HTMLResponse)
def review_page(run_id: str = ""):
    prefill  = run_id or ""
    auto_load = "true" if prefill else "false"

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Falcon — Manual Review (QA)</title>
<style>{GLOBAL_CSS}{REVIEW_CSS}</style></head>
<body>
{nav_html("review")}
<div class="review-layout">

  <!-- Sidebar -->
  <div class="review-sidebar">
    <div>
      <div class="slabel">Run ID</div>
      <input id="run-id-input" class="run-input" type="text"
             placeholder="2026-06-25_094523_dace81" value="{prefill}">
    </div>
    <button class="load-btn" id="load-btn" onclick="loadReview()">Load Review</button>

    <div class="err-block" id="err-block"></div>

    <div class="meta-block" id="meta-block">
      <div class="slabel" style="margin-bottom:10px;">Run Info</div>
      <div class="mrow"><span class="mkey">Run ID<br></span>
        <code style="font-size:11px;font-weight:600;color:var(--tx);letter-spacing:.02em;" id="m-run-id"></code>
      </div>
      <div class="mrow"><span class="mkey">Slug </span><span class="mval" id="m-slug"></span></div>
      <div class="mrow"><span class="mkey">Task </span><span class="mval" id="m-task"></span></div>
      <div class="mrow" id="m-grade-row"><span class="mkey">Grade </span><span id="m-grade"></span></div>
      <div class="mrow" id="m-wr-row"><span class="mkey">Win rate </span><span class="mval" id="m-wr"></span></div>
      <div style="margin-top:12px;">
        <a id="m-full-link" href="#" target="_blank" style="font-size:12px;color:var(--ac);">Open full page ↗</a>
      </div>
    </div>
  </div>

  <!-- Main panel -->
  <div class="review-main">
    <div class="placeholder" id="placeholder">
      <span style="font-size:28px;">🔍</span>
      <span>Enter a run ID and click Load Review</span>
    </div>

    <!-- Loading cover with progress bar -->
    <div class="loading-cover" id="loading-cover">
      <span class="progress-pct" id="prog-pct">0%</span>
      <div class="progress-track">
        <div class="progress-fill" id="prog-fill"></div>
      </div>
      <div class="progress-msg" id="prog-msg">Connecting to Modal volume…</div>
    </div>

    <iframe class="review-iframe" id="review-iframe" src=""></iframe>
  </div>
</div>

<script>
{REVIEW_JS}
var autoLoad = {auto_load};
if (autoLoad) window.addEventListener("DOMContentLoaded", loadReview);
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# API: POST /api/review/load
# ---------------------------------------------------------------------------

def modal_get_file(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["modal","volume","get","--force", MODAL_VOLUME, remote, str(local)],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def modal_ls(path: str) -> str:
    r = subprocess.run(
        ["modal","volume","ls", MODAL_VOLUME, path],
        capture_output=True, text=True,
    )
    return r.stdout + r.stderr


def discover_slug_task(run_id: str) -> tuple[str, str]:
    for line in modal_ls(f"runs/{run_id}/judge/").splitlines():
        m = re.search(r'([^/\s]+)_(instructions|translation)_verdicts\.jsonl', line.strip())
        if m: return m.group(1), m.group(2)
    for line in modal_ls(f"runs/{run_id}/regional/").splitlines():
        m = re.search(r'([^/\s]+)_(instructions|translation)_outputs\.jsonl', line.strip())
        if m: return m.group(1), m.group(2)
    return "", ""


def load_generate_review():
    gr_path = REPO_ROOT / "scripts" / "generate_review.py"
    spec = importlib.util.spec_from_file_location("generate_review", gr_path)
    gr   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gr)
    return gr


def _review_html_path(slug: str, task: str, run_id: str) -> Path:
    return REVIEW_DIR / f"{slug}_{task}_{run_id}" / "review.html"


def _review_url(slug: str, task: str, run_id: str) -> str:
    return f"/static/review/{slug}_{task}_{run_id}/review.html"


def _grade_from_qual_json(run_id: str) -> tuple[Optional[str], Optional[float]]:
    try:
        for ev in json.loads(QUAL_JSON.read_text()).get("evaluations", []):
            if ev.get("run_id") == run_id:
                wr = ev.get("judge_win_rate")
                return ev.get("grade"), (round(wr, 1) if wr is not None else None)
    except Exception:
        pass
    return None, None


def _grade_from_cache_meta(run_id: str) -> tuple[Optional[str], Optional[float]]:
    meta_file = CACHE_DIR / run_id / "meta.json"
    try:
        m = json.loads(meta_file.read_text())
        return m.get("grade"), m.get("judge_win_rate")
    except Exception:
        return None, None


def _generate_review_html(run_id: str, slug: str, task: str,
                           gemma4_path: Path, regional_path: Path, verdicts_path: Path) -> str:
    gr               = load_generate_review()
    gemma4_records   = gr.load_jsonl(gemma4_path)
    regional_records = gr.load_jsonl(regional_path)
    verdict_records  = gr.load_jsonl(verdicts_path)
    aggregated       = gr.aggregate_verdicts(verdict_records)
    meta             = SLUG_META.get(slug)
    model_display    = meta[2] if meta else slug
    return gr.generate_html(
        run_id=run_id, slug=slug, task=task, model_display=model_display,
        regional_records=regional_records, gemma4_records=gemma4_records,
        aggregated=aggregated,
    )


@app.post("/api/review/load")
async def load_review_api(request: Request):
    body   = await request.json()
    run_id = (body.get("run_id") or "").strip()
    if not run_id:
        return JSONResponse({"error": "run_id is required"})

    # ── Step 0: check pre-generated static HTML (committed to git, works on Vercel) ──
    for folder in STATIC_REVIEW_DIR.iterdir() if STATIC_REVIEW_DIR.exists() else []:
        name = folder.name
        if run_id in name:
            for t in ("instructions", "translation"):
                marker = f"_{t}_{run_id}"
                if name.endswith(marker):
                    pre_slug = name[: -len(marker)]
                    pre_html = folder / "review.html"
                    if pre_html.exists():
                        grade, win_rate = _grade_from_qual_json(run_id)
                        if grade is None:
                            grade, win_rate = _grade_from_cache_meta(run_id)
                        return JSONResponse({
                            "url": f"/static/review_static/{name}/review.html",
                            "slug": pre_slug, "task": t, "run_id": run_id,
                            "grade": grade, "win_rate": win_rate,
                        })

    # ── Step 1: scan for known slug/task from cache meta OR existing HTML folder ──
    slug = task = ""
    cache_run_dir = CACHE_DIR / run_id
    meta_file     = cache_run_dir / "meta.json"

    if meta_file.exists():
        try:
            m    = json.loads(meta_file.read_text())
            slug = m.get("slug", "")
            task = m.get("task", "")
        except Exception:
            pass

    if not slug or not task:
        # Try parsing from existing review folder name
        for folder in REVIEW_DIR.iterdir() if REVIEW_DIR.exists() else []:
            name = folder.name
            if run_id in name:
                for t in ("instructions", "translation"):
                    marker = f"_{t}_{run_id}"
                    if name.endswith(marker):
                        slug = name[: -len(marker)]
                        task = t
                        break
            if slug:
                break

    # ── Step 2: if HTML already generated, return it immediately ──
    if slug and task:
        html_path = _review_html_path(slug, task, run_id)
        if html_path.exists():
            grade, win_rate = _grade_from_qual_json(run_id)
            if grade is None:
                grade, win_rate = _grade_from_cache_meta(run_id)
            return JSONResponse({
                "url": _review_url(slug, task, run_id),
                "slug": slug, "task": task, "run_id": run_id,
                "grade": grade, "win_rate": win_rate,
            })

    # ── Step 3: try generating from data/modal_cache/ ──
    if cache_run_dir.exists() and slug and task:
        verdicts_path  = cache_run_dir / "verdicts.jsonl"
        gemma4_path    = cache_run_dir / "gemma4.jsonl"
        regional_path  = cache_run_dir / "regional.jsonl"

        if verdicts_path.exists() and gemma4_path.exists() and regional_path.exists():
            try:
                gr = load_generate_review()
            except Exception as e:
                return JSONResponse({"error": f"Failed to load generate_review.py: {e}"})
            try:
                html_content = _generate_review_html(
                    run_id, slug, task, gemma4_path, regional_path, verdicts_path
                )
                out_file = _review_html_path(slug, task, run_id)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text(html_content, encoding="utf-8")
                grade, win_rate = _grade_from_qual_json(run_id)
                if grade is None:
                    grade, win_rate = _grade_from_cache_meta(run_id)
                return JSONResponse({
                    "url": _review_url(slug, task, run_id),
                    "slug": slug, "task": task, "run_id": run_id,
                    "grade": grade, "win_rate": win_rate,
                })
            except Exception as e:
                return JSONResponse({"error": f"Failed to generate review HTML: {e}"})

    # ── Step 4: Vercel — no Modal access, nothing to fall back on ──
    if IS_VERCEL:
        return JSONResponse({
            "error": (
                f"Run {run_id} is not pre-loaded in the hosted version. "
                "Run `python scripts/download_all_runs.py` locally, then redeploy."
            )
        })

    # ── Step 5: Local — download from Modal ──
    if not slug or not task:
        slug, task = discover_slug_task(run_id)
    if not slug or not task:
        return JSONResponse({"error": f"Could not discover slug/task for run ID '{run_id}'. Is it in phase2a-outputs?"})

    with tempfile.TemporaryDirectory(prefix="dash_review_") as tmp:
        tmp_path      = Path(tmp)
        gemma4_path   = tmp_path / "gemma4.jsonl"
        regional_path = tmp_path / "regional.jsonl"
        verdicts_path = tmp_path / "verdicts.jsonl"

        ok1 = modal_get_file(f"runs/{run_id}/gemma4/{task}_outputs.jsonl", gemma4_path)
        ok2 = modal_get_file(f"runs/{run_id}/regional/{slug}_{task}_outputs.jsonl", regional_path)
        ok3 = modal_get_file(f"runs/{run_id}/judge/{slug}_{task}_verdicts.jsonl", verdicts_path)

        if not (ok1 and ok2 and ok3):
            missing = [n for ok, n in [(ok1,"gemma4 outputs"),(ok2,"regional outputs"),(ok3,"judge verdicts")] if not ok]
            return JSONResponse({"error": f"Could not download: {', '.join(missing)}. Run ID: {run_id}"})

        try:
            gr = load_generate_review()
        except Exception as e:
            return JSONResponse({"error": f"Failed to load generate_review.py: {e}"})

        try:
            html_content = _generate_review_html(
                run_id, slug, task, gemma4_path, regional_path, verdicts_path
            )
        except Exception as e:
            return JSONResponse({"error": f"Failed to generate review HTML: {e}"})

        out_file = _review_html_path(slug, task, run_id)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(html_content, encoding="utf-8")

    grade, win_rate = _grade_from_qual_json(run_id)
    return JSONResponse({
        "url": _review_url(slug, task, run_id),
        "slug": slug, "task": task, "run_id": run_id,
        "grade": grade, "win_rate": win_rate,
    })


# ---------------------------------------------------------------------------
# API: GET /api/qualitative/refresh
# ---------------------------------------------------------------------------

_RUN_ID_RE = re.compile(r'\b(\d{4}-\d{2}-\d{2}_\d{6}_[a-f0-9]+)\b')


def list_modal_run_ids() -> list[str]:
    """Return all run_ids found in the phase2a-outputs/runs/ directory."""
    output = modal_ls("runs/")
    return list(dict.fromkeys(_RUN_ID_RE.findall(output)))  # preserve order, deduplicate


def compute_grade(win_rate_pct: float) -> str:
    if win_rate_pct >= 60: return "A"
    if win_rate_pct >= 50: return "B"
    if win_rate_pct >= 40: return "C"
    if win_rate_pct >= 20: return "D"
    return "E"


def compute_stats_from_verdicts(verdicts_path: Path) -> dict:
    """Read a verdicts JSONL, aggregate, return win_rate + grade."""
    try:
        gr = load_generate_review()
        records    = gr.load_jsonl(verdicts_path)
        aggregated = gr.aggregate_verdicts(records)
    except Exception:
        return {}

    regional = gemma4 = ties = no_verdict = 0
    for dims in aggregated.values():
        winner = gr.overall_winner(dims)
        if winner == "regional":   regional   += 1
        elif winner == "gemma4":   gemma4     += 1
        elif winner == "tie":      ties       += 1
        else:                      no_verdict += 1

    total_judged = regional + gemma4 + ties
    if total_judged == 0:
        return {}
    win_rate = round(regional / total_judged * 100, 1)
    gemma4_wr = round(gemma4 / total_judged * 100, 1)
    return {
        "judge_win_rate":  win_rate,
        "gemma4_win_rate": gemma4_wr,
        "grade":           compute_grade(win_rate),
    }


def parse_report_file(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    run_id = data.get("run_id"); task = data.get("task")
    if not run_id or not task:
        return None

    if "classification" in data:
        wr = data.get("judge_win_rate")
        win_rate = round(float(wr)*100,1) if wr is not None and float(wr)<=1.0 else (round(float(wr),1) if wr is not None else None)
        ev = {"run_id":run_id,"task":task,"grade":data.get("classification"),"judge_win_rate":win_rate}
        for k in ["bleu_regional","bleu_gemma4","chrf_regional","chrf_gemma4"]:
            if k in data: ev[k]=data[k]
        return ev

    results = data.get("results",{})
    if not results: return None
    slug = next(iter(results)); res = results[slug]
    wr_raw = res.get("judge_win_rate")
    win_rate = round(float(wr_raw)*100,1) if wr_raw is not None else None
    ev = {"run_id":run_id,"task":task,"grade":res.get("classification"),"judge_win_rate":win_rate}
    if task=="instructions":
        for k in ["lang_adherence_regional","lang_adherence_gemma4","format_compliance_regional",
                  "format_compliance_gemma4","length_accuracy_regional","length_accuracy_gemma4",
                  "tone_register_regional","tone_register_gemma4"]:
            v = res.get(k); ev[k] = round(v*100) if v is not None else None
    else:
        for k in ["bleu_regional","bleu_gemma4","chrf_regional","chrf_gemma4"]:
            ev[k] = res.get(k)
    return ev


def _merge_fields(target: dict, source: dict) -> None:
    METRIC_FIELDS = [
        "grade","judge_win_rate","gemma4_win_rate",
        "bleu_regional","bleu_gemma4","chrf_regional","chrf_gemma4",
        "lang_adherence_regional","lang_adherence_gemma4",
        "format_compliance_regional","format_compliance_gemma4",
        "length_accuracy_regional","length_accuracy_gemma4",
        "tone_register_regional","tone_register_gemma4",
    ]
    for f in METRIC_FIELDS:
        if f in source and source[f] is not None:
            target[f] = source[f]
    target["status"] = "complete"


def scan_modal_new_runs(existing_run_ids: set[str]) -> list[dict]:
    """
    Scan Modal volume for run_ids not yet in qualitative_results.json.
    For each new completed run, build an evaluation entry by:
      1. Discovering slug + task from the judge/ directory listing.
      2. Looking up language / region / model from SLUG_META.
      3. Downloading the verdicts file to compute grade + win rate.
      4. Trying the summary report for BLEU/chrF (optional).
    Returns a list of new evaluation dicts ready to append.
    """
    try:
        all_run_ids = list_modal_run_ids()
    except Exception:
        return []

    new_entries: list[dict] = []

    for run_id in all_run_ids:
        if run_id in existing_run_ids:
            continue

        # Discover slug + task — skip if judge dir not present (run still going)
        slug, task = discover_slug_task(run_id)
        if not slug or not task:
            continue

        # Look up metadata from SLUG_META
        meta = SLUG_META.get(slug)
        if meta:
            language, region, model = meta
        else:
            # Unknown slug — use slug itself, region unknown
            language = slug.replace("-", " ").replace("_", " ").title()
            region   = "Unknown"
            model    = slug

        with tempfile.TemporaryDirectory(prefix="scan_") as tmp:
            tmp_path      = Path(tmp)
            verdicts_path = tmp_path / "verdicts.jsonl"

            ok = modal_get_file(
                f"runs/{run_id}/judge/{slug}_{task}_verdicts.jsonl",
                verdicts_path,
            )
            if not ok:
                continue

            stats = compute_stats_from_verdicts(verdicts_path)
            if not stats:
                continue  # verdicts file present but empty / parse failed

            entry: dict = {
                "language": language,
                "region":   region,
                "model":    model,
                "task":     task,
                "run_id":   run_id,
                "status":   "complete",
                "notes":    "",
                **stats,
            }

            # Try to pull BLEU / chrF from summary report
            report_path = tmp_path / "report.json"
            if modal_get_file(f"runs/{run_id}/reports/{slug}_summary.json", report_path):
                try:
                    rdata = json.loads(report_path.read_text())
                    for k in ["bleu_regional","bleu_gemma4","chrf_regional","chrf_gemma4"]:
                        if k in rdata:
                            entry[k] = rdata[k]
                except Exception:
                    pass

        new_entries.append(entry)

    return new_entries


@app.get("/api/qualitative/refresh")
def qualitative_refresh():
    data        = json.loads(QUAL_JSON.read_text()) if QUAL_JSON.exists() else {"evaluations":[]}
    evaluations = data.get("evaluations", [])
    idx         = {ev.get("run_id"): ev for ev in evaluations if ev.get("run_id")}

    # 1. Update existing entries from local data/reports/ files
    reports_dir = REPO_ROOT / "data" / "reports"
    if reports_dir.exists():
        for f in sorted(reports_dir.glob("*.json")):
            parsed = parse_report_file(f)
            if not parsed or not parsed.get("run_id"): continue
            existing = idx.get(parsed["run_id"])
            if existing:
                _merge_fields(existing, parsed)

    # 2. Scan Modal for brand-new run_ids
    new_entries = scan_modal_new_runs(set(idx.keys()))
    new_run_labels: list[str] = []
    for entry in new_entries:
        evaluations.append(entry)
        idx[entry["run_id"]] = entry
        new_run_labels.append(f"{entry['language']} {entry['task']} ({entry['run_id'][:10]}…)")

    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    data["evaluations"]  = evaluations
    QUAL_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return JSONResponse({
        "last_updated": data["last_updated"],
        "html":         render_qual_cards(evaluations),
        "new_runs":     new_run_labels,
    })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n  Falcon Dashboard → http://localhost:{PORT}\n")
    uvicorn.run("dashboard:app", host="0.0.0.0", port=PORT, reload=True,
                app_dir=str(REPO_ROOT/"scripts"))
