"""
Interactive world map: best tokenizer per language.
Modern dark-theme UI with sidebar detail panel, region filters, and metric bars.

Output: docs/language-map.html
Usage:  python experiments/language_map.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "data" / "results.csv"
OUT         = ROOT / "docs" / "language-map.html"

# ── Language → ISO-3166-1 alpha-3 primary countries ──────────────────────────
LANG_COUNTRIES = {
    "Hindi":                ["IND"],
    "Bengali":              ["BGD"],
    "Tamil":                ["IND", "LKA", "SGP"],
    "Telugu":               ["IND"],
    "Kannada":              ["IND"],
    "Malayalam":            ["IND"],
    "Marathi":              ["IND"],
    "Gujarati":             ["IND"],
    "Punjabi":              ["PAK"],
    "Odia":                 ["IND"],
    "Assamese":             ["IND"],
    "Urdu":                 ["PAK"],
    "Nepali":               ["NPL"],
    "Sinhala":              ["LKA"],
    "Maithili":             ["IND"],
    "Arabic":               ["SAU","EGY","DZA","SDN","IRQ","MAR","YEM",
                             "SYR","TUN","JOR","LBY","LBN","ARE","OMN",
                             "KWT","QAT","BHR","MRT"],
    "Persian":              ["IRN","AFG"],
    "Turkish":              ["TUR"],
    "Hebrew":               ["ISR"],
    "Kurdish":              ["IRQ"],
    "Azerbaijani":          ["AZE"],
    "Uzbek":                ["UZB"],
    "Kazakh":               ["KAZ"],
    "Chinese":              ["CHN","TWN"],
    "Japanese":             ["JPN"],
    "Korean":               ["KOR"],
    "Vietnamese":           ["VNM"],
    "Thai":                 ["THA"],
    "Indonesian":           ["IDN"],
    "Malay":                ["MYS","BRN"],
    "Tagalog":              ["PHL"],
    "Burmese":              ["MMR"],
    "Khmer":                ["KHM"],
    "Swahili":              ["KEN","TZA","UGA","RWA"],
    "Amharic":              ["ETH"],
    "Hausa":                ["NGA","NER"],
    "Yoruba":               ["NGA"],
    "Igbo":                 ["NGA"],
    "Zulu":                 ["ZAF"],
    "Xhosa":                ["ZAF"],
    "Somali":               ["SOM"],
    "Wolof":                ["SEN"],
    "Shona":                ["ZWE"],
    "French":               ["FRA","BEL","LUX","CAN","CHE"],
    "German":               ["DEU","AUT"],
    "Spanish":              ["ESP"],
    "Portuguese":           ["PRT","AGO","MOZ"],
    "Italian":              ["ITA"],
    "Dutch":                ["NLD"],
    "Polish":               ["POL"],
    "Russian":              ["RUS","BLR"],
    "Ukrainian":            ["UKR"],
    "Romanian":             ["ROU","MDA"],
    "Swedish":              ["SWE"],
    "Czech":                ["CZE"],
    "Greek":                ["GRC","CYP"],
    "Lat.Am. Spanish":      ["MEX","COL","ARG","VEN","CHL","ECU","GTM",
                             "CUB","BOL","DOM","HND","PRY","SLV","NIC",
                             "CRI","PAN","URY"],
    "Brazilian Portuguese": ["BRA"],
    "Quechua":              ["PER"],
    "Haitian Creole":       ["HTI"],
    "Māori":                ["NZL"],
    "Samoan":               ["WSM"],
    "Tok Pisin":            ["PNG"],
}

ISO_NAMES = {
    "IND":"India","PAK":"Pakistan","BGD":"Bangladesh","NPL":"Nepal",
    "LKA":"Sri Lanka","SGP":"Singapore","IRN":"Iran","AFG":"Afghanistan",
    "SAU":"Saudi Arabia","EGY":"Egypt","DZA":"Algeria","SDN":"Sudan",
    "IRQ":"Iraq","MAR":"Morocco","YEM":"Yemen","SYR":"Syria","TUN":"Tunisia",
    "JOR":"Jordan","LBY":"Libya","LBN":"Lebanon","ARE":"UAE","OMN":"Oman",
    "KWT":"Kuwait","QAT":"Qatar","BHR":"Bahrain","MRT":"Mauritania",
    "TUR":"Turkey","ISR":"Israel","AZE":"Azerbaijan","UZB":"Uzbekistan",
    "KAZ":"Kazakhstan","CHN":"China","TWN":"Taiwan","JPN":"Japan",
    "KOR":"South Korea","VNM":"Vietnam","THA":"Thailand","IDN":"Indonesia",
    "MYS":"Malaysia","BRN":"Brunei","PHL":"Philippines","MMR":"Myanmar",
    "KHM":"Cambodia","KEN":"Kenya","TZA":"Tanzania","UGA":"Uganda",
    "RWA":"Rwanda","ETH":"Ethiopia","NGA":"Nigeria","NER":"Niger",
    "ZAF":"South Africa","SOM":"Somalia","SEN":"Senegal","ZWE":"Zimbabwe",
    "FRA":"France","BEL":"Belgium","LUX":"Luxembourg","CAN":"Canada",
    "CHE":"Switzerland","DEU":"Germany","AUT":"Austria","ESP":"Spain",
    "PRT":"Portugal","AGO":"Angola","MOZ":"Mozambique","ITA":"Italy",
    "NLD":"Netherlands","POL":"Poland","RUS":"Russia","BLR":"Belarus",
    "UKR":"Ukraine","ROU":"Romania","MDA":"Moldova","SWE":"Sweden",
    "CZE":"Czech Republic","GRC":"Greece","CYP":"Cyprus","MEX":"Mexico",
    "COL":"Colombia","ARG":"Argentina","VEN":"Venezuela","CHL":"Chile",
    "ECU":"Ecuador","GTM":"Guatemala","CUB":"Cuba","BOL":"Bolivia",
    "DOM":"Dominican Rep.","HND":"Honduras","PRY":"Paraguay","SLV":"El Salvador",
    "NIC":"Nicaragua","CRI":"Costa Rica","PAN":"Panama","URY":"Uruguay",
    "BRA":"Brazil","PER":"Peru","HTI":"Haiti","NZL":"New Zealand",
    "WSM":"Samoa","PNG":"Papua New Guinea",
}

MODEL_COLORS = {
    "Gemma-4":             "#4A90D9",
    "No Candidate Tested": "#3D3D50",
    "Multiple Winners":    "#F5A623",
    "Tamil-Mistral-7B":   "#FF6B35",
    "MahaMarathi-7B":     "#E63946",
    "Ambari-7B":          "#FF9F1C",
    "Gujju-Llama-7B":     "#F4A261",
    "Jais-2-8B":          "#9B5DE5",
    "DictaLM-2.0-7B":     "#C77DFF",
    "Polyglot-Ko-12B":    "#00B4D8",
    "MaLLaM-5B":          "#0096C7",
    "Swahili-Gemma-7B":   "#52B788",
    "Walia-LLM-7B":       "#2D6A4F",
    "Lucie-7B":           "#3D5A80",
    "Viking-7B":          "#98C1D9",
    "CSMPT-7B":           "#A8DADC",
    "Meltemi-7B":         "#457B9D",
    "Tucano-2b4":         "#E76F51",
    "Goldfish-mri-39M":   "#2EC4B6",
    "Goldfish-tpi-125M":  "#80CED7",
}

LANG_REGION = {
    lang: entry["region"]
    for entry in [
        {"region":"Indic","langs":["Hindi","Bengali","Tamil","Telugu","Kannada","Malayalam","Marathi","Gujarati","Punjabi","Odia","Assamese","Urdu","Nepali","Sinhala","Maithili"]},
        {"region":"Middle East","langs":["Arabic","Persian","Turkish","Hebrew","Kurdish","Azerbaijani","Uzbek","Kazakh"]},
        {"region":"East Asia","langs":["Chinese","Japanese","Korean"]},
        {"region":"SEA","langs":["Vietnamese","Thai","Indonesian","Malay","Tagalog","Burmese","Khmer"]},
        {"region":"Africa","langs":["Swahili","Amharic","Hausa","Yoruba","Igbo","Zulu","Xhosa","Somali","Wolof","Shona"]},
        {"region":"Europe","langs":["French","German","Spanish","Portuguese","Italian","Dutch","Polish","Russian","Ukrainian","Romanian","Swedish","Czech","Greek"]},
        {"region":"Americas","langs":["Lat.Am. Spanish","Brazilian Portuguese","Quechua","Haitian Creole"]},
        {"region":"Oceania","langs":["Māori","Samoan","Tok Pisin"]},
    ]
    for lang in entry["langs"]
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_results():
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_per_language(rows):
    g4   = {r["language"]: r for r in rows if r["tokenizer_name"] == "Gemma-4"}
    cand = {r["language"]: r for r in rows
            if r["tokenizer_name"] not in ("Gemma-4", "BLOOM", "mT5")}
    out = {}
    for lang, gr in g4.items():
        cr  = cand.get(lang)
        gf  = float(gr["fertility"])
        gvc = float(gr["vocab_coverage"])
        if cr is None:
            out[lang] = {"best":"No Candidate Tested","candidate":"—",
                         "g4_fertility":gf,"g4_vcov":gvc,
                         "cand_fertility":None,"vcov":None,"rt":None,"bfr":None}
        else:
            cf  = float(cr["fertility"])
            cv  = float(cr["vocab_coverage"])
            rt  = float(cr["roundtrip_pass_rate"])
            bfr = float(cr["byte_fallback_rate"])
            wins = cf < gf and cv >= 80 and rt >= 95
            out[lang] = {
                "best":          cr["tokenizer_name"] if wins else "Gemma-4",
                "candidate":     cr["tokenizer_name"],
                "g4_fertility":  gf, "g4_vcov": gvc,
                "cand_fertility":cf, "vcov":cv, "rt":rt, "bfr":bfr,
            }
    return out


def build_country_data(lang_best):
    country_langs = defaultdict(list)
    for lang, info in lang_best.items():
        for iso3 in LANG_COUNTRIES.get(lang, []):
            country_langs[iso3].append((lang, info))

    records = []
    for iso3, entries in country_langs.items():
        winners = [info["best"] for _, info in entries
                   if info["best"] not in ("Gemma-4","No Candidate Tested")]
        unique_w = list(dict.fromkeys(winners))

        if len(unique_w) > 1:
            display = "Multiple Winners"
        elif len(unique_w) == 1:
            display = unique_w[0]
        elif any(info["best"] == "Gemma-4" for _, info in entries):
            display = "Gemma-4"
        else:
            display = "No Candidate Tested"

        # Primary region (from first language)
        primary_region = LANG_REGION.get(entries[0][0], "Other") if entries else "Other"

        lang_details = []
        for lang, info in entries:
            lang_details.append({
                "language":      lang,
                "best":          info["best"],
                "candidate":     info["candidate"],
                "g4_fertility":  info["g4_fertility"],
                "g4_vcov":       info["g4_vcov"],
                "cand_fertility":info["cand_fertility"],
                "vcov":          info["vcov"],
                "rt":            info["rt"],
                "bfr":           info["bfr"],
            })

        records.append({
            "iso3":         iso3,
            "country_name": ISO_NAMES.get(iso3, iso3),
            "best_model":   display,
            "region":       primary_region,
            "lang_details": lang_details,
        })

    return records


# ── HTML generation ───────────────────────────────────────────────────────────

def make_colorscale(model_list, dim="#1E2030"):
    n   = len(model_list) + 1      # +1 for dimmed slot at z=-1
    eps = 1e-4
    cs  = []
    for i in range(n):
        lo = i / n
        hi = (i + 1) / n - eps
        color = dim if i == 0 else (MODEL_COLORS.get(model_list[i-1], "#888"))
        cs.append([lo,  color])
        cs.append([hi,  color])
    cs[-1][0] = 1.0
    return cs


def generate_html(country_records):
    # ── Stats ──
    all_langs       = {ld["language"] for c in country_records for ld in c["lang_details"]}
    winner_langs    = {ld["language"] for c in country_records for ld in c["lang_details"]
                       if ld["best"] not in ("Gemma-4","No Candidate Tested","—")}
    total_countries = len(country_records)

    # ── Ordered model list (determines z indices) ──
    model_order = ["Gemma-4","No Candidate Tested","Multiple Winners"] + [
        m for m in MODEL_COLORS if m not in ("Gemma-4","No Candidate Tested","Multiple Winners")
    ]
    # Only keep models that actually appear
    present = {c["best_model"] for c in country_records}
    model_list = [m for m in model_order if m in present]

    model_idx   = {m: i for i, m in enumerate(model_list)}
    colorscale  = make_colorscale(model_list)

    # Add idx and z to each record
    for i, c in enumerate(country_records):
        c["idx"] = i
        c["z"]   = model_idx.get(c["best_model"], 0)

    # ── Legend HTML ──
    legend_items = "".join(
        f'<div class="legend-item">'
        f'<div class="legend-dot" style="background:{MODEL_COLORS.get(m,"#888")}"></div>'
        f'<span>{m}</span></div>'
        for m in model_list
    )

    countries_json  = json.dumps(country_records, ensure_ascii=False)
    colorscale_json = json.dumps(colorscale)
    colors_json     = json.dumps(MODEL_COLORS)
    model_idx_json  = json.dumps(model_idx)
    zmin            = -1
    zmax            = len(model_list) - 1

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Falcon Language Map</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-3.0.0.min.js" charset="utf-8"></script>
<style>
:root {{
  --bg:        #0D1117;
  --surface:   #161B22;
  --surface2:  #21262D;
  --border:    #30363D;
  --text:      #C9D1D9;
  --muted:     #8B949E;
  --accent:    #58A6FF;
  --green:     #3FB950;
  --yellow:    #D29922;
  --radius:    10px;
  --sb-w:      360px;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Inter', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}}

/* ── Header ── */
header {{
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 0 20px;
  height: 52px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
  z-index: 10;
}}
.logo {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  font-size: 14px;
  color: white;
  text-decoration: none;
}}
.logo-icon {{
  width: 26px; height: 26px;
  background: linear-gradient(135deg, #58A6FF 0%, #7C6AF7 100%);
  border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px;
}}
.header-sep {{ color: var(--border); font-size: 18px; }}
.header-sub {{ color: var(--muted); font-size: 12px; font-weight: 400; }}
.hspacer {{ flex: 1; }}
.header-chip {{
  font-size: 11px; color: var(--muted);
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 3px 10px;
}}

/* ── Stats bar ── */
.stats-bar {{
  display: flex;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}}
.stat {{
  flex: 1;
  padding: 10px 18px;
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 2px;
}}
.stat:last-child {{ border-right: none; }}
.stat-val {{
  font-size: 20px;
  font-weight: 700;
  color: white;
  line-height: 1;
}}
.stat-lbl {{
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}

/* ── Main layout ── */
.main {{
  display: flex;
  flex: 1;
  overflow: hidden;
}}

/* ── Map column ── */
.map-col {{
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-width: 0;
}}

/* ── Filter bar ── */
.filter-bar {{
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
  flex-wrap: wrap;
}}
.filter-lbl {{
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding-right: 4px;
}}
.pill {{
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--muted);
  transition: all 0.15s ease;
}}
.pill:hover {{ border-color: var(--accent); color: var(--accent); }}
.pill.active {{
  background: var(--accent);
  border-color: var(--accent);
  color: white;
}}

/* ── Map container ── */
#map {{
  flex: 1;
  width: 100%;
  min-height: 0;
}}

/* ── Legend ── */
.legend-bar {{
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 14px;
  background: var(--surface);
  border-top: 1px solid var(--border);
  flex-shrink: 0;
  flex-wrap: wrap;
  overflow: hidden;
}}
.legend-ttl {{
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-right: 4px;
  white-space: nowrap;
}}
.legend-chip {{
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--text);
  padding: 3px 8px;
  border-radius: 5px;
  background: var(--surface2);
  border: 1px solid var(--border);
  white-space: nowrap;
  cursor: pointer;
  transition: border-color 0.15s;
}}
.legend-chip:hover {{ border-color: var(--accent); }}
.legend-dot {{
  width: 9px; height: 9px;
  border-radius: 2px;
  flex-shrink: 0;
}}

/* ── Sidebar ── */
.sidebar {{
  width: var(--sb-w);
  flex-shrink: 0;
  background: var(--surface);
  border-left: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  transition: width 0.22s cubic-bezier(.4,0,.2,1),
              opacity 0.22s ease;
}}
.sidebar.hidden {{
  width: 0;
  opacity: 0;
  border-left: none;
  pointer-events: none;
}}
.sb-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px 12px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}}
.sb-title {{ font-size: 15px; font-weight: 600; color: white; }}
.sb-close {{
  width: 24px; height: 24px;
  border-radius: 6px;
  background: var(--surface2);
  border: 1px solid var(--border);
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px;
  color: var(--muted);
  transition: all 0.15s;
  flex-shrink: 0;
}}
.sb-close:hover {{ color: white; border-color: var(--muted); }}
.sb-body {{
  flex: 1;
  overflow-y: auto;
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}}
.sb-empty {{
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--muted);
  text-align: center;
  padding: 20px;
}}
.sb-empty-icon {{ font-size: 36px; opacity: 0.35; }}
.sb-empty p {{ font-size: 13px; line-height: 1.6; }}

/* ── Language card ── */
.lc {{
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  transition: border-color 0.15s;
}}
.lc:hover {{ border-color: #444D57; }}
.lc-top {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 4px;
}}
.lc-name {{ font-size: 13px; font-weight: 600; color: white; }}
.badge {{
  font-size: 10px; font-weight: 500;
  padding: 2px 7px;
  border-radius: 10px;
}}
.badge-win  {{ background:rgba(63,185,80,.15); color:#3FB950; border:1px solid rgba(63,185,80,.25); }}
.badge-g4   {{ background:rgba(88,166,255,.12); color:#58A6FF; border:1px solid rgba(88,166,255,.2); }}
.badge-none {{ background:rgba(139,148,158,.1); color:var(--muted); border:1px solid var(--border); }}
.lc-model   {{ font-size: 11px; color: var(--muted); margin-bottom: 10px; }}
.metrics    {{ display: flex; flex-direction: column; gap: 7px; }}
.m-row      {{ display: flex; flex-direction: column; gap: 3px; }}
.m-labels   {{ display: flex; justify-content: space-between; font-size: 10px; color: var(--muted); }}
.m-labels span:last-child {{ color: var(--text); font-weight: 500; }}
.m-bar      {{ height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }}
.m-fill     {{ height: 100%; border-radius: 2px; transition: width .5s cubic-bezier(.4,0,.2,1); }}
.fill-g4    {{ background: #4A90D9; }}
.fill-cand  {{ background: #3FB950; }}
.fill-vcov  {{ background: #9B5DE5; }}
.fill-rt    {{ background: #2EC4B6; }}
.divider    {{ height: 1px; background: var(--border); margin: 2px 0; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

/* ── Responsive ── */
@media (max-width: 860px) {{
  .stats-bar {{ display: none; }}
  .sidebar {{
    position: absolute; right: 0; top: 52px; bottom: 0;
    z-index: 20;
    box-shadow: -8px 0 30px rgba(0,0,0,.6);
  }}
  .header-sub, .header-chip {{ display: none; }}
}}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🌍</div>
    Falcon Language Map
  </div>
  <span class="header-sep">/</span>
  <span class="header-sub">Tokenizer Evaluation · FLORES-200</span>
  <div class="hspacer"></div>
  <span class="header-chip">{len(all_langs)} languages · {total_countries} countries</span>
</header>

<div class="stats-bar">
  <div class="stat">
    <span class="stat-val">{len(all_langs)}</span>
    <span class="stat-lbl">Languages Evaluated</span>
  </div>
  <div class="stat">
    <span class="stat-val" style="color:var(--green)">{len(winner_langs)}</span>
    <span class="stat-lbl">Candidates Beat Gemma-4</span>
  </div>
  <div class="stat">
    <span class="stat-val">{total_countries}</span>
    <span class="stat-lbl">Countries Mapped</span>
  </div>
  <div class="stat">
    <span class="stat-val">3</span>
    <span class="stat-lbl">Baselines (G4 · BLOOM · mT5)</span>
  </div>
  <div class="stat">
    <span class="stat-val">~1 012</span>
    <span class="stat-lbl">Sentences per Language</span>
  </div>
</div>

<div class="main">
  <div class="map-col">
    <div class="filter-bar">
      <span class="filter-lbl">Region</span>
      <button class="pill active" onclick="setRegion('All',this)">All</button>
      <button class="pill" onclick="setRegion('Indic',this)">Indic</button>
      <button class="pill" onclick="setRegion('Middle East',this)">Middle East</button>
      <button class="pill" onclick="setRegion('East Asia',this)">East Asia</button>
      <button class="pill" onclick="setRegion('SEA',this)">SEA</button>
      <button class="pill" onclick="setRegion('Africa',this)">Africa</button>
      <button class="pill" onclick="setRegion('Europe',this)">Europe</button>
      <button class="pill" onclick="setRegion('Americas',this)">Americas</button>
      <button class="pill" onclick="setRegion('Oceania',this)">Oceania</button>
    </div>
    <div id="map"></div>
    <div class="legend-bar">
      <span class="legend-ttl">Legend</span>
      {legend_items}
    </div>
  </div>

  <div class="sidebar hidden" id="sb">
    <div class="sb-header">
      <span class="sb-title" id="sb-title">—</span>
      <div class="sb-close" onclick="closeSB()">✕</div>
    </div>
    <div class="sb-body" id="sb-body">
      <div class="sb-empty">
        <div class="sb-empty-icon">🗺️</div>
        <p>Click any country on the map<br>to see language details</p>
      </div>
    </div>
  </div>
</div>

<script>
const COUNTRIES   = {countries_json};
const MODEL_IDX   = {model_idx_json};
const MODEL_COLORS= {colors_json};
const COLORSCALE  = {colorscale_json};
const ZMIN = {zmin}, ZMAX = {zmax};

const REGION_VIEW = {{
  "All":         {{ lat: 15,  lon: 10,  scale: 0.85 }},
  "Indic":       {{ lat: 22,  lon: 80,  scale: 3.2  }},
  "Middle East": {{ lat: 27,  lon: 44,  scale: 3.0  }},
  "East Asia":   {{ lat: 36,  lon: 118, scale: 3.0  }},
  "SEA":         {{ lat:  8,  lon: 112, scale: 3.0  }},
  "Africa":      {{ lat:  2,  lon: 22,  scale: 2.4  }},
  "Europe":      {{ lat: 52,  lon: 14,  scale: 4.2  }},
  "Americas":    {{ lat:  5,  lon:-68,  scale: 1.9  }},
  "Oceania":     {{ lat:-25,  lon:145,  scale: 3.2  }},
}};

// ── Build initial z array ──────────────────────────────────────────────────
function zFor(region) {{
  return COUNTRIES.map(c =>
    (region === "All" || c.region === region) ? c.z : -1
  );
}}

const trace = {{
  type: "choropleth",
  locations: COUNTRIES.map(c => c.iso3),
  z: zFor("All"),
  text: COUNTRIES.map(c => c.country_name),
  customdata: COUNTRIES.map((c, i) => i),
  colorscale: COLORSCALE,
  zmin: ZMIN, zmax: ZMAX,
  showscale: false,
  hovertemplate: "<b>%{{text}}</b><extra></extra>",
  marker: {{ line: {{ color: "#0D1117", width: 0.6 }} }},
}};

const layout = {{
  paper_bgcolor: "#0D1117",
  margin: {{ l:0, r:0, t:0, b:0 }},
  geo: {{
    showframe: false,
    showcoastlines: true,
    coastlinecolor: "#30363D",
    showland: true,
    landcolor: "#21262D",
    showocean: true,
    oceancolor: "#141B2D",
    showlakes: true,
    lakecolor: "#141B2D",
    showcountries: true,
    countrycolor: "#30363D",
    bgcolor: "#0D1117",
    projection: {{ type: "natural earth" }},
  }},
  dragmode: "pan",
  uirevision: "map",
}};

const config = {{
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ["select2d","lasso2d","autoScale2d","toImage"],
  modeBarButtonsToAdd: [{{
    name: "Reset view",
    icon: Plotly.Icons.home,
    click: () => setRegion("All", document.querySelector(".pill.active")),
  }}],
  displaylogo: false,
  scrollZoom: true,
}};

Plotly.newPlot("map", [trace], layout, config);

// ── Click → sidebar ──────────────────────────────────────────────────────
document.getElementById("map").on("plotly_click", e => {{
  if (!e.points.length) return;
  const c = COUNTRIES[e.points[0].customdata];
  if (c) openSB(c);
}});

// ── Region filter ─────────────────────────────────────────────────────────
function setRegion(region, btn) {{
  if (btn) {{
    document.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
  }}
  Plotly.restyle("map", {{ z: [zFor(region)] }});
  const v = REGION_VIEW[region] || REGION_VIEW["All"];
  Plotly.relayout("map", {{
    "geo.center.lat": v.lat,
    "geo.center.lon": v.lon,
    "geo.projection.scale": v.scale,
  }});
}}

// ── Sidebar ───────────────────────────────────────────────────────────────
function openSB(country) {{
  document.getElementById("sb-title").textContent = country.country_name;
  document.getElementById("sb-body").innerHTML = buildCards(country);
  document.getElementById("sb").classList.remove("hidden");
}}
function closeSB() {{
  document.getElementById("sb").classList.add("hidden");
}}

function buildCards(country) {{
  if (!country.lang_details?.length)
    return `<div class="sb-empty"><p>No evaluation data for this country.</p></div>`;

  return country.lang_details.map(d => {{
    const isWin = d.best !== "Gemma-4" && d.best !== "No Candidate Tested";
    const [badgeClass, badgeTxt] = isWin
      ? ["badge-win",  "✅ Candidate wins"]
      : d.best === "Gemma-4"
        ? ["badge-g4",  "Gemma-4 best"]
        : ["badge-none","No candidate"];

    let metrics = "";
    if (d.cand_fertility !== null) {{
      const maxF   = Math.max(d.g4_fertility, d.cand_fertility, 0.01);
      const g4pct  = (d.g4_fertility   / maxF * 100).toFixed(0);
      const cpct   = (d.cand_fertility / maxF * 100).toFixed(0);
      const candLabel = d.candidate !== "—" ? d.candidate : "Candidate";
      metrics = `
        <div class="metrics">
          <div class="m-row">
            <div class="m-labels"><span>Fertility · Gemma-4</span><span>${{d.g4_fertility}}</span></div>
            <div class="m-bar"><div class="m-fill fill-g4"  style="width:${{g4pct}}%"></div></div>
          </div>
          <div class="m-row">
            <div class="m-labels"><span>Fertility · ${{candLabel}}</span><span>${{d.cand_fertility}}</span></div>
            <div class="m-bar"><div class="m-fill fill-cand" style="width:${{cpct}}%"></div></div>
          </div>
          <div class="divider"></div>
          <div class="m-row">
            <div class="m-labels"><span>Vocab Coverage</span><span>${{d.vcov}}%</span></div>
            <div class="m-bar"><div class="m-fill fill-vcov" style="width:${{d.vcov}}%"></div></div>
          </div>
          <div class="m-row">
            <div class="m-labels"><span>Roundtrip Fidelity</span><span>${{d.rt}}%</span></div>
            <div class="m-bar"><div class="m-fill fill-rt"   style="width:${{d.rt}}%"></div></div>
          </div>
        </div>`;
    }}

    return `
      <div class="lc">
        <div class="lc-top">
          <span class="lc-name">${{d.language}}</span>
          <span class="badge ${{badgeClass}}">${{badgeTxt}}</span>
        </div>
        <div class="lc-model">${{d.best}}</div>
        ${{metrics}}
      </div>`;
  }}).join("");
}}
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows            = load_results()
    lang_best       = best_per_language(rows)
    country_records = build_country_data(lang_best)
    html            = generate_html(country_records)

    OUT.write_text(html, encoding="utf-8")
    size_kb = OUT.stat().st_size // 1024
    print(f"Map written → {OUT}  ({size_kb} KB)")

    from collections import Counter
    counts = Counter(c["best_model"] for c in country_records)
    print("\nCountries per winner:")
    for m, n in counts.most_common():
        print(f"  {m:35s}  {n}")


if __name__ == "__main__":
    main()
