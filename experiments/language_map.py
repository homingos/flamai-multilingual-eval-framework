"""
Interactive world map: best tokenizer per language.
v3 — dark theme, India state layer, full 7-metric sidebar, stats popovers,
     region filters, collapsible floating legend.

Output: docs/viz/language-map.html
Usage:  python experiments/language_map.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "data" / "results.csv"
OUT         = ROOT / "docs" / "viz" / "language-map.html"

# ── Language → ISO-3166-1 alpha-3 primary countries ──────────────────────────
LANG_COUNTRIES = {
    # English → Gemma-4 (injected as hardcoded winner — not tested since Gemma-4 is production)
    "English":              ["USA","GBR","AUS","IRL","GHA","ZMB","MWI","BWA",
                             "NAM","LSO","SWZ","JAM","TTO","GUY","BLZ","BHS",
                             "BRB","FJI","SLB","VUT","LCA","VCT","ATG","GRD",
                             "KNA","DMA","SYC"],
    "Hindi":                ["IND"],
    "Bengali":              ["BGD"],
    "Tamil":                ["IND","LKA","SGP"],
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
                             "KWT","QAT","BHR","MRT","COM"],
    "Persian":              ["IRN","AFG"],
    "Turkish":              ["TUR","AZE"],
    "Hebrew":               ["ISR"],
    "Kurdish":              ["IRQ"],
    "Azerbaijani":          ["AZE"],
    "Uzbek":                ["UZB","TJK"],
    "Kazakh":               ["KAZ"],
    "Chinese":              ["CHN","TWN","HKG","MAC"],
    "Japanese":             ["JPN"],
    "Korean":               ["KOR","PRK"],
    "Vietnamese":           ["VNM"],
    "Thai":                 ["THA"],
    "Indonesian":           ["IDN","TLS"],
    "Malay":                ["MYS","BRN","SGP"],
    "Tagalog":              ["PHL"],
    "Burmese":              ["MMR"],
    "Khmer":                ["KHM"],
    "Swahili":              ["KEN","TZA","UGA","RWA","BDI","COD"],
    "Amharic":              ["ETH"],
    "Hausa":                ["NGA","NER"],
    "Yoruba":               ["NGA"],
    "Igbo":                 ["NGA"],
    "Zulu":                 ["ZAF","LSO"],
    "Xhosa":                ["ZAF"],
    "Somali":               ["SOM","DJI","ETH"],
    "Wolof":                ["SEN","GMB"],
    "Shona":                ["ZWE"],
    "French":               ["FRA","BEL","LUX","CAN","CHE",
                             "CIV","CMR","MLI","BFA","COG","GAB",
                             "TGO","BEN","GIN","MDG","CAF","TCD",
                             "GNQ","RWA","BDI","DJI","MUS","REU",
                             "CPV","GNB","STP","HTI"],
    "German":               ["DEU","AUT","LIE"],
    "Spanish":              ["ESP"],
    "Portuguese":           ["PRT","AGO","MOZ","CPV","GNB","STP","TLS"],
    "Italian":              ["ITA","SMR","VAT"],
    "Dutch":                ["NLD","SUR","BEL"],
    "Polish":               ["POL"],
    "Russian":              ["RUS","BLR","KGZ"],
    "Ukrainian":            ["UKR"],
    "Romanian":             ["ROU","MDA"],
    "Swedish":              ["SWE","FIN"],
    "Czech":                ["CZE"],
    "Greek":                ["GRC","CYP"],
    "Lat.Am. Spanish":      ["MEX","COL","ARG","VEN","CHL","ECU","GTM",
                             "CUB","BOL","DOM","HND","PRY","SLV","NIC",
                             "CRI","PAN","URY","PRI"],
    "Brazilian Portuguese": ["BRA"],
    "Quechua":              ["PER"],
    "Haitian Creole":       ["HTI"],
    "Māori":                ["NZL"],
    "Samoan":               ["WSM","ASM"],
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
    "KOR":"South Korea","PRK":"North Korea","VNM":"Vietnam","THA":"Thailand",
    "IDN":"Indonesia","MYS":"Malaysia","BRN":"Brunei","PHL":"Philippines",
    "MMR":"Myanmar","KHM":"Cambodia","TLS":"Timor-Leste","KEN":"Kenya",
    "TZA":"Tanzania","UGA":"Uganda","RWA":"Rwanda","BDI":"Burundi",
    "ETH":"Ethiopia","NGA":"Nigeria","NER":"Niger","ZAF":"South Africa",
    "SOM":"Somalia","SEN":"Senegal","ZWE":"Zimbabwe","GMB":"Gambia",
    "FRA":"France","BEL":"Belgium","LUX":"Luxembourg","CAN":"Canada",
    "CHE":"Switzerland","DEU":"Germany","AUT":"Austria","LIE":"Liechtenstein",
    "ESP":"Spain","PRT":"Portugal","AGO":"Angola","MOZ":"Mozambique",
    "ITA":"Italy","SMR":"San Marino","VAT":"Vatican City",
    "NLD":"Netherlands","SUR":"Suriname","POL":"Poland",
    "RUS":"Russia","BLR":"Belarus","KGZ":"Kyrgyzstan","TJK":"Tajikistan",
    "UKR":"Ukraine","ROU":"Romania","MDA":"Moldova","SWE":"Sweden",
    "FIN":"Finland","CZE":"Czech Republic","GRC":"Greece","CYP":"Cyprus",
    "MEX":"Mexico","COL":"Colombia","ARG":"Argentina","VEN":"Venezuela",
    "CHL":"Chile","ECU":"Ecuador","GTM":"Guatemala","CUB":"Cuba",
    "BOL":"Bolivia","DOM":"Dominican Rep.","HND":"Honduras","PRY":"Paraguay",
    "SLV":"El Salvador","NIC":"Nicaragua","CRI":"Costa Rica","PAN":"Panama",
    "URY":"Uruguay","PRI":"Puerto Rico",
    "BRA":"Brazil","PER":"Peru","HTI":"Haiti","NZL":"New Zealand",
    "WSM":"Samoa","ASM":"American Samoa","PNG":"Papua New Guinea",
    # English-speaking
    "USA":"United States","GBR":"United Kingdom","AUS":"Australia",
    "IRL":"Ireland","GHA":"Ghana","ZMB":"Zambia","MWI":"Malawi",
    "BWA":"Botswana","NAM":"Namibia","LSO":"Lesotho","SWZ":"Eswatini",
    "JAM":"Jamaica","TTO":"Trinidad & Tobago","GUY":"Guyana","BLZ":"Belize",
    "BHS":"Bahamas","BRB":"Barbados","FJI":"Fiji","SLB":"Solomon Islands",
    "VUT":"Vanuatu","LCA":"Saint Lucia","VCT":"St. Vincent","ATG":"Antigua & Barbuda",
    "GRD":"Grenada","KNA":"St. Kitts & Nevis","DMA":"Dominica","SYC":"Seychelles",
    # Francophone Africa
    "CIV":"Ivory Coast","CMR":"Cameroon","MLI":"Mali","BFA":"Burkina Faso",
    "COD":"DR Congo","COG":"Congo","GAB":"Gabon","TGO":"Togo","BEN":"Benin",
    "GIN":"Guinea","MDG":"Madagascar","CAF":"Cent. African Rep.","TCD":"Chad",
    "GNQ":"Equatorial Guinea","MUS":"Mauritius","REU":"Réunion",
    "CPV":"Cape Verde","GNB":"Guinea-Bissau","STP":"São Tomé & Príncipe",
    # Other
    "HKG":"Hong Kong","MAC":"Macau","COM":"Comoros",
}

# Indian state → primary evaluated language (ST_NM keys from GeoJSON)
INDIA_STATE_LANG = {
    "Andhra Pradesh":   "Telugu",
    "Assam":            "Assamese",
    "Bihar":            "Hindi",
    "Chhattisgarh":     "Hindi",
    "Gujarat":          "Gujarati",
    "Haryana":          "Hindi",
    "Himachal Pradesh": "Hindi",
    "Jammu & Kashmir":  "Urdu",
    "Jharkhand":        "Hindi",
    "Karnataka":        "Kannada",
    "Kerala":           "Malayalam",
    "Madhya Pradesh":   "Hindi",
    "Maharashtra":      "Marathi",
    "Odisha":           "Odia",
    "Puducherry":       "Tamil",
    "Punjab":           "Punjabi",
    "Rajasthan":        "Hindi",
    "Tamil Nadu":       "Tamil",
    "Telangana":        "Telugu",
    "Tripura":          "Bengali",
    "Uttar Pradesh":    "Hindi",
    "Uttarakhand":      "Hindi",
    "West Bengal":      "Bengali",
    "Delhi":            "Hindi",
}

MODEL_COLORS = {
    "Gemma-4":             "#4A90D9",
    "No Candidate Tested": "#2E3347",
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
    "BanglaLLama-3.1-8B": "#E9C46A",
}

LANG_REGION = {
    lang: r
    for r, langs in {
        "Indic":       ["Hindi","Bengali","Tamil","Telugu","Kannada","Malayalam","Marathi","Gujarati","Punjabi","Odia","Assamese","Urdu","Nepali","Sinhala","Maithili"],
        "Middle East": ["Arabic","Persian","Turkish","Hebrew","Kurdish","Azerbaijani","Uzbek","Kazakh"],
        "East Asia":   ["Chinese","Japanese","Korean"],
        "SEA":         ["Vietnamese","Thai","Indonesian","Malay","Tagalog","Burmese","Khmer"],
        "Africa":      ["Swahili","Amharic","Hausa","Yoruba","Igbo","Zulu","Xhosa","Somali","Wolof","Shona"],
        "Europe":      ["French","German","Spanish","Portuguese","Italian","Dutch","Polish","Russian","Ukrainian","Romanian","Swedish","Czech","Greek"],
        "Americas":    ["Lat.Am. Spanish","Brazilian Portuguese","Quechua","Haitian Creole"],
        "Oceania":     ["Māori","Samoan","Tok Pisin"],
        "Other":       ["English"],
    }.items()
    for lang in langs
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_results():
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_per_language(rows):
    g4   = {r["language"]: r for r in rows if r["tokenizer_name"] == "Gemma-4"}
    cand = {r["language"]: r for r in rows
            if r["tokenizer_name"] not in ("Gemma-4","BLOOM","mT5")}
    # English wasn't tested — Gemma-4 IS the production English model
    out = {"English": {"best": "Gemma-4", "candidate": "—",
                       "g4": None, "cand": None}}
    for lang, gr in g4.items():
        cr = cand.get(lang)

        def f(r, k): return round(float(r[k]), 2) if r else None

        g4_metrics = {
            "fertility":       f(gr,"fertility"),
            "compression":     f(gr,"compression_ratio"),
            "bfr":             f(gr,"byte_fallback_rate"),
            "unk":             f(gr,"unknown_rate"),
            "vcov":            f(gr,"vocab_coverage"),
            "rt":              f(gr,"roundtrip_pass_rate"),
            "avg_tok":         f(gr,"avg_tokens_per_sent"),
        }
        if cr is None:
            out[lang] = {"best":"No Candidate Tested","candidate":"—",
                         "g4": g4_metrics, "cand": None}
        else:
            cand_metrics = {
                "fertility":   f(cr,"fertility"),
                "compression": f(cr,"compression_ratio"),
                "bfr":         f(cr,"byte_fallback_rate"),
                "unk":         f(cr,"unknown_rate"),
                "vcov":        f(cr,"vocab_coverage"),
                "rt":          f(cr,"roundtrip_pass_rate"),
                "avg_tok":     f(cr,"avg_tokens_per_sent"),
            }
            gf, cf = g4_metrics["fertility"], cand_metrics["fertility"]
            cv, rt = cand_metrics["vcov"], cand_metrics["rt"]
            wins = cf < gf and cv >= 80 and rt >= 95
            out[lang] = {
                "best":      cr["tokenizer_name"] if wins else "Gemma-4",
                "candidate": cr["tokenizer_name"],
                "g4":        g4_metrics,
                "cand":      cand_metrics,
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

        if len(unique_w) > 1:   display = "Multiple Winners"
        elif len(unique_w) == 1: display = unique_w[0]
        elif any(info["best"] == "Gemma-4" for _, info in entries):
            display = "Gemma-4"
        else: display = "No Candidate Tested"

        primary_region = LANG_REGION.get(entries[0][0], "Other") if entries else "Other"
        lang_details = [
            {"language": lang, "best": info["best"], "candidate": info["candidate"],
             "g4": info["g4"], "cand": info["cand"]}
            for lang, info in entries
        ]
        records.append({
            "iso3": iso3, "country_name": ISO_NAMES.get(iso3, iso3),
            "best_model": display, "region": primary_region,
            "lang_details": lang_details,
        })
    return records


def build_india_states(lang_best):
    rows = []
    for state, lang in INDIA_STATE_LANG.items():
        info = lang_best.get(lang)
        if info is None:
            continue
        rows.append({"state": state, "lang": lang, "best": info["best"],
                     "candidate": info["candidate"],
                     "g4": info["g4"], "cand": info["cand"]})
    return rows


def make_colorscale(model_list, dim="#1A1E2E"):
    n = len(model_list) + 1
    eps = 1e-4
    cs = []
    for i in range(n):
        lo = i / n
        hi = (i + 1) / n - eps
        color = dim if i == 0 else MODEL_COLORS.get(model_list[i - 1], "#888")
        cs += [[lo, color], [hi, color]]
    cs[-1][0] = 1.0
    return cs


# ── HTML ──────────────────────────────────────────────────────────────────────

def generate_html(country_records, india_states, lang_best):
    all_langs    = sorted(lang_best.keys())
    winner_langs = [(lang, info["best"]) for lang, info in lang_best.items()
                    if info["best"] not in ("Gemma-4","No Candidate Tested")]
    winner_langs.sort(key=lambda x: LANG_REGION.get(x[0],""))

    model_order = (
        ["Gemma-4","No Candidate Tested","Multiple Winners"]
        + [m for m in MODEL_COLORS if m not in ("Gemma-4","No Candidate Tested","Multiple Winners")]
    )
    present     = {c["best_model"] for c in country_records} | {s["best"] for s in india_states}
    model_list  = [m for m in model_order if m in present]
    model_idx   = {m: i for i, m in enumerate(model_list)}
    colorscale  = make_colorscale(model_list)

    for i, c in enumerate(country_records):
        c["idx"] = i
        c["z"]   = model_idx.get(c["best_model"], 0)

    for s in india_states:
        s["z"] = model_idx.get(s["best"], 0)

    legend_items = "".join(
        f'<div class="lchip" title="{m}">'
        f'<div class="ldot" style="background:{MODEL_COLORS.get(m,"#888")}"></div>'
        f'<span>{m}</span></div>'
        for m in model_list
    )

    # Stats popover data
    all_lang_rows = "".join(
        f'<tr><td>{lang}</td><td class="reg">{LANG_REGION.get(lang,"")}</td>'
        f'<td class="mdl" style="color:{MODEL_COLORS.get(info["best"],"#aaa")}">'
        f'{info["best"]}</td></tr>'
        for lang, info in sorted(lang_best.items(), key=lambda x: LANG_REGION.get(x[0],""))
    )
    winner_rows = "".join(
        f'<tr><td>{lang}</td><td class="reg">{LANG_REGION.get(lang,"")}</td>'
        f'<td class="mdl" style="color:{MODEL_COLORS.get(mdl,"#aaa")}">{mdl}</td>'
        f'<td class="num">{lang_best[lang]["g4"]["fertility"]} → {lang_best[lang]["cand"]["fertility"]}</td></tr>'
        for lang, mdl in winner_langs
    )

    D = json.dumps
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
  --bg:#0D1117; --sf:#161B22; --sf2:#21262D; --bd:#30363D;
  --tx:#C9D1D9; --mu:#8B949E; --ac:#58A6FF; --gr:#3FB950;
  --sb:340px;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--tx);
  height:100vh;display:flex;flex-direction:column;overflow:hidden}}

/* Header */
header{{display:flex;align-items:center;gap:12px;padding:0 18px;height:50px;
  background:var(--sf);border-bottom:1px solid var(--bd);flex-shrink:0;position:relative;z-index:30}}
.logo{{display:flex;align-items:center;gap:7px;font-weight:600;font-size:14px;color:white}}
.logo-icon{{width:26px;height:26px;background:linear-gradient(135deg,#58A6FF,#7C6AF7);
  border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:13px}}
.hsep{{color:var(--bd);font-size:16px}}
.hsub{{color:var(--mu);font-size:12px}}
.hsp{{flex:1}}
.hchip{{font-size:11px;color:var(--mu);background:var(--sf2);border:1px solid var(--bd);
  border-radius:20px;padding:3px 10px}}

/* Stats bar */
.stats{{display:flex;background:var(--sf);border-bottom:1px solid var(--bd);flex-shrink:0;position:relative;z-index:20}}
.st{{flex:1;padding:9px 16px;border-right:1px solid var(--bd);cursor:pointer;
  position:relative;transition:background .15s;user-select:none}}
.st:last-child{{border-right:none}}
.st:hover{{background:var(--sf2)}}
.stv{{font-size:19px;font-weight:700;color:white;line-height:1;display:flex;align-items:center;gap:5px}}
.stv .arr{{font-size:10px;color:var(--mu);margin-top:1px}}
.stl{{font-size:10px;color:var(--mu);text-transform:uppercase;letter-spacing:.4px;margin-top:2px}}

/* Popovers */
.pop{{position:absolute;top:calc(100% + 4px);left:0;min-width:520px;max-width:680px;
  background:var(--sf);border:1px solid var(--bd);border-radius:10px;
  box-shadow:0 8px 32px rgba(0,0,0,.6);z-index:100;overflow:hidden;
  animation:fadeIn .15s ease}}
.pop.hidden{{display:none}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(-4px)}}to{{opacity:1;transform:none}}}}
.pop-head{{padding:10px 14px 8px;border-bottom:1px solid var(--bd);
  font-size:12px;font-weight:600;color:white;display:flex;justify-content:space-between;align-items:center}}
.pop-close{{cursor:pointer;color:var(--mu);font-size:14px;padding:2px 4px}}
.pop-close:hover{{color:white}}
.pop-body{{max-height:340px;overflow-y:auto;padding:8px 0}}
.ptbl{{width:100%;border-collapse:collapse;font-size:11px}}
.ptbl th{{padding:5px 12px;color:var(--mu);font-weight:500;text-align:left;
  border-bottom:1px solid var(--bd);text-transform:uppercase;letter-spacing:.4px}}
.ptbl td{{padding:5px 12px;border-bottom:1px solid rgba(48,54,61,.5)}}
.ptbl tr:last-child td{{border-bottom:none}}
.ptbl tr:hover td{{background:var(--sf2)}}
td.reg{{color:var(--mu);font-size:10px}}
td.mdl{{font-weight:500;font-size:11px}}
td.num{{font-family:monospace;font-size:11px;color:var(--mu)}}

/* Main */
.main{{display:flex;flex:1;overflow:hidden}}
.map-col{{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}}

/* Filter bar */
.fbar{{display:flex;align-items:center;gap:6px;padding:7px 14px;
  background:var(--sf);border-bottom:1px solid var(--bd);flex-shrink:0;flex-wrap:wrap}}
.flbl{{font-size:10px;color:var(--mu);text-transform:uppercase;letter-spacing:.4px;padding-right:4px}}
.pill{{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:500;
  cursor:pointer;border:1px solid var(--bd);background:transparent;color:var(--mu);transition:all .15s}}
.pill:hover{{border-color:var(--ac);color:var(--ac)}}
.pill.active{{background:var(--ac);border-color:var(--ac);color:white}}

/* Map */
#map{{flex:1;width:100%;min-height:0}}

/* Floating legend */
.legend-fab{{position:absolute;bottom:58px;left:14px;z-index:25;
  background:var(--sf);border:1px solid var(--bd);border-radius:8px;overflow:hidden;
  box-shadow:0 4px 16px rgba(0,0,0,.5)}}
.legend-header{{padding:6px 12px;font-size:11px;font-weight:600;
  color:var(--mu);letter-spacing:.04em;text-transform:uppercase;
  border-bottom:1px solid var(--bd)}}
.legend-body{{padding:8px 10px;
  max-height:260px;overflow-y:auto;display:flex;flex-direction:column;gap:4px}}
.lchip{{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--tx);
  padding:3px 6px;border-radius:5px;cursor:default;white-space:nowrap}}
.lchip:hover{{background:var(--sf2)}}
.ldot{{width:9px;height:9px;border-radius:2px;flex-shrink:0}}

/* Sidebar */
.sidebar{{width:var(--sb);flex-shrink:0;background:var(--sf);border-left:1px solid var(--bd);
  display:flex;flex-direction:column;overflow:hidden;
  transition:width .22s cubic-bezier(.4,0,.2,1),opacity .22s ease}}
.sidebar.hidden{{width:0;opacity:0;border-left:none;pointer-events:none}}
.sbh{{display:flex;align-items:center;justify-content:space-between;
  padding:13px 16px 11px;border-bottom:1px solid var(--bd);flex-shrink:0}}
.sbhtitle{{font-size:14px;font-weight:600;color:white}}
.sbhsub{{font-size:10px;color:var(--mu);margin-top:1px}}
.sbclose{{width:24px;height:24px;border-radius:6px;background:var(--sf2);
  border:1px solid var(--bd);cursor:pointer;display:flex;align-items:center;
  justify-content:center;font-size:11px;color:var(--mu);transition:all .15s;flex-shrink:0}}
.sbclose:hover{{color:white;border-color:var(--mu)}}
.sbb{{flex:1;overflow-y:auto;padding:12px 14px;display:flex;flex-direction:column;gap:10px}}
.sb-empty{{flex:1;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:10px;color:var(--mu);text-align:center;padding:20px}}
.sb-empty-icon{{font-size:34px;opacity:.3}}
.sb-empty p{{font-size:12px;line-height:1.6}}

/* Language cards */
.lc{{background:var(--sf2);border:1px solid var(--bd);border-radius:8px;padding:11px 13px}}
.lc-top{{display:flex;align-items:flex-start;justify-content:space-between;gap:8px;margin-bottom:3px}}
.lc-name{{font-size:13px;font-weight:600;color:white;flex-shrink:0}}
.badge{{font-size:10px;font-weight:600;padding:2px 7px;border-radius:10px;
  white-space:nowrap;max-width:160px;overflow:hidden;text-overflow:ellipsis;text-align:right}}
.bwin{{background:rgba(63,185,80,.12);color:#3FB950;border:1px solid rgba(63,185,80,.22)}}
.bg4 {{background:rgba(88,166,255,.1);color:#58A6FF;border:1px solid rgba(88,166,255,.18)}}
.bnone{{background:rgba(139,148,158,.08);color:var(--mu);border:1px solid var(--bd)}}
.lc-cand{{font-size:10px;color:var(--mu);margin-bottom:9px}}

/* Metric section */
.msec{{margin-bottom:8px}}
.msec-title{{font-size:9px;text-transform:uppercase;letter-spacing:.5px;color:var(--mu);
  margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid var(--bd)}}
.mrow{{display:flex;flex-direction:column;gap:3px;margin-bottom:5px}}
.mlabels{{display:flex;justify-content:space-between;font-size:10px;color:var(--mu)}}
.mlabels span:last-child{{color:var(--tx);font-weight:500}}
.mbar{{height:3px;background:var(--bd);border-radius:2px;overflow:hidden;margin-top:2px}}
.mfill{{height:100%;border-radius:2px;transition:width .5s cubic-bezier(.4,0,.2,1)}}
.fg4 {{background:#4A90D9}}.fc{{background:#3FB950}}.fv{{background:#9B5DE5}}
.fr {{background:#2EC4B6}}.fb{{background:#E76F51}}.fk{{background:#F5A623}}
.fco{{background:#98C1D9}}

/* Comparison mini-table */
.ctbl{{width:100%;border-collapse:collapse;font-size:10px;margin-top:4px}}
.ctbl th{{color:var(--mu);padding:3px 5px;text-align:left;font-weight:500;
  border-bottom:1px solid var(--bd)}}
.ctbl td{{padding:3px 5px;border-bottom:1px solid rgba(48,54,61,.4)}}
.ctbl tr:last-child td{{border-bottom:none}}
.ctbl .dir-good{{color:#3FB950}}.ctbl .dir-bad{{color:#F85149}}.ctbl .dir-neu{{color:var(--mu)}}
.ctbl .v{{font-family:monospace;text-align:right}}

::-webkit-scrollbar{{width:4px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:var(--bd);border-radius:2px}}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🌍</div>
    Falcon Language Map
  </div>
  <span class="hsep">/</span>
  <span class="hsub">Tokenizer Evaluation · FLORES-200</span>
  <div class="hsp"></div>
  <span class="hchip">{len(all_langs)} languages · {len(country_records)} countries</span>
</header>

<div class="stats" id="statsbar">
  <div class="st" onclick="togglePop('pop-langs',this)">
    <div class="stv">{len(all_langs)} <span class="arr">↗</span></div>
    <div class="stl">Languages Tested</div>
    <div class="pop hidden" id="pop-langs">
      <div class="pop-head">All {len(all_langs)} languages evaluated
        <span class="pop-close" onclick="event.stopPropagation();closePop('pop-langs')">✕</span></div>
      <div class="pop-body">
        <table class="ptbl">
          <tr><th>Language</th><th>Region</th><th>Best LLM</th></tr>
          {all_lang_rows}
        </table>
      </div>
    </div>
  </div>
  <div class="st" onclick="togglePop('pop-winners',this)">
    <div class="stv" style="color:var(--gr)">{len(winner_langs)} <span class="arr">↗</span></div>
    <div class="stl">Candidates Beat Gemma-4</div>
    <div class="pop hidden" id="pop-winners">
      <div class="pop-head">{len(winner_langs)} regional candidates outperform Gemma-4
        <span class="pop-close" onclick="event.stopPropagation();closePop('pop-winners')">✕</span></div>
      <div class="pop-body">
        <table class="ptbl">
          <tr><th>Language</th><th>Region</th><th>Winner Model</th><th>Fertility G4→Cand</th></tr>
          {winner_rows}
        </table>
      </div>
    </div>
  </div>
  <div class="st">
    <div class="stv">{len(country_records)}</div>
    <div class="stl">Countries Mapped</div>
  </div>
  <div class="st">
    <div class="stv">3</div>
    <div class="stl">Baselines (G4 · BLOOM · mT5)</div>
  </div>
  <div class="st">
    <div class="stv">~1 012</div>
    <div class="stl">Sentences / Language</div>
  </div>
</div>

<div class="main">
  <div class="map-col" style="position:relative">
    <div class="fbar">
      <span class="flbl">Region</span>
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

    <!-- Floating legend -->
    <div class="legend-fab">
      <div class="legend-header">Legend</div>
      <div class="legend-body">
        {legend_items}
      </div>
    </div>
  </div>

  <div class="sidebar hidden" id="sb">
    <div class="sbh">
      <div>
        <div class="sbhtitle" id="sb-title">—</div>
        <div class="sbhsub" id="sb-sub"></div>
      </div>
      <div class="sbclose" onclick="closeSB()">✕</div>
    </div>
    <div class="sbb" id="sb-body">
      <div class="sb-empty">
        <div class="sb-empty-icon">🗺️</div>
        <p>Click any country or Indian state<br>to see language details</p>
      </div>
    </div>
  </div>
</div>

<script>
const COUNTRIES  = {D(country_records, ensure_ascii=False)};
const INDIA_ST   = {D(india_states,    ensure_ascii=False)};
const MODEL_IDX  = {D(model_idx)};
const MCOLORS    = {D(MODEL_COLORS)};
const COLORSCALE = {D(colorscale)};
const ZMIN={-1}, ZMAX={len(model_list)-1};

const REGION_VIEW = {{
  "All":        {{lat:15, lon:10,  sc:0.85}},
  "Indic":      {{lat:22, lon:80,  sc:3.2}},
  "Middle East":{{lat:27, lon:44,  sc:3.0}},
  "East Asia":  {{lat:36, lon:118, sc:3.0}},
  "SEA":        {{lat:8,  lon:112, sc:3.0}},
  "Africa":     {{lat:2,  lon:22,  sc:2.4}},
  "Europe":     {{lat:52, lon:14,  sc:4.2}},
  "Americas":   {{lat:5,  lon:-68, sc:1.9}},
  "Oceania":    {{lat:-25,lon:145, sc:3.2}},
}};

let activeRegion = "All";
let indiaGeoJson = null;
let mapReady     = false;

// ── Colorscale helpers ────────────────────────────────────────────────────
function zFor(region, dataset) {{
  return dataset.map(c =>
    (region==="All" || c.region===region) ? c.z : -1
  );
}}

// ── Build world trace ─────────────────────────────────────────────────────
const worldTrace = {{
  type:"choropleth",
  locations: COUNTRIES.map(c=>c.iso3),
  z:         zFor("All", COUNTRIES),
  text:      COUNTRIES.map(c=>c.country_name),
  customdata:COUNTRIES.map((_,i)=>({{"type":"country","idx":i}})),
  colorscale:COLORSCALE, zmin:ZMIN, zmax:ZMAX, showscale:false,
  hovertemplate:"<b>%{{text}}</b><extra></extra>",
  marker:{{line:{{color:"#0D1117",width:0.6}}}},
  name:"Countries",
}};

const layout = {{
  paper_bgcolor:"#0D1117",
  margin:{{l:0,r:0,t:0,b:0}},
  geo:{{
    showframe:false, showcoastlines:true, coastlinecolor:"#30363D",
    showland:true, landcolor:"#1C2333",
    showocean:true, oceancolor:"#0D1520",
    showlakes:true, lakecolor:"#0D1520",
    showcountries:true, countrycolor:"#30363D",
    bgcolor:"#0D1117",
    projection:{{type:"natural earth"}},
  }},
  dragmode:"pan", uirevision:"map",
}};

const config = {{
  responsive:true,
  displayModeBar:true,
  modeBarButtonsToRemove:["select2d","lasso2d","autoScale2d","toImage"],
  displaylogo:false, scrollZoom:true,
}};

Plotly.newPlot("map",[worldTrace],layout,config).then(()=>{{
  mapReady = true;
  loadIndiaStates();
}});

// ── India states layer ────────────────────────────────────────────────────
function loadIndiaStates() {{
  fetch("https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson")
    .then(r=>r.json())
    .then(gj=>{{
      indiaGeoJson = gj;
      const stateTrace = {{
        type:"choropleth",
        geojson: gj,
        featureidkey:"properties.ST_NM",
        locations: INDIA_ST.map(s=>s.state),
        z:         INDIA_ST.map(s=>s.z),
        text:      INDIA_ST.map(s=>s.state+"<br>"+s.lang+" · "+s.best),
        customdata:INDIA_ST.map((_,i)=>({{"type":"state","idx":i}})),
        colorscale:COLORSCALE, zmin:ZMIN, zmax:ZMAX, showscale:false,
        hovertemplate:"<b>%{{text}}</b><extra></extra>",
        marker:{{line:{{color:"rgba(13,17,23,.8)",width:0.8}}}},
        name:"India States",
      }};
      Plotly.addTraces("map", stateTrace);
      // Re-attach click handler (addTraces clears it in some versions)
      attachClick();
    }})
    .catch(()=>console.warn("India GeoJSON unavailable — state layer skipped"));
}}

// ── Click handler ─────────────────────────────────────────────────────────
function attachClick() {{
  const el = document.getElementById("map");
  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on("plotly_click", e=>{{
    if (!e.points.length) return;
    const cd = e.points[0].customdata;
    if (!cd) return;
    if (cd.type==="country") openSB(COUNTRIES[cd.idx], false);
    else if (cd.type==="state") openSB(INDIA_ST[cd.idx], true);
  }});
}}

document.getElementById("map").on("plotly_click", e=>{{
  if (!e.points.length) return;
  const cd = e.points[0].customdata;
  if (!cd) return;
  if (cd.type==="country") openSB(COUNTRIES[cd.idx], false);
  else if (cd.type==="state") openSB(INDIA_ST[cd.idx], true);
}});

// ── Region filter ─────────────────────────────────────────────────────────
function setRegion(region, btn) {{
  activeRegion = region;
  document.querySelectorAll(".pill").forEach(p=>p.classList.remove("active"));
  btn.classList.add("active");
  const updates = {{ z:[zFor(region,COUNTRIES)] }};
  Plotly.restyle("map", updates, [0]);
  if (indiaGeoJson) {{
    const indZ = INDIA_ST.map(s=>(region==="All"||region==="Indic") ? s.z : -1);
    Plotly.restyle("map", {{z:[indZ]}}, [1]);
  }}
  const v = REGION_VIEW[region]||REGION_VIEW["All"];
  Plotly.relayout("map",{{
    "geo.center.lat":v.lat,"geo.center.lon":v.lon,"geo.projection.scale":v.sc
  }});
  closeAllPops();
}}

// ── Sidebar ───────────────────────────────────────────────────────────────
function openSB(data, isState) {{
  const title  = isState ? data.state  : data.country_name;
  const sub    = isState ? `Primary language: ${{data.lang}}` : ``;
  document.getElementById("sb-title").textContent = title;
  document.getElementById("sb-sub").textContent   = sub;

  const details = isState
    ? [{{language:data.lang, best:data.best, candidate:data.candidate,
         g4:data.g4, cand:data.cand}}]
    : (data.lang_details||[]);

  document.getElementById("sb-body").innerHTML = buildCards(details);
  document.getElementById("sb").classList.remove("hidden");
}}

function closeSB() {{ document.getElementById("sb").classList.add("hidden"); }}

function buildCards(details) {{
  if (!details.length)
    return `<div class="sb-empty"><p>No data available.</p></div>`;
  return details.map(d=>{{
    const isWin  = d.best!=="Gemma-4" && d.best!=="No Candidate Tested";
    const [bc,bt] = isWin ? ["bwin", d.best]
      : d.best==="Gemma-4" ? ["bg4","Gemma-4 best"] : ["bnone","No candidate"];

    if (!d.g4 && !d.cand) return `
      <div class="lc">
        <div class="lc-top"><span class="lc-name">${{d.language}}</span>
          <span class="badge ${{bc}}">${{bt}}</span></div>
        <div class="lc-cand" style="color:var(--mu);font-style:italic">
          Production language — not independently benchmarked</div>
      </div>`;

    if (!d.cand) return `
      <div class="lc">
        <div class="lc-top"><span class="lc-name">${{d.language}}</span>
          <span class="badge ${{bc}}">${{bt}}</span></div>
        <div class="lc-cand">No candidate evaluated</div>
        <div class="msec"><div class="msec-title">Gemma-4 baseline</div>
          ${{metricRow("Fertility","fg4",d.g4.fertility,4,"lower")}}
          ${{metricRow("Compression ratio","fco",d.g4.compression,5,"higher")}}
          ${{metricRow("Vocab coverage","fv",d.g4.vcov,100,"higher")}}
        </div>
      </div>`;

    const maxF = Math.max(d.g4.fertility, d.cand.fertility, 0.01);
    return `
      <div class="lc">
        <div class="lc-top"><span class="lc-name">${{d.language}}</span>
          <span class="badge ${{bc}}" title="${{d.best}}">${{bt}}</span></div>
        <div class="lc-cand">vs ${{d.candidate}}</div>

        <div class="msec">
          <div class="msec-title">Fertility — lower is better</div>
          ${{dualBar("Gemma-4","fg4",d.g4.fertility,maxF)}}
          ${{dualBar(d.candidate,"fc",d.cand.fertility,maxF)}}
        </div>

        <div class="msec">
          <div class="msec-title">All metrics comparison</div>
          <table class="ctbl">
            <tr><th>Metric</th><th>Gemma-4</th><th>${{shortName(d.candidate)}}</th><th>Δ</th></tr>
            ${{cmpRow("Fertility",         d.g4.fertility,  d.cand.fertility,  "lower")}}
            ${{cmpRow("Compression",       d.g4.compression,d.cand.compression,"higher")}}
            ${{cmpRow("Vocab coverage %",  d.g4.vcov,       d.cand.vcov,      "higher")}}
            ${{cmpRow("Roundtrip %",       d.g4.rt,         d.cand.rt,        "higher")}}
            ${{cmpRow("Byte fallback %",   d.g4.bfr,        d.cand.bfr,       "lower")}}
            ${{cmpRow("UNK rate %",        d.g4.unk,        d.cand.unk,       "lower")}}
            ${{cmpRow("Avg tokens/seg",    d.g4.avg_tok,    d.cand.avg_tok,   "lower")}}
          </table>
        </div>
      </div>`;
  }}).join("");
}}

function shortName(s) {{ return s && s.length>14 ? s.slice(0,13)+"…" : (s||"—"); }}

function metricRow(label, cls, val, max, dir) {{
  if (val==null) return "";
  const pct = Math.min((dir==="higher" ? val : (max-val)) / max * 100, 100).toFixed(0);
  return `<div class="mrow">
    <div class="mlabels"><span>${{label}}</span><span>${{val}}</span></div>
    <div class="mbar"><div class="mfill ${{cls}}" style="width:${{pct}}%"></div></div>
  </div>`;
}}

function dualBar(label, cls, val, maxVal) {{
  const pct = Math.min(val/maxVal*100,100).toFixed(0);
  return `<div class="mrow">
    <div class="mlabels"><span>${{shortName(label)}}</span><span>${{val}}</span></div>
    <div class="mbar"><div class="mfill ${{cls}}" style="width:${{pct}}%"></div></div>
  </div>`;
}}

function cmpRow(label, g4, cand, dir) {{
  if (g4==null||cand==null) return "";
  const delta = cand - g4;
  const pct   = g4!==0 ? (delta/Math.abs(g4)*100).toFixed(1) : "—";
  const better = dir==="lower" ? delta<0 : delta>0;
  const neutral= Math.abs(delta)<0.01;
  const cls   = neutral?"dir-neu":better?"dir-good":"dir-bad";
  const sign  = delta>0?"+":"";
  return `<tr>
    <td>${{label}}</td>
    <td class="v">${{g4}}</td>
    <td class="v">${{cand}}</td>
    <td class="${{cls}} v">${{neutral?"=":sign+pct+"%"}}</td>
  </tr>`;
}}

// ── Stats popovers ────────────────────────────────────────────────────────
function togglePop(id, stEl) {{
  const pop = document.getElementById(id);
  const wasHidden = pop.classList.contains("hidden");
  closeAllPops();
  if (wasHidden) pop.classList.remove("hidden");
}}
function closePop(id) {{
  document.getElementById(id).classList.add("hidden");
}}
function closeAllPops() {{
  document.querySelectorAll(".pop").forEach(p=>p.classList.add("hidden"));
}}
document.addEventListener("click", e=>{{
  if (!e.target.closest(".st") && !e.target.closest(".pop"))
    closeAllPops();
}});
</script>
</body>
</html>"""


def main():
    rows            = load_results()
    lang_best       = best_per_language(rows)
    country_records = build_country_data(lang_best)
    india_states    = build_india_states(lang_best)

    html = generate_html(country_records, india_states, lang_best)
    OUT.write_text(html, encoding="utf-8")
    print(f"Map written → {OUT}  ({OUT.stat().st_size//1024} KB)")
    print(f"India states: {len(india_states)} states mapped")

    from collections import Counter
    counts = Counter(c["best_model"] for c in country_records)
    print("\nCountries per category:")
    for m, n in counts.most_common():
        print(f"  {m:35s}  {n}")


if __name__ == "__main__":
    main()
