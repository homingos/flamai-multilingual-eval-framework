"""
Interactive world map: best tokenizer per language.

Colors each country by the winning LLM from our FLORES-200 evaluation.
Hover shows per-language metric comparisons vs Gemma-4.

Output: docs/language-map.html
Usage:  python experiments/language_map.py
"""

import csv
from collections import defaultdict
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "data" / "results.csv"
OUT         = ROOT / "docs" / "language-map.html"

# ── Language → ISO-3166-1 alpha-3 primary countries ──────────────────────────
LANG_COUNTRIES = {
    # Indic
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
    # Middle East
    "Arabic":               ["SAU", "EGY", "DZA", "SDN", "IRQ", "MAR", "YEM",
                             "SYR", "TUN", "JOR", "LBY", "LBN", "ARE", "OMN",
                             "KWT", "QAT", "BHR", "MRT"],
    "Persian":              ["IRN", "AFG"],
    "Turkish":              ["TUR"],
    "Hebrew":               ["ISR"],
    "Kurdish":              ["IRQ"],
    "Azerbaijani":          ["AZE"],
    "Uzbek":                ["UZB"],
    "Kazakh":               ["KAZ"],
    # East Asia
    "Chinese":              ["CHN", "TWN"],
    "Japanese":             ["JPN"],
    "Korean":               ["KOR"],
    # SEA
    "Vietnamese":           ["VNM"],
    "Thai":                 ["THA"],
    "Indonesian":           ["IDN"],
    "Malay":                ["MYS", "BRN"],
    "Tagalog":              ["PHL"],
    "Burmese":              ["MMR"],
    "Khmer":                ["KHM"],
    # Africa
    "Swahili":              ["KEN", "TZA", "UGA", "RWA"],
    "Amharic":              ["ETH"],
    "Hausa":                ["NGA", "NER"],
    "Yoruba":               ["NGA"],
    "Igbo":                 ["NGA"],
    "Zulu":                 ["ZAF"],
    "Xhosa":                ["ZAF"],
    "Somali":               ["SOM"],
    "Wolof":                ["SEN"],
    "Shona":                ["ZWE"],
    # Europe
    "French":               ["FRA", "BEL", "LUX", "CAN", "CHE"],
    "German":               ["DEU", "AUT"],
    "Spanish":              ["ESP"],
    "Portuguese":           ["PRT", "AGO", "MOZ"],
    "Italian":              ["ITA"],
    "Dutch":                ["NLD"],
    "Polish":               ["POL"],
    "Russian":              ["RUS", "BLR"],
    "Ukrainian":            ["UKR"],
    "Romanian":             ["ROU", "MDA"],
    "Swedish":              ["SWE"],
    "Czech":                ["CZE"],
    "Greek":                ["GRC", "CYP"],
    # Americas
    "Lat.Am. Spanish":      ["MEX", "COL", "ARG", "VEN", "CHL", "ECU", "GTM",
                             "CUB", "BOL", "DOM", "HND", "PRY", "SLV", "NIC",
                             "CRI", "PAN", "URY"],
    "Brazilian Portuguese": ["BRA"],
    "Quechua":              ["PER"],
    "Haitian Creole":       ["HTI"],
    # Oceania
    "Māori":                ["NZL"],
    "Samoan":               ["WSM"],
    "Tok Pisin":            ["PNG"],
}

# ── Color palette ─────────────────────────────────────────────────────────────
# Special categories
BASE_COLORS = {
    "Gemma-4":             "#5B9BD5",   # steel blue
    "No Candidate Tested": "#D6D6D6",   # light gray
    "Multiple Winners":    "#FFD700",   # gold
}
# One color per winning regional model
MODEL_COLORS = {
    # Indic
    "Tamil-Mistral-7B":    "#FF6B35",
    "MahaMarathi-7B":      "#E63946",
    "Ambari-7B":           "#FF9F1C",
    "Gujju-Llama-7B":      "#F4A261",
    # Middle East
    "Jais-2-8B":           "#9B5DE5",
    "DictaLM-2.0-7B":      "#C77DFF",
    # East Asia
    "Polyglot-Ko-12B":     "#00B4D8",
    # SEA
    "MaLLaM-5B":           "#0096C7",
    # Africa
    "Swahili-Gemma-7B":    "#52B788",
    "Walia-LLM-7B":        "#1B4332",
    # Europe
    "Lucie-7B":            "#3D5A80",
    "Viking-7B":           "#98C1D9",
    "CSMPT-7B":            "#293241",
    "Meltemi-7B":          "#457B9D",
    # Americas
    "Tucano-2b4":          "#E76F51",
    # Oceania
    "Goldfish-mri-39M":    "#2EC4B6",
    "Goldfish-tpi-125M":   "#CBF3F0",
}
ALL_COLORS = {**BASE_COLORS, **MODEL_COLORS}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results():
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_per_language(rows):
    """Returns {language: {best, candidate, g4_fertility, cand_fertility, vcov, rt}}"""
    g4   = {r["language"]: r for r in rows if r["tokenizer_name"] == "Gemma-4"}
    cand = {r["language"]: r for r in rows
            if r["tokenizer_name"] not in ("Gemma-4", "BLOOM", "mT5")}

    out = {}
    for lang, gr in g4.items():
        cr = cand.get(lang)
        gf = float(gr["fertility"])
        if cr is None:
            out[lang] = {"best": "No Candidate Tested", "candidate": "—",
                         "g4_fertility": gf, "cand_fertility": None,
                         "vcov": None, "rt": None}
        else:
            cf = float(cr["fertility"])
            cv = float(cr["vocab_coverage"])
            rt = float(cr["roundtrip_pass_rate"])
            wins = cf < gf and cv >= 80 and rt >= 95
            out[lang] = {
                "best":          cr["tokenizer_name"] if wins else "Gemma-4",
                "candidate":     cr["tokenizer_name"],
                "g4_fertility":  gf,
                "cand_fertility": cf,
                "vcov":          cv,
                "rt":            rt,
            }
    return out


# ── Country aggregation ───────────────────────────────────────────────────────

def build_country_rows(lang_best):
    """One row per ISO country: best_model, hover_html, languages."""
    country_langs = defaultdict(list)
    for lang, info in lang_best.items():
        for iso3 in LANG_COUNTRIES.get(lang, []):
            country_langs[iso3].append((lang, info))

    rows = []
    for iso3, entries in country_langs.items():
        winners = [info["best"] for _, info in entries
                   if info["best"] not in ("Gemma-4", "No Candidate Tested")]
        unique_winners = list(dict.fromkeys(winners))  # preserve order, dedupe

        if len(unique_winners) > 1:
            display = "Multiple Winners"
        elif len(unique_winners) == 1:
            display = unique_winners[0]
        elif any(info["best"] == "Gemma-4" for _, info in entries):
            display = "Gemma-4"
        else:
            display = "No Candidate Tested"

        lines = []
        for lang, info in entries:
            if info["cand_fertility"] is not None:
                mark = " ✅" if info["best"] not in ("Gemma-4", "No Candidate Tested") else ""
                lines.append(
                    f"<b>{lang}</b>{mark}: {info['best']}<br>"
                    f"&nbsp;&nbsp;Fertility {info['g4_fertility']} → {info['cand_fertility']} "
                    f"| VCov {info['vcov']}% | RT {info['rt']}%"
                )
            else:
                lines.append(
                    f"<b>{lang}</b>: No candidate tested "
                    f"(G4 fertility {info['g4_fertility']})"
                )

        rows.append({
            "iso3":       iso3,
            "best_model": display,
            "hover_html": "<br>".join(lines),
            "languages":  ", ".join(lang for lang, _ in entries),
        })
    return rows


# ── Map generation ────────────────────────────────────────────────────────────

def make_map(country_rows):
    import plotly.express as px

    iso3s      = [r["iso3"]       for r in country_rows]
    models     = [r["best_model"] for r in country_rows]
    hovers     = [r["hover_html"] for r in country_rows]
    lang_lists = [r["languages"]  for r in country_rows]

    # Ordered category list — special categories first, then models
    present_models = list(dict.fromkeys(models))
    category_order = (
        [m for m in ["Gemma-4", "No Candidate Tested", "Multiple Winners"] if m in present_models]
        + [m for m in MODEL_COLORS if m in present_models]
    )

    fig = px.choropleth(
        locations=iso3s,
        color=models,
        color_discrete_map=ALL_COLORS,
        category_orders={"color": category_order},
        custom_data=[hovers, lang_lists],
        title="Best Tokenizer per Language — Global Evaluation (FLORES-200)",
        projection="natural earth",
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[1]}</b><br><br>"
            "%{customdata[0]}"
            "<extra></extra>"
        ),
        marker_line_color="white",
        marker_line_width=0.4,
    )

    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        legend_title_text="Best tokenizer",
        legend=dict(
            orientation="v",
            x=1.01,
            y=0.5,
            font_size=11,
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#AAAAAA",
            showland=True,
            landcolor="#F0F0F0",
            showocean=True,
            oceancolor="#E8F4FD",
            showlakes=True,
            lakecolor="#E8F4FD",
            projection_type="natural earth",
            bgcolor="white",
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=220, t=60, b=0),
        height=620,
    )

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows         = load_results()
    lang_best    = best_per_language(rows)
    country_rows = build_country_rows(lang_best)
    fig          = make_map(country_rows)

    fig.write_html(OUT, include_plotlyjs="cdn", full_html=True)
    print(f"Map written → {OUT}")

    # Quick summary
    from collections import Counter
    counts = Counter(r["best_model"] for r in country_rows)
    print("\nCountries per category:")
    for model, n in counts.most_common():
        print(f"  {model:35s}  {n} countries")


if __name__ == "__main__":
    main()
