"""
Generate the global tokenizer evaluation detailed report.
Matches the format of docs/pdfs/Tokenizer evaluation — detailed report - document_pdf.pdf

Output: docs/llm-evaluation.md

Usage:
  python experiments/detailed_report.py
"""

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
RESULTS   = ROOT / "data" / "results.csv"
SUMMARY   = ROOT / "data" / "summary.json"
OUT       = ROOT / "docs" / "llm-evaluation.md"

METRICS = [
    ("fertility",           "Fertility (tokens / whitespace word)",  "Lower is better"),
    ("compression_ratio",   "Compression ratio (chars / token)",     "Higher is better"),
    ("byte_fallback_rate",  "Byte fallback rate (%)",                "Lower is better"),
    ("unknown_rate",        "UNK rate (%)",                          "Lower is better"),
    ("vocab_coverage",      "Vocabulary coverage (%)",               "Higher is better"),
    ("roundtrip_pass_rate", "Roundtrip fidelity (%)",                "Higher is better"),
    ("avg_tokens_per_sent", "Avg tokens / segment",                  "Lower is better"),
]

REGION_ORDER = ["Indic", "Middle East", "East Asia", "SEA", "Africa", "Europe", "Americas", "Oceania"]


def load_results():
    with open(RESULTS, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_summary():
    with open(SUMMARY, encoding="utf-8") as f:
        return json.load(f)


def pivot(rows, metric):
    """Build {language: {tokenizer: value}} for one metric."""
    table = defaultdict(dict)
    for r in rows:
        table[r["language"]][r["tokenizer_name"]] = float(r[metric])
    return table


def tokenizer_order(rows):
    """Baselines first, then regional candidates sorted by language."""
    baselines = ["Gemma-4", "BLOOM", "mT5"]
    candidates = sorted({r["tokenizer_name"] for r in rows if r["tokenizer_name"] not in baselines})
    return baselines + candidates


def languages_by_region(rows):
    """OrderedDict: region → [language, ...]"""
    region_langs = defaultdict(list)
    seen = set()
    for r in rows:
        key = (r["region"], r["language"])
        if key not in seen:
            region_langs[r["region"]].append(r["language"])
            seen.add(key)
    return region_langs


def gemma_lookup(rows):
    return {r["language"]: r for r in rows if r["tokenizer_name"] == "Gemma-4"}


def verdict(r, g):
    cf = float(r["fertility"]);        gf = float(g["fertility"])
    cv = float(r["vocab_coverage"]);   gv = float(g["vocab_coverage"])
    rt = float(r["roundtrip_pass_rate"])
    bfr = float(r["byte_fallback_rate"])
    if cf < gf and cv >= 80 and rt >= 95:
        return "✅ Candidate wins"
    wins = sum([cf < gf, cv > gv, bfr < 1.0, rt >= 99.0])
    return "⚠️ Mixed" if wins >= 2 else "❌ Gemma-4 wins"


def md_table(headers, rows_data):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows_data:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def generate():
    rows    = load_results()
    summary = load_summary()
    toks    = tokenizer_order(rows)
    reg_map = languages_by_region(rows)
    g4      = gemma_lookup(rows)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []

    lines += [
        f"# Global Tokenizer Evaluation — Detailed Report",
        f"",
        f"**Generated:** {now}  ·  **Source:** `data/results.csv`  ·  **Corpus:** FLORES-200 devtest (~1012 sentences/language)",
        f"",
        f"> **Caveats.** Fertility uses whitespace \"words\", which is imperfect for languages without whitespace",
        f"> word boundaries (Japanese, Thai, Burmese, Khmer). For those languages, use `avg_tokens_per_sent`",
        f"> as the primary metric instead of fertility. Vocabulary coverage is computed from characters present",
        f"> in the FLORES-200 corpus only.",
        f"",
        f"---",
        f"",
    ]

    # ── 0. Per-language best LLM summary ─────────────────────────────────────
    ALL_LANGS = [
        ("Hindi","Indic"),("Bengali","Indic"),("Tamil","Indic"),("Telugu","Indic"),
        ("Kannada","Indic"),("Malayalam","Indic"),("Marathi","Indic"),("Gujarati","Indic"),
        ("Punjabi","Indic"),("Odia","Indic"),("Assamese","Indic"),("Urdu","Indic"),
        ("Nepali","Indic"),("Sinhala","Indic"),("Maithili","Indic"),
        ("Arabic","Middle East"),("Persian","Middle East"),("Turkish","Middle East"),
        ("Hebrew","Middle East"),("Kurdish","Middle East"),("Azerbaijani","Middle East"),
        ("Uzbek","Middle East"),("Kazakh","Middle East"),
        ("Chinese","East Asia"),("Japanese","East Asia"),("Korean","East Asia"),
        ("Vietnamese","SEA"),("Thai","SEA"),("Indonesian","SEA"),("Malay","SEA"),
        ("Tagalog","SEA"),("Burmese","SEA"),("Khmer","SEA"),
        ("Swahili","Africa"),("Amharic","Africa"),("Hausa","Africa"),("Yoruba","Africa"),
        ("Igbo","Africa"),("Zulu","Africa"),("Xhosa","Africa"),("Somali","Africa"),
        ("Wolof","Africa"),("Shona","Africa"),
        ("French","Europe"),("German","Europe"),("Spanish","Europe"),("Portuguese","Europe"),
        ("Italian","Europe"),("Dutch","Europe"),("Polish","Europe"),("Russian","Europe"),
        ("Ukrainian","Europe"),("Romanian","Europe"),("Swedish","Europe"),("Czech","Europe"),
        ("Greek","Europe"),
        ("Lat.Am. Spanish","Americas"),("Brazilian Portuguese","Americas"),("Quechua","Americas"),
        ("Nahuatl","Americas"),("Haitian Creole","Americas"),
        ("Māori","Oceania"),("Samoan","Oceania"),("Hawaiian","Oceania"),("Tok Pisin","Oceania"),
    ]

    cand_map = {r["language"]: r for r in rows if r["tokenizer_name"] not in ("Gemma-4","BLOOM","mT5")}

    def _verdict(r, g):
        cf = float(r["fertility"]); gf = float(g["fertility"])
        cv = float(r["vocab_coverage"])
        rt = float(r["roundtrip_pass_rate"]); bfr = float(r["byte_fallback_rate"])
        if cf < gf and cv >= 80 and rt >= 95:
            return "Candidate"
        wins = sum([cf < gf, cv > float(g["vocab_coverage"]), bfr < 1.0, rt >= 99.0])
        return "Mixed → Gemma-4" if wins >= 2 else "Gemma-4"

    summary_rows = []
    for lang, region in ALL_LANGS:
        if lang in ("Nahuatl","Hawaiian"):
            summary_rows.append([lang, region, "—", "Not in FLORES-200", "—"])
            continue
        g = g4.get(lang)
        c = cand_map.get(lang)
        if g is None:
            summary_rows.append([lang, region, "Gemma-4", "No test data", "—"])
            continue
        if c is None:
            summary_rows.append([lang, region, "Gemma-4", "No regional candidate tested", "—"])
            continue
        v = _verdict(c, g)
        best = c["tokenizer_name"] if v == "Candidate" else "Gemma-4"
        summary_rows.append([lang, region, best, c["tokenizer_name"], f"fertility {g['fertility']}→{c['fertility']}, vcov {c['vocab_coverage']}%"])

    lines += [
        "## 1. Best LLM per language — summary",
        "",
        "For languages where the regional candidate tokenizer was not tested (gated repo, missing package, or no model found),",
        "Gemma-4 is the current default. Re-run after fixing those to get a full comparison.",
        "",
        md_table(
            ["#", "Language", "Region", "Best tokenizer", "Candidate tested", "Key metrics"],
            [[i+1] + r for i, r in enumerate(summary_rows)],
        ),
        "",
        "---",
        "",
    ]

    # ── 1. Metric glossary ────────────────────────────────────────────────────
    lines += [
        "## 2. Metric glossary",
        "",
        "All metrics computed over the same FLORES-200 devtest segments per language.",
        "",
        "### 1.1 Quick reference: which direction is \"better\"?",
        "",
        md_table(
            ["Metric", "Better", "Target"],
            [
                ["Fertility",             "**Lower**",  "1.0–2.5 for script-based languages"],
                ["Compression ratio",     "**Higher**", "> 3.0 chars/token"],
                ["Byte fallback rate (%)", "**Lower**", "0% ideal; any > 0 = unhandled script"],
                ["UNK rate (%)",          "**Lower**",  "0%; any UNK = vocab miss"],
                ["Vocabulary coverage (%)", "**Higher**", "> 80% of unique chars as single tokens"],
                ["Roundtrip fidelity (%)", "**Higher**", "100% = lossless encode→decode"],
                ["Avg tokens / segment",  "**Lower**",  "same story as fertility (inference cost)"],
            ],
        ),
        "",
        "### 1.2 Fertility",
        "**What it is:** Tokens per whitespace-separated 'word': `total_tokens / total_words`.",
        "**Direction:** Lower is better. High fertility = same document uses more tokens (higher API cost, less room in context).",
        "",
        "### 1.3 Compression ratio",
        "**What it is:** Characters per token: `total_chars / total_tokens`.",
        "**Direction:** Higher is better. Low compression often means byte-level or very fine splitting.",
        "",
        "### 1.4 Byte fallback rate",
        "**What it is:** Percentage of output tokens that represent raw byte fallback, not normal subwords.",
        "**Direction:** Lower is better. 0% means no detected byte-fallback tokens.",
        "",
        "### 1.5 UNK rate",
        "**What it is:** Percentage of token ids equal to the tokenizer's `unk_token_id`.",
        "**Direction:** Lower is better; 0% is the target.",
        "",
        "### 1.6 Vocabulary coverage",
        "**What it is:** Among all unique characters in that language's test text, the fraction that encode to exactly **one** token when passed alone to the tokenizer.",
        "**Direction:** Higher is better. High coverage means many script characters are first-class tokens.",
        "",
        "### 1.7 Roundtrip fidelity",
        "**What it is:** Per segment: after `encode` then `decode`, does the text match the original?",
        "**Direction:** Higher is better; 100% means no lossy tokenization on this test set.",
        "",
        "### 1.8 Avg tokens per segment",
        "**What it is:** `total_tokens / total_sentences`.",
        "**Direction:** Lower is better for cost/latency. Use this as the primary efficiency metric for non-whitespace languages (Japanese, Thai, Burmese, Khmer).",
        "",
        "---",
        "",
    ]

    # ── 2. Overall summary table ───────────────────────────────────────────────
    lines += [
        "## 3. Aggregate summary (all languages)",
        "",
        "**Unweighted** averages treat each language equally. **Character-weighted** averages weight by `total_chars` so languages with more text influence the score more.",
        "",
        md_table(
            ["Tokenizer", "Languages tested", "Avg fertility", "Avg compression", "Avg byte fallback %", "Avg vocab coverage %", "Avg roundtrip %"],
            sorted(
                [
                    [
                        tok,
                        s["languages_tested"],
                        s["avg_fertility"],
                        s["avg_compression_ratio"],
                        s["avg_byte_fallback_rate"],
                        s["avg_vocab_coverage"],
                        s["avg_roundtrip_pass_rate"],
                    ]
                    for tok, s in summary.items()
                    if s["languages_tested"] > 1
                ],
                key=lambda x: x[2],
            ),
        ),
        "",
        "---",
        "",
    ]

    # ── 3. Candidate vs Gemma-4 verdict table ─────────────────────────────────
    candidates_rows = [r for r in rows if r["tokenizer_name"] not in ("Gemma-4", "BLOOM", "mT5")]

    lines += [
        "## 4. Regional candidate vs Gemma-4 — verdict per language",
        "",
        "Primary signals: fertility and vocab coverage. Secondary: byte fallback rate and roundtrip fidelity.",
        "",
        md_table(
            ["Language", "Region", "Candidate", "Fertility G4", "Fertility cand", "Vcov G4 %", "Vcov cand %", "BFR cand %", "RT cand %", "Verdict"],
            [
                [
                    r["language"],
                    r["region"],
                    r["tokenizer_name"],
                    g4[r["language"]]["fertility"],
                    r["fertility"],
                    g4[r["language"]]["vocab_coverage"],
                    r["vocab_coverage"],
                    r["byte_fallback_rate"],
                    r["roundtrip_pass_rate"],
                    verdict(r, g4[r["language"]]),
                ]
                for r in candidates_rows
                if r["language"] in g4
            ],
        ),
        "",
        "---",
        "",
    ]

    # ── 4. Pivot tables per metric ────────────────────────────────────────────
    lines += ["## 5. Comparison across metrics (pivot tables)", ""]
    lines += ["Rows = languages. Columns = tokenizers. Scan across a row to compare models on one language; scan down a column to see one model across languages.", ""]

    baseline_toks = ["Gemma-4", "BLOOM", "mT5"]

    for metric_key, metric_label, direction in METRICS:
        pv = pivot(rows, metric_key)
        lines += [f"### {metric_label}", f"*{direction}.*", ""]

        for region in REGION_ORDER:
            langs = reg_map.get(region, [])
            if not langs:
                continue

            # Collect candidate tokenizers that appear in this region
            region_cands = []
            for lang in langs:
                for r in candidates_rows:
                    if r["language"] == lang and r["tokenizer_name"] not in region_cands:
                        region_cands.append(r["tokenizer_name"])

            col_toks = baseline_toks + region_cands
            headers = ["language"] + col_toks

            table_rows = []
            for lang in langs:
                row_vals = [lang]
                for tok in col_toks:
                    val = pv.get(lang, {}).get(tok, "—")
                    row_vals.append(val if val == "—" else str(val))
                table_rows.append(row_vals)

            lines += [f"**{region}**", ""]
            lines += [md_table(headers, table_rows), ""]

    # ── 5. Full per-language results ──────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 6. Complete per-language results",
        "",
        "Every tokenizer × language combination from `data/results.csv`.",
        "",
        md_table(
            ["tokenizer", "language", "region", "fertility", "compression", "byte_fallback%", "unk%", "vcov%", "roundtrip%", "avg_tok/sent", "total_tokens", "total_chars", "sentences"],
            [
                [
                    r["tokenizer_name"], r["language"], r["region"],
                    r["fertility"], r["compression_ratio"], r["byte_fallback_rate"],
                    r["unknown_rate"], r["vocab_coverage"], r["roundtrip_pass_rate"],
                    r["avg_tokens_per_sent"], r["total_tokens"], r["total_chars"], r["total_sentences"],
                ]
                for r in rows
            ],
        ),
        "",
        "---",
        "",
        "## 7. Raw files",
        "",
        "- `data/results.csv` — machine-readable detail (one row per tokenizer × language)",
        "- `data/summary.json` — per-tokenizer aggregates (unweighted + character-weighted)",
        "",
        f"*Regenerate this report:* `python experiments/detailed_report.py`",
    ]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written → {OUT}  ({len(lines)} lines)")


if __name__ == "__main__":
    generate()
