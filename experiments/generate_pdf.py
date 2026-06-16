"""
Generate a professional PDF report from docs/llm-evaluation.md
Uses weasyprint for HTML→PDF conversion with a dark-navy styled report.
"""

import os
import re
import base64
import sys

# Ensure homebrew dylibs (pango, cairo) are visible to weasyprint on macOS
os.environ.setdefault("DYLD_LIBRARY_PATH", "/opt/homebrew/lib")

try:
    from weasyprint import HTML, CSS
except OSError:
    # Try again after setting the env var (in case the import happened before)
    import ctypes
    import ctypes.util
    # Force-load the dylib manually
    try:
        ctypes.CDLL("/opt/homebrew/lib/libpango-1.0.dylib")
    except Exception:
        pass
    from weasyprint import HTML, CSS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(BASE, "docs", "llm-evaluation.md")
MAP_PATH = os.path.join(BASE, "docs", "world-map.png")
OUT_PDF = os.path.join(BASE, "docs", "falcon-tokenizer-eval-report.pdf")

# ---------------------------------------------------------------------------
# Load files
# ---------------------------------------------------------------------------
with open(MD_PATH, "r", encoding="utf-8") as f:
    md_text = f.read()

# Embed world-map image as base64 data URI
map_b64 = ""
if os.path.exists(MAP_PATH):
    with open(MAP_PATH, "rb") as f:
        map_b64 = base64.b64encode(f.read()).decode("ascii")

# ---------------------------------------------------------------------------
# Minimal Markdown → HTML parser
# ---------------------------------------------------------------------------

def escape_html(text):
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))

def inline_markup(text):
    """Convert inline **bold**, `code`, and simple text."""
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text

def parse_table(lines):
    """Convert markdown table lines to HTML table."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        # Skip separator rows (--- cells)
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if all(re.match(r"^[-:]+$", c) for c in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""

    header = rows[0]
    data_rows = rows[1:]

    thead = "<thead><tr>" + "".join(f"<th>{escape_html(h)}</th>" for h in header) + "</tr></thead>"

    tbody_rows = []
    for i, row in enumerate(data_rows):
        # Detect winner rows: check if any cell contains "✅" or "Candidate wins"
        # For section 1 table: highlight rows where "Best tokenizer" != "Gemma-4"
        # We do a generic approach: green-tint if row text contains "wins" but not "Gemma-4 wins"
        row_text = " ".join(row)
        is_winner = ("✅" in row_text or "Candidate wins" in row_text)
        is_gemma_row = ("❌" in row_text or "Gemma-4 wins" in row_text)

        row_class = "winner-row" if is_winner else ("gemma-row" if is_gemma_row else "")
        bg = f' class="{row_class}"' if row_class else (' class="alt-row"' if i % 2 == 0 else "")

        cells_html = ""
        for j, cell in enumerate(row):
            processed = inline_markup(escape_html(cell))
            cells_html += f"<td>{processed}</td>"
        tbody_rows.append(f"<tr{bg}>{cells_html}</tr>")

    tbody = "<tbody>" + "\n".join(tbody_rows) + "</tbody>"
    return f'<div class="table-wrapper"><table>{thead}{tbody}</table></div>'

def md_to_html(text, map_b64=""):
    """Convert markdown to HTML, inserting the world map after section 1 header."""
    lines = text.split("\n")
    html_parts = []
    in_table = False
    table_lines = []
    in_blockquote = False
    in_list = False
    in_code_block = False
    skip_hr = False
    map_inserted = False
    section_count = 0

    # We'll track when we've finished section 1's table so we insert the map
    after_section1_table = False
    in_section1 = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # ---- Code blocks ----
        if line.strip().startswith("```"):
            if not in_code_block:
                in_code_block = True
                html_parts.append("<pre><code>")
            else:
                in_code_block = False
                html_parts.append("</code></pre>")
            i += 1
            continue
        if in_code_block:
            html_parts.append(escape_html(line) + "\n")
            i += 1
            continue

        # ---- Horizontal rule ----
        if re.match(r"^---+\s*$", line.strip()):
            # Close any open list/blockquote
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            if in_blockquote:
                html_parts.append("</blockquote>")
                in_blockquote = False
            # Insert map after the first --- following section 1's table
            if in_section1 and not map_inserted and after_section1_table and map_b64:
                html_parts.append(MAP_HTML.format(map_b64=map_b64))
                map_inserted = True
                in_section1 = False
            html_parts.append('<hr>')
            i += 1
            continue

        # ---- Headers ----
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            if in_table:
                html_parts.append(parse_table(table_lines))
                table_lines = []
                in_table = False
                after_section1_table = True
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            if in_blockquote:
                html_parts.append("</blockquote>")
                in_blockquote = False

            level = len(m.group(1))
            heading_text = m.group(2).strip()
            heading_id = re.sub(r"[^a-z0-9]+", "-", heading_text.lower()).strip("-")

            # Track section numbering for page breaks and map insertion
            if level == 2:
                section_count += 1
                page_break = ' class="section-break"' if section_count > 1 else ''
                # Track section 1 specifically
                if "1." in heading_text or section_count == 1:
                    in_section1 = True
                    after_section1_table = False
                else:
                    in_section1 = False

                html_parts.append(f'<h{level}{page_break} id="{heading_id}">{inline_markup(escape_html(heading_text))}</h{level}>')
            else:
                html_parts.append(f'<h{level} id="{heading_id}">{inline_markup(escape_html(heading_text))}</h{level}>')
            i += 1
            continue

        # ---- Tables ----
        if line.strip().startswith("|"):
            if not in_table:
                in_table = True
                table_lines = []
                if in_list:
                    html_parts.append("</ul>")
                    in_list = False
                if in_blockquote:
                    html_parts.append("</blockquote>")
                    in_blockquote = False
            table_lines.append(line)
            i += 1
            continue
        else:
            if in_table:
                html_parts.append(parse_table(table_lines))
                table_lines = []
                in_table = False
                after_section1_table = True

        # ---- Blockquotes ----
        if line.startswith(">"):
            if not in_blockquote:
                in_blockquote = True
                html_parts.append("<blockquote>")
            content = line.lstrip("> ").strip()
            html_parts.append(f"<p>{inline_markup(escape_html(content))}</p>")
            i += 1
            continue
        else:
            if in_blockquote:
                html_parts.append("</blockquote>")
                in_blockquote = False

        # ---- Lists ----
        if re.match(r"^[-*]\s+", line):
            if not in_list:
                in_list = True
                html_parts.append("<ul>")
            content = re.sub(r"^[-*]\s+", "", line).strip()
            html_parts.append(f"<li>{inline_markup(escape_html(content))}</li>")
            i += 1
            continue
        else:
            if in_list and line.strip():
                html_parts.append("</ul>")
                in_list = False

        # ---- Blank line ----
        if not line.strip():
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            i += 1
            continue

        # ---- Paragraph ----
        html_parts.append(f"<p>{inline_markup(escape_html(line.strip()))}</p>")
        i += 1

    # Close any open elements
    if in_table:
        html_parts.append(parse_table(table_lines))
    if in_list:
        html_parts.append("</ul>")
    if in_blockquote:
        html_parts.append("</blockquote>")

    # If map wasn't inserted yet (e.g., file ended), append it at the right place
    # (already handled by --- detection above)

    return "\n".join(html_parts)

# ---------------------------------------------------------------------------
# World map HTML snippet
# ---------------------------------------------------------------------------
MAP_HTML = """
<div class="world-map-section">
  <img src="data:image/png;base64,{map_b64}" alt="World Map — Best LLM per country" class="world-map-img">
  <p class="map-caption">Best LLM per country — 163 countries mapped</p>
</div>
"""

# ---------------------------------------------------------------------------
# CSS stylesheet
# ---------------------------------------------------------------------------
CSS_STYLE = """
@page {
    size: A4;
    margin: 20mm;
    @bottom-center {
        content: "Flam AI · Falcon Language Support · Task 1 — Regional LLM Evaluation  |  Page " counter(page) " of " counter(pages);
        font-family: sans-serif;
        font-size: 8pt;
        color: #666;
    }
}

* {
    box-sizing: border-box;
}

body {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    font-size: 9pt;
    line-height: 1.5;
    color: #1a1a2e;
    margin: 0;
    padding: 0;
}

/* ---- Cover / header bar ---- */
.report-header {
    background: #0D1117;
    color: #ffffff;
    padding: 24px 28px 20px 28px;
    margin: -20mm -20mm 20px -20mm;
    border-bottom: 4px solid #1565c0;
}

.report-header h1 {
    margin: 0 0 6px 0;
    font-size: 22pt;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    border: none;
}

.report-header .subtitle {
    font-size: 10pt;
    color: #8b949e;
    margin: 0;
    font-weight: 400;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.report-header .meta {
    font-size: 8pt;
    color: #6e7681;
    margin-top: 10px;
    border-top: 1px solid #30363d;
    padding-top: 8px;
}

/* ---- Headings ---- */
h1, h2, h3, h4 {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    color: #1a237e;
    margin-top: 1.4em;
    margin-bottom: 0.5em;
    font-weight: 700;
}

h2 {
    font-size: 16pt;
    padding-left: 12px;
    border-left: 4px solid #1565c0;
    line-height: 1.3;
}

h3 {
    font-size: 12pt;
    padding-left: 10px;
    border-left: 3px solid #42a5f5;
}

h4 {
    font-size: 10pt;
    color: #283593;
}

/* Page breaks before major sections */
h2.section-break {
    page-break-before: always;
    padding-top: 4px;
}

/* ---- Paragraphs ---- */
p {
    margin: 0.4em 0 0.7em 0;
}

/* ---- Blockquotes ---- */
blockquote {
    background: #f0f4ff;
    border-left: 4px solid #3f51b5;
    margin: 12px 0;
    padding: 10px 14px;
    border-radius: 0 4px 4px 0;
    font-size: 8.5pt;
    color: #333;
}

blockquote p {
    margin: 0 0 4px 0;
}

blockquote p:last-child {
    margin-bottom: 0;
}

/* ---- Horizontal rule ---- */
hr {
    border: none;
    border-top: 1px solid #e0e4f0;
    margin: 16px 0;
}

/* ---- Lists ---- */
ul {
    padding-left: 18px;
    margin: 6px 0;
}

li {
    margin-bottom: 3px;
}

/* ---- Code ---- */
code {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 8pt;
    background: #f1f3f9;
    color: #c7254e;
    padding: 1px 4px;
    border-radius: 3px;
}

pre {
    background: #0d1117;
    color: #c9d1d9;
    padding: 12px 14px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 7.5pt;
    line-height: 1.6;
}

pre code {
    background: none;
    color: inherit;
    padding: 0;
}

/* ---- Tables ---- */
.table-wrapper {
    overflow-x: auto;
    margin: 10px 0 14px 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    font-size: 7.5pt;
    page-break-inside: auto;
}

thead tr {
    background: #1a237e;
    color: #ffffff;
}

thead th {
    padding: 6px 8px;
    text-align: left;
    font-weight: 600;
    font-size: 7pt;
    letter-spacing: 0.3px;
    border: 1px solid #283593;
    white-space: nowrap;
}

tbody td {
    padding: 5px 8px;
    border: 1px solid #e0e4ee;
    vertical-align: top;
    line-height: 1.4;
}

tbody tr {
    background: #ffffff;
}

tbody tr.alt-row {
    background: #f8f9fa;
}

/* Winner rows: green tint + left accent */
tbody tr.winner-row {
    background: #f0fff4;
    border-left: 3px solid #2e7d32;
}

tbody tr.winner-row td:first-child {
    border-left: 3px solid #2e7d32;
}

/* Gemma wins: subtle red/orange tint */
tbody tr.gemma-row {
    background: #fff8f6;
}

tbody tr:hover {
    background: #e8edf8;
}

strong {
    font-weight: 700;
}

em {
    font-style: italic;
}

/* ---- World map ---- */
.world-map-section {
    margin: 16px 0 20px 0;
    text-align: center;
    page-break-inside: avoid;
}

.world-map-img {
    width: 100%;
    max-width: 100%;
    border-radius: 6px;
    border: 1px solid #e0e4ee;
    display: block;
}

.map-caption {
    font-size: 8pt;
    color: #666;
    font-style: italic;
    margin-top: 6px;
    text-align: center;
}
"""

# ---------------------------------------------------------------------------
# Cover / header HTML
# ---------------------------------------------------------------------------
HEADER_HTML = """
<div class="report-header">
  <h1>Falcon Tokenizer Evaluation Report</h1>
  <p class="subtitle">FLORES-200 &middot; 63 Languages &middot; 17 Regional Winners</p>
  <p class="meta">
    Generated: 2026-06-16 11:50 UTC &nbsp;&bull;&nbsp;
    Source: <code>data/results.csv</code> &nbsp;&bull;&nbsp;
    Corpus: FLORES-200 devtest (~1012 sentences/language) &nbsp;&bull;&nbsp;
    Project: Flam AI &mdash; Falcon Language Support
  </p>
</div>
"""

# ---------------------------------------------------------------------------
# Build full HTML
# ---------------------------------------------------------------------------
body_html = md_to_html(md_text, map_b64=map_b64)

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Falcon Tokenizer Evaluation Report</title>
</head>
<body>
{HEADER_HTML}
{body_html}
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Generate PDF
# ---------------------------------------------------------------------------
print("Building PDF…")
html_doc = HTML(string=full_html, base_url=BASE)
css_doc = CSS(string=CSS_STYLE)
html_doc.write_pdf(OUT_PDF, stylesheets=[css_doc])
print(f"PDF written to: {OUT_PDF}")
