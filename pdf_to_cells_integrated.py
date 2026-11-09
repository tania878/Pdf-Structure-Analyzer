#!/usr/bin/env python3
"""
pdf_to_cells.py - Convert agreements / contracts / purchase-order PDFs into
structured cells (headings, subheadings, paragraphs, tables, images) with
links preserved, then render them with configurable font settings.

Install:   pip install pymupdf
Usage:     python pdf_to_cells.py contract.pdf --out output
           python pdf_to_cells.py contract.pdf --inspect      (print font details)

Outputs (in --out):
    structure.json   structured nodes in reading order (the "intermediate model")
    styles.json      font settings per cell type (edit this, re-run, no re-parsing needed)
    preview.html     formatted cells using styles.json
    images/          extracted images

Scanned pages are OCRed automatically when native text extraction is empty.
OCR uses RapidOCR (ONNX Runtime) and pypdfium2 for page rendering.
"""
import argparse
import collections
import html
import json
import re
from pathlib import Path

import fitz  # PyMuPDF: native text/layout extraction

# OCR dependencies are imported lazily only when a scanned page is encountered.
# Install: python -m pip install rapidocr onnxruntime pypdfium2
_RAPID_OCR = None

def get_rapid_ocr():
    """Create one reusable RapidOCR engine (ONNX Runtime backend)."""
    global _RAPID_OCR
    if _RAPID_OCR is None:
        try:
            from rapidocr import RapidOCR
            from rapidocr import EngineType
            _RAPID_OCR = RapidOCR(params={
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Cls.engine_type": EngineType.ONNXRUNTIME,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
            })
        except ImportError as exc:
            raise RuntimeError(
                "Scanned page detected. Install OCR dependencies with: "
                "python -m pip install rapidocr onnxruntime pypdfium2"
            ) from exc
    return _RAPID_OCR


from structured_elements_integrated import StructuredDocument
import classifier as clf   # Level 1 features + Level 2 rules (classifier.py)

# ----------------------------------------------------------------------------
# Settings: font features per cell type (written to styles.json if missing)
# ----------------------------------------------------------------------------
DEFAULT_STYLES = {
    "h1": {"font_family": "Arial", "font_size": 22, "bold": True, "italic": False, "color": "#111111"},
    "h2": {"font_family": "Arial", "font_size": 18, "bold": True, "italic": False, "color": "#222222"},
    "h3": {"font_family": "Arial", "font_size": 15, "bold": True, "italic": False, "color": "#333333"},
    "h4": {"font_family": "Arial", "font_size": 13, "bold": True, "italic": True, "color": "#444444"},
    "paragraph": {"font_family": "Georgia", "font_size": 12, "bold": False, "italic": False, "color": "#000000"},
    "table_header": {"font_family": "Arial", "font_size": 12, "bold": True, "italic": False, "color": "#000000", "background": "#e8e8e8"},
    "table_cell": {"font_family": "Arial", "font_size": 11, "bold": False, "italic": False, "color": "#000000"},
    "link_color": "#0645ad",
}

# Strong heading patterns in contracts / agreements / POs
NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,3})[.)]?\s+\S")
KEYWORD_H1 = re.compile(
    r"^(ARTICLE|SECTION|SCHEDULE|ANNEXURE|ANNEX|EXHIBIT|APPENDIX|RECITALS|WHEREAS|TERMS AND CONDITIONS)\b",
    re.I,
)
PAGE_NUM = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.I)


# ----------------------------------------------------------------------------
# Low-level helpers
# ----------------------------------------------------------------------------
def is_bold(span):
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def is_italic(span):
    return bool(span["flags"] & 2) or "italic" in span["font"].lower()


def hex_color(c):
    return f"#{c:06x}"


def center_in(bbox, rects):
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1 for r in rects)


def link_for_span(span_bbox, links):
    """Return URI if this span sits inside a hyperlink rectangle."""
    r = fitz.Rect(span_bbox)
    area = r.get_area()
    if area <= 0:
        return None
    for lk in links:
        uri = lk.get("uri")
        if uri and (r & lk["from"]).get_area() / area > 0.5:
            return uri
    return None


def make_line(spans, links):
    runs = []
    for s in spans:
        uri = link_for_span(s["bbox"], links)
        if runs and runs[-1].get("uri") == uri:
            runs[-1]["text"] += s["text"]
        else:
            run = {"text": s["text"]}
            if uri:
                run["uri"] = uri
            runs.append(run)
    text = "".join(s["text"] for s in spans).strip()
    chars = collections.Counter()
    for s in spans:
        chars[round(s["size"], 1)] += len(s["text"].strip())
    bbox = fitz.Rect(spans[0]["bbox"])
    for s in spans[1:]:
        bbox |= fitz.Rect(s["bbox"])
    visible_chars = sum(len(s["text"].strip()) for s in spans)
    bold_chars = sum(len(s["text"].strip()) for s in spans if is_bold(s))
    bold_ratio = bold_chars / visible_chars if visible_chars else 0.0
    return {
        "text": text,
        "runs": runs,
        "size": chars.most_common(1)[0][0],
        "bold": bold_ratio >= 0.80,
        "bold_ratio": round(bold_ratio, 3),
        "italic": all(is_italic(s) for s in spans if s["text"].strip()),
        "font": spans[0]["font"],
        "color": hex_color(spans[0]["color"]),
        "bbox": [round(v, 1) for v in bbox],
    }


# ----------------------------------------------------------------------------
# Pass 1: per-page raw extraction (lines, tables, images)
# ----------------------------------------------------------------------------
def cell_text(page, cell):
    """Text inside one table cell rectangle (None = merged/empty cell)."""
    if not cell:
        return ""
    x0, y0, x1, y1 = cell
    r = fitz.Rect(x0 + 1, y0 + 1, x1 - 1, y1 - 1)
    if r.is_empty:
        return ""
    return " ".join(page.get_text("text", clip=r).split())


def ocr_page_lines(page, pdfium_page, dpi=220):
    """OCR a scanned page with RapidOCR/ONNX and pypdfium2 rendering.

    RapidOCR returns one quadrilateral per detected text line. Coordinates are
    mapped from rendered pixels to PDF points for the existing classifier.
    """
    try:
        import numpy as np
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "Scanned page detected. Install OCR dependencies with: "
            "python -m pip install rapidocr onnxruntime pypdfium2"
        ) from exc

    scale = dpi / 72.0
    bitmap = pdfium_page.render(scale=scale, rotation=0)
    image = np.asarray(bitmap.to_pil().convert("RGB"))
    result = get_rapid_ocr()(image)
    if result is None or result.boxes is None or result.txts is None:
        return []

    # Use actual rendered dimensions to avoid small rounding discrepancies.
    px_per_pt_x = image.shape[1] / page.rect.width
    px_per_pt_y = image.shape[0] / page.rect.height
    lines = []
    for box, text, confidence in zip(result.boxes, result.txts, result.scores):
        text = (text or "").strip()
        if not text or float(confidence) < 0.25:
            continue
        xs = [float(pt[0]) for pt in box]
        ys = [float(pt[1]) for pt in box]
        x0, x1 = min(xs) / px_per_pt_x, max(xs) / px_per_pt_x
        y0, y1 = min(ys) / px_per_pt_y, max(ys) / px_per_pt_y
        # OCR has no reliable font metadata; retain geometry and estimate line
        # height so the existing layout-aware classifier can still operate.
        size = max(6.0, (y1 - y0) * 0.78)
        lines.append({
            "text": text,
            "runs": [{"text": text}],
            "size": round(size, 1),
            "bold": False,
            "bold_ratio": 0.0,
            "italic": False,
            "font": "OCR",
            "color": "#000000",
            "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
            "ocr": True,
            "ocr_confidence": round(float(confidence) * 100, 1),
            "page_w": page.rect.width,
            "page_h": page.rect.height,
        })

    # Reading order: top-to-bottom, then left-to-right.
    lines.sort(key=lambda ln: (round(ln["bbox"][1] / 4) * 4, ln["bbox"][0]))
    return lines


def extract_page(page, img_dir, pdfium_page=None):
    links = [l for l in page.get_links() if l.get("uri")]

    tables, table_rects = [], []
    try:
        for t in page.find_tables().tables:
            try:   # read each cell's text straight from the page: keeps spaces and decimals
                rows = [[cell_text(page, c) for c in row.cells] for row in t.rows]
            except Exception:
                rows = None
            if not rows or not any(any(r) for r in rows):
                rows = [[(c or "").strip() for c in row] for row in t.extract()]
            if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                table_rects.append(fitz.Rect(t.bbox))
                tables.append({"type": "table", "bbox": [round(v, 1) for v in t.bbox],
                               "header": rows[0], "rows": rows[1:]})
    except Exception:
        pass  # older PyMuPDF without find_tables -> tables fall back to text

    blocks_out = []  # each: {"kind": "text", "lines": [...]} or image node
    for bi, b in enumerate(page.get_text("dict")["blocks"]):
        if b["type"] == 1:  # image
            w, h = b["bbox"][2] - b["bbox"][0], b["bbox"][3] - b["bbox"][1]
            if w < 10 or h < 10:
                continue
            # A scanned page is commonly represented as one image covering
            # almost the entire page. Do not render that background again
            # beside the OCR text; retain smaller embedded figures/images.
            if (w * h) >= 0.70 * page.rect.width * page.rect.height:
                continue
            name = f"p{page.number + 1}_img{bi}.{b.get('ext', 'png')}"
            (img_dir / name).write_bytes(b["image"])
            blocks_out.append({"kind": "node", "node": {
                "type": "image", "src": f"images/{name}",
                "bbox": [round(v, 1) for v in b["bbox"]]}})
        elif b["type"] == 0:
            lines = []
            for l in b["lines"]:
                spans = [s for s in l["spans"]
                         if s["text"].strip() and not center_in(s["bbox"], table_rects)]
                if spans:
                    ln = make_line(spans, links)
                    ln["page_w"], ln["page_h"] = page.rect.width, page.rect.height
                    lines.append(ln)
            if lines:
                blocks_out.append({"kind": "text", "lines": lines})

    # Scanned pages usually have image blocks but no extractable text. OCR only
    # when native extraction is effectively empty; digital pages are unchanged.
    native_count = sum(len(b["lines"]) for b in blocks_out if b["kind"] == "text")
    native_chars = sum(len(ln["text"].strip()) for b in blocks_out if b["kind"] == "text" for ln in b["lines"])
    if native_count == 0 or native_chars < 20:
        if pdfium_page is None:
            raise RuntimeError("PDFium page renderer was not provided for OCR.")
        ocr_lines = ocr_page_lines(page, pdfium_page)
        if ocr_lines:
            blocks_out.append({"kind": "text", "lines": ocr_lines})

    for t in tables:
        blocks_out.append({"kind": "node", "node": t})
    return blocks_out


def find_repeated_margin_text(all_pages, page_heights):
    """Headers/footers: same text in top/bottom 8% on >= half the pages."""
    count = collections.Counter()
    for pno, blocks in enumerate(all_pages):
        H = page_heights[pno]
        seen = set()
        for b in blocks:
            if b["kind"] != "text":
                continue
            for ln in b["lines"]:
                y0, y1 = ln["bbox"][1], ln["bbox"][3]
                if y1 < H * 0.08 or y0 > H * 0.92:
                    seen.add(re.sub(r"\d+", "#", ln["text"]))
        count.update(seen)
    n = len(all_pages)
    return {t for t, c in count.items() if n >= 2 and c >= max(2, n * 0.5)}


# ----------------------------------------------------------------------------
# Pass 2: classify lines into heading levels / body
# ----------------------------------------------------------------------------
def build_nodes(all_pages, skip_text):
    """Segregate lines by role using classifier.py, then group them into cells."""
    ctx = clf.make_context(all_pages)
    for blocks in all_pages:
        clf.annotate_page(blocks, ctx, skip_text)      # Level 1 features + Level 2 rules
    clf.assign_levels(all_pages, ctx)                  # heading levels, title

    nodes = []
    for pno, blocks in enumerate(all_pages, start=1):
        items = []

        def add(node):
            items.append((node["bbox"][1], node["bbox"][0], node))

        for b in blocks:
            if b["kind"] == "node":
                n = dict(b["node"], page=pno)
                add(n)
                continue
            buf = []

            def flush():
                if buf:
                    runs = []
                    for ln in buf:
                        runs += ln["runs"] + [{"text": " "}]
                    c = [ln["cls"] for ln in buf]
                    add({"type": "paragraph", "text": " ".join(l["text"] for l in buf), "runs": runs,
                         "role": "list_item" if c[0]["label"] == "list_item" else "paragraph",
                         "confidence": min(x["confidence"] for x in c),
                         "needs_review": any(x["needs_review"] for x in c), "page": pno,
                         "bbox": [min(l["bbox"][0] for l in buf), buf[0]["bbox"][1],
                                  max(l["bbox"][2] for l in buf), buf[-1]["bbox"][3]]})
                    buf.clear()

            # A text block can contain multiple paragraphs. Native PDF extraction
            # often separates these into blocks, but OCR returns all detected
            # lines together. Use the vertical gap between OCR/native line boxes
            # to close the current paragraph before appending the next line.
            line_heights = [
                max(1.0, ln["bbox"][3] - ln["bbox"][1])
                for ln in b["lines"] if ln.get("text", "").strip()
            ]
            typical_line_height = (
                sorted(line_heights)[len(line_heights) // 2] if line_heights else 10.0
            )
            paragraph_gap_threshold = max(2.0, typical_line_height * 0.55)

            for ln in b["lines"]:
                # If the next line starts substantially below the previous
                # line, it is likely a new paragraph rather than a wrapped line.
                if buf:
                    vertical_gap = ln["bbox"][1] - buf[-1]["bbox"][3]
                    if vertical_gap > paragraph_gap_threshold:
                        flush()

                c = ln["cls"]
                lab = c["label"]
                if lab == "skip":
                    continue
                if lab == "heading":
                    flush()
                    add({"type": f"h{c['level']}", "level": c["level"], "role": c["role"],
                         "text": ln["text"], "runs": ln["runs"], "confidence": c["confidence"],
                         "needs_review": c["needs_review"], "page": pno, "bbox": ln["bbox"]})
                elif lab == "field":
                    flush()
                    fields = c["fields"]
                    for fd in fields:
                        text = f"{fd['label']}: {fd['value']}"
                        add({"type": "paragraph", "role": "field", "label": fd["label"], "value": fd["value"],
                             "text": text, "runs": ln["runs"] if len(fields) == 1 else [{"text": text}],
                             "confidence": c["confidence"], "needs_review": False, "page": pno,
                             "bbox": ln["bbox"]})
                elif lab == "signature":
                    flush()
                    add({"type": "paragraph", "role": "signature", "text": ln["text"], "runs": ln["runs"],
                         "confidence": c["confidence"], "needs_review": False, "page": pno, "bbox": ln["bbox"]})
                elif lab == "list_item":
                    flush()
                    buf.append(ln)
                else:                                   # body text (continues an open list item too)
                    buf.append(ln)
            flush()
        items.sort(key=lambda t: (round(t[0]), t[1]))   # reading order
        nodes += [n for _, _, n in items]
    for i, n in enumerate(nodes):
        n["id"] = i
    return nodes


# ----------------------------------------------------------------------------
# Render: reads the StructuredDocument; every element is its own cell.
# Settings panel: change a type (e.g. h1) -> applies to ALL cells of that type,
# or switch scope to "Only the selected cell" for a per-element override.
# ----------------------------------------------------------------------------
def runs_html(runs):
    out = []
    for r in runs:
        t = html.escape(r["text"])
        if r.get("uri"):
            out.append(f'<a href="{html.escape(r["uri"])}" target="_blank">{t}</a>')
        else:
            out.append(t)
    return "".join(out)


HTML_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PDF cells preview</title>
<style>
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f4f5f7}
.wrap{display:flex;gap:20px;align-items:flex-start;max-width:1200px;margin:0 auto;padding:20px}
#doc{flex:1;min-width:0}
#panel{width:280px;position:sticky;top:20px;background:#fff;border:1px solid #ccc;border-radius:8px;padding:14px;font-size:14px}
#panel h3{margin:0 0 6px}
#panel label{display:block;margin:10px 0 3px;font-weight:600}
#panel label.plain{font-weight:400;margin:4px 0}
#panel select,#panel input[type=number]{width:100%;padding:5px;box-sizing:border-box}
#panel input[type=color]{width:60px;height:30px;padding:0;border:1px solid #aaa}
#panel .row{display:flex;gap:16px}
#panel button{margin-top:12px;margin-right:6px;padding:6px 10px;cursor:pointer}
#selinfo{display:block;color:#555;margin-top:4px}
.cell{position:relative;background:#fff;border:1px solid #ddd;border-radius:6px;padding:18px 12px 10px;margin:8px 0;cursor:pointer}
.cell .tag{position:absolute;top:2px;left:8px;font:10px monospace;color:#888;text-transform:uppercase}
.cell.review{border-color:#e8a000}
.cell.sel{outline:2px solid #9db9f5;background:#f3f7ff}
.cell.picked{outline:3px solid #2f6fed}
.cell h1,.cell h2,.cell h3,.cell h4{margin:0}
table{border-collapse:collapse;width:100%}
td,th{border:1px solid #999;padding:4px 8px;text-align:left;vertical-align:top}
img{max-width:100%}
@media print{body{background:#fff}.wrap{display:block;padding:0;max-width:none}#panel{display:none}
.cell{border:none!important;outline:none!important;background:none!important;padding:0;margin:4px 0;break-inside:avoid}
.cell .tag{display:none}th,td{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
@media(max-width:800px){.wrap{flex-direction:column}#panel{width:auto;position:static}}
</style>
<style id="dyn"></style></head><body>
<div class="wrap">
<div id="doc">
__CELLS__
</div>
<div id="panel">
  <h3>Font settings</h3>
  <small>Choose an element type, or click a cell. Changes apply instantly.</small>
  <label>Element type</label><select id="type"></select>
  <label>Apply to</label>
  <label class="plain"><input type="radio" name="scope" value="type" id="scopeType" checked> All cells of this type</label>
  <label class="plain"><input type="radio" name="scope" value="cell" id="scopeCell" disabled> Only the selected cell</label>
  <small id="selinfo">No cell selected</small>
  <label>Font family</label><select id="family"></select>
  <label>Font size (px)</label><input type="number" id="size" min="6" max="80" step="0.5">
  <div class="row"><label class="plain"><input type="checkbox" id="bold"> Bold</label><label class="plain"><input type="checkbox" id="italic"> Italic</label></div>
  <label>Text color</label><input type="color" id="color">
  <div id="bgrow"><label>Background</label><input type="color" id="bg"></div>
  <label>Link color</label><input type="color" id="link">
  <button id="clear">Clear cell override</button>
  <button id="reset">Reset all</button>
  <button id="save">Download styles.json</button>
  <button id="print">Print / Save as PDF</button>
  <button id="srvexport" style="display:none">Export PDF (server)</button>
</div></div>
<script>
const DEFAULTS = __STYLES__;
const OVERRIDES0 = __OVERRIDES__;
const TYPES = ["h1","h2","h3","h4","paragraph","table_header","table_cell"];
const NAMES = {h1:"Heading (h1)",h2:"Subheading (h2)",h3:"Sub-subheading (h3)",h4:"Level 4 (h4)",
  paragraph:"Content / paragraph",table_header:"Table header",table_cell:"Table cell"};
const FONTS = ["Arial","Calibri","Georgia","Times New Roman","Verdana","Tahoma","Courier New","Trebuchet MS"];
const clone = o => JSON.parse(JSON.stringify(o));
let styles = clone(DEFAULTS), overrides = clone(OVERRIDES0);
let cur = "h1", selEid = null;
const $ = id => document.getElementById(id);
const scope = () => document.querySelector("input[name=scope]:checked").value;

function eff(t, eid){
  const s = Object.assign({}, styles[t]), o = (eid && overrides[eid]) || {};
  for (const k in o) s[k === "text_color" ? "color" : k] = o[k];
  return s;
}
function apply(){
  let css = "";
  for (const t of TYPES){
    const s = styles[t];
    css += `[data-style="${t}"]{font-family:'${s.font_family}',sans-serif;font-size:${s.font_size}px;` +
           `font-weight:${s.bold ? 700 : 400};font-style:${s.italic ? "italic" : "normal"};color:${s.color};` +
           `${s.background ? "background:" + s.background + ";" : ""}}\n`;
  }
  for (const eid in overrides){
    const o = overrides[eid]; let d = "";
    if (o.font_family) d += `font-family:'${o.font_family}',sans-serif;`;
    if (o.font_size) d += `font-size:${o.font_size}px;`;
    if ("bold" in o) d += `font-weight:${o.bold ? 700 : 400};`;
    if ("italic" in o) d += `font-style:${o.italic ? "italic" : "normal"};`;
    if ("underline" in o) d += `text-decoration:${o.underline ? "underline" : "none"};`;
    if (o.alignment) d += `text-align:${o.alignment};`;
    if (o.text_color) d += `color:${o.text_color};`;
    if (o.background) d += `background:${o.background};`;
    if (d) css += `[data-eid="${eid}"][data-style]{${d}}\n`;
  }
  css += `.cell a{color:${styles.link_color}}`;
  $("dyn").textContent = css;
}
function ensureFont(name){
  if (![...$("family").options].some(o => o.value === name)){
    const o = document.createElement("option"); o.value = o.textContent = name; $("family").appendChild(o);
  }
}
function load(){
  const s = eff(cur, scope() === "cell" ? selEid : null);
  ensureFont(s.font_family);
  $("type").value = cur; $("family").value = s.font_family; $("size").value = s.font_size;
  $("bold").checked = !!s.bold; $("italic").checked = !!s.italic; $("color").value = s.color;
  $("bgrow").style.display = (cur === "table_header") ? "block" : "none";
  if (s.background) $("bg").value = s.background;
  $("link").value = styles.link_color;
  $("scopeCell").disabled = !selEid;
  $("selinfo").textContent = selEid ? `Selected cell: ${selEid}` + (overrides[selEid] ? " (has override)" : "") : "No cell selected";
  document.querySelectorAll(".cell").forEach(c => {
    c.classList.toggle("sel", (c.dataset.types || "").split(" ").includes(cur));
    c.classList.toggle("picked", !!selEid && !!c.querySelector(`[data-eid="${selEid}"]`));
  });
}
function set(prop, val){
  if (scope() === "cell" && selEid){
    (overrides[selEid] = overrides[selEid] || {})[prop === "color" ? "text_color" : prop] = val;
  } else {
    styles[cur][prop] = val;
  }
  apply(); load();
}

TYPES.forEach(t => { const o = document.createElement("option"); o.value = t; o.textContent = NAMES[t]; $("type").appendChild(o); });
FONTS.forEach(f => { const o = document.createElement("option"); o.value = o.textContent = f; $("family").appendChild(o); });
document.querySelectorAll("input[name=scope]").forEach(r => r.onchange = load);
$("type").onchange = e => { cur = e.target.value; selEid = null; $("scopeType").checked = true; load(); };
$("family").onchange = e => set("font_family", e.target.value);
$("size").oninput = e => { const v = parseFloat(e.target.value); if (v > 0) set("font_size", v); };
$("bold").onchange = e => set("bold", e.target.checked);
$("italic").onchange = e => set("italic", e.target.checked);
$("color").oninput = e => set("color", e.target.value);
$("bg").oninput = e => set("background", e.target.value);
$("link").oninput = e => { styles.link_color = e.target.value; apply(); };
$("print").onclick = () => window.print();
$("clear").onclick = () => { if (selEid){ delete overrides[selEid]; apply(); load(); } };
$("reset").onclick = () => { styles = clone(DEFAULTS); overrides = clone(OVERRIDES0); apply(); load(); };
function payload(){
  const out = Object.assign({}, styles);
  const ov = {}; for (const k in overrides) if (Object.keys(overrides[k]).length) ov[k] = overrides[k];
  if (Object.keys(ov).length) out.element_overrides = ov;
  return out;
}
function saveBlob(blob, name){
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
}
$("save").onclick = () => saveBlob(new Blob([JSON.stringify(payload(), null, 2)], {type: "application/json"}), "styles.json");
if (location.protocol.startsWith("http") && location.pathname.startsWith("/jobs/")) $("srvexport").style.display = "inline-block";
$("srvexport").onclick = async () => {
  $("srvexport").textContent = "Exporting...";
  try {
    const r = await fetch("export", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload())});
    if (!r.ok) throw new Error(await r.text());
    saveBlob(await r.blob(), "exported.pdf");
  } catch (err) { alert("Export failed: " + err.message); }
  $("srvexport").textContent = "Export PDF (server)";
};
$("doc").addEventListener("click", e => {
  const el = e.target.closest("[data-style]");
  if (!el) return;
  cur = el.dataset.style;
  selEid = el.dataset.eid || null;
  if (!selEid) $("scopeType").checked = true;
  load();
});
apply(); load();
</script></body></html>"""


def tag_text(e):
    parts = [e.type]
    role = e.metadata.get("role")
    if role and role not in (e.type, "paragraph"):
        parts.append(role.replace("_", " "))
    if e.metadata.get("needs_review"):
        parts.append("check")
    return html.escape(" \u00b7 ".join(parts))


def render_html(doc, styles):
    """Render a StructuredDocument. Each element becomes one cell."""
    cells = []
    for e in doc.elements:
        t = e.type
        rv = " review" if e.metadata.get("needs_review") else ""
        runs = e.metadata.get("runs") or [{"text": str(e.content)}]
        if t in ("h1", "h2", "h3", "h4"):
            cells.append(f'<div class="cell{rv}" data-types="{t}"><span class="tag">{tag_text(e)}</span>'
                         f'<{t} data-style="{t}" data-eid="{e.id}">{runs_html(runs)}</{t}></div>')
        elif t == "image":
            cells.append(f'<div class="cell" data-types=""><span class="tag">image</span>'
                         f'<img src="{html.escape(str(e.content))}"></div>')
        elif t == "table":
            header = e.content.get("header", [])
            rows = e.content.get("rows", [])
            h = "".join(f'<th data-style="table_header">{html.escape(c)}</th>' for c in header)
            body = "".join("<tr>" + "".join(f'<td data-style="table_cell">{html.escape(c)}</td>' for c in r) + "</tr>"
                           for r in rows)
            cells.append(f'<div class="cell" data-types="table_header table_cell"><span class="tag">table</span>'
                         f'<table><tr>{h}</tr>{body}</table></div>')
        elif isinstance(e.content, str):      # paragraph and any other text-like element
            cells.append(f'<div class="cell{rv}" data-types="paragraph"><span class="tag">{tag_text(e)}</span>'
                         f'<div data-style="paragraph" data-eid="{e.id}">{runs_html(runs)}</div></div>')
    overrides = {e.id: e.style for e in doc.elements if e.style}
    def js(o):
        return json.dumps(o).replace("</", "<\\/")
    return (HTML_TEMPLATE.replace("__STYLES__", js(styles))
                         .replace("__OVERRIDES__", js(overrides))
                         .replace("__CELLS__", "\n".join(cells)))


# ----------------------------------------------------------------------------
# Export: rebuild a PDF from the cells, using the same fonts/sizes/colors as
# the preview (type styles + per-cell overrides).  Needs:  pip install reportlab
# ----------------------------------------------------------------------------
FONT_DIRS = [r"C:\Windows\Fonts", "/usr/share/fonts", "/usr/share/fonts/truetype",
             "/Library/Fonts", "/System/Library/Fonts/Supplemental",
             str(Path.home() / "Library/Fonts"), str(Path.home() / ".fonts")]
# family -> (regular, bold, italic, bold-italic) font files
FONT_FILES = {
    "Arial": ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"),
    "Calibri": ("calibri.ttf", "calibrib.ttf", "calibrii.ttf", "calibriz.ttf"),
    "Georgia": ("georgia.ttf", "georgiab.ttf", "georgiai.ttf", "georgiaz.ttf"),
    "Times New Roman": ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"),
    "Verdana": ("verdana.ttf", "verdanab.ttf", "verdanai.ttf", "verdanaz.ttf"),
    "Tahoma": ("tahoma.ttf", "tahomabd.ttf", None, None),
    "Courier New": ("cour.ttf", "courbd.ttf", "couri.ttf", "courbi.ttf"),
    "Trebuchet MS": ("trebuc.ttf", "trebucbd.ttf", "trebucit.ttf", "trebucbi.ttf"),
}
# built-in PDF fonts used when a TrueType file is not found
BUILTIN = {
    "sans": ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"),
    "serif": ("Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"),
    "mono": ("Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique"),
}
_font_cache = {}


def _find_font_file(name):
    if not name:
        return None
    for d in FONT_DIRS:
        p = Path(d)
        if p.is_dir():
            for f in p.rglob("*") if "share" in d else p.iterdir():
                if f.name.lower() == name.lower():
                    return str(f)
    return None


def resolve_font(family, bold, italic):
    """Return a registered PDF font name for family + bold/italic."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    idx = (1 if bold else 0) + (2 if italic else 0)
    key = (family, idx)
    if key in _font_cache:
        return _font_cache[key]
    name = None
    files = FONT_FILES.get(family)
    if files:
        path = _find_font_file(files[idx]) or _find_font_file(files[0])
        if path:
            name = f"{family.replace(' ', '')}-{idx}"
            try:
                pdfmetrics.registerFont(TTFont(name, path))
            except Exception:
                name = None
    if not name:
        kind = "mono" if "Courier" in family else "serif" if family in ("Georgia", "Times New Roman") else "sans"
        name = BUILTIN[kind][idx]
    _font_cache[key] = name
    return name


def effective_style(styles, element):
    """Type style + this element's own overrides (text_color -> color)."""
    s = dict(styles.get(element.type) or styles.get("paragraph"))
    for k, v in (element.style or {}).items():
        s["color" if k == "text_color" else k] = v
    return s


def _markup(runs, link_color):
    out = []
    for r in runs:
        t = html.escape(r["text"])
        if r.get("uri"):
            out.append(f'<a href="{html.escape(r["uri"])}" color="{link_color}"><u>{t}</u></a>')
        else:
            out.append(t)
    return "".join(out)


def export_pdf(doc, styles, out_dir, pdf_path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    out_dir = Path(out_dir)
    margin = 50
    frame_w = A4[0] - 2 * margin
    align = {"left": TA_LEFT, "center": TA_CENTER, "right": TA_RIGHT}
    link_color = styles.get("link_color", "#0645ad")

    def pstyle(s, name, space_before=0, space_after=6):
        size = float(s["font_size"])
        return ParagraphStyle(
            name, fontName=resolve_font(s["font_family"], s.get("bold"), s.get("italic")),
            fontSize=size, leading=size * 1.3, textColor=colors.HexColor(s["color"]),
            alignment=align.get(s.get("alignment", "left"), TA_LEFT),
            spaceBefore=space_before, spaceAfter=space_after)

    story = []
    for e in doc.elements:
        t = e.type
        runs = e.metadata.get("runs") or [{"text": str(e.content)}]
        if t in ("h1", "h2", "h3", "h4"):
            s = effective_style(styles, e)
            story.append(Paragraph(_markup(runs, link_color),
                                   pstyle(s, f"{t}_{e.id}", s["font_size"] * 0.6, s["font_size"] * 0.3)))
        elif t == "image":
            path = out_dir / str(e.content)
            if path.exists():
                w, h = ImageReader(str(path)).getSize()
                if e.bbox:                       # keep the original size on the page
                    w, h = e.bbox[2] - e.bbox[0], e.bbox[3] - e.bbox[1]
                scale = min(1.0, frame_w / w)
                story += [Image(str(path), width=w * scale, height=h * scale, hAlign="LEFT"), Spacer(1, 6)]
        elif t == "table":
            hs, cs = styles["table_header"], styles["table_cell"]
            hp, cp = pstyle(hs, f"th_{e.id}", 0, 0), pstyle(cs, f"td_{e.id}", 0, 0)
            header = e.content.get("header", [])
            rows = e.content.get("rows", [])
            data = [[Paragraph(html.escape(c), hp) for c in header]] + \
                   [[Paragraph(html.escape(c), cp) for c in r] for r in rows]
            ncol = max(len(r) for r in data)
            lens = [max([len(str(x)) for x in col] + [4]) for col in zip(*([header] + rows))] if header else [10] * ncol
            total = sum(lens)
            widths = [max(frame_w * 0.08, frame_w * l / total) for l in lens]
            scale = frame_w / sum(widths)
            tbl = Table(data, colWidths=[w * scale for w in widths], repeatRows=1)
            tbl.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(hs.get("background", "#e8e8e8"))),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
            story += [tbl, Spacer(1, 8)]
        elif isinstance(e.content, str):
            story.append(Paragraph(_markup(runs, link_color), pstyle(effective_style(styles, e), f"p_{e.id}")))

    SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=margin, rightMargin=margin,
                      topMargin=margin, bottomMargin=margin,
                      title=Path(doc.source_pdf).stem or "Exported").build(story)
    return Path(pdf_path)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def inspect(pdf_path):
    """Step 2 helper: print every line with its font details."""
    doc = fitz.open(pdf_path)
    for page in doc:
        print(f"\n=== Page {page.number + 1} ===")
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                spans = [s for s in l["spans"] if s["text"].strip()]
                if spans:
                    ln = make_line(spans, [])
                    print(f"size={ln['size']:<5} bold={ln['bold']!s:<5} font={ln['font'][:22]:<22} | {ln['text'][:80]}")


def convert(pdf_path, out_dir, make_pdf=False):
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    styles_file = out / "styles.json"
    if not styles_file.exists():
        styles_file.write_text(json.dumps(DEFAULT_STYLES, indent=2))
    loaded = json.loads(styles_file.read_text())
    overrides = loaded.pop("element_overrides", {})   # per-cell overrides saved from the preview
    styles = {**DEFAULT_STYLES, **loaded}

    doc = fitz.open(pdf_path)
    try:
        import pypdfium2 as pdfium
    except ImportError:
        pdfium = None
    pdfium_doc = pdfium.PdfDocument(str(pdf_path)) if pdfium is not None else None

    all_pages = []
    for page_index, page in enumerate(doc):
        pdfium_page = pdfium_doc[page_index] if pdfium_doc is not None else None
        all_pages.append(extract_page(page, out / "images", pdfium_page))
    ocr_pages = sum(any(ln.get("ocr") for b in blocks if b["kind"] == "text" for ln in b["lines"])
                    for blocks in all_pages)
    if ocr_pages:
        print(f"OCR used on {ocr_pages} page(s).")
    skip = find_repeated_margin_text(all_pages, [p.rect.height for p in doc])
    nodes = build_nodes(all_pages, skip)
    clf.write_csv(all_pages, out / "classification.csv")   # why each line got its label

    # Convert real extracted PDF nodes into the structured data model.
    structured_doc = StructuredDocument(str(Path(pdf_path).resolve()), type_styles=styles)
    for node in nodes:
        structured_doc.add_extracted_item(node)
    for eid, o in overrides.items():
        structured_doc.update_style(eid, **o)

    (out / "structure.json").write_text(
        json.dumps(nodes, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    structured_doc.save_json(out / "structured_elements.json")
    (out / "preview.html").write_text(render_html(structured_doc, styles), encoding="utf-8")

    if make_pdf:
        pdf_out = export_pdf(structured_doc, styles, out, out / "exported.pdf")
        print(f"Exported PDF -> {pdf_out}")

    counts = collections.Counter(n["type"] for n in nodes)
    if pdfium_doc is not None:
        pdfium_doc.close()
    doc.close()
    print(f"Done: {dict(counts)}  ->  {out}/preview.html")
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out", default="output")
    ap.add_argument("--pdf-export", dest="make_pdf", action="store_true", help="also export exported.pdf (needs: pip install reportlab)")
    ap.add_argument("--inspect", action="store_true", help="print font info per line and exit")
    a = ap.parse_args()
    inspect(a.pdf) if a.inspect else convert(a.pdf, a.out, a.make_pdf)
