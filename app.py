"""
Web upload for the PDF -> cells project.

Install (once):
    pip install fastapi uvicorn python-multipart pymupdf reportlab
Run:
    python -m uvicorn app:app --reload
Open:
    http://127.0.0.1:8000

Flow: upload a PDF -> it is converted to cells -> preview page with the font
settings panel opens -> "Export PDF (server)" downloads the styled PDF.
Each upload gets its own folder under ./jobs/<id>/ (keep it private; this
app is meant for local use - add login/HTTPS before putting it on a network).
"""
import re
import shutil
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from pdf_to_cells_integrated import DEFAULT_STYLES, convert

JOBS = Path("jobs")
JOBS.mkdir(exist_ok=True)
MAX_MB = 25
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
FONTS = {"Arial", "Calibri", "Georgia", "Times New Roman", "Verdana", "Tahoma",
         "Courier New", "Trebuchet MS", "Helvetica"}

app = FastAPI(title="PDF to cells")

INDEX = """<!doctype html><html><head><meta charset="utf-8"><title>PDF to cells</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:Segoe UI,Arial,sans-serif;background:#f4f5f7;margin:0}
.box{max-width:520px;margin:12vh auto;background:#fff;border:1px solid #ddd;border-radius:10px;padding:28px}
h1{margin:0 0 6px;font-size:24px} p{color:#555}
input[type=file]{display:block;margin:18px 0;width:100%}
button{background:#2f6fed;color:#fff;border:0;border-radius:6px;padding:10px 18px;font-size:15px;cursor:pointer}
#msg{margin-top:14px;color:#2f6fed;display:none}
</style></head><body><div class="box">
<h1>PDF to cells</h1>
<p>Upload an agreement, contract or purchase order (text-based PDF, up to 25 MB).</p>
<form action="/upload" method="post" enctype="multipart/form-data"
      onsubmit="document.getElementById('msg').style.display='block';this.querySelector('button').disabled=true">
  <input type="file" name="file" accept="application/pdf,.pdf" required>
  <button type="submit">Upload and convert</button>
</form>
<div id="msg">Converting... this can take a few seconds.</div>
</div></body></html>"""


def clean_style(src):
    """Keep only known keys with safe values (the page embeds these)."""
    out = {}
    for k, v in (src or {}).items():
        if k == "font_family" and v in FONTS:
            out[k] = v
        elif k == "font_size" and isinstance(v, (int, float)) and not isinstance(v, bool) and 4 <= v <= 200:
            out[k] = v
        elif k in ("bold", "italic", "underline") and isinstance(v, bool):
            out[k] = v
        elif k in ("color", "text_color", "background") and isinstance(v, str) and HEX.match(v):
            out[k] = v
        elif k == "alignment" and v in ("left", "center", "right"):
            out[k] = v
    return out


def sanitize(data):
    styles = {}
    for t, base in DEFAULT_STYLES.items():
        if t == "link_color":
            v = data.get(t)
            styles[t] = v if isinstance(v, str) and HEX.match(v) else base
        else:
            src = data.get(t)
            styles[t] = {**base, **clean_style(src if isinstance(src, dict) else {})}
    overrides = {}
    raw = data.get("element_overrides")
    if isinstance(raw, dict):
        for eid, o in raw.items():
            if re.fullmatch(r"el_\d{4,}", str(eid)) and isinstance(o, dict):
                c = clean_style(o)
                if c:
                    overrides[eid] = c
    if overrides:
        styles["element_overrides"] = overrides
    return styles


def job_folder(job):
    if not JOB_ID.match(job):
        raise HTTPException(404, "Not found")
    folder = JOBS / job
    if not (folder / "input.pdf").exists():
        raise HTTPException(404, "Not found")
    return folder


@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    if file.file.read(5) != b"%PDF-":
        raise HTTPException(400, "Please upload a PDF file.")
    file.file.seek(0)
    job = uuid.uuid4().hex
    folder = JOBS / job
    folder.mkdir()
    too_big, size = False, 0
    with open(folder / "input.pdf", "wb") as f:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_MB * 1024 * 1024:
                too_big = True
                break
            f.write(chunk)
    if too_big:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(413, f"File is larger than {MAX_MB} MB.")
    try:
        counts = convert(str(folder / "input.pdf"), str(folder))
    except Exception as err:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(422, f"Could not read this PDF: {err}")
    if not counts:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(422, "No text found. Scanned PDFs need OCR first (for example with ocrmypdf).")
    return RedirectResponse(f"/jobs/{job}/preview.html", status_code=303)


@app.post("/jobs/{job}/export")
def export(job: str, data: dict = Body(...)):
    """Receive the styles chosen in the preview, rebuild, and return the PDF."""
    folder = job_folder(job)
    import json
    (folder / "styles.json").write_text(json.dumps(sanitize(data), indent=2))
    try:
        convert(str(folder / "input.pdf"), str(folder), make_pdf=True)
    except ImportError:
        raise HTTPException(500, "reportlab is not installed. Run: pip install reportlab")
    return FileResponse(folder / "exported.pdf", media_type="application/pdf", filename="exported.pdf")


# Mounted last so the routes above win. Serves jobs/<id>/preview.html, images/, exported.pdf.
app.mount("/jobs", StaticFiles(directory=JOBS), name="jobs")
