"""
classifier.py - segregate PDF lines into roles, in three levels.

Level 1  Structural features  (measured for every line)
         font size, weight, family, coordinates, distance from previous block,
         text length, numbering pattern, page position, alignment, indentation
Level 2  Semantic rules       (what the line means in agreements / contracts / POs)
         known section words, field labels ("PO Number:"), signature lines,
         list markers, clause numbering, document title
Level 3  Review hook          lines whose score is close to the threshold are
         flagged needs_review=True; send those to an LLM / RAG reviewer later.

Roles produced:  title, section_heading, clause_heading, field, list_item,
                 signature, paragraph          (headings also get level 1-4)
"""
import collections
import csv
import re
import statistics

PAGE_NUM = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.I)

# ---- Level 2 vocabulary ----------------------------------------------------
NUM_DECIMAL = re.compile(r"^(\d+(?:\.\d+){0,3})[.)]?\s+\S")
NUM_ARTICLE = re.compile(r"^(ARTICLE|SECTION|CLAUSE|SCHEDULE|ANNEXURE|ANNEX|EXHIBIT|APPENDIX)\b", re.I)
NUM_ALPHA = re.compile(r"^\(([a-z]|[ivx]{1,4})\)\s+\S", re.I)
NUM_BULLET = re.compile(r"^[\u2022\u25cf\u25aa\u25e6\u00b7*\-\u2013\u2014]\s+\S")
STRUCTURAL = re.compile(
    r"^(WHEREAS|RECITALS|BACKGROUND|NOW,? THEREFORE|IN WITNESS WHEREOF|TERMS AND CONDITIONS)\b", re.I)

SECTION_WORDS = {
    "abstract", "background", "recitals", "whereas", "definitions", "interpretation", "scope",
    "scope of work", "services", "term", "term and termination", "termination", "fees", "payment",
    "payment terms", "invoicing", "confidentiality", "intellectual property", "warranties",
    "representations and warranties", "indemnity", "indemnification", "limitation of liability",
    "liability", "force majeure", "governing law", "jurisdiction", "dispute resolution",
    "arbitration", "notices", "assignment", "amendment", "severability", "entire agreement",
    "miscellaneous", "general", "parties", "signatures", "signature", "order details",
    "terms and conditions", "delivery", "shipping", "inspection", "acceptance", "returns",
    "cancellation", "special instructions", "order summary", "line items",
}
FIELD_LABELS = {
    "po number", "po no", "po #", "purchase order number", "purchase order no", "order number",
    "order no", "date", "order date", "po date", "delivery date", "due date", "effective date",
    "expiry date", "vendor", "supplier", "seller", "buyer", "customer", "client", "bill to",
    "ship to", "shipping address", "billing address", "payment terms", "shipping method",
    "currency", "reference", "ref", "contact", "phone", "email", "attention", "attn", "tax id",
    "gst", "gstin", "pan", "subtotal", "total", "tax", "party",
}
_FIELD_ALT = "|".join(re.escape(l) for l in sorted(FIELD_LABELS, key=len, reverse=True))
FIELD_RE = re.compile(rf"(?:(?<=\s)|^)(?P<label>{_FIELD_ALT})\s*:", re.I)
GENERIC_FIELD = re.compile(r"^(?P<label>[A-Za-z][A-Za-z0-9 /&.#()'-]{0,30}?)\s*:\s*(?P<value>\S.*)$")
SIGNATURE_START = ("signed", "signature", "authorized signature", "authorised signature",
                   "for and on behalf of", "witness")

HEADING_THRESHOLD = 3.0


# ---- helpers ----------------------------------------------------------------
def numbering_info(text):
    t = text.strip()
    m = NUM_DECIMAL.match(t)
    if m:
        return "decimal", min(m.group(1).count(".") + 1, 4)
    if NUM_ARTICLE.match(t):
        return "article", 1
    if NUM_ALPHA.match(t):
        return "alpha", 0
    if NUM_BULLET.match(t):
        return "bullet", 0
    return "", 0


def core_text(text):
    t = re.sub(r"^(\d+(\.\d+)*[.)]?|\([a-z0-9]+\))\s+", "", text.strip(), flags=re.I)
    return t.strip(" :.-\u2013\u2014").lower()


def detect_fields(text):
    """'PO Number: X  Date: Y'  ->  [{'label':'PO Number','value':'X'}, ...]"""
    ms = list(FIELD_RE.finditer(text))
    if ms and ms[0].start() == 0:
        out = []
        for i, m in enumerate(ms):
            end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
            out.append({"label": text[m.start():m.end() - 1].strip(),
                        "value": text[m.end():end].strip()})
        if all(o["value"] for o in out):
            return out
    g = GENERIC_FIELD.match(text)
    if g and len(g["label"].split()) <= 3 and len(text.split()) <= 12 and not text.rstrip().endswith("."):
        return [{"label": g["label"].strip(), "value": g["value"].strip()}]
    return []


# ---- Level 1: context + features ---------------------------------------------
def make_context(all_pages):
    size_chars, left, right = collections.Counter(), collections.Counter(), []
    bold_chars = collections.Counter()
    for blocks in all_pages:
        for b in blocks:
            if b["kind"] != "text":
                continue
            for ln in b["lines"]:
                size_chars[ln["size"]] += len(ln["text"])
                if ln["bold"]:
                    bold_chars[ln["size"]] += len(ln["text"])
    body = size_chars.most_common(1)[0][0] if size_chars else 11
    for blocks in all_pages:
        for b in blocks:
            if b["kind"] != "text":
                continue
            for ln in b["lines"]:
                if ln["size"] == body and len(ln["text"].split()) >= 5:
                    left[round(ln["bbox"][0] / 2) * 2] += len(ln["text"])
                    right.append(ln["bbox"][2])
    return {"body_size": body,
            "left": left.most_common(1)[0][0] if left else 50,
            "right": max(right) if right else 545,
            "body_bold": bold_chars[body] > 0.6 * size_chars[body] if size_chars else False}


def annotate_page(blocks, ctx, skip_text):
    """Measure the Level-1 features and classify every line of one page."""
    lines = [ln for b in blocks if b["kind"] == "text" for ln in b["lines"]]
    for ln in lines:
        key = re.sub(r"\d+", "#", ln["text"])
        ln["skip"] = key in skip_text or bool(PAGE_NUM.match(ln["text"].strip()))
    live = sorted([l for l in lines if not l["skip"]], key=lambda l: (round(l["bbox"][1]), l["bbox"][0]))
    for i, ln in enumerate(live):
        x0, y0, x1, y1 = ln["bbox"]
        pw, ph = ln.get("page_w", 595), ln.get("page_h", 842)
        prev = live[i - 1] if i else None
        nxt = live[i + 1] if i + 1 < len(live) else None
        gap_before = max(0.0, y0 - prev["bbox"][3]) if prev and abs(y0 - prev["bbox"][1]) > 2 else 0.0
        gap_after = max(0.0, nxt["bbox"][1] - y1) if nxt and abs(y0 - nxt["bbox"][1]) > 2 else 0.0
        cx = (x0 + x1) / 2
        if abs(cx - pw / 2) <= pw * 0.03 and x0 - ctx["left"] > pw * 0.06:
            align = "center"
        elif x1 >= ctx["right"] - 3 and x0 - ctx["left"] > pw * 0.25:
            align = "right"
        else:
            align = "left"
        text = ln["text"]
        ntype, ndepth = numbering_info(text)
        alpha = [c for c in text if c.isalpha()]
        # Use a character-weighted bold ratio where the extractor provides it.
        # This distinguishes a fully bold heading from a bold lead-in followed
        # by regular paragraph text on the same line.
        bold_ratio = float(ln.get("bold_ratio", 1.0 if ln.get("bold") else 0.0))
        is_bold_line = bold_ratio >= 0.80
        ln["f"] = {
            "font_size": ln["size"], "size_ratio": round(ln["size"] / ctx["body_size"], 2),
            "bold": is_bold_line, "bold_ratio": round(bold_ratio, 2),
            "italic": ln["italic"], "font_family": ln["font"],
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "gap_before": round(gap_before, 1), "gap_after": round(gap_after, 1),
            "n_chars": len(text), "n_words": len(text.split()),
            "numbering": f"{ntype}:{ndepth}" if ntype else "",
            "y_rel": round(y0 / ph, 3), "first_on_page": i == 0,
            "alignment": align, "indent": round(x0 - ctx["left"], 1),
            "is_caps": bool(alpha) and text.upper() == text,
            "ends_sentence": text.rstrip().endswith((".", ";", ",")),
            "_ntype": ntype, "_ndepth": ndepth,
        }
        ln["cls"] = classify(ln, ctx)
    for ln in lines:
        if ln["skip"]:
            ln["cls"] = {"label": "skip", "score": 0, "confidence": 1.0, "needs_review": False, "reasons": []}


# ---- Level 1 + 2: classification ------------------------------------------------
def classify(ln, ctx):
    f, text = ln["f"], ln["text"]
    words = f["n_words"]
    low = text.strip().lower()
    base = {"score": 0.0, "confidence": 0.9, "needs_review": False, "reasons": [], "depth": 0}

    # Level 2 rule: signature lines
    if re.search(r"_{4,}", text) or low.startswith(SIGNATURE_START):
        return dict(base, label="signature", reasons=["signature line"])

    # Level 2 rule: "Label: value" fields (PO number, date, vendor, ...)
    fields = detect_fields(text)
    if fields:
        return dict(base, label="field", fields=fields, reasons=["label: value"])

    # Level 2 rule: list items  (a) ...  / bullets
    if f["_ntype"] in ("alpha", "bullet"):
        return dict(base, label="list_item", reasons=[f"{f['_ntype']} marker"])

    # Heading score: structural evidence + semantic evidence
    s, why = 0.0, []
    r = f["size_ratio"]
    if r >= 1.25:
        s += 3; why.append("font much larger")
    elif r >= 1.08:
        s += 1.5; why.append("font larger")
    # Bold is useful even when the document body is mostly bold; combine it
    # with short-line/spacing evidence instead of suppressing it globally.
    if f["bold"]:
        s += 1.5 if not ctx["body_bold"] else 0.75
        why.append("bold")
        # Many academic notes use the same point size for headings and body.
        # A complete short bold line, separated from the previous line, is a
        # strong heading signal even without numbering or a keyword match.
        if words <= 12 and not f["ends_sentence"] and not text.rstrip().endswith(":"):
            s += 1.0
            why.append("short standalone bold line")
    if f["is_caps"] and words <= 10:
        s += 1; why.append("caps")
    if f["_ntype"] in ("decimal", "article"):
        s += 2; why.append("numbered")
    core = core_text(text)
    kw = (core in SECTION_WORDS
          or any(core.startswith(word + " ") for word in SECTION_WORDS if len(word) >= 5)
          or bool(STRUCTURAL.match(text))) and words <= 12
    if kw:
        s += 1.5; why.append("known section word")
    if f["gap_before"] > 0.8 * f["font_size"]:
        s += 1; why.append("space above")
    if f["alignment"] == "center" and words <= 10:
        s += 1; why.append("centered")
    if words > 15:
        s -= 3; why.append("long text")
    if f["ends_sentence"]:
        s -= 2; why.append("ends like a sentence")

    # A short numbered section/clause line is strong heading evidence.
    if f["_ntype"] in ("decimal", "article") and words <= 12 and not f["ends_sentence"]:
        s = max(s, HEADING_THRESHOLD + 0.5)
        why.append("short numbered section")
    is_head = s >= HEADING_THRESHOLD
    margin = abs(s - HEADING_THRESHOLD)
    conf = round(min(1.0, 0.5 + margin / 5), 2)
    depth = f["_ndepth"] if f["_ntype"] == "decimal" else 1 if (f["_ntype"] == "article" or STRUCTURAL.match(text)) else 0
    return dict(base, label="heading" if is_head else "paragraph", score=round(s, 1), confidence=conf,
                needs_review=margin < 1.0, reasons=why, depth=depth)


# ---- heading levels ---------------------------------------------------------------
def assign_levels(all_pages, ctx):
    """
    Assign hierarchical H1-H4 levels across the whole document.

    Numbering is the strongest structural signal. A standalone document title
    shifts numbered sections down one level (e.g. title=H1, 1.=H2, 1.1=H3).
    For unnumbered headings, use their visual size relative to the document's
    other heading styles. This avoids treating every distinct font size as a
    separate level when numbering already establishes the hierarchy.
    """
    heads = []
    for pno, blocks in enumerate(all_pages):
        for b in blocks:
            if b["kind"] == "text":
                heads.extend((pno, ln) for ln in b["lines"]
                             if ln.get("cls", {}).get("label") == "heading")
    heads.sort(key=lambda t: (t[0], round(t[1]["bbox"][1]), t[1]["bbox"][0]))
    lines = [ln for _, ln in heads]
    if not lines:
        return

    # A standalone first line, larger than the following headings, is the
    # document title. Do not mistake a numbered first section for a title.
    first = lines[0]
    later = lines[1:]
    title = bool(
        later
        and not first["f"]["_ntype"]
        and first["cls"].get("depth", 0) == 0
        and first["size"] > max(x["size"] for x in later)
        and first["f"]["n_words"] <= 12
    )
    title_offset = 1 if title else 0
    if title:
        first["cls"]["level"] = 1
        first["cls"]["role"] = "title"

    numbered = [ln for ln in lines if ln["f"]["_ntype"] in ("decimal", "article")]
    # Use the explicit numbering depth as the backbone of the hierarchy.
    # E.g. with a document title: 1 -> H2, 1.1 -> H3, 1.1.1 -> H4.
    depth_levels = {}
    for ln in numbered:
        depth = ln["f"]["_ndepth"] if ln["f"]["_ntype"] == "decimal" else 1
        level = max(1, depth + title_offset)
        ln["cls"]["level"] = min(level, 4)
        depth_levels.setdefault(depth, []).append(ln["size"])

    # Map observed heading font sizes to structural levels. This is used for
    # unnumbered headings and as a fallback when numbering is absent.
    size_to_levels = {}
    for depth, vals in depth_levels.items():
        size_to_levels.setdefault(round(statistics.median(vals), 1), []).append(depth + title_offset)

    unnumbered = [ln for ln in lines
                  if ln["f"]["_ntype"] not in ("decimal", "article")
                  and not (title and ln is first)]
    if unnumbered:
        # Academic notes often encode hierarchy with capitalization rather
        # than font size: ALL-CAPS topic headings, followed by Title Case
        # subsection headings, may share exactly the same font size.
        has_caps_groups = (any(ln["f"]["is_caps"] for ln in unnumbered)
                           and any(not ln["f"]["is_caps"] for ln in unnumbered))
        known = [(sz, min(levels)) for sz, levels in size_to_levels.items()]
        for ln in unnumbered:
            sz = ln["size"]
            if has_caps_groups:
                # Uppercase top-level topic headings; mixed/title case headings
                # are subordinate. Size/indent can still reveal deeper levels.
                level = 1 if ln["f"]["is_caps"] else 2
                if not ln["f"]["is_caps"]:
                    caps_sizes = [x["size"] for x in unnumbered if x["f"]["is_caps"]]
                    if caps_sizes and sz < statistics.median(caps_sizes) - max(1.5, sz * 0.10):
                        level = 3
            elif known:
                nearest_size, nearest_level = min(known, key=lambda item: abs(item[0] - sz))
                delta = sz - nearest_size
                level = nearest_level
                if delta >= max(1.5, nearest_size * 0.10):
                    level -= 1
                elif delta <= -max(1.5, nearest_size * 0.10):
                    level += 1
            else:
                sizes = sorted({round(x["size"], 1) for x in unnumbered}, reverse=True)
                size_rank = {value: i + 1 + title_offset for i, value in enumerate(sizes)}
                level = size_rank[round(sz, 1)]
            ln["cls"]["level"] = max(1, min(4, level))

    for ln in lines:
        if ln["cls"].get("role") == "title":
            continue
        ln["cls"]["level"] = max(1, min(4, int(ln["cls"].get("level", 1))))
        ln["cls"]["role"] = ("clause_heading"
                             if ln["f"]["_ntype"] in ("decimal", "article")
                             else "section_heading")


# ---- report ------------------------------------------------------------------------
FEATURE_COLUMNS = ["font_size", "size_ratio", "bold", "italic", "font_family", "x0", "y0", "x1", "y1",
                   "gap_before", "gap_after", "n_chars", "n_words", "numbering", "y_rel",
                   "first_on_page", "alignment", "indent"]


def write_csv(all_pages, path):
    """classification.csv: every line with its Level-1 features and the decision."""
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["page", "text", "label", "level", "role", "score", "confidence",
                    "needs_review", "reasons"] + FEATURE_COLUMNS)
        for pno, blocks in enumerate(all_pages, start=1):
            for b in blocks:
                if b["kind"] != "text":
                    continue
                for ln in b["lines"]:
                    c = ln.get("cls", {})
                    f = ln.get("f", {})
                    w.writerow([pno, ln["text"][:90], c.get("label"), c.get("level", ""), c.get("role", ""),
                                c.get("score"), c.get("confidence"), c.get("needs_review"),
                                "; ".join(c.get("reasons", []))] + [f.get(k, "") for k in FEATURE_COLUMNS])


def review_with_llm(lines):
    """Level 3 hook (not implemented yet): send lines with needs_review=True to
    Claude / a RAG store of labelled examples and return corrected labels."""
    return lines
