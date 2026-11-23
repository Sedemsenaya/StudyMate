"""
StudyMate v4 - PDF -> source units (the "what does the document say" half of the IR).

Two readers, picked automatically per PDF:

  slides  (landscape pages, e.g. Stephane Maarek decks)
          pypdfium2 reads each page's text runs WITH their positions, so we know
          the title, every bullet (and its indent level) and which text is part of
          the picture. The picture region is cropped from the page and kept, so a
          slide's AWS diagram is never lost. ~10-30 ms per page, no AI, no GPU.
          Divider slides ("Amazon Route 53") split a big deck into lessons.

  docs    (portrait pages: notes, papers, textbooks)
          Docling -> markdown (v3 reader) when it is installed, otherwise
          pypdfium2 plain text. Sections are split at headings later.

Both run inside worker processes (pdfium is not thread-safe), see batch.py.
"""
import base64
import io
import re
from pathlib import Path

BOILER = re.compile(r"(©\s*Stephane Maarek|NOT FOR DISTRIBUTION|^Maarek$|datacumulus\.com|"
                    r"^www\.|^https?://links\.)", re.I)
BULLET = re.compile(r"^\s*(?:[•▪◦●○■➢]|\d{1,2}[.)](?=\s))\s*")
SKIP_TITLES = re.compile(r"^(disclaimer|table of contents)", re.I)


def is_slide_deck(pdf):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf))
    try:
        w, h = doc[0].get_size()
        return w > h * 1.15
    finally:
        doc.close()


def _clean(s):
    s = s.replace("\r", " ").replace("￾", "-").replace("­", "")
    return re.sub(r"\s+", " ", s).strip()


def _garbled(s):
    """Math typeset with private fonts comes out as '!"#$%&'. Not readable text."""
    letters = sum(c.isalnum() for c in s)
    return len(s) > 3 and letters / len(s) < 0.55


def _lines(tp):
    """Text lines in reading order with their boxes: [{"s", "l", "b", "r", "t", "h"}]."""
    n = tp.count_chars()
    text = tp.get_text_range(0, n)
    exact = True
    out, pos = [], 0            # pos counts pdfium chars: astral chars (𝑡𝑟𝑎) take 2
    for line in text.split("\r\n"):
        idx = []
        for c in line:
            if not c.isspace():
                idx.append(pos)
            pos += 2 if ord(c) > 0xFFFF else 1
        pos += 2
        s = _clean(line)
        if not s or not idx:
            continue
        if exact:
            bx = [tp.get_charbox(k) for k in idx if k < n]
            bx = [b for b in bx if b[2] > b[0]] or bx
            l, b = min(x[0] for x in bx), min(x[1] for x in bx)
            r, t = max(x[2] for x in bx), max(x[3] for x in bx)
            gap = max([abs(bx[i + 1][0] - bx[i][2]) for i in range(len(bx) - 1)] or [0])
        else:
            l = b = r = t = gap = 0
        out.append({"s": s, "l": l, "b": b, "r": r, "t": t, "h": t - b, "gap": gap})
    return out


def read_slide(page, index, crop_figures=True, scale=1.6):
    """One slide -> {"page", "title", "bullets":[{"text","level"}], "labels":[...], "figure"}"""
    import pypdfium2.raw as raw
    W, H = page.get_size()
    lines = [u for u in _lines(page.get_textpage())
             if not BOILER.search(u["s"]) and not (u["l"] > W - 40)]
    title, title_b = "", H * 0.85
    if lines and (lines[0]["t"] > H * 0.75 or lines[0]["t"] == 0) and not BULLET.match(lines[0]["s"]):
        title, title_b = lines[0]["s"], lines[0]["b"] or H * 0.85
        lines = lines[1:]
        while lines and lines[0]["t"] > title_b - 4 and lines[0]["h"] > 18:   # 2-line title
            title += " " + lines[0]["s"]
            title_b = lines[0]["b"]
            lines = lines[1:]

    if " • " in title:                      # "Table of Contents • A • B" on one line
        head, *items = title.split(" • ")
        title = head
        lines = [{"s": "• " + it, "l": 60, "b": 0, "r": W / 2, "t": 0, "h": 0} for it in items] + lines
    bullets, labels, cur = [], [], None
    lefts = sorted({round(u["l"] / 10) for u in lines if BULLET.match(u["s"])})
    for u in lines:
        if BULLET.match(u["s"]) and u["l"] < W * 0.45:
            txt = BULLET.sub("", u["s"])
            cur = {"text": txt, "level": lefts.index(round(u["l"] / 10)), "box": dict(u)}
            bullets.append(cur)
            continue
        if cur and cur["box"]["l"] - 4 < u["l"] < cur["box"]["l"] + 60 and \
                -8 <= cur["box"]["b"] - u["t"] < max(14, cur["box"]["h"] * 1.3):
            cur["text"] += " " + u["s"]
            cur["box"]["b"] = u["b"]
            continue
        cur = None
        labels.append(u)
    for bl in bullets:
        if _garbled(bl["text"]):
            bl["text"] = re.sub(r"\S*[!\"#$%&*+/]\S*", " ", bl["text"])
        bl["text"] = _clean(bl["text"])
    bullets = [b for b in bullets if len(b["text"]) > 1]
    if not bullets:     # slides written as plain paragraphs
        for u in list(labels):
            if u["r"] - u["l"] > W * 0.5 and u["l"] < W * 0.2 and u.get("gap", 0) < 40:
                bullets.append({"text": u["s"], "level": 0, "box": u})
                labels.remove(u)
    labels = [u for u in labels if not _garbled(u["s"])]
    if not title and not bullets and 1 <= len(labels) <= 2:       # centred divider title
        title = labels.pop(0)["s"]
        centered = True

    col = [b["box"] for b in bullets]
    def in_column(l, b, r, t):
        return any(l >= c["l"] - 25 and r <= max(c["r"], c["l"] + 30) + 25 and
                   b >= c["b"] - 8 and t <= c["t"] + 8 for c in col)
    figure = None
    centered = locals().get("centered", False)
    boxes = [(u["l"], u["b"], u["r"], u["t"]) for u in labels
             if u["r"] > u["l"] and not in_column(u["l"], u["b"], u["r"], u["t"])]
    col = [b["box"] for b in bullets]
    for o in page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE, raw.FPDF_PAGEOBJ_PATH], max_depth=2):
        try:
            l, b, r, t = o.get_pos()
        except Exception:
            continue
        a = (r - l) * (t - b)
        if a < 120 or (r - l) > W * 0.9 or t > title_b + 2 or in_column(l, b, r, t) \
                or t < H * 0.11 or l > W - 35:
            continue                     # dots, footer stripe, side watermark
            continue
        boxes.append((l, b, r, t))
    if crop_figures and boxes:
        l = max(0, min(x[0] for x in boxes) - 10)
        b = max(0, min(x[1] for x in boxes) - 10)
        r = min(W * 0.972, max(x[2] for x in boxes) + 10)       # side watermark
        t = min(title_b - 2, max(x[3] for x in boxes) + 10)
        if (r - l) > 80 and (t - b) > 60 and (r - l) * (t - b) > W * H * 0.05:
            img = page.render(scale=scale, crop=(l, b, W - r, H - t)).to_pil()
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "PNG", optimize=True)
            figure = {"png": base64.b64encode(buf.getvalue()).decode(),
                      "w": img.width, "h": img.height,
                      "box": [round(l), round(H - t), round(r), round(H - b)]}
    return {"page": index, "title": title,
            "bullets": [{"text": b["text"], "level": b["level"]} for b in bullets],
            "labels": [u["s"] for u in labels],
            "figure": figure, "centered_title": centered}


def is_divider(sl):
    """Section title slide: a title and (almost) nothing else."""
    return bool(sl["title"]) and not sl["bullets"] and len(sl["labels"]) <= 1 \
        and len(sl["title"]) < 70 and sl.get("centered_title", True)


def read_slides(pdf, pages=None, crop_figures=True):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf))
    try:
        idx = pages or range(len(doc))
        return [read_slide(doc[i], i + 1, crop_figures) for i in idx]
    finally:
        doc.close()


def page_count(pdf):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf))
    n = len(doc)
    doc.close()
    return n


def split_deck(slides, title_fallback, max_slides=40):
    """Cut a deck at divider slides -> [{"title", "slides":[...]}]."""
    lessons, cur = [], {"title": title_fallback, "slides": []}
    for sl in slides:
        if is_divider(sl):
            if cur["slides"]:
                lessons.append(cur)
            cur = {"title": sl["title"], "slides": [], "first_page": sl["page"]}
        elif (sl["title"] or sl["bullets"]) and not SKIP_TITLES.match(sl["title"]):
            cur["slides"].append(sl)
    if cur["slides"]:
        lessons.append(cur)
    if max_slides:                       # long sections -> "Part 1", "Part 2" (more parallel work)
        out = []
        for ls in lessons:
            s = ls["slides"]
            k = (len(s) + max_slides - 1) // max_slides
            if k <= 1:
                out.append(ls)
                continue
            size = (len(s) + k - 1) // k
            for i in range(k):
                out.append(dict(ls, title=f"{ls['title']} - Part {i + 1}",
                                slides=s[i * size:(i + 1) * size]))
        lessons = out
    return lessons


def read_document(pdf, use_docling=True, pages=None):
    """Portrait documents -> markdown. pages: 0-based page indexes (None = all)."""
    if pages:                                          # Docling converts whole files: use pdfium
        return pdfium_markdown(pdf, pages), "pdfium"
    if use_docling:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
            opts = PdfPipelineOptions()
            opts.do_ocr = False
            opts.do_table_structure = True
            conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(
                pipeline_options=opts)})
            return conv.convert(str(pdf)).document.export_to_markdown(), "docling"
        except Exception as ex:                        # not installed / no models: plain text
            print(f"[extract] docling unavailable ({type(ex).__name__}); using pdfium text")
    return pdfium_markdown(pdf), "pdfium"


def pdfium_markdown(pdf, page_idx=None):
    """No Docling: rebuild headings / list items / paragraphs from line sizes and indents."""
    import statistics
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf))
    pages = [_lines(doc[i].get_textpage()) for i in (page_idx or range(len(doc)))]
    doc.close()
    allx = [u for p in pages for u in p if u["h"] > 0]
    if not allx:
        return ""
    med_h = statistics.median(u["h"] for u in allx)
    left = statistics.mode(round(u["l"]) for u in allx)
    counts = {}
    for p in pages:                     # running headers/footers repeat on every page
        for u in p:
            k = re.sub(r"\d+", "#", u["s"])
            counts[k] = counts.get(k, 0) + 1
    out, para = [], ""
    numbers = list(page_idx or range(len(pages)))
    for pi, p in enumerate(pages):
        out.append(f"<!--page:{numbers[pi] + 1}-->")          # lets sections know their page
        for u in p:
            s = u["s"]
            if counts.get(re.sub(r"\d+", "#", s), 0) > max(2, len(pages) * 0.6) and len(pages) > 2:
                continue
            if u["h"] > med_h * 1.25 and len(s) < 90:
                kind = "head"
            elif BULLET.match(s) or u["l"] > left + 8 and len(s) > 2:
                kind = "item"
            else:
                kind = "text"
            if kind == "text" and para and not re.search(r"[.!?:]$", para):
                para += " " + s
                continue
            if kind == "item" and para.startswith("- ") and not re.search(r"[.!?:]$", para) \
                    and s[:1].islower():
                para += " " + s
                continue
            if para:
                out.append(para)
            if kind == "head":
                out.append("## " + s)
                para = ""
                continue
            para = "- " + BULLET.sub("", s) if kind == "item" else s
        if para.startswith("## "):
            out.append(para)
            para = ""
    if para:
        out.append(para)
    return "\n".join(out)


if __name__ == "__main__":
    import json
    import sys
    sl = read_slides(sys.argv[1], [int(p) - 1 for p in sys.argv[2:]] or None, crop_figures=False)
    print(json.dumps(sl, indent=1, ensure_ascii=False)[:6000])
