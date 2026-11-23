"""
StudyMate v4 - practice-test reader (Udemy / Tutorials Dojo style "Results" PDFs).

A results PDF is a list of questions, each with the same parts:

    Question 12 Skipped|Correct|Incorrect
    <scenario + the ask>
    <options, one per paragraph>      "Correct answer" / "Correct selection" marks the right one(s)
    Overall explanation
    Correct option: ...  Incorrect options: ...  Reference: ...
    Domain
    <exam domain>

read_quiz() returns one dict per question with the scenario, the options (and which are
correct), the explanation, the domain, its pages and the pictures on those pages.
Options are rebuilt from wrapped lines with the same rule a text layout uses: a line
was wrapped if the next word would not have fitted on it.
No AI, ~1-3 s for a 170-page PDF.
"""
import base64
import hashlib
import io
import re

from extract import _lines

QHEAD = re.compile(r"^Question\s+(\d+)\s*(Skipped|Correct|Incorrect)?\s*$", re.I)
MARK_CORRECT = re.compile(r"^(Correct answer|Correct selection|Your answer is correct|Your selection is correct)$", re.I)
MARK_OTHER = re.compile(r"^(Your answer is incorrect|Your selection is incorrect)$", re.I)
OPTION_VERB = re.compile(r"^(Use|Create|Configure|Set up|Set|Enable|Deploy|Store|Leverage|Make sure|Delete|"
                         r"Convert|Attach|Launch|Migrate|Provision|Implement|Build|Add|Apply|Assign|Install|"
                         r"Move|Replace|Run|Select|Send|Turn|Update|Upload|Write|Change|Copy|Define|"
                         r"Establish|Purchase|Remove|Restrict|Share|Specify|Switch|Tag|Grant|Encrypt|"
                         r"Develop|Distribute|Increase|Modify|Schedule|Host|Register|Integrate|Spin up|"
                         r"Opt|Place|Put|Keep|Offload|Activate|Allow|Disable|Ensure|Ingest|Stream|Replicate|"
                         r"Refactor|Rehost|Archive|Partition|Subscribe|Invoke|Trigger|Monitor|Analyze|"
                         r"Package|Containerize|Mount|Enforce|Rotate|Revoke|Request|Generate|Process)\b")
EXPL = re.compile(r"^Overall explanation$", re.I)


def looks_like_quiz(pdf, sample_pages=6):
    """True if the first pages have 'Question N' headers and 'Overall explanation'."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf))
    try:
        text = "\n".join(doc[i].get_textpage().get_text_range() for i in range(min(sample_pages, len(doc))))
    finally:
        doc.close()
    return len(re.findall(r"^Question\s+\d+", text, re.M)) >= 2 and "Overall explanation" in text


def _page_images(doc, i, min_side=160, max_side=1400):
    """Pictures on page i as PNG base64 (smasks applied), big enough to matter."""
    import pypdfium2.raw as raw
    out = []
    page = doc[i]
    for o in page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE], max_depth=3):
        try:
            img = o.get_bitmap(render=True).to_pil().convert("RGB")
        except Exception:
            try:
                img = o.get_bitmap().to_pil().convert("RGB")
            except Exception:
                continue
        if min(img.size) < min_side * 0.6 or max(img.size) < min_side:
            continue
        if max(img.size) > max_side:
            k = max_side / max(img.size)
            img = img.resize((int(img.width * k), int(img.height * k)))
        buf = io.BytesIO()
        img.save(buf, "PNG", optimize=True)
        raw_png = buf.getvalue()
        out.append({"page": i + 1, "w": img.width, "h": img.height,
                    "png": base64.b64encode(raw_png).decode(),
                    "sha": hashlib.sha1(raw_png).hexdigest()[:16]})
    return out


def _join(lines, right_margin):
    """Wrapped PDF lines -> paragraphs (a line ending early ends its paragraph)."""
    paras, cur = [], ""
    for k, u in enumerate(lines):
        cur = (cur + " " + u["s"]).strip() if cur else u["s"]
        nxt = lines[k + 1] if k + 1 < len(lines) else None
        if nxt is None:
            break
        cw = (u["r"] - u["l"]) / max(1, len(u["s"]))          # average char width on this line
        first = nxt["s"].split(" ", 1)[0]
        wrapped = u["r"] + (len(first) + 1) * cw > right_margin - 2
        # the geometry is ambiguous near the margin: the first word decides
        if re.search(r"\b(into|configure|and|or|to|the|a|an|of|with|for|from|using|via|in|on|by|as)$", u["s"]):
            wrapped = True                      # a sentence can't end on "into"
        elif re.match(r"^(?:[a-z]|www\.|https?:|\(|&)", nxt["s"]) and not re.search(r"[.?!]$", u["s"]):
            wrapped = True                      # "...routes traffic to" + "www.example.com"
        elif OPTION_VERB.match(nxt["s"]) and not re.search(r"(,|\band|\bor|\bto|\bthe|\ba|\ban)$", u["s"]):
            wrapped = False                     # "...static assets" + "Configure AWS Lambda ..."
        elif u["s"].endswith(")") and re.match(r"^(AWS|Amazon|Simple|Elastic)\b", nxt["s"]) and len(u["s"]) > 40:
            wrapped = False                     # "...(AWS Managed Microsoft AD)" + "Simple AD"
        if not wrapped:
            paras.append(cur)
            cur = ""
    if cur:
        paras.append(cur)
    return [re.sub(r"\s+", " ", p.replace("￾", "-")).strip() for p in paras if p.strip()]


def _short(s, n=90):
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def read_quiz(pdf, pages=None, images=True):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf))
    try:
        idx = list(pages) if pages else list(range(len(doc)))
        seq = []                                        # (page, line)
        for i in idx:
            for u in _lines(doc[i].get_textpage()):
                if re.fullmatch(r"\d{1,3}", u["s"]):    # page numbers
                    continue
                seq.append((i, u))
        right = max((u["r"] for _, u in seq), default=540)
        starts = [k for k, (_, u) in enumerate(seq) if QHEAD.match(u["s"])]
        qs = []
        for n, a in enumerate(starts):
            b = starts[n + 1] if n + 1 < len(starts) else len(seq)
            qs.append(_question(seq[a:b], right))
        if images:
            pics = {}
            for i in sorted({p for q in qs for p in q["pages"]}):
                pics[i] = _page_images(doc, i - 1)
            seen = set()
            for q in qs:
                q["images"] = []
                for p in q["pages"]:
                    for im in pics.get(p, []):
                        if im["sha"] not in seen:
                            seen.add(im["sha"])
                            q["images"].append(im)
        return [q for q in qs if q["options"] or q["stem"]]
    finally:
        doc.close()


def _question(chunk, right):
    head = QHEAD.match(chunk[0][1]["s"])
    num, status = int(head.group(1)), (head.group(2) or "").title()
    body = chunk[1:]
    pages = sorted({p + 1 for p, _ in chunk})
    # drop the trailing "Domain / <name>" (it belongs to this question)
    domain = ""
    for k in range(len(body) - 1, -1, -1):
        if body[k][1]["s"] == "Domain":
            domain = " ".join(u["s"] for _, u in body[k + 1:k + 3] if not QHEAD.match(u["s"]))
            domain = domain.split(" Question ")[0].strip()
            body = body[:k]
            break
    e = next((k for k, (_, u) in enumerate(body) if EXPL.match(u["s"])), len(body))
    ask, expl = body[:e], body[e + 1:]
    # options: blocks separated by markers; inside a block, paragraphs from wrapping
    groups, cur, flag = [], [], None
    for _, u in ask:
        if MARK_CORRECT.match(u["s"]) or MARK_OTHER.match(u["s"]):
            if cur:
                groups.append((flag, cur))
            cur, flag = [], "correct" if MARK_CORRECT.match(u["s"]) else "other"
            continue
        cur.append(u)
    if cur:
        groups.append((flag, cur))
    paras = []                                         # (text, marked_correct)
    for flag, lines in groups:
        ps = _join(lines, right)
        for j, p in enumerate(ps):
            paras.append((p, flag == "correct" and j == 0))
    # the ask ends at the paragraph with the question ("...?", "(Select two)")
    q_end = 0
    for j, (p, _) in enumerate(paras):
        if re.search(r"\?\s*(\(Select (two|three|2|3)\))?$|\(Select (two|three|2|3)\)$|:$", p, re.I):
            q_end = j + 1
    if q_end == 0:                                     # no question mark: 4 last paragraphs are options
        q_end = max(1, len(paras) - 4)
    stem = " ".join(p for p, _ in paras[:q_end])
    options = [{"text": p, "correct": c} for p, c in paras[q_end:] if not c or True]
    # the explanation, as paragraphs, with its usual parts pulled out
    ex_text = "\n".join(_join([u for _, u in expl], right))
    ex_text = re.sub(r"\n?via - https?://\S+", "", ex_text)
    correct_part = re.split(r"\nIncorrect options?:", ex_text)[0]
    correct_part = re.sub(r"^Correct options?:\s*", "", correct_part).strip()
    wrong_part = ""
    m = re.search(r"Incorrect options?:\s*(.*?)(?:\nReferences?:|$)", ex_text, re.S)
    if m:
        wrong_part = m.group(1).strip()
    refs = re.findall(r"https?://\S+", ex_text.split("Reference")[-1]) if "Reference" in ex_text else []
    # why each wrong option is wrong: the text after "<option> - "
    for o in options:
        o["why"] = ""
        if o["correct"]:
            continue
        key = o["text"][:60]
        pos = wrong_part.replace("\n", " ").find(key)
        if pos >= 0:
            after = wrong_part.replace("\n", " ")[pos + len(o["text"]):]
            after = re.sub(r"^\s*[-–—]\s*", "", after)
            nxt = min([after.find(x["text"][:40]) for x in options
                       if x is not o and not x["correct"] and after.find(x["text"][:40]) > 0] or [len(after)])
            o["why"] = after[:nxt].strip()
    return {"number": num, "status": status, "domain": domain, "pages": pages,
            "stem": stem, "options": options,
            "correct_explanation": correct_part, "wrong_explanation": wrong_part,
            "references": refs[:3], "images": []}


if __name__ == "__main__":
    import json
    import sys
    import time
    t = time.perf_counter()
    qs = read_quiz(sys.argv[1], images="--no-images" not in sys.argv)
    print(f"{len(qs)} questions in {time.perf_counter() - t:.1f}s")
    for q in qs[:int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3]:
        show = dict(q, images=[f"{im['w']}x{im['h']} p{im['page']}" for im in q["images"]])
        print(json.dumps(show, indent=1, ensure_ascii=False)[:2500])
