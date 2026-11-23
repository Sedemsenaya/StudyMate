"""
StudyMate v4 - the visual selection engine + the lesson IR.

ChatGPT's advice, implemented: do not ask "which subject is this?" Ask "what
kind of idea is this?" and pick the visual from that. The rules below look only
at the SHAPE of the content (definitions, steps, a parent with parts, two things
compared, a figure, a formula, a cycle), so the same engine works for AWS,
biology, maths or history.

IR (intermediate representation) for one scene, consumed by every renderer:
{
  "id": "L03-S07",
  "title": "Routing Policies – Weighted",
  "concept": "routing policy",            # what the scene is about
  "idea_kind": "definition|process|parts|comparison|figure|formula|cycle|list",
  "visual_type": "list|flow|diagram|table|figure+list|...",
  "narration": "...the source's sentences, word for word...",
  "note": "key fact for the sticky note",
  "visuals": [ v3 visual dicts ],          # what scene.py / visuals.py draw
  "source": {"pdf": "...", "pages": [209]},
  "units": ["every source bullet/label this scene must cover"]
}
"""
import re

TERM = re.compile(r"^(.{2,60}?)\s*(?::|\s[–—-]\s|=>|→)\s+(.{3,})$")
KEY = re.compile(r"\b(only|must|can't|cannot|not|never|always|default|free|exam|max(?:imum)?|"
                 r"min(?:imum)?|up to|important|required|\d+\s?%|\d+ ?(?:ms|seconds?|minutes?|"
                 r"hours?|days?|GB|TB|MB))\b", re.I)
STEP = re.compile(r"^(?:step\s*)?\d{1,2}[.)]\s|^(first|then|next|finally)\b", re.I)
CYCLE = re.compile(r"\b(cycle|loop|repeat|again)\b", re.I)
FORMULA = re.compile(r"[=^√∑]|\b(sqrt|log|sin|cos|tan)\b")
MAX_ITEMS = 7


def _tree(bullets):
    """Flat bullets with levels -> [{"text", "kids": [...]}] (2 levels kept)."""
    roots = []
    for b in bullets:
        if b["level"] == 0 or not roots:
            roots.append({"text": b["text"], "kids": []})
        else:
            roots[-1]["kids"].append(b["text"])
    return roots


def classify(title, roots, figure):
    """What kind of idea is this? (subject independent)"""
    flat = [r["text"] for r in roots] + [k for r in roots for k in r["kids"]]
    if figure:
        return "figure"
    if not flat:
        return "title"
    parents = [r for r in roots if r["kids"]]
    if re.search(r"\bvs\.?\b|versus|compar", title, re.I) and len(parents) >= 2:
        return "comparison"
    if sum(bool(STEP.match(t)) for t in flat) >= 2 or re.search(r"\b(how|steps?|process|flow|workflow|lifecycle)\b", title, re.I) and len(roots) >= 3 and not parents:
        return "cycle" if CYCLE.search(title) else "process"
    if len(flat) >= 3 and sum(bool(TERM.match(t)) for t in flat) >= 0.6 * len(flat):
        return "definition"
    if len(parents) == 1 and len(parents[0]["kids"]) >= 3 and len(roots) <= 3:
        return "parts"
    if sum(bool(FORMULA.search(t)) and len(t) < 60 for t in flat) >= 2:
        return "formula"
    return "list"


def _term_items(texts):
    out = []
    for t in texts:
        m = TERM.match(t)
        out.append({"term": m.group(1).strip(), "definition": m.group(2).strip()} if m else t)
    return out


def _flat_items(roots):
    items = []
    for r in roots:
        if r["kids"]:
            items.append(r["text"].rstrip(":") + ": " + "; ".join(r["kids"]))
        else:
            items.append(r["text"])
    return items


def visual_for(kind, title, roots):
    """Idea kind -> v3 visual dict(s)."""
    if kind == "comparison":
        parents = [r for r in roots if r["kids"]][:3]
        n = max(len(p["kids"]) for p in parents)
        rows = [[""] + [p["kids"][i] if i < len(p["kids"]) else "" for p in parents] for i in range(n)]
        cols = [""] + [p["text"].rstrip(": ") for p in parents]
        # drop the empty first column when rows have no headers
        rows = [r[1:] for r in rows]
        return [{"type": "table", "columns": cols[1:], "rows": rows}], "table"
    if kind in ("process", "cycle"):
        steps = [re.sub(STEP, "", r["text"]).strip() for r in roots][:6]
        return [{"type": "cycle" if kind == "cycle" else "flow", "steps": steps}], kind == "cycle" and "cycle" or "flow"
    if kind == "parts":
        p = next(r for r in roots if r["kids"])
        nodes = [{"id": "hub", "label": p["text"].rstrip(": ")}] + \
                [{"id": f"k{i}", "label": k} for i, k in enumerate(p["kids"][:6])]
        edges = [{"from": "hub", "to": f"k{i}"} for i in range(min(6, len(p["kids"])))]
        vis = [{"type": "diagram", "nodes": nodes, "edges": edges}]
        others = [r["text"] for r in roots if r is not p]
        if others:
            vis.append({"type": "list", "items": others})
        return vis, "diagram"
    if kind == "definition":
        flat = [r["text"] for r in roots] + [k for r in roots for k in r["kids"]]
        return [{"type": "list", "items": _term_items(flat)}], "list(terms)"
    if kind == "formula":
        lines = [{"math": t} for r in roots for t in [r["text"]] + r["kids"]][:5]
        return [{"type": "equation", "lines": lines}], "equation"
    return [{"type": "list", "items": _term_items(_flat_items(roots))}], "list"


def note_for(texts):
    best = [t for t in texts if KEY.search(t) and len(t) < 110]
    return min(best, key=len) if best else ""


def speakable(s):
    s = re.sub(r"\s*=>\s*|\s*→\s*", " maps to ", s)
    s = re.sub(r"\be\.g\.,?", "for example,", s, flags=re.I)
    s = re.sub(r"\bi\.e\.,?", "that is,", s, flags=re.I)
    s = re.sub(r"\baka\.?", "also known as", s, flags=re.I)
    s = s.replace("…", ".").replace("&", " and ").replace("%", " percent")
    s = re.sub(r"\s+/\s+", ", ", s)
    s = re.sub(r"\s[–—]\s", ", ", s)
    s = re.sub(r"!!+", "!", s)
    return re.sub(r"\s+", " ", s).strip()


def _sentence(t):
    t = t.strip()
    return t if re.search(r"[.!?:]$", t) else t + "."


def slide_scenes(slide, pdf_name, lesson_idx):
    """One slide -> 1..n IR scenes. Word for word: every bullet and figure label is used."""
    title = slide["title"] or f"Slide {slide['page']}"
    roots = _tree(slide["bullets"])
    fig = slide.get("figure")
    kind = classify(title, roots, fig)
    texts = [b["text"] for b in slide["bullets"]]
    units = texts + slide.get("labels", [])
    base = {"source": {"pdf": pdf_name, "pages": [slide["page"]]}, "concept": title}

    # narration = the slide's own words, in order
    parts = []
    for r in roots:
        if r["kids"]:
            parts.append(r["text"].rstrip(":") + ": " + ", ".join(k.rstrip(".") for k in r["kids"]) + ".")
        else:
            parts.append(_sentence(r["text"]))
    labels = []
    for l in slide.get("labels", []):
        if l not in labels and len(l) > 2 and l not in texts and re.search(r"[A-Za-z0-9]{2}", l):
            labels.append(l)
    narration = " ".join(parts)
    if fig and labels:
        narration += (" " if narration else "") + "In the diagram: " + ", ".join(labels) + "."
    note = note_for(texts)
    scenes = []
    if kind == "figure":
        figv = {"type": "figure", "png": fig["png"], "w": fig["w"], "h": fig["h"],
                "caption": f"Slide {slide['page']}"}
        if roots and len(roots) <= 5 and sum(len(t) for t in texts) < 420:
            sub_kind = classify(title, roots, None)
            vis, vt = visual_for(sub_kind, title, roots)
            scenes.append(dict(base, title=title, idea_kind="figure", visual_type="figure+" + vt,
                               visuals=[figv, vis[0]], narration=narration, note=note, units=units))
        elif roots:      # too much text to share the screen: text scene, then the figure
            sub_kind = classify(title, roots, None)
            vis, vt = visual_for(sub_kind, title, roots)
            scenes.append(dict(base, title=title, idea_kind=sub_kind, visual_type=vt, visuals=vis,
                               narration=" ".join(parts), note=note, units=texts))
            scenes.append(dict(base, title=title + " (diagram)", idea_kind="figure",
                               visual_type="figure", visuals=[figv],
                               narration="In the diagram: " + ", ".join(labels) + "." if labels
                               else "Here is the diagram from the slide.", note="",
                               units=slide.get("labels", [])))
        else:
            scenes.append(dict(base, title=title, idea_kind="figure", visual_type="figure",
                               visuals=[figv], narration=narration or title + ".", note="",
                               units=units))
    else:
        vis, vt = visual_for(kind, title, roots)
        if labels:
            narration += " " + " ".join(_sentence(l) for l in labels)
        scenes.append(dict(base, title=title, idea_kind=kind, visual_type=vt, visuals=vis,
                           narration=narration or title + ".", note=note, units=units))
    return scenes


def plan_slides(lesson, pdf_name, lesson_idx=1):
    """Rules planner: a deck lesson -> IR lesson (no LLM needed, instant)."""
    scenes = []
    for sl in lesson["slides"]:
        scenes += slide_scenes(sl, pdf_name, lesson_idx)
    for i, s in enumerate(scenes, 1):
        s["id"] = f"L{lesson_idx:02d}-S{i:03d}"
    return {"title": lesson["title"], "subject": re.sub(r"\s*-\s*Part \d+$", "", lesson["title"]),
            "source_pdf": pdf_name, "scenes": scenes}


# ---------------------------------------------------------------- documents (no LLM)

def _doc_sections(md, max_chars=900):
    """Headings (markdown or 'Title (Slides 1-3)' style lines) start sections; long ones are cut."""
    md = re.sub(r"<!--\s*image\s*-->", "", md)
    lines = [l.rstrip() for l in md.replace("\r", "").split("\n")]
    has_md = any(l.startswith("#") for l in lines)
    page = 0
    secs, cur = [], {"head": "", "lines": [], "page": 0}
    for l in lines:
        s = l.strip()
        m = re.fullmatch(r"<!--page:(\d+)-->", s)
        if m:
            page = int(m.group(1))
            if not cur["lines"] and not cur["head"]:
                cur["page"] = page
            continue
        if not s:
            continue
        is_head = s.startswith("#") or (not has_md and len(s) < 70 and not re.search(r"[.,;:]$", s)
                                         and re.match(r"^[A-Z0-9]", s) and len(s.split()) <= 9
                                         and not re.match(r"^[-*•]", s) and cur["lines"]
                                         and re.search(r"[.:!?)]$", cur["lines"][-1]))
        if is_head:
            if cur["lines"] or cur["head"]:
                secs.append(cur)
            cur = {"head": re.sub(r"^#+\s*", "", s), "lines": [], "page": page}
        else:
            cur["lines"].append(s)
    secs.append(cur)
    out = []
    for sc in secs:
        # join wrapped lines into sentences / list items
        items, buf = [], ""
        for l in sc["lines"]:
            if re.match(r"^([-*•]|\d+[.)])\s", l) or (buf and re.search(r"[.!?:]$", buf)):
                if buf:
                    items.append(buf)
                buf = re.sub(r"^[-*•]\s*", "", l)
            else:
                buf = (buf + " " + l).strip()
        if buf:
            items.append(buf)
        chunk, n = [], 0
        for it in items:
            if chunk and n + len(it) > max_chars:
                out.append((sc["head"], chunk, sc.get("page", 0)))
                chunk, n = [], 0
            chunk.append(it)
            n += len(it)
        if chunk or sc["head"]:
            out.append((sc["head"], chunk, sc.get("page", 0)))
    return out


def plan_document_rules(md, title, pdf_name, lesson_idx=1):
    """Fallback for portrait docs when Ollama is off: sections -> scenes, word for word."""
    scenes = []
    for head, items, page in _doc_sections(md):
        if not items:
            continue
        sl = {"page": page, "title": head[:80] or title, "labels": [], "figure": None,
              "bullets": [{"text": t, "level": 0} for t in items]}
        scenes += slide_scenes(sl, pdf_name, lesson_idx)
    for i, s in enumerate(scenes, 1):
        s["id"] = f"L{lesson_idx:02d}-S{i:03d}"
    return {"title": title, "subject": title, "source_pdf": pdf_name, "scenes": scenes}
