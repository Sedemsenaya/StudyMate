"""
StudyMate scene builder: lesson JSON (from the LLM) -> .excalidraw files.

    python scene_builder.py lesson.json out_folder

Pure Python, no Excalidraw library needed. A .excalidraw file is JSON:
{"type": "excalidraw", "version": 2, "elements": [...], "files": {}}.

What it does per scene:
- picks a layout function from scene["layout"] to position the elements
- draws each element as a shape with its label bound inside (moves with it)
- draws connections as arrows bound to their shapes, with labels
- adds a title, a sticky note (key_fact) and a caption bar (narration)
- adds an icon above elements that have one (resolve_icon hook, emoji for now)
"""
import json
import math
import re
import random
import sys
import uuid
from pathlib import Path

from mathviz import default_x_range, equation_elements, looks_like_latex, plot_elements
from icons import is_life_science  # noqa: F401  (kept for subject checks)

W, H = 1600, 900
INK = "#1e1e1e"
FONT = 5          # Excalifont (hand-drawn)
CHAR_W = 0.55     # rough text width per font-size unit

# area where the diagram goes (title above, caption below, sticky on the right)
AREA = (80, 170, 1180, 680)   # x0, y0, x1, y1


# ---------------- low-level element helpers ----------------

def _id():
    return uuid.uuid4().hex[:16]


def _base(kind, x, y, w, h, stroke=INK, fill="transparent", sw=2, **kw):
    e = {"id": _id(), "type": kind, "x": x, "y": y, "width": w, "height": h, "angle": 0,
         "strokeColor": stroke, "backgroundColor": fill, "fillStyle": "solid",
         "strokeWidth": sw, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
         "groupIds": [], "frameId": None, "roundness": None,
         "seed": random.randint(1, 2**30), "version": 1,
         "versionNonce": random.randint(1, 2**30), "isDeleted": False,
         "boundElements": [], "updated": 1, "link": None, "locked": False}
    e.update(kw)
    return e


def _text_size(s, size):
    lines = s.split("\n")
    return max(len(l) for l in lines) * size * CHAR_W, len(lines) * size * 1.25


def text(x, y, s, size=24, color=INK, align="left", container=None):
    w, h = _text_size(s, size)
    if align == "center":
        x -= w / 2
    return _base("text", x, y, w, h, color, "transparent", 1,
                 text=s, originalText=s, fontSize=size, fontFamily=FONT,
                 textAlign="center" if container else align,
                 verticalAlign="middle" if container else "top",
                 containerId=container, lineHeight=1.25, autoResize=True)


def wrap(s, n):
    out, cur = [], ""
    for word in str(s).split():
        if cur and len(cur) + len(word) + 1 > n:
            out.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    out.append(cur)
    return "\n".join(out)


# ---------------- shapes for scene elements ----------------

SHAPE = {"rectangle": ("rectangle", None), "rounded_rectangle": ("rectangle", {"type": 3}),
         "circle": ("ellipse", None), "ellipse": ("ellipse", None),
         "diamond": ("diamond", None), "equation": ("rectangle", {"type": 3}),
         "text": (None, None), "icon_placeholder": (None, None)}


def element_size(el):
    label = wrap(el.get("text", ""), 16)
    tw, th = _text_size(label, 22)
    t = el.get("type", "rectangle")
    if t == "circle":
        d = max(tw, th) + 50
        return d, d
    if t == "diamond":
        return tw * 1.6 + 40, th * 1.6 + 50
    if t == "ellipse":
        return tw + 80, th + 60
    return max(tw + 40, 150), th + 36


def ink_for(bg):
    try:
        r, g, b = (int(bg.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return INK
    return "#ffffff" if 0.299 * r + 0.587 * g + 0.114 * b < 140 else INK


def draw_element(el, cx, cy, w, h):
    """Return (shape_or_text_element, [all elements]) centered on (cx, cy)."""
    kind, roundness = SHAPE.get(el.get("type"), ("rectangle", {"type": 3}))
    label = wrap(el.get("text", ""), 16)
    color = el.get("color") or "#e9ecef"
    if kind is None:  # plain text / icon-only element: no box
        t = text(cx, cy - _text_size(label, 24)[1] / 2, label, 24, INK, "center")
        return t, [t]
    shape = _base(kind, cx - w / 2, cy - h / 2, w, h, INK, color, 2, roundness=roundness)
    tw, th = _text_size(label, 22)
    t = text(cx - tw / 2, cy - th / 2, label, 22, ink_for(color), container=shape["id"])
    t["width"], t["height"] = tw, th
    shape["boundElements"].append({"id": t["id"], "type": "text"})
    return shape, [shape, t]


# ---------------- arrows ----------------

def _edge_point(box, tx, ty, kind):
    """Point on the border of box (x, y, w, h) toward (tx, ty)."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    if kind == "ellipse":
        a, b = w / 2, h / 2
        k = 1 / math.sqrt((dx / a) ** 2 + (dy / b) ** 2)
    elif kind == "diamond":
        k = 1 / (abs(dx) / (w / 2) + abs(dy) / (h / 2))
    else:
        k = min((w / 2) / abs(dx) if dx else 1e9, (h / 2) / abs(dy) if dy else 1e9)
    return cx + dx * k, cy + dy * k


def draw_arrow(a, b, label="", dashed=False):
    """Arrow from shape a to shape b, bound to both, with an optional label."""
    ab = (a["x"], a["y"], a["width"], a["height"])
    bb = (b["x"], b["y"], b["width"], b["height"])
    bc = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
    ac = (ab[0] + ab[2] / 2, ab[1] + ab[3] / 2)
    x1, y1 = _edge_point(ab, *bc, a["type"])
    x2, y2 = _edge_point(bb, *ac, b["type"])
    L = math.hypot(x2 - x1, y2 - y1) or 1
    gap = 8
    x1 += (x2 - x1) / L * gap; y1 += (y2 - y1) / L * gap
    x2 -= (x2 - x1) / L * gap; y2 -= (y2 - y1) / L * gap
    arr = _base("arrow", x1, y1, abs(x2 - x1), abs(y2 - y1), "#495057", "transparent", 2,
                points=[[0, 0], [x2 - x1, y2 - y1]], lastCommittedPoint=None,
                startBinding={"elementId": a["id"], "focus": 0, "gap": gap},
                endBinding={"elementId": b["id"], "focus": 0, "gap": gap},
                startArrowhead=None, endArrowhead="arrow", roundness={"type": 2},
                strokeStyle="dashed" if dashed else "solid")
    a["boundElements"].append({"id": arr["id"], "type": "arrow"})
    b["boundElements"].append({"id": arr["id"], "type": "arrow"})
    out = [arr]
    if label:
        lbl = wrap(label, 18)
        tw, th = _text_size(lbl, 16)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        t = text(mx - tw / 2, my - th / 2, lbl, 16, "#495057", container=arr["id"])
        t["width"], t["height"] = tw, th
        arr["boundElements"].append({"id": t["id"], "type": "text"})
        out.append(t)
    return out


# ---------------- layouts: return {id: (cx, cy)} ----------------

def _area():
    x0, y0, x1, y1 = AREA
    return x0, y0, x1, y1, (x0 + x1) / 2, (y0 + y1) / 2


def layout_row(els, conns):
    """sequence / flowchart / timeline: left to right, wrapping to a 2nd row."""
    x0, y0, x1, y1, cx, cy = _area()
    n = len(els)
    per_row = n if n <= 5 else math.ceil(n / 2)
    rows = math.ceil(n / per_row)
    pos = {}
    for i, el in enumerate(els):
        r, c = divmod(i, per_row)
        cols = min(per_row, n - r * per_row)
        x = x0 + (x1 - x0) * (c + 0.5) / cols
        y = cy if rows == 1 else y0 + (y1 - y0) * (r + 0.5) / rows
        pos[el["id"]] = (x, y)
    return pos


def layout_cycle(els, conns):
    x0, y0, x1, y1, cx, cy = _area()
    rx, ry = (x1 - x0) / 2 - 130, (y1 - y0) / 2 - 70
    n = len(els)
    return {el["id"]: (cx + rx * math.cos(-math.pi / 2 + 2 * math.pi * i / n),
                       cy + ry * math.sin(-math.pi / 2 + 2 * math.pi * i / n))
            for i, el in enumerate(els)}


def layout_hierarchy(els, conns):
    """Tree: roots are elements nothing points to (by contains/part_of/leads_to)."""
    x0, y0, x1, y1, cx, cy = _area()
    ids = [e["id"] for e in els]
    children = {i: [] for i in ids}
    has_parent = set()
    for c in conns:
        a, b = c.get("from"), c.get("to")
        if c.get("relationship") == "part_of":
            a, b = b, a
        if a in children and b in children and b not in has_parent and a != b:
            children[a].append(b)
            has_parent.add(b)
    roots = [i for i in ids if i not in has_parent] or ids[:1]
    levels, seen, frontier = [], set(), roots
    while frontier:
        levels.append(frontier)
        seen.update(frontier)
        frontier = [c for p in frontier for c in children[p] if c not in seen]
    rest = [i for i in ids if i not in seen]
    if rest:
        levels.append(rest)
    # each level needs room for a box plus an icon above it; a plain chain reads better as a row
    if (y1 - y0) / len(levels) < 160 or all(len(l) == 1 for l in levels):
        return layout_row(els, conns)
    pos = {}
    for li, lvl in enumerate(levels):
        y = y0 + (y1 - y0) * (li + 0.5) / len(levels)
        for k, i in enumerate(lvl):
            pos[i] = (x0 + (x1 - x0) * (k + 0.5) / len(lvl), y)
    return pos


def layout_comparison(els, conns):
    """Side by side columns; element i goes to column i % columns."""
    x0, y0, x1, y1, cx, cy = _area()
    cols = 2 if len(els) <= 4 or len(els) % 3 else 3
    rows = math.ceil(len(els) / cols)
    return {el["id"]: (x0 + (x1 - x0) * (i % cols + 0.5) / cols,
                       y0 + (y1 - y0) * (i // cols + 0.5) / rows)
            for i, el in enumerate(els)}


def anatomy_container(els):
    return next((e for e in els if e.get("type") == "ellipse"), els[0])


def layout_anatomy(els, conns):
    """Big container in the middle; its parts inside it, everything else beside it."""
    x0, y0, x1, y1, cx, cy = _area()
    box = anatomy_container(els)
    linked = {c.get("from") for c in conns if c.get("to") == box["id"]} | \
             {c.get("to") for c in conns if c.get("from") == box["id"]}
    # structures go inside the container; processes (rounded boxes) go beside it
    inside = [e for e in els if e is not box and (e["id"] in linked or e.get("type") != "rounded_rectangle")]
    outside = [e for e in els if e is not box and e not in inside]
    pos = {box["id"]: (cx, cy)}
    for i, el in enumerate(inside):
        if len(inside) == 1:
            pos[el["id"]] = (cx, cy + 20)
            continue
        a = math.pi + 2 * math.pi * i / len(inside)
        pos[el["id"]] = (cx + 170 * math.cos(a), cy + 30 + 80 * math.sin(a))
    for i, el in enumerate(outside):
        side = -1 if i % 2 == 0 else 1
        row = i // 2
        rows = math.ceil(len(outside) / 2)
        pos[el["id"]] = (cx + side * 440, y0 + (y1 - y0) * (row + 0.5) / max(rows, 1))
    return pos


def layout_grid(els, conns):
    """concept_map / network / anything else: balanced grid.
    If one element links to all the others (a hub), it goes on top, the rest in a row below."""
    x0, y0, x1, y1, cx, cy = _area()
    n = len(els)
    degree = {}
    for c in conns:
        for k in (c.get("from"), c.get("to")):
            degree[k] = degree.get(k, 0) + 1
    hub = max(els, key=lambda e: degree.get(e["id"], 0)) if els else None
    if hub and n >= 3 and degree.get(hub["id"], 0) >= n - 1 and n <= 7:
        rest = [e for e in els if e is not hub]
        pos = {hub["id"]: (cx, y0 + (y1 - y0) * 0.25)}
        for i, e in enumerate(rest):
            pos[e["id"]] = (x0 + (x1 - x0) * (i + 0.5) / len(rest), y0 + (y1 - y0) * 0.78)
        return pos
    cols = math.ceil(math.sqrt(n * 1.6))
    rows = math.ceil(n / cols)
    pos = {}
    for i, el in enumerate(els):
        r, c = divmod(i, cols)
        in_row = min(cols, n - r * cols)
        pos[el["id"]] = (x0 + (x1 - x0) * (c + 0.5) / in_row,
                         y0 + (y1 - y0) * (r + 0.5) / rows)
    return pos


LAYOUTS = {"sequence": layout_row, "flowchart": layout_row, "timeline": layout_row,
           "equation": layout_row, "cycle": layout_cycle, "hierarchy": layout_hierarchy,
           "architecture": layout_hierarchy, "comparison": layout_comparison,
           "table": layout_comparison, "anatomy": layout_anatomy}


# ---------------- icons ----------------

ICONS = None          # set to an icons.IconResolver to get real icons
SUBJECT = None        # lesson subject, used to filter icon sources
ICON_SIZE = 90


def place_icon(name, x, y, size=ICON_SIZE):
    """Icon for `name` in a size x size box at (x, y). Returns (elements, files, credit)."""
    if not name or ICONS is None:
        return [], {}, None
    hit = ICONS.find(name, SUBJECT)
    if hit is None:
        return [], {}, None
    els, files = hit.to_excalidraw(x, y, size)
    return els, files, hit.credit or None


# ---------------- scene chrome ----------------

def chrome(scene, n_total, subject):
    els = [text(30, 22, f"{subject} · Scene {scene.get('scene', '')}", 20, "#868e96"),
           text(W / 2, 60, scene.get("title", ""), 44, INK, "center")]
    fact = scene.get("key_fact")
    if fact:
        s = wrap(fact, 24)
        lines = s.count("\n") + 1
        els.append(_base("rectangle", 1230, 180, 320, 40 + lines * 30, "#e8c547", "#fff3bf", 1.5,
                         angle=0.03, roundness={"type": 3}))
        els.append(text(1250, 200, s, 22))
    narr = scene.get("narration")
    if narr:
        size, width = 20, 100                  # longer beginner narration -> smaller, wider text
        c = wrap(narr, width)
        while c.count("\n") + 1 > 5 and size > 15:
            size -= 1
            width = int(width * 1.08)
            c = wrap(narr, width)
        cl = c.count("\n") + 1
        lh = size * 1.4
        bh = cl * lh + 30
        els.append(_base("rectangle", 120, H - 30 - bh, W - 240, bh, "#343a40", "#343a40", 1,
                         roundness={"type": 3}))
        els.append(text(145, H - 30 - bh + 15, c, size, "#ffffff"))
    return els


# ---------------- build one scene ----------------

def build_scene(scene, subject="Lesson", n_total=1):
    random.seed(scene.get("scene", 0))
    els_in = scene.get("elements", [])
    conns = scene.get("connections", [])
    if not els_in:
        return chrome(scene, n_total, subject), {}, []
    pos = LAYOUTS.get(scene.get("layout"), layout_grid)(els_in, conns)
    out, shapes, files, credits = [], {}, {}, []
    if scene.get("layout") == "graph":
        return build_graph_scene(scene, subject, n_total)
    if is_triangle_scene(els_in):
        return build_triangle_scene(scene, subject, n_total)
    if scene.get("layout") == "anatomy" and str(SUBJECT or subject).lower().startswith(("math", "algebra", "geometry", "trig", "calcul", "statist")):
        scene = dict(scene, layout="concept_map")      # abstract subjects have no "parts inside a body"
    if scene.get("layout") in ("comparison", "table"):
        table = comparison_table(els_in, conns)
        if table:
            return build_table_scene(scene, subject, n_total, *table)
    used_icons = set()
    big = anatomy_container(els_in)["id"] if scene.get("layout") == "anatomy" else None
    els_in = sorted(els_in, key=lambda e: e["id"] != big)   # container drawn first (behind)
    for el in els_in:
        w, h = (640, 380) if el["id"] == big else element_size(el)
        cx, cy = pos[el["id"]]
        rendered = equation_elements(el["text"], 0, 0, height=54) \
            if el.get("type") == "equation" and looks_like_latex(el.get("text")) else None
        if rendered:
            eq, eq_files, ew = rendered
            eq[0]["x"], eq[0]["y"] = cx - ew / 2, cy - 27
            eq[0]["boundElements"] = []
            shapes[el["id"]] = eq[0]
            out += eq
            files.update(eq_files)
            continue
        shape, parts = draw_element(el, cx, cy, w, h)
        if el["id"] == big:                      # label at the top edge, unbound
            parts[1]["containerId"] = None
            shape["boundElements"] = []
            parts[1]["y"] = cy - h / 2 + 18
        shapes[el["id"]] = shape
        out += parts
        if el["id"] == big:                       # icon inside the container, top-left
            ix, iy, isz = cx - w / 2 + 70, cy - h / 2 + 30, 80
        else:                                     # icon sitting above the box
            ix, iy, isz = cx - ICON_SIZE / 2, cy - h / 2 - ICON_SIZE - 6, ICON_SIZE
        icon_name = (el.get("icon") or "").strip().lower()
        if icon_name in used_icons:              # same picture twice adds nothing
            icon_name = ""
        used_icons.add(icon_name)
        ie, ifiles, credit = place_icon(icon_name, ix, iy, isz)
        out += ie
        files.update(ifiles)
        if credit and credit not in credits:
            credits.append(credit)
    for c in conns:
        if big and big in (c.get("from"), c.get("to")):
            other = c["to"] if c.get("from") == big else c.get("from")
            o = shapes.get(other)
            if o and o["type"] != "text" and _inside(o, shapes[big]):
                continue                        # being drawn inside already says "contains"
        a, b = shapes.get(c.get("from")), shapes.get(c.get("to"))
        if a and b and a is not b and a["type"] != "text" and b["type"] != "text":
            out += draw_arrow(a, b, c.get("label", ""), c.get("style") == "dashed")
    return chrome(scene, n_total, subject) + out, files, credits


def _inside(el, box):
    cx, cy = el["x"] + el["width"] / 2, el["y"] + el["height"] / 2
    rx, ry = box["width"] / 2, box["height"] / 2
    bx, by = box["x"] + rx, box["y"] + ry
    return ((cx - bx) / rx) ** 2 + ((cy - by) / ry) ** 2 < 1


TRI_WORDS = {"hyp": ("hypotenuse",), "opp": ("opposite", "leg a", "side a"), "adj": ("adjacent", "leg b", "side b", "base")}


def _tri_role(text):
    t = " " + str(text).lower() + " "
    for role, words in TRI_WORDS.items():
        if any(w in t for w in words):
            return role
    return None


def is_triangle_scene(els):
    roles = {_tri_role(e.get("text", "")) for e in els}
    return "hyp" in roles and len(roles & {"opp", "adj"}) >= 1


def build_triangle_scene(scene, subject, n_total):
    """Trig / Pythagoras scenes: draw a real right triangle, put the side elements ON its sides,
    and list the other elements (sine, cosine, a^2+b^2=c^2 ...) beside it."""
    out, files = [], {}
    A, B, Cv = (170, 640), (650, 640), (650, 250)          # right angle at B, angle x at A
    tri = _base("line", A[0], A[1], B[0] - A[0], B[1] - Cv[1], INK, "#e7f5ff", 2,
                points=[[0, 0], [B[0] - A[0], 0], [Cv[0] - A[0], Cv[1] - A[1]], [0, 0]],
                lastCommittedPoint=None, startBinding=None, endBinding=None,
                startArrowhead=None, endArrowhead=None)
    out.append(tri)
    out.append(_base("rectangle", B[0] - 28, B[1] - 28, 28, 28, INK, "transparent", 1.5))
    ang = math.atan2(A[1] - Cv[1], Cv[0] - A[0])
    arc = [[(A[0] + 70 * math.cos(-ang * i / 12)) - (A[0] + 70), 70 * math.sin(-ang * i / 12)] for i in range(13)]
    out.append(_base("line", A[0] + 70, A[1], 70, 70 * math.sin(ang), "#e8590c", "transparent", 2,
                     points=arc, lastCommittedPoint=None, startBinding=None, endBinding=None,
                     startArrowhead=None, endArrowhead=None, roundness={"type": 2}))
    out.append(text(A[0] + 82, A[1] - 42, "x", 26, "#e8590c"))
    sides = {"opp": ((B[0], B[1]), (Cv[0], Cv[1]), (B[0] + 20, (B[1] + Cv[1]) / 2 - 18)),
             "adj": ((A[0], A[1]), (B[0], B[1]), ((A[0] + B[0]) / 2 - 80, A[1] + 16)),
             "hyp": ((A[0], A[1]), (Cv[0], Cv[1]), ((A[0] + Cv[0]) / 2 - 250, (A[1] + Cv[1]) / 2 - 40))}
    rest, used = [], set()
    for el in scene.get("elements", []):
        role = _tri_role(el.get("text", ""))
        if role and role not in used:
            used.add(role)
            (x1, y1), (x2, y2), (lx, ly) = sides[role]
            col = el.get("color") or "#495057"
            out.append(_base("line", x1, y1, abs(x2 - x1), abs(y2 - y1), col if col != "#ffffff" else INK, "transparent", 6,
                             points=[[0, 0], [x2 - x1, y2 - y1]], lastCommittedPoint=None, startBinding=None,
                             endBinding=None, startArrowhead=None, endArrowhead=None, opacity=70))
            lbl = wrap(el.get("text", ""), 18)
            w, h = _text_size(lbl, 22)
            box = _base("rectangle", lx, ly, w + 24, h + 16, INK, col, 1.5, roundness={"type": 3})
            t = text(lx + 12, ly + 8, lbl, 22, ink_for(col), container=box["id"])
            t["width"], t["height"] = w, h
            box["boundElements"].append({"id": t["id"], "type": "text"})
            out += [box, t]
        else:
            rest.append(el)
    rest = [e for e in rest if "triangle" not in str(e.get("text", "")).lower()]   # it's drawn already
    y = 260
    for el in rest[:5]:
        rendered = equation_elements(el["text"], 860, y, height=50) \
            if el.get("type") == "equation" and looks_like_latex(el.get("text")) else None
        if rendered:
            eq, f, _ = rendered
            out += eq
            files.update(f)
            y += 90
        else:
            w, h = element_size(el)
            _, parts = draw_element(el, 860 + w / 2, y + h / 2, w, h)
            out += parts
            y += h + 30
    return chrome(scene, n_total, subject) + out, files, []


def build_graph_scene(scene, subject, n_total):
    """layout == "graph": plot every element that has an "expression" (or whose text
    is a formula in x); other elements become notes beside the plot."""
    funcs, names, notes, files = [], [], [], {}
    for el in scene.get("elements", []):
        expr = el.get("expression") or (
            el.get("text") if re.match(r"\s*(y|f\(x\)|g\(x\))\s*=", el.get("text", "")) else None)
        if expr:
            funcs.append(expr)
            names.append(el.get("text") or expr)
        else:
            notes.append(el)
    xr = tuple(scene.get("x_range") or default_x_range(funcs))
    # axis marks the model wrote as elements ("0", "pi/2", "2pi") and the bare axis name are not notes
    notes = [e for e in notes if not re.fullmatch(r"[-+0-9.\s/]*(pi|π)?[-+0-9.\s/]*(degrees?|°)?",
                                                   str(e.get("text", "")).strip().lower())
             and not re.match(r"\s*(angle|x|input)\s*(\(x\))?\s*$", str(e.get("text", "")).lower())]
    yr = tuple(scene["y_range"]) if scene.get("y_range") else None
    out = plot_elements(funcs, 150, 200, 600, 420, x_range=xr, y_range=yr, labels=names)
    y = 200 + 34 * len(funcs) + 30
    for el in notes:
        rendered = equation_elements(el["text"], 930, y, height=56) \
            if el.get("type") == "equation" and looks_like_latex(el.get("text")) else None
        if rendered:
            eq, f, _ = rendered
            out += eq
            files.update(f)
            y += 80
        else:
            w, h = element_size(el)
            _, parts = draw_element(el, 930 + w / 2, y + h / 2, w, h)
            out += parts
            y += h + 25
    return chrome(scene, n_total, subject) + out, files, []


def comparison_table(els, conns):
    """Detect 'A and B each point to the same features' and return
    (subjects, features, cells) for a table, or None if it isn't that shape."""
    by_id = {e["id"]: e for e in els}
    targets = {}
    for c in conns:
        a, b = c.get("from"), c.get("to")
        if a in by_id and b in by_id:
            targets.setdefault(a, {})[b] = c.get("label", "")
    subjects = [i for i, t in targets.items() if len(t) >= 2]
    if len(subjects) < 2:
        return None
    shared = set.intersection(*(set(targets[s]) for s in subjects))
    if len(shared) < 2:
        return None
    features = [e["id"] for e in els if e["id"] not in subjects
                and any(e["id"] in targets[s] for s in subjects)]
    cells = {(s, f): targets[s].get(f, "—") or "✓" for s in subjects for f in features}
    leftovers = [e for e in els if e["id"] not in subjects and e["id"] not in features]
    # a column whose cells all say the same thing ("value of") tells the reader nothing
    subjects = [s for s in subjects if len({cells[(s, f)] for f in features}) > 1] or subjects
    for (s, f), v in list(cells.items()):           # "sin(90)" under "90 degrees" -> "sin(90°) = 1"
        cells[(s, f)] = _eval_trig_cell(v, by_id[s].get("text", ""))
    return [by_id[s] for s in subjects], [by_id[f] for f in features], cells, leftovers


def _eval_trig_cell(v, header=""):
    m = re.fullmatch(r"\s*(sin|cos|tan)\s*\(\s*(-?[0-9.]+)\s*(°|degrees?)?\s*\)\s*", str(v))
    if not m:
        return v
    fn, num = m.group(1), float(m.group(2))
    deg = bool(m.group(3)) or "degree" in header.lower() or "°" in header or num in (30, 45, 60, 90, 120, 180, 270, 360)
    val = getattr(math, fn)(math.radians(num) if deg else num)
    if abs(val) > 1e9:
        return f"{v} = undefined"
    val = round(val, 3) + 0.0
    return f"{fn}({num:g}{'°' if deg else ''}) = {val:g}"


def build_table_scene(scene, subject, n_total, subjects, features, cells, leftovers=()):
    """Feature rows x subject columns. Reads cleanly and animates row by row."""
    out, files, credits = [], {}, []
    first_w, col_w = 280, 300
    n_rows = len(features) + 1
    row_h = min(80, (AREA[3] - AREA[1]) / n_rows)
    total_w = first_w + col_w * len(subjects)
    x0 = (AREA[0] + AREA[2]) / 2 - total_w / 2 + 40
    y0 = (AREA[1] + AREA[3]) / 2 - row_h * n_rows / 2 + 20

    def cell(x, y, w, h, label, fill, size=22, bold_ink=None):
        box = _base("rectangle", x, y, w, h, INK, fill, 1.5)
        lbl = wrap(label, int(w / (size * CHAR_W)) - 1)
        tw, th = _text_size(lbl, size)
        t = text(x + w / 2 - tw / 2, y + h / 2 - th / 2, lbl, size,
                 bold_ink or ink_for(fill), container=box["id"])
        t["width"], t["height"] = tw, th
        box["boundElements"].append({"id": t["id"], "type": "text"})
        return [box, t]

    # header row: subjects, with their icons above
    out += cell(x0, y0, first_w, row_h, "", "#f8f9fa")
    for k, s in enumerate(subjects):
        cx = x0 + first_w + k * col_w
        out += cell(cx, y0, col_w, row_h, s.get("text", ""), s.get("color") or "#e9ecef", 26)
        ie, f, cr = place_icon(s.get("icon"), cx + col_w / 2 - 35, y0 - 78, 70)
        out += ie; files.update(f); credits += [cr] if cr else []
    # one row per feature
    for r, f in enumerate(features, 1):
        y = y0 + r * row_h
        stripe = "#ffffff" if r % 2 else "#f1f3f5"
        out += cell(x0, y, first_w, row_h, f.get("text", ""), f.get("color") or "#e9ecef", 22)
        for k, s in enumerate(subjects):
            out += cell(x0 + first_w + k * col_w, y, col_w, row_h, cells[(s["id"], f["id"])], stripe, 20)
    # anything that didn't fit the table (e.g. an identity) goes underneath it
    y = y0 + row_h * (len(features) + 1) + 24
    for el in leftovers[:2]:
        rendered = equation_elements(el["text"], x0, y, height=44) if looks_like_latex(el.get("text")) else None
        if rendered:
            out += rendered[0]
            files.update(rendered[1])
            y += 64
        else:
            w, h = element_size(el)
            _, parts = draw_element(el, x0 + w / 2, y + h / 2, w, h)
            out += parts
            y += h + 16
    return chrome(scene, n_total, subject) + out, files, credits


def to_file(elements, path, files=None):
    doc = {"type": "excalidraw", "version": 2, "source": "studymate",
           "elements": elements,
           "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
           "files": files or {}}
    Path(path).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def build_lesson(lesson, out_dir, start_index=1, icons=None):
    """Write one .excalidraw per scene. Pass icons=IconResolver(...) for real icons.
    Also writes credits.txt listing every icon source used (needed for CC-BY icons)."""
    global ICONS, SUBJECT
    if icons is not None:
        ICONS = icons
    SUBJECT = lesson.get("subject")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subject = lesson.get("subject") or lesson.get("section_title") or "Lesson"
    paths, all_credits = [], []
    for i, scene in enumerate(lesson.get("scenes", []), start_index):
        p = out_dir / f"scene_{i:02d}.excalidraw"
        els, files, credits = build_scene(scene, subject, len(lesson["scenes"]))
        to_file(els, p, files)
        paths.append(p)
        all_credits += [c for c in credits if c not in all_credits]
    if all_credits:                      # merge with credits from earlier sections
        cf = out_dir / "credits.txt"
        old = cf.read_text(encoding="utf-8").splitlines()[1:] if cf.exists() else []
        merged = old + [c for c in all_credits if c not in old]
        cf.write_text("Icons:\n" + "\n".join(merged), encoding="utf-8")
    return paths


if __name__ == "__main__":
    # python scene_builder.py lesson.json out_dir [excalidraw-libraries dir] [bioicons dir]
    lesson = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    resolver = None
    if len(sys.argv) > 3:
        from icons import IconResolver
        resolver = IconResolver(sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    for p in build_lesson(lesson, sys.argv[2] if len(sys.argv) > 2 else "scenes_out", icons=resolver):
        print("wrote", p)