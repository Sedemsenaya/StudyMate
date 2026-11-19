"""
StudyMate v3 visual builders.

The model is free to describe a visual however it likes; draw_visual() reads
whatever keys are there, guesses the closest builder, and never crashes (a
visual it cannot read becomes a key-point list instead of an empty scene).

Builders (type names the model can use, plus many aliases):
  flow          steps in order, boxes + arrows          {"steps": [...], "labels": [...]}
  diagram       any boxes connected by arrows           {"nodes": [...], "edges": [...]}
  tree          hierarchy / BST, optional search path   {"root": {...children}} / nodes+edges
  table         comparison or data table                {"columns": [...], "rows": [[...]]}
  plot          function graph(s) / data lines          {"functions": [{"expr", "label"}]}
  bar           bar chart                               {"bars": [{"label", "value"}]}
  array         list cells, one row per step (traces)   {"rows": [{"items", "highlight"}]}
  stack/queue   pile of items / line of items           {"items": [...]}
  equation      formula + worked lines                  {"lines": [...]}
  right_triangle  labelled triangle, optional squares   {"a", "b", "c", "angle", "squares"}
  unit_circle   unit circle with (cos x, sin x)         {"angle": 40}
  shapes        area shapes with formulas               {"items": [{"shape", "formula"}]}
  cycle         loop of stages                          {"steps": [...]}
  timeline      dated events                            {"events": [...]}
  list          key points, numbered                    {"items": [...]}
"""
import math
import re

from canvas import (BLUE, FILLS, GRAY, GREEN, INK, LINE_COLORS, MUTED, ORANGE, RED,
                    STRONG_GREEN, YELLOW, fill_for, fit_text, text_size, wrap)
from mathtext import pretty

# ---------------------------------------------------------------- helpers


def _s(x):
    """Anything -> display string."""
    if x is None:
        return ""
    if isinstance(x, dict):
        for k in ("label", "text", "name", "title", "value", "step", "event", "item"):
            if k in x and x[k] not in (None, ""):
                return _s(x[k])
        return ", ".join(f"{k}: {_s(v)}" for k, v in x.items())
    if isinstance(x, (list, tuple)):
        return ", ".join(_s(i) for i in x)
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def _list(v, *keys):
    for k in keys:
        val = v.get(k)
        if isinstance(val, list) and val:
            return val
        if isinstance(val, str) and val.strip():
            parts = [p.strip() for p in re.split(r"\n|;|\s->\s|→", val) if p.strip()]
            if len(parts) > 1:
                return parts
    return []


def _num(x, default=None):
    try:
        return float(re.sub(r"[^\d.\-eE]", "", str(x)) or "nan") if not isinstance(
            x, (int, float)) else float(x)
    except ValueError:
        return default


def _idx_list(val, n):
    """[1, 3] | "0-4" | 3 | ["16"] (values) -> set of indices."""
    out = set()
    if val is None:
        return out
    if isinstance(val, (int, float)):
        val = [val]
    if isinstance(val, str):
        val = re.split(r"[,\s]+", val)
    for item in val:
        if isinstance(item, (int, float)) and 0 <= int(item) < n:
            out.add(int(item))
        elif isinstance(item, str) and re.fullmatch(r"\d+\s*-\s*\d+", item.strip()):
            a, b = map(int, item.split("-"))
            out.update(i for i in range(a, b + 1) if 0 <= i < n)
        elif isinstance(item, str) and item.strip().isdigit() and int(item) < n:
            out.add(int(item))
    return out


def node_box(cv, cx, cy, w, h, label, step, fill, shape="rect", size=22, sub=None):
    x, y = cx - w / 2, cy - h / 2
    if shape in ("ellipse", "circle", "oval"):
        cv.ellipse(x, y, w, h, step, fill)
        cv.label_in(x + w * 0.1, y + h * 0.1, w * 0.8, h * 0.8, label, step, size, pad=2)
    elif shape == "diamond":
        cv.diamond(x, y, w, h, step, fill)
        cv.label_in(x + w * 0.2, y + h * 0.2, w * 0.6, h * 0.6, label, step, size, pad=2)
    else:
        cv.rect(x, y, w, h, step, fill, rounded=shape != "square")
        if sub:
            cv.label_in(x, y, w, h * 0.55, label, step, size)
            cv.label_in(x, y + h * 0.5, w, h * 0.5, sub, step, max(13, size - 6), MUTED)
        else:
            cv.label_in(x, y, w, h, label, step, size)
    return (x, y, w, h)


def _label_dims(labels, max_w, size=21, min_w=130):
    """Box width/height that fits the longest label at a readable size."""
    longest = max((len(pretty(l)) for l in labels), default=8)
    w = max(min_w, min(max_w, longest * size * 0.55 + 40))
    lines = max(math.ceil(len(pretty(l)) * size * 0.55 / (w - 30)) for l in labels) if labels else 1
    h = max(64, min(150, lines * size * 1.3 + 30))
    return w, h


# ---------------------------------------------------------------- flow


def build_flow(cv, v, box, step):
    items = _list(v, "steps", "items", "nodes", "stages", "sequence")
    labels = [_s(i) for i in items]
    icons = [i.get("icon") if isinstance(i, dict) else None for i in items]
    subs = [_s(i.get("detail") or i.get("description") or "") if isinstance(i, dict) else ""
            for i in items]
    arrow_labels = [_s(a) for a in (v.get("labels") or v.get("arrow_labels") or [])]
    n = len(labels)
    if not n:
        return step
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    per_row = n if n <= 5 else math.ceil(n / 2)
    rows = math.ceil(n / per_row)
    longest_arrow = max((len(pretty(a)) for a in arrow_labels), default=0)
    gap = 70 if per_row <= 4 else 46
    if longest_arrow:
        gap = max(gap, min(170, min(longest_arrow, 14) * 16 * 0.55 + 30))
    w_avail = min(260, (W - gap * (per_row - 1)) / per_row)
    has_sub = any(subs)
    w, h = _label_dims(labels, w_avail, 22, min(130, w_avail))
    w = max(w, min(w_avail, 230))
    h = max(h, 84) + (40 if has_sub else 0)
    icon_sz = 64 if any(icons) and cv.icons else 0
    row_h = h + icon_sz + 70
    total_w = per_row * w + (per_row - 1) * gap
    top = y0 + max(0, (H - rows * row_h) / 2) + icon_sz
    prev = None
    for i, lab in enumerate(labels):
        r, c = divmod(i, per_row)
        cx = x0 + (W - total_w) / 2 + c * (w + gap) + w / 2
        cy = top + r * row_h + h / 2
        fill = fill_for(i, items[i].get("color") if isinstance(items[i], dict) else None)
        if i == n - 1 and v.get("highlight_last", True) and n > 2:
            fill = GREEN
        b = node_box(cv, cx, cy, w, h, lab, step, fill, sub=subs[i] or None)
        if icons[i] and cv.icons:
            cv.icon(icons[i], cx - icon_sz / 2, cy - h / 2 - icon_sz - 4, icon_sz, step)
        if prev:
            cv.arrow_between(prev, b, step, arrow_labels[i - 1] if i - 1 < len(arrow_labels) else "")
        prev = b
        step += 1
    return step


# ---------------------------------------------------------------- diagram (nodes + edges)


def _graph(v):
    nodes = v.get("nodes") or v.get("boxes") or v.get("items") or []
    edges = v.get("edges") or v.get("connections") or v.get("arrows") or v.get("links") or []
    norm = []
    for i, n in enumerate(nodes):
        if isinstance(n, dict):
            nid = _s(n.get("id") or n.get("label") or n.get("name") or i)
            norm.append({"id": nid, "label": _s(n.get("label") or n.get("text") or n.get("name")
                                                 or nid),
                         "icon": n.get("icon"), "shape": n.get("shape", "rect"),
                         "color": n.get("color"), "detail": n.get("detail") or n.get("description"),
                         "highlight": n.get("highlight")})
        else:
            norm.append({"id": _s(n), "label": _s(n)})
    ids = {n["id"] for n in norm}
    by_label = {n["label"].lower(): n["id"] for n in norm}
    E = []
    for e in edges:
        if isinstance(e, dict):
            a = _s(e.get("from") or e.get("source") or e.get("start"))
            b = _s(e.get("to") or e.get("target") or e.get("end"))
            lab = _s(e.get("label") or e.get("text") or "")
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            a, b, lab = _s(e[0]), _s(e[1]), _s(e[2]) if len(e) > 2 else ""
        else:
            continue
        a, b = by_label.get(a.lower(), a), by_label.get(b.lower(), b)
        for x in (a, b):                           # edge to an unknown node: create it
            if x not in ids:
                norm.append({"id": x, "label": x})
                ids.add(x)
        E.append((a, b, lab, isinstance(e, dict) and e.get("dashed")))
    return norm, E


def _levels(nodes, edges):
    ids = [n["id"] for n in nodes]
    indeg = {i: 0 for i in ids}
    out = {i: [] for i in ids}
    for a, b, *_ in edges:
        if a in out and b in indeg and a != b:
            out[a].append(b)
            indeg[b] += 1
    roots = [i for i in ids if indeg[i] == 0] or ids[:1]
    level = {r: 0 for r in roots}
    queue = list(roots)
    seen = set(roots)
    while queue:
        a = queue.pop(0)
        for b in out[a]:
            if b not in seen:
                level[b] = level[a] + 1
                seen.add(b)
                queue.append(b)
    for i in ids:                                   # disconnected nodes
        level.setdefault(i, 0)
    return level


def build_diagram(cv, v, box, step):
    nodes, edges = _graph(v)
    if not nodes:
        return step
    # a plain chain is clearer as a flow
    if edges and len(edges) == len(nodes) - 1 and all(
            edges[i][1] == edges[i + 1][0] for i in range(len(edges) - 1)) and len(nodes) <= 8:
        steps = [{"label": n["label"], "icon": n.get("icon"), "detail": n.get("detail")}
                 for n in nodes]
        order = [edges[0][0]] + [e[1] for e in edges]
        lookup = {n["id"]: s for n, s in zip(nodes, steps)}
        if set(order) == set(lookup):
            return build_flow(cv, {"steps": [lookup[i] for i in order],
                                   "labels": [e[2] for e in edges]}, box, step)
    level = _levels(nodes, edges)
    rows = {}
    for n in nodes:
        rows.setdefault(level[n["id"]], []).append(n)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    nrows = len(rows)
    widest = max(len(r) for r in rows.values())
    gap = 40
    w = min(250, (W - gap * (widest - 1)) / widest)
    w, h = _label_dims([n["label"] for n in nodes], w, 20, min(120, w))
    has_icons = any(n.get("icon") for n in nodes) and cv.icons
    icon_sz = 56 if has_icons else 0
    row_gap = max(30, (H - nrows * (h + icon_sz)) / max(1, nrows))
    row_gap = min(row_gap, 120)
    total_h = nrows * (h + icon_sz) + (nrows - 1) * row_gap
    top = y0 + (H - total_h) / 2 + icon_sz
    boxes = {}
    for r, lv in enumerate(sorted(rows)):
        row = rows[lv]
        rw = len(row) * w + (len(row) - 1) * gap
        for c, n in enumerate(row):
            cx = x0 + (W - rw) / 2 + c * (w + gap) + w / 2
            cy = top + r * (h + icon_sz + row_gap) + h / 2
            fill = GREEN if n.get("highlight") else fill_for(r * 3 + c, n.get("color"))
            boxes[n["id"]] = node_box(cv, cx, cy, w, h, n["label"], step, fill,
                                      n.get("shape") or "rect")
            if n.get("icon") and cv.icons:
                cv.icon(n["icon"], cx - icon_sz / 2, cy - h / 2 - icon_sz - 2, icon_sz, step)
        step += 1
        for a, b, lab, dashed in edges:            # arrows into this row
            if b in boxes and a in boxes and level.get(b) == lv and a != b:
                cv.arrow_between(boxes[a], boxes[b], step - 1, lab, dashed)
    for a, b, lab, dashed in edges:                # back / same-row edges
        if a in boxes and b in boxes and a != b and level.get(b) <= level.get(a):
            cv.arrow_between(boxes[a], boxes[b], step, lab, True)
    return step + 1


# ---------------------------------------------------------------- tree


def _tree_from(v):
    root = v.get("root") or v.get("tree")
    if isinstance(root, (str, int, float)) and v.get("children"):
        root = {"label": _s(root), "children": v["children"]}
    if root is None and v.get("label") and v.get("children"):
        root = v
    if root is None and (v.get("nodes") or v.get("edges")):
        nodes, edges = _graph(v)
        kids = {n["id"]: [] for n in nodes}
        has_parent = set()
        for a, b, lab, _ in edges:
            kids.setdefault(a, []).append((b, lab))
            has_parent.add(b)
        label = {n["id"]: n["label"] for n in nodes}
        rid = next((n["id"] for n in nodes if n["id"] not in has_parent), nodes[0]["id"])

        def mk(i, lab="", depth=0):
            return {"label": label.get(i, i), "edge": lab,
                    "children": [mk(c, l, depth + 1) for c, l in kids.get(i, []) if depth < 8]}
        return mk(rid)
    return root


def _norm_tree(t):
    if isinstance(t, dict):
        kids = t.get("children") or t.get("left_right") or []
        if not kids and ("left" in t or "right" in t):
            kids = [t.get("left"), t.get("right")]
        return {"label": _s(t.get("label") or t.get("value") or t.get("name") or ""),
                "edge": _s(t.get("edge") or t.get("edge_label") or ""),
                "children": [_norm_tree(k) for k in kids if k not in (None, "", {})]}
    return {"label": _s(t), "edge": "", "children": []}


def build_tree(cv, v, box, step):
    root = _tree_from(v)
    if not root:
        return build_diagram(cv, v, box, step)
    root = _norm_tree(root)
    path = [_s(p).lower() for p in (v.get("path") or v.get("highlight") or v.get("search_path")
                                    or [])]
    notes = [_s(n) for n in (v.get("notes") or v.get("steps") or [])]
    x0, y0, x1, y1 = box
    if notes:
        tree_x1 = x0 + (x1 - x0) * 0.66
    else:
        tree_x1 = x1
    leaves, depth_of = [], {}

    def walk(n, d):
        n["depth"] = d
        if not n["children"]:
            leaves.append(n)
        for c in n["children"]:
            walk(c, d + 1)
    walk(root, 0)
    max_d = max(n["depth"] for n in leaves) if leaves else 0
    W, H = tree_x1 - x0, y1 - y0
    slot = W / max(1, len(leaves))
    small = all(len(_s(n["label"])) <= 4 for n in _iter(root))
    nw = min(slot * 0.8, 72 if small else 190)
    nh = 60 if small else 64
    if not small:
        nw, nh = _label_dims([n["label"] for n in _iter(root)], nw, 19, min(100, nw))
    level_h = min(150, (H - nh) / max(1, max_d)) if max_d else 0
    for i, lf in enumerate(leaves):
        lf["cx"] = x0 + slot * (i + 0.5)

    def place(n):
        if n["children"]:
            for c in n["children"]:
                place(c)
            n["cx"] = sum(c["cx"] for c in n["children"]) / len(n["children"])
        n["cy"] = y0 + (H - max_d * level_h - nh) / 2 + nh / 2 + n["depth"] * level_h
    place(root)
    boxes = {}
    by_depth = {}
    for n in _iter(root):
        by_depth.setdefault(n["depth"], []).append(n)
    for d in sorted(by_depth):
        for n in by_depth[d]:
            on_path = path and n["label"].lower() in path
            fill = BLUE if d == 0 else (GREEN if not n["children"] and not small else YELLOW
                                        if small else fill_for(d))
            if small:
                fill = BLUE if d == 0 else "#f8f9fa"
            b = node_box(cv, n["cx"], n["cy"], nw, nh, n["label"], step, fill,
                         "ellipse" if small or d == 0 else "rect", 22 if small else 19)
            boxes[id(n)] = b
            for c in n["children"]:
                pass
        for n in by_depth[d]:
            for c in n["children"]:
                pass
        if d > 0:
            for p in by_depth[d - 1]:
                for c in p["children"]:
                    cv.arrow_between(boxes[id(p)], boxes[id(c)], step, c.get("edge", ""))
        step += 1
    if path:                                   # highlight the search path, one node per step
        prev = None
        ordered = []
        for lab in path:
            n = next((m for m in _iter(root) if m["label"].lower() == lab), None)
            if n:
                ordered.append(n)
        for k, n in enumerate(ordered):
            x, y, w, h = boxes[id(n)]
            last = k == len(ordered) - 1
            cv.ellipse(x - 6, y - 6, w + 12, h + 12, step, "transparent",
                       "#2f9e44" if last else "#f08c00", 3)
            if last:
                cv.ellipse(x, y, w, h, step, STRONG_GREEN, "#2f9e44", 2)
                cv.label_in(x + w * 0.1, y + h * 0.1, w * 0.8, h * 0.8, n["label"], step, 22)
            if k < len(notes):
                _note_line(cv, notes[k], tree_x1 + 20, y0 + 30 + k * 52, x1 - tree_x1 - 20, step,
                           last)
            step += 1
        for k in range(len(ordered), len(notes)):
            _note_line(cv, notes[k], tree_x1 + 20, y0 + 30 + k * 52, x1 - tree_x1 - 20, step)
            step += 1
    elif notes:
        for k, nt in enumerate(notes):
            _note_line(cv, nt, tree_x1 + 20, y0 + 30 + k * 52, x1 - tree_x1 - 20, step)
            step += 1
    return step


def _note_line(cv, s, x, y, w, step, good=False):
    wrapped, sz = fit_text(pretty(s), w, 48, 19, 13)
    cv.text(x, y, wrapped, step, sz, "#2f9e44" if good else INK, math=False)


def _iter(n):
    yield n
    for c in n["children"]:
        yield from _iter(c)


# ---------------------------------------------------------------- table


def build_table(cv, v, box, step):
    cols = [_s(c) for c in (v.get("columns") or v.get("headers") or v.get("header") or [])]
    rows = v.get("rows") or v.get("data") or []
    if not rows and isinstance(v.get("items"), list) and v["items"] and isinstance(
            v["items"][0], dict):
        cols = cols or list(v["items"][0].keys())
        rows = [[it.get(c, "") for c in cols] for it in v["items"]]
    rows = [[_s(c) for c in (r if isinstance(r, (list, tuple)) else
                             [r.get(c, "") for c in cols] if isinstance(r, dict) else [r])]
            for r in rows]
    if not rows and not cols:
        return step
    ncol = max([len(cols)] + [len(r) for r in rows])
    cols = (cols + [""] * ncol)[:ncol]
    rows = [(r + [""] * ncol)[:ncol] for r in rows]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    # column widths proportional to content (clamped)
    need = [max([len(pretty(cols[c]))] + [len(pretty(r[c])) for r in rows]) for c in range(ncol)]
    need = [min(max(n, 6), 40) for n in need]
    total = sum(need)
    widths = [max(110, W * n / total) for n in need]
    scale = min(1.0, W / sum(widths))
    widths = [w * scale for w in widths]
    nrow = len(rows) + 1
    row_h = min(78, max(40, H / nrow))
    size = 20 if row_h >= 56 else 17
    tw = sum(widths)
    tx = x0 + (W - tw) / 2
    ty = y0 + max(0, (H - nrow * row_h) / 2)
    first_col_is_label = not cols[0].strip() or all(len(r[0]) < 22 for r in rows)
    head_fills = [ORANGE, GREEN, BLUE, YELLOW, "#e5dbff", "#c3fae8"]
    # header
    x = tx
    for c in range(ncol):
        fill = "#f8f9fa" if (c == 0 and first_col_is_label and not cols[0].strip()) else \
            head_fills[(c - (1 if first_col_is_label else 0)) % len(head_fills)]
        cv.rect(x, ty, widths[c], row_h, step, fill, rounded=False)
        cv.label_in(x, ty, widths[c], row_h, cols[c], step, size + 2, pad=8)
        x += widths[c]
    step += 1
    for r, row in enumerate(rows):
        x = tx
        y = ty + (r + 1) * row_h
        for c in range(ncol):
            fill = "#f1f3f5" if (c == 0 and first_col_is_label) else "#ffffff"
            cv.rect(x, y, widths[c], row_h, step, fill, rounded=False)
            cv.label_in(x, y, widths[c], row_h, row[c], step, size, pad=8)
            x += widths[c]
        step += 1
    return step


# ---------------------------------------------------------------- plot


def _make_fn(expr, var_hint="x"):
    """Return f(x) for an expression string such as 'n log n', 'x^2', 'sin(x)'. None if bad."""
    e = _s(expr).strip()
    e = re.sub(r"^\s*[yf]\s*(\(\s*[xn]\s*\))?\s*=", "", e)          # y = ..., f(x) = ...
    m = re.fullmatch(r"\s*O\((.*)\)\s*", e)                        # O(n log n) -> n log n
    if m:
        e = m.group(1)
    e = e.replace("^", "**").replace("√", "sqrt").replace("π", "pi").replace("·", "*")
    e = re.sub(r"log2\s*\(", "log2(", e)
    e = re.sub(r"\blog\s+([a-z])\b", r"log(\1)", e)                # log n -> log(n)
    e = re.sub(r"\b(sin|cos|tan|sqrt|exp|ln|log2|log)\s+([a-z])\b", r"\1(\2)", e)
    if e.strip() in ("1", "c", "const", "constant"):
        e = "1"
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (implicit_multiplication_application,
                                                parse_expr, standard_transformations,
                                                convert_xor)
        x, n, t = sp.symbols("x n t")
        local = {"log2": lambda a: sp.log(a, 2), "ln": sp.log, "e": sp.E, "pi": sp.pi,
                 "x": x, "n": n, "t": t}
        ex = parse_expr(e, local_dict=local, transformations=standard_transformations + (
            implicit_multiplication_application, convert_xor))
        free = list(ex.free_symbols)
        var = free[0] if free else x
        f = sp.lambdify(var, ex, "math")
        const = not free

        def safe(xv):
            try:
                y = float(f(xv)) if not const else float(ex)
                return y if math.isfinite(y) else None
            except Exception:
                return None
        return safe
    except Exception:
        ns = {k: getattr(math, k) for k in ("sin", "cos", "tan", "sqrt", "exp", "log", "log2",
                                            "pi", "e")}
        ns["ln"] = math.log
        e2 = re.sub(r"(?<![A-Za-z_])(\d+(?:\.\d+)?)\s*([A-Za-z(])", r"\1*\2", e)
        e2 = re.sub(r"([\w)])\s+(?=[A-Za-z(])", r"\1*", e2)

        def safe(xv):
            try:
                y = float(eval(e2, {"__builtins__": {}}, dict(ns, x=xv, n=xv, t=xv)))
                return y if math.isfinite(y) else None
            except Exception:
                return None
        return safe if safe(1.0) is not None or safe(2.0) is not None else None


def _ticks(lo, hi, target=6):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=mag * 10)
    start = math.ceil(lo / step) * step
    out, t = [], start
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


def _fmt(t):
    if abs(t - round(t)) < 1e-9:
        return str(int(round(t)))
    return f"{t:.2f}".rstrip("0").rstrip(".")


def build_plot(cv, v, box, step):
    funcs = v.get("functions") or v.get("curves") or v.get("lines") or v.get("series") or []
    if isinstance(funcs, (str, dict)):
        funcs = [funcs]
    if not funcs and v.get("expr"):
        funcs = [{"expr": v["expr"], "label": v.get("label", "")}]
    series = []
    for f in funcs:
        if isinstance(f, str):
            f = {"expr": f}
        label = _s(f.get("label") or f.get("name") or f.get("expr") or "")
        pts = f.get("points") or f.get("data")
        if pts:
            P = [(float(p[0]), float(p[1])) for p in pts
                 if isinstance(p, (list, tuple)) and len(p) >= 2]
            if P:
                series.append({"label": label, "points": P, "data": True})
            continue
        fn = _make_fn(f.get("expr") or f.get("y") or f.get("function") or label)
        if fn:
            series.append({"label": label, "fn": fn})
    if not series:
        return step
    xr = v.get("x_range") or v.get("xrange") or v.get("domain")
    if not (isinstance(xr, (list, tuple)) and len(xr) == 2):
        if any(s.get("data") for s in series):
            xs = [p[0] for s in series if s.get("data") for p in s["points"]]
            xr = [min(xs), max(xs)]
        else:
            trig = any(re.search(r"sin|cos|tan", s["label"]) for s in series)
            xr = [0, 2 * math.pi] if trig else ([0, 20] if any("n" in s["label"] for s in series)
                                                 else [-5, 5])
    xr = [_num(xr[0], 0), _num(xr[1], 10)]
    if isinstance(v.get("x_range"), (list, tuple)) and any("pi" in _s(a) for a in v["x_range"]):
        xr = [_make_fn(_s(a))(0) if _make_fn(_s(a)) else b for a, b in zip(v["x_range"], xr)]
    N = 200
    for s in series:
        if not s.get("data"):
            s["points"] = [(xr[0] + (xr[1] - xr[0]) * i / N,
                            s["fn"](xr[0] + (xr[1] - xr[0]) * i / N)) for i in range(N + 1)]
    yr = v.get("y_range") or v.get("yrange")
    if isinstance(yr, (list, tuple)) and len(yr) == 2:
        yr = [_num(yr[0], 0), _num(yr[1], 1)]
    else:
        ys = sorted(p[1] for s in series for p in s["points"] if p[1] is not None)
        lo, hi = ys[0], ys[int(len(ys) * 0.9)] if len(series) > 2 else ys[-1]
        hi = max(hi, ys[0] + 1e-6)
        pad = (hi - lo) * 0.12 or 1
        yr = [min(0, lo) if lo >= -pad else lo - pad, hi + pad]
    x0, y0, x1, y1 = box
    legend_w = min(260, (x1 - x0) * 0.3) if len(series) > 1 or series[0]["label"] else 0
    px0, py0, px1, py1 = x0 + 60, y0 + 30, x1 - legend_w - 30, y1 - 50

    def X(xv):
        return px0 + (xv - xr[0]) / (xr[1] - xr[0]) * (px1 - px0)

    def Y(yv):
        return py1 - (yv - yr[0]) / (yr[1] - yr[0]) * (py1 - py0)
    # axes (through 0 when 0 is inside the range)
    ax_y = Y(0) if yr[0] < 0 < yr[1] else py1
    ax_x = X(0) if xr[0] < 0 < xr[1] else px0
    cv.line([(px0 - 10, ax_y), (px1 + 20, ax_y)], step, INK, 2, arrow_end=True)
    cv.line([(ax_x, py1 + 10), (ax_x, py0 - 20)], step, INK, 2, arrow_end=True)
    cv.text(px1 + 26, ax_y, v.get("x_label") or v.get("xlabel") or "x", step, 18, MUTED,
            "left", "middle")
    cv.text(ax_x + 10, py0 - 26, v.get("y_label") or v.get("ylabel") or "y", step, 18, MUTED)
    trig_ticks = abs(xr[1] - 2 * math.pi) < 0.01 or abs(xr[1] - math.pi) < 0.01
    xt = [i * math.pi / 2 for i in range(int(xr[0] // (math.pi / 2)), int(xr[1] // (math.pi / 2))
                                         + 1)] if trig_ticks else _ticks(*xr)
    for t in xt:
        if abs(t) < 1e-9 and ax_x != px0:
            continue
        lab = _pi_label(t) if trig_ticks else _fmt(t)
        cv.line([(X(t), ax_y - 5), (X(t), ax_y + 5)], step, INK, 1, roughness=0)
        cv.text(X(t), ax_y + 10, lab, step, 15, MUTED, "center", math=False)
    for t in _ticks(*yr, target=5):
        if abs(t) < 1e-9 and ax_y != py1:
            continue
        cv.line([(ax_x - 5, Y(t)), (ax_x + 5, Y(t))], step, INK, 1, roughness=0)
        cv.text(ax_x - 10, Y(t), _fmt(t), step, 15, MUTED, "right", "middle", math=False)
    step += 1
    for k, s in enumerate(series):
        color = LINE_COLORS[k % len(LINE_COLORS)]
        seg = []
        segs = []
        for xv, yv in s["points"]:
            if yv is None or not (yr[0] - (yr[1] - yr[0]) * 0.02 <= yv <= yr[1]):
                if len(seg) > 1:
                    segs.append(seg)
                seg = []
                continue
            seg.append((X(xv), Y(yv)))
        if len(seg) > 1:
            segs.append(seg)
        for sg in segs:
            if len(sg) > 60:                      # Excalidraw lines look best with ~60 points
                stride = math.ceil(len(sg) / 60)
                sg = sg[::stride] + ([sg[-1]] if (len(sg) - 1) % stride else [])
            ln = cv.line(sg, step, color, 2.5, roughness=0)
            ln["roundness"] = {"type": 2}
            if s.get("data"):
                for (a, b) in sg:
                    cv.ellipse(a - 4, b - 4, 8, 8, step, color, color, 1)
        if legend_w and s["label"]:
            ly = py0 + 10 + k * 32
            cv.line([(px1 + 50, ly + 11), (px1 + 80, ly + 11)], step, color, 3, roughness=0)
            cv.text(px1 + 88, ly, s["label"], step, 18, color)
        step += 1
    for k, mark in enumerate(v.get("marks") or v.get("annotations") or []):
        if isinstance(mark, dict) and "x" in mark:
            mx = _num(mark["x"], None)
            my = _num(mark.get("y"), None)
            if mx is None:
                continue
            if my is None:
                cv.line([(X(mx), py0), (X(mx), py1)], step, MUTED, 1, dashed=True)
                cv.text(X(mx) + 6, py0, mark.get("label", ""), step, 16, MUTED)
            else:
                cv.ellipse(X(mx) - 6, Y(my) - 6, 12, 12, step, "#e03131", "#e03131", 1)
                cv.text(X(mx) + 10, Y(my) - 26, mark.get("label", ""), step, 16, "#c92a2a")
            step += 1
    return step


def _pi_label(t):
    k = round(t / (math.pi / 2))
    return {0: "0", 1: "π/2", 2: "π", 3: "3π/2", 4: "2π", -1: "-π/2", -2: "-π"}.get(k, f"{k}π/2")


# ---------------------------------------------------------------- bar chart


def build_bar(cv, v, box, step):
    bars = v.get("bars") or v.get("data") or v.get("items") or []
    pairs = []
    for b in bars:
        if isinstance(b, dict):
            pairs.append((_s(b.get("label") or b.get("name")), _num(b.get("value"), 0) or 0))
        elif isinstance(b, (list, tuple)) and len(b) >= 2:
            pairs.append((_s(b[0]), _num(b[1], 0) or 0))
    if not pairs:
        return step
    x0, y0, x1, y1 = box
    top = max(p[1] for p in pairs) or 1
    px0, py0, px1, py1 = x0 + 50, y0 + 30, x1 - 20, y1 - 60
    cv.line([(px0, py1), (px1, py1)], step, INK, 2)
    cv.line([(px0, py1), (px0, py0 - 10)], step, INK, 2)
    if v.get("y_label"):
        cv.text(px0 + 8, py0 - 30, v["y_label"], step, 17, MUTED)
    step += 1
    slot = (px1 - px0) / len(pairs)
    for i, (lab, val) in enumerate(pairs):
        h = (py1 - py0) * val / top
        bx = px0 + i * slot + slot * 0.18
        cv.rect(bx, py1 - h, slot * 0.64, h, step, fill_for(i), rounded=False)
        cv.text(bx + slot * 0.32, py1 - h - 26, _fmt(val), step, 18, INK, "center")
        cv.label_in(px0 + i * slot, py1 + 4, slot, 54, lab, step, 17)
        step += 1
    return step


# ---------------------------------------------------------------- array / trace


def build_array(cv, v, box, step):
    rows = v.get("rows") or v.get("frames") or v.get("steps") or []
    if not rows and v.get("items"):
        rows = [{"items": v["items"], "highlight": v.get("highlight"),
                 "label": v.get("label", ""), "note": v.get("note", "")}]
    norm = []
    base_items = None
    for r in rows:
        if isinstance(r, dict):
            items = r.get("items") or r.get("array") or r.get("values") or base_items or []
            norm.append({"items": [_s(i) for i in items],
                         "hl": r.get("highlight") if r.get("highlight") is not None else r.get(
                             "mid", r.get("current")),
                         "found": r.get("found"), "dim": r.get("dim") or r.get("discarded"),
                         "active": r.get("active") or r.get("range") or r.get("window"),
                         "label": _s(r.get("label") or r.get("step") or ""),
                         "note": _s(r.get("note") or r.get("caption") or r.get("text") or "")})
            base_items = items
        elif isinstance(r, (list, tuple)):
            norm.append({"items": [_s(i) for i in r], "hl": None, "found": None, "dim": None,
                         "active": None, "label": "", "note": ""})
    if not norm:
        return step
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    n = max(len(r["items"]) for r in norm)
    label_w = 110 if any(r["label"] for r in norm) else 0
    note_w = min(320, W * 0.32) if any(r["note"] for r in norm) else 0
    cell = min(74, (W - label_w - note_w - 20) / max(1, n))
    row_h = min(130, H / max(1, len(norm)))
    ch = min(cell, 66, row_h - 34)
    ax = x0 + label_w + (W - label_w - note_w - n * cell) / 2
    show_idx = v.get("show_index", len(norm) <= 4)
    for k, r in enumerate(norm):
        cy = y0 + (H - len(norm) * row_h) / 2 + k * row_h + row_h / 2
        if r["label"]:
            cv.text(x0, cy, r["label"], step, 19, MUTED, "left", "middle")
        hl = _idx_list(r["hl"], len(r["items"]))
        found = _idx_list(r["found"], len(r["items"]))
        dim = _idx_list(r["dim"], len(r["items"]))
        act = _idx_list(r["active"], len(r["items"]))
        if act:
            dim |= set(range(len(r["items"]))) - act
        # values given instead of indices ("highlight": "16")
        for key, target in (("hl", hl), ("found", found)):
            val = r[key]
            if val is not None and not target:
                vals = [_s(x) for x in (val if isinstance(val, list) else [val])]
                target.update(i for i, it in enumerate(r["items"]) if it in vals)
        for i, it in enumerate(r["items"]):
            x = ax + i * cell
            fill = STRONG_GREEN if i in found else YELLOW if i in hl else \
                GRAY if i in dim else "#ffffff"
            cv.rect(x, cy - ch / 2, cell, ch, step, fill, MUTED if i in dim else INK,
                    1.5, rounded=False)
            cv.label_in(x, cy - ch / 2, cell, ch, it, step, 22,
                        MUTED if i in dim else INK, pad=4)
            if show_idx and k == 0:
                cv.text(x + cell / 2, cy - ch / 2 - 22, str(i), step, 13, MUTED, "center")
        if r["note"]:
            wrapped, sz = fit_text(pretty(r["note"]), note_w - 20, row_h - 6, 19, 13)
            good = bool(found) or re.search(r"found|done|result|=\s*\w+!", r["note"], re.I)
            cv.text(ax + n * cell + 24, cy, wrapped, step, sz, "#2f9e44" if good else INK,
                    "left", "middle", math=False)
        step += 1
    return step


# ---------------------------------------------------------------- stack & queue


def build_stack(cv, v, box, step, kind="stack"):
    items = [_s(i) for i in (v.get("items") or v.get("elements") or v.get("values") or [])]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    title = v.get("label") or v.get("title") or ("Stack (LIFO)" if kind == "stack" else
                                                  "Queue (FIFO)")
    cv.text(cx, y0, title, step, 24, INK, "center")
    if kind == "stack":
        n = max(1, len(items))
        bw = min(200, W * 0.5)
        bh = min(62, (H - 110) / n)
        bottom = y1 - 20
        left = cx - bw / 2
        cv.line([(left - 14, bottom - n * bh - 30), (left - 14, bottom + 8),
                 (left + bw + 14, bottom + 8), (left + bw + 14, bottom - n * bh - 30)],
                step, INK, 2.5, kind="poly")
        step += 1
        for i, it in enumerate(items):
            y = bottom - (i + 1) * bh
            top_item = i == len(items) - 1
            cv.rect(left, y, bw, bh - 4, step, YELLOW if top_item else BLUE)
            cv.label_in(left, y, bw, bh - 4, it, step, 22)
            step += 1
        ty = bottom - n * bh
        push, pop = v.get("push_label", "push"), v.get("pop_label", "pop (top only)")
        cv.arrow(left - 90, ty - 70, left + 30, ty - 6, step, "")
        cv.text(left - 95, ty - 96, push, step, 19, "#1971c2", "left")
        cv.arrow(left + bw - 30, ty - 6, left + bw + 90, ty - 70, step, "")
        cv.text(left + bw + 40, ty - 100, pop, step, 19, "#c92a2a", "left")
        step += 1
    else:
        n = max(1, len(items))
        bw = min(90, (W - 160) / n)
        bh = 66
        cy = y0 + H / 2
        left = cx - n * bw / 2
        cv.line([(left - 10, cy - bh / 2 - 10), (left + n * bw + 10, cy - bh / 2 - 10)],
                step, INK, 2.5)
        cv.line([(left - 10, cy + bh / 2 + 10), (left + n * bw + 10, cy + bh / 2 + 10)],
                step, INK, 2.5)
        step += 1
        for i, it in enumerate(items):
            x = left + i * bw
            cv.rect(x + 3, cy - bh / 2, bw - 6, bh, step, YELLOW if i == 0 else GREEN)
            cv.label_in(x + 3, cy - bh / 2, bw - 6, bh, it, step, 22)
            step += 1
        deq, enq = v.get("dequeue_label", "dequeue"), v.get("enqueue_label", "enqueue")
        cv.arrow(left - 14, cy, left - 110, cy, step)
        cv.text(left - 62, cy + 18, f"{deq}\n(front)", step, 18, "#c92a2a", "center")
        cv.arrow(left + n * bw + 120, cy, left + n * bw + 16, cy, step)
        cv.text(left + n * bw + 68, cy + 18, f"{enq}\n(back)", step, 18, "#1971c2", "center")
        step += 1
    return step


# ---------------------------------------------------------------- equation / worked example


def build_equation(cv, v, box, step):
    lines = v.get("lines") or v.get("steps") or v.get("equations") or v.get("work") or []
    if not lines:
        for k in ("formula", "equation", "expr", "latex", "text"):
            if v.get(k):
                lines = [v[k]]
                break
    if isinstance(lines, str):
        lines = [l for l in lines.split("\n") if l.strip()]
    rows = []
    for l in lines:
        if isinstance(l, dict):
            rows.append((_s(l.get("math") or l.get("line") or l.get("equation") or
                            l.get("formula") or l.get("text")),
                         _s(l.get("note") or l.get("why") or l.get("explain") or "")))
        else:
            rows.append((_s(l), ""))
    if not rows:
        return step
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    has_notes = any(n for _, n in rows)
    eq_w = W * (0.58 if has_notes else 1.0)
    row_h = min(110, H / len(rows))
    size = int(min(46, row_h * 0.5))
    longest = max(len(pretty(m)) for m, _ in rows)
    size = int(max(20, min(size, eq_w / (longest * 0.58 + 1))))
    top = y0 + (H - len(rows) * row_h) / 2
    cx = x0 + eq_w / 2
    boxed = v.get("boxed", "first" if len(rows) > 1 else "all")
    for i, (m, note) in enumerate(rows):
        cy = top + i * row_h + row_h / 2
        t = cv.text(cx, cy, m, step, size if i == 0 else int(size * 0.85), INK, "center",
                    "middle")
        if t and ((boxed == "first" and i == 0) or boxed == "all" or
                  (boxed == "last" and i == len(rows) - 1)):
            cv.rect(t["x"] - 24, t["y"] - 14, t["width"] + 48, t["height"] + 28, step,
                    "transparent", "#1971c2" if i == 0 else "#2f9e44", 2)
        if note:
            wrapped, sz = fit_text(pretty(note), W - eq_w - 20, row_h - 8, 20, 13)
            cv.text(x0 + eq_w + 20, cy, wrapped, step, sz, "#495057", "left", "middle",
                    math=False)
        step += 1
    return step


# ---------------------------------------------------------------- right triangle


def build_right_triangle(cv, v, box, step):
    x0, y0, x1, y1 = box
    a_lab = _s(v.get("a") or v.get("opposite") or "a")
    b_lab = _s(v.get("b") or v.get("adjacent") or "b")
    c_lab = _s(v.get("c") or v.get("hypotenuse") or "c")
    ang = _s(v.get("angle") or v.get("angle_label") or "x")
    a = _num(re.findall(r"=\s*([\d.]+)", a_lab)[-1] if re.findall(r"=\s*([\d.]+)", a_lab)
             else v.get("a_value", 3), 3) or 3
    b = _num(re.findall(r"=\s*([\d.]+)", b_lab)[-1] if re.findall(r"=\s*([\d.]+)", b_lab)
             else v.get("b_value", 4), 4) or 4
    if re.fullmatch(r"[\d.]+", a_lab.strip()):
        a = float(a_lab)
    if re.fullmatch(r"[\d.]+", b_lab.strip()):
        b = float(b_lab)
    squares = bool(v.get("squares") or v.get("show_squares"))
    W, H = x1 - x0, y1 - y0
    # scale: leave room for squares (b below, a to the right, c up-left)
    span_w = b + (a if squares else 0) + (a if squares else 0) * 0.9
    span_h = a + (b if squares else 0) + (b * 0.9 if squares else 0)
    s = min((W - 160) / span_w, (H - 80) / span_h) if squares else min((W - 260) / b,
                                                                         (H - 80) / a)
    s = min(s, 90)
    if squares:
        Cx = x0 + 60 + (a * 0.9 + b) * s           # right angle corner
        Cy = y0 + 30 + (a + b * 0.9) * s
    else:
        Cx = x0 + (W + b * s) / 2
        Cy = y0 + (H + a * s) / 2
    A = (Cx - b * s, Cy)        # angle x here
    B = (Cx, Cy - a * s)        # top
    C = (Cx, Cy)                # right angle
    cv.polygon([A, C, B], step, "#e7f5ff", INK, 2.5)
    q = 18
    cv.line([(C[0] - q, C[1]), (C[0] - q, C[1] - q), (C[0], C[1] - q)], step, INK, 1.5,
            kind="poly")
    step += 1
    cv.text((A[0] + C[0]) / 2, C[1] + 12 + (b * s if squares else 0), b_lab, step, 21,
            "#1971c2", "center")
    step += 1
    cv.text(C[0] + 14 + (a * s if squares else 0), (B[1] + C[1]) / 2, a_lab, step, 21,
            "#c92a2a", "left", "middle")
    step += 1
    mx, my = (A[0] + B[0]) / 2, (A[1] + B[1]) / 2
    cv.text(mx - 16, my - 16, c_lab, step, 21, "#2f9e44", "right", "bottom")
    step += 1
    cv.text(A[0] + 52, A[1] - 12, ang, step, 22, INK, "left", "bottom")
    cv.line([(A[0] + 40, A[1]), (A[0] + 38, A[1] - 14)], step, INK, 1.5)
    step += 1
    if squares:
        # square on b (below), on a (right), on c (outward from hypotenuse)
        cv.polygon([A, C, (C[0], C[1] + b * s), (A[0], A[1] + b * s)], step, "#d0ebff",
                   "#1971c2")
        cv.text((A[0] + C[0]) / 2, C[1] + b * s / 2, _fmt(b * b), step, 26, "#1971c2",
                "center", "middle")
        step += 1
        cv.polygon([C, B, (B[0] + a * s, B[1]), (C[0] + a * s, C[1])], step, "#ffe3e3",
                   "#c92a2a")
        cv.text(C[0] + a * s / 2, (B[1] + C[1]) / 2 + 30, _fmt(a * a), step, 26, "#c92a2a",
                "center", "middle")
        step += 1
        dx, dy = B[0] - A[0], B[1] - A[1]          # outward normal (to upper-left)
        nx, ny = dy, -dx
        D, E = (B[0] + nx, B[1] + ny), (A[0] + nx, A[1] + ny)
        cv.polygon([A, B, D, E], step, "#d3f9d8", "#2f9e44")
        cx_, cy_ = (A[0] + D[0]) / 2, (A[1] + D[1]) / 2
        cv.text(cx_, cy_, _fmt(a * a + b * b), step, 28, "#2f9e44", "center", "middle")
        step += 1
    for k, f in enumerate(v.get("formulas") or v.get("ratios") or []):
        cv.text(x1 - 10, y0 + 20 + k * 44, _s(f), step, 24, INK, "right")
        step += 1
    return step


# ---------------------------------------------------------------- unit circle


def build_unit_circle(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    deg = _num(v.get("angle") or v.get("angle_deg") or 40, 40)
    if deg is None or abs(deg) > 720:
        deg = 40
    R = min(W * 0.32, H * 0.38)
    cx, cy = x0 + W * 0.4, y0 + H / 2
    cv.line([(cx - R - 50, cy), (cx + R + 60, cy)], step, INK, 1.5, arrow_end=True)
    cv.line([(cx, cy + R + 40), (cx, cy - R - 50)], step, INK, 1.5, arrow_end=True)
    for lab, (px, py, al) in {"(1, 0)": (cx + R + 8, cy + 6, "left"),
                              "(-1, 0)": (cx - R - 8, cy + 6, "right"),
                              "(0, 1)": (cx + 8, cy - R - 34, "left"),
                              "(0, -1)": (cx + 8, cy + R + 6, "left")}.items():
        cv.text(px, py, lab, step, 16, MUTED, al)
    step += 1
    cv.ellipse(cx - R, cy - R, 2 * R, 2 * R, step, "transparent", "#1971c2", 2.5)
    step += 1
    t = math.radians(deg)
    px, py = cx + R * math.cos(t), cy - R * math.sin(t)
    cv.line([(cx, cy), (px, py)], step, INK, 2.5)
    cv.text((cx + px) / 2 - 10, (cy + py) / 2 - 10, v.get("radius_label", "1"), step, 20, INK,
            "right", "bottom")
    cv.text(cx + 38, cy - 8, v.get("angle_label", "x"), step, 20, INK, "left", "bottom")
    step += 1
    cv.line([(px, py), (px, cy)], step, "#c92a2a", 2.5, dashed=True)
    cv.text(px + 10, (py + cy) / 2, "sin x", step, 20, "#c92a2a", "left", "middle")
    cv.line([(cx, cy + 3), (px, cy + 3)], step, "#2f9e44", 3)
    cv.text((cx + px) / 2, cy + 12, "cos x", step, 20, "#2f9e44", "center")
    step += 1
    cv.ellipse(px - 7, py - 7, 14, 14, step, "#e03131", "#e03131", 1)
    cv.text(px + 14, py - 34, v.get("point_label", "(cos x, sin x)"), step, 22, INK)
    step += 1
    notes = v.get("notes") or ["point on circle:", "x-coordinate = cos x",
                               "y-coordinate = sin x"]
    for k, n in enumerate(notes):
        cv.text(x0 + W * 0.78, y0 + H * 0.3 + k * 36, _s(n), step, 19, INK)
    return step + 1


# ---------------------------------------------------------------- area shapes


def build_shapes(cv, v, box, step):
    items = v.get("items") or v.get("shapes") or []
    if not items:
        return step
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    col = W / len(items)
    size = min(col * 0.55, H * 0.42)
    for i, it in enumerate(items):
        it = it if isinstance(it, dict) else {"shape": _s(it)}
        kind = _s(it.get("shape") or it.get("type") or it.get("label")).lower()
        cx = x0 + col * (i + 0.5)
        top = y0 + 30
        fill = fill_for(i, it.get("color"))
        dims = it.get("dims") or it.get("labels") or {}
        if "circle" in kind:
            r = size / 2
            cv.ellipse(cx - r, top, 2 * r, 2 * r, step, fill)
            cv.line([(cx, top + r), (cx + r, top + r)], step, INK, 2)
            cv.text(cx + r / 2, top + r - 6, _s(dims.get("r", "r")) if isinstance(dims, dict)
                    else "r", step, 20, INK, "center", "bottom")
        elif "tri" in kind:
            pts = [(cx - size / 2, top + size), (cx + size / 2, top + size),
                   (cx - size * 0.1, top)]
            cv.polygon(pts, step, fill)
            cv.line([(cx - size * 0.1, top), (cx - size * 0.1, top + size)], step, MUTED, 1.5,
                    dashed=True)
            cv.text(cx - size * 0.1 + 6, top + size / 2, _s(dims.get("h", "h"))
                    if isinstance(dims, dict) else "h", step, 18, INK, "left", "middle")
            cv.text(cx, top + size + 4, _s(dims.get("b", "base")) if isinstance(dims, dict)
                    else "base", step, 17, MUTED, "center")
        else:
            w, h = size * 1.2, size * 0.72
            cv.rect(cx - w / 2, top + (size - h) / 2, w, h, step, fill, rounded=False)
            if isinstance(dims, dict):
                cv.text(cx, top + (size - h) / 2 - 26, _s(dims.get("l", dims.get("w", "l"))),
                        step, 17, MUTED, "center")
                cv.text(cx + w / 2 + 8, top + size / 2, _s(dims.get("w", dims.get("h", "w"))),
                        step, 17, MUTED, "left", "middle")
        name = _s(it.get("label") or it.get("name") or kind.title())
        cv.text(cx, top + size + 36, name, step, 24, INK, "center")
        y = top + size + 74
        for f in [it.get("formula")] + list(it.get("formulas") or []) + [it.get("example")]:
            if f:
                wrapped, sz = fit_text(pretty(f), col - 20, 60, 21, 14)
                t = cv.text(cx, y, wrapped, step, sz, "#1971c2" if y < top + size + 80 else
                            "#495057", "center", math=False)
                y += (t["height"] if t else 0) + 8
        step += 1
    return step


# ---------------------------------------------------------------- cycle


def build_cycle(cv, v, box, step):
    items = [_s(i) for i in _list(v, "steps", "stages", "items", "nodes")]
    n = len(items)
    if n < 2:
        return build_list(cv, v, box, step)
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    R = min(x1 - x0, y1 - y0) * 0.36
    w, h = _label_dims(items, min(210, 2 * math.pi * R / n * 0.8), 19, 110)
    boxes = []
    for i, lab in enumerate(items):
        t = -math.pi / 2 + 2 * math.pi * i / n
        b = node_box(cv, cx + R * 1.25 * math.cos(t), cy + R * math.sin(t), w, h, lab, step,
                     fill_for(i))
        if boxes:
            cv.arrow_between(boxes[-1], b, step)
        boxes.append(b)
        step += 1
    cv.arrow_between(boxes[-1], boxes[0], step - 1)
    if v.get("center") or v.get("title"):
        cv.text(cx, cy, v.get("center") or v.get("title"), step - 1, 22, MUTED, "center",
                "middle")
    return step


# ---------------------------------------------------------------- timeline


def build_timeline(cv, v, box, step):
    ev = v.get("events") or v.get("items") or v.get("steps") or []
    pairs = []
    for e in ev:
        if isinstance(e, dict):
            pairs.append((_s(e.get("date") or e.get("time") or e.get("year") or ""),
                          _s(e.get("label") or e.get("event") or e.get("text") or "")))
        else:
            pairs.append(("", _s(e)))
    if not pairs:
        return step
    x0, y0, x1, y1 = box
    cy = (y0 + y1) / 2
    cv.line([(x0 + 10, cy), (x1 - 10, cy)], step, INK, 2.5, arrow_end=True)
    step += 1
    slot = (x1 - x0 - 40) / len(pairs)
    for i, (d, lab) in enumerate(pairs):
        x = x0 + 20 + slot * (i + 0.5)
        up = i % 2 == 0
        cv.ellipse(x - 8, cy - 8, 16, 16, step, "#1971c2", "#1971c2", 1)
        cv.text(x, cy + (-48 if up else 16), d, step, 19, "#1971c2", "center",
                "bottom" if up else "top")
        wrapped, sz = fit_text(pretty(lab), slot - 10, 110, 19, 13)
        cv.text(x, cy + (-60 if up else 44), wrapped, step, sz, INK, "center",
                "bottom" if up else "top", math=False)
        step += 1
    return step


# ---------------------------------------------------------------- list


def build_list(cv, v, box, step):
    items = v.get("items") or v.get("points") or v.get("bullets") or v.get("terms") or []
    if isinstance(items, str):
        items = [i for i in re.split(r"\n|(?<=[.;])\s+", items) if i.strip()]
    rows = []
    for it in items:
        if isinstance(it, dict):
            term = _s(it.get("term") or it.get("label") or it.get("title") or "")
            desc = _s(it.get("definition") or it.get("text") or it.get("description") or "")
            rows.append((term, desc) if term and desc else ("", term or desc or _s(it)))
        else:
            rows.append(("", _s(it)))
    if not rows:
        return step
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    row_h = min(110, H / len(rows))
    for i, (term, desc) in enumerate(rows):
        cy = y0 + (H - len(rows) * row_h) / 2 + i * row_h + row_h / 2
        r = 22
        cv.ellipse(x0 + 10, cy - r, 2 * r, 2 * r, step, fill_for(i))
        cv.text(x0 + 10 + r, cy, str(i + 1), step, 20, INK, "center", "middle")
        tx = x0 + 70
        avail = W - 80
        if term:
            tw, _ = text_size(pretty(term), 22)
            tw = min(tw, avail * 0.35)
            wrapped_t, sz_t = fit_text(pretty(term), tw + 2, row_h - 8, 22, 14)
            cv.text(tx, cy, wrapped_t, step, sz_t, "#1971c2", "left", "middle", math=False)
            tx += tw + 24
            avail -= tw + 24
        wrapped, sz = fit_text(pretty(desc), avail, row_h - 8, 22, 13)
        cv.text(tx, cy, wrapped, step, sz, INK, "left", "middle", math=False)
        step += 1
    return step


# ---------------------------------------------------------------- dispatch

BUILDERS = {
    "flow": build_flow, "diagram": build_diagram, "tree": build_tree, "table": build_table,
    "plot": build_plot, "bar": build_bar, "array": build_array,
    "stack": lambda cv, v, b, s: build_stack(cv, v, b, s, "stack"),
    "queue": lambda cv, v, b, s: build_stack(cv, v, b, s, "queue"),
    "equation": build_equation, "right_triangle": build_right_triangle,
    "unit_circle": build_unit_circle, "shapes": build_shapes, "cycle": build_cycle,
    "timeline": build_timeline, "list": build_list,
}
ALIASES = {
    "flowchart": "flow", "sequence": "flow", "process": "flow", "steps": "flow",
    "pipeline": "flow", "chain": "flow",
    "concept_map": "diagram", "concept": "diagram", "network": "diagram", "graph_nodes": "diagram",
    "architecture": "diagram", "mindmap": "diagram", "mind_map": "diagram",
    "relationship": "diagram", "anatomy": "diagram", "boxes": "diagram",
    "hierarchy": "tree", "bst": "tree", "binary_tree": "tree", "org_chart": "tree",
    "comparison": "table", "compare": "table", "matrix": "table", "grid": "table",
    "chart": "plot", "line_chart": "plot", "function": "plot", "function_plot": "plot",
    "graph_plot": "plot", "line": "plot", "curve": "plot", "functions": "plot",
    "bar_chart": "bar", "histogram": "bar", "bars": "bar",
    "trace": "array", "binary_search": "array", "list_trace": "array", "memory": "array",
    "cells": "array", "sorting": "array",
    "lifo": "stack", "fifo": "queue",
    "formula": "equation", "math": "equation", "derivation": "equation", "worked_example":
    "equation", "calculation": "equation", "proof": "equation", "identity": "equation",
    "triangle": "right_triangle", "pythagoras": "right_triangle", "pythagorean":
    "right_triangle", "trig_triangle": "right_triangle",
    "circle": "unit_circle", "unitcircle": "unit_circle",
    "area": "shapes", "geometry": "shapes",
    "loop": "cycle", "circular": "cycle",
    "history": "timeline", "dates": "timeline",
    "bullets": "list", "key_points": "list", "points": "list", "definitions": "list",
    "definition": "list", "summary": "list", "text": "list", "glossary": "list",
}


def kind_of(v):
    t = _s(v.get("type") or v.get("kind") or v.get("visual") or "").lower().strip()
    t = re.sub(r"[\s\-]+", "_", t)
    if t == "graph":                       # "graph" means plot if it has functions
        t = "plot" if (v.get("functions") or v.get("curves") or v.get("expr")) else "diagram"
    t = ALIASES.get(t, t)
    if t in BUILDERS:
        return t
    # guess from the keys
    if v.get("functions") or v.get("curves") or v.get("expr"):
        return "plot"
    if v.get("columns") or v.get("headers") or v.get("rows") and isinstance(
            (v.get("rows") or [None])[0], list):
        return "table"
    if v.get("root") or v.get("children"):
        return "tree"
    if v.get("nodes") or v.get("edges"):
        return "diagram"
    if v.get("bars"):
        return "bar"
    if v.get("lines") or v.get("formula") or v.get("equation"):
        return "equation"
    if v.get("events"):
        return "timeline"
    if v.get("steps") and all(isinstance(s, (str, dict)) for s in v["steps"]):
        return "flow"
    return "list"


def draw_visual(cv, v, box, step, fallback_items=None):
    """Draw one visual into box. Never raises; falls back to a key-point list."""
    if not isinstance(v, dict):
        v = {"type": "list", "items": v if isinstance(v, list) else [v]}
    kind = kind_of(v)
    n_before = len(cv.elements)
    try:
        new_step = BUILDERS[kind](cv, v, box, step)
        if len(cv.elements) > n_before:
            return new_step
        print(f"[visuals] {kind}: nothing drawn, using key points")
    except Exception as ex:
        import traceback
        print(f"[visuals] {kind} failed ({ex}); using key points")
        traceback.print_exc(limit=2)
        del cv.elements[n_before:]
    if fallback_items:
        return build_list(cv, {"items": fallback_items}, box, step)
    return step
