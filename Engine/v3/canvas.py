"""
Tiny Excalidraw element factory.

Every element gets customData = {"step": k}: the renderer draws elements of
step 0 first, then step 1, ... so a diagram builds up piece by piece, exactly
like the reference video. Special steps: "title", "note", "bar", "page".
"""
import random
import uuid

from mathtext import pretty

INK = "#1e1e1e"
MUTED = "#868e96"
FONT = 5            # Excalifont, the hand-drawn Excalidraw font
CHAR_W = 0.55       # average glyph width per px of font size (Excalifont)
LINE_H = 1.25

# soft fills that match the reference video
BLUE, GREEN, YELLOW, ORANGE = "#d0ebff", "#b2f2bb", "#ffec99", "#ffd8a8"
RED, VIOLET, GRAY, TEAL, PINK = "#ffc9c9", "#e5dbff", "#e9ecef", "#c3fae8", "#fcc2d7"
FILLS = [BLUE, GREEN, YELLOW, ORANGE, VIOLET, TEAL, PINK, RED]
STRONG_GREEN = "#69db7c"
# curve / legend colours (stroke)
LINE_COLORS = ["#1971c2", "#e8590c", "#2f9e44", "#9c36b5", "#c92a2a", "#0c8599", "#f08c00"]
NAMED = {"blue": BLUE, "green": GREEN, "yellow": YELLOW, "orange": ORANGE, "red": RED,
         "purple": VIOLET, "violet": VIOLET, "gray": GRAY, "grey": GRAY, "teal": TEAL,
         "pink": PINK}


def fill_for(i, color=None):
    if isinstance(color, str):
        c = color.strip().lower()
        if c in NAMED:
            return NAMED[c]
        if c.startswith("#") and len(c) in (4, 7):
            return c
    return FILLS[i % len(FILLS)]


def wrap(s, max_chars):
    """Greedy word wrap that keeps existing newlines."""
    max_chars = max(4, int(max_chars))
    out = []
    for para in str(s).split("\n"):
        line = ""
        for word in para.split():
            while len(word) > max_chars:                 # very long token
                if line:
                    out.append(line)
                    line = ""
                out.append(word[:max_chars])
                word = word[max_chars:]
            if not line:
                line = word
            elif len(line) + 1 + len(word) <= max_chars:
                line += " " + word
            else:
                out.append(line)
                line = word
        out.append(line)
    return "\n".join(out)


def text_size(s, size):
    lines = str(s).split("\n")
    return max(len(l) for l in lines) * size * CHAR_W, len(lines) * size * LINE_H


def fit_text(s, w, h, size=24, min_size=12):
    """Largest font size (<= size) at which s, wrapped, fits in w x h. Returns (wrapped, size)."""
    s = str(s)
    while True:
        wrapped = wrap(s, w / (size * CHAR_W))
        tw, th = text_size(wrapped, size)
        if (tw <= w and th <= h) or size <= min_size:
            return wrapped, size
        size -= 1


class Canvas:
    def __init__(self, seed=1):
        self.elements = []
        self.files = {}
        self.rng = random.Random(seed)
        self.icons = None          # optional IconResolver (Engine/visualization/icons.py)
        self.credits = []

    # ---------------- primitives ----------------
    def _base(self, kind, x, y, w, h, step, stroke=INK, fill="transparent", sw=2, **kw):
        el = {"id": uuid.uuid4().hex[:20], "type": kind, "x": float(x), "y": float(y),
              "width": float(max(w, 1)), "height": float(max(h, 1)), "angle": 0,
              "strokeColor": stroke, "backgroundColor": fill,
              "fillStyle": "solid", "strokeWidth": sw, "strokeStyle": "solid",
              "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
              "roundness": None, "seed": self.rng.randint(1, 2 ** 31 - 1), "version": 1,
              "versionNonce": self.rng.randint(1, 2 ** 31 - 1), "isDeleted": False,
              "boundElements": None, "updated": 1, "link": None, "locked": False,
              "customData": {"step": step}}
        el.update(kw)
        self.elements.append(el)
        return el

    def rect(self, x, y, w, h, step, fill="transparent", stroke=INK, sw=2, rounded=True, **kw):
        return self._base("rectangle", x, y, w, h, step, stroke, fill, sw,
                          roundness={"type": 3} if rounded else None, **kw)

    def ellipse(self, x, y, w, h, step, fill="transparent", stroke=INK, sw=2, **kw):
        return self._base("ellipse", x, y, w, h, step, stroke, fill, sw, **kw)

    def diamond(self, x, y, w, h, step, fill="transparent", stroke=INK, sw=2, **kw):
        return self._base("diamond", x, y, w, h, step, stroke, fill, sw, **kw)

    def text(self, x, y, s, step, size=22, color=INK, align="left", valign="top", math=True):
        """x is the left edge (align=left), centre (center) or right edge (right)."""
        s = pretty(s) if math else str(s)
        if not s.strip():
            return None
        w, h = text_size(s, size)
        x0 = x if align == "left" else x - w / 2 if align == "center" else x - w
        y0 = y if valign == "top" else y - h / 2 if valign == "middle" else y - h
        return self._base("text", x0, y0, w, h, step, color, "transparent", 1,
                          text=s, originalText=s, fontSize=size, fontFamily=FONT,
                          textAlign=align, verticalAlign="top", containerId=None,
                          autoResize=True, lineHeight=LINE_H)

    def label_in(self, x, y, w, h, s, step, size=22, color=INK, pad=10, min_size=12):
        """Centered, wrapped, auto-shrunk label inside the box (x, y, w, h)."""
        wrapped, sz = fit_text(pretty(s), w - 2 * pad, h - 2 * pad, size, min_size)
        return self.text(x + w / 2, y + h / 2, wrapped, step, sz, color, "center", "middle",
                         math=False)

    def line(self, pts, step, stroke=INK, sw=2, fill="transparent", dashed=False,
             roughness=1, arrow_end=False, arrow_start=False, kind=None):
        x0, y0 = pts[0]
        rel = [[px - x0, py - y0] for px, py in pts]
        xs, ys = [p[0] for p in rel], [p[1] for p in rel]
        etype = "arrow" if (arrow_end or arrow_start) else "line"
        return self._base(etype,
                          x0, y0, max(xs) - min(xs), max(ys) - min(ys), step, stroke, fill, sw,
                          points=rel, strokeStyle="dashed" if dashed else "solid",
                          roughness=roughness, lastCommittedPoint=None,
                          startBinding=None, endBinding=None,
                          startArrowhead="arrow" if arrow_start else None,
                          endArrowhead="arrow" if arrow_end else None,
                          roundness={"type": 2} if len(pts) > 2 and kind != "poly" else None)

    def polygon(self, pts, step, fill="transparent", stroke=INK, sw=2):
        pts = list(pts) + [pts[0]]
        el = self.line(pts, step, stroke, sw, fill, kind="line")
        el["roundness"] = None
        return el

    def arrow(self, x1, y1, x2, y2, step, label="", dashed=False, color=INK, size=16):
        a = self.line([(x1, y1), (x2, y2)], step, color, 2, dashed=dashed, arrow_end=True)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            vertical = abs(y2 - y1) > abs(x2 - x1)
            wrapped = wrap(pretty(label), 14 if not vertical else 22)
            if vertical:
                self.text(mx + 8, my, wrapped, step, size, "#495057", "left", "middle", math=False)
            else:
                self.text(mx, my - 6, wrapped, step, size, "#495057", "center", "bottom",
                          math=False)
        return a

    def arrow_between(self, a, b, step, label="", dashed=False, gap=6):
        """Arrow from the edge of box a to the edge of box b (boxes are (x, y, w, h))."""
        (x1, y1), (x2, y2) = edge_point(a, b, gap), edge_point(b, a, gap)
        return self.arrow(x1, y1, x2, y2, step, label, dashed)

    # ---------------- icons ----------------
    def icon(self, name, x, y, size, step):
        """Place an icon from the user's IconResolver (if configured). Returns True if placed."""
        if not name or not self.icons:
            return False
        try:
            hit = self.icons.find(str(name), getattr(self, "subject", None))
            if not hit:
                return False
            els, files = hit.to_excalidraw(x, y, size)
        except Exception as ex:                      # icons are a bonus, never a crash
            print(f"[icons] {name!r}: {ex}")
            return False
        for e in els:
            e.setdefault("customData", {})
            e["customData"] = dict(e.get("customData") or {}, step=step)
            e.pop("index", None)
        self.elements += els
        self.files.update(files or {})
        credit = getattr(hit, "credit", None)
        if credit and credit not in self.credits:
            self.credits.append(credit)
        return True


def edge_point(box, toward, gap=6):
    """Point on the border of box (x, y, w, h) in the direction of the centre of `toward`."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    tx, ty = toward[0] + toward[2] / 2, toward[1] + toward[3] / 2
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = (w / 2 + gap) / abs(dx) if dx else float("inf")
    sy = (h / 2 + gap) / abs(dy) if dy else float("inf")
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s
