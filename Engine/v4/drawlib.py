"""
StudyMate v4 - drawn illustrations (no icons needed): the engine draws the thing itself.

Every drawing is plain Excalidraw shapes, so it is hand-drawn and animates stroke by
stroke. Things that MOVE or CHANGE use two customData keys understood by render4:
    "until": k   -> the element disappears when step k starts (packet at hop 1, 2, 3 ...)
    "fade": True -> the element appears at once instead of being drawn

Visuals registered with v3 visuals.py (usable by the AI, the rules and hand-written plans):
  cell         an animal or plant cell with its organelles drawn and labelled one by one
  mitosis      one big cell that goes through the phases (chromosomes condense, line up,
               split, two nuclei form, the cell pinches in two) + a phase strip
  packet_flow  devices drawn in a row, a packet (envelope) travelling hop by hop
  sequence     devices with lifelines, messages one after another (TCP handshake, DNS...)
  code_trace   the code with the running line highlighted, the variables and the array
               changing step by step (Python is really executed: tracer.py)
  device       draw(...) any of: server, database, laptop, phone, router, switch, cloud,
               user, users, file, browser, queue, firewall, internet, building, chip
"""
import math
import random
import re

import visuals                                      # v3
from canvas import INK, MUTED, fit_text, text_size

GREEN, RED, BLUE, YELLOW = "#b2f2bb", "#ffc9c9", "#d0ebff", "#ffec99"
INKS = ["#1971c2", "#e03131", "#2f9e44", "#9c36b5"]


def _tag(el, until=None, fade=False):
    if el is None:
        return el
    if until is not None:
        el["customData"]["until"] = until
    if fade:
        el["customData"]["fade"] = True
    return el


class _Track:
    """Collect the elements a drawing adds, so they can all be given until/fade at once."""

    def __init__(self, cv):
        self.cv, self.n = cv, len(cv.elements)

    def tag(self, until=None, fade=False):
        for el in self.cv.elements[self.n:]:
            _tag(el, until, fade)


def _blob(cx, cy, rx, ry, n=28, wobble=0.0, seed=1, pinch=0.0):
    """Closed smooth outline (ellipse-ish). pinch > 0 squeezes the middle (dividing cell)."""
    rnd = random.Random(seed)
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        r = 1 + rnd.uniform(-wobble, wobble)
        x = cx + rx * r * math.cos(a)
        y = cy + ry * r * math.sin(a) * (1 - pinch * math.exp(-(math.cos(a) / 0.35) ** 2))
        pts.append((x, y))
    return pts + [pts[0]]


def _poly(cv, pts, step, stroke=INK, fill="transparent", sw=2, smooth=True):
    el = cv.line(pts, step, stroke, sw, fill)
    if not smooth:
        el["roundness"] = None
    return el


# ======================================================================== devices

def draw_device(cv, kind, cx, cy, s, step, fill=None):
    """Draw a device of size ~s centred at (cx, cy). Returns its box (x, y, w, h)."""
    k = kind
    if k == "server":
        w, h = s * 0.62, s * 0.95
        x, y = cx - w / 2, cy - h / 2
        cv.rect(x, y, w, h, step, fill or "#e9ecef")
        for i in range(3):
            yy = y + h * (0.1 + i * 0.29)
            cv.rect(x + w * 0.1, yy, w * 0.8, h * 0.22, step, "#ffffff", sw=1.5)
            cv.ellipse(x + w * 0.18, yy + h * 0.07, s * 0.06, s * 0.06, step, "#40c057", "#2b8a3e", 1)
            cv.line([(x + w * 0.42, yy + h * 0.11), (x + w * 0.8, yy + h * 0.11)], step, MUTED, 1.5)
        return (x, y, w, h)
    if k == "database":
        w, h = s * 0.75, s * 0.9
        x, y = cx - w / 2, cy - h / 2
        e = h * 0.24
        cv.rect(x, y + e / 2, w, h - e, step, fill or "#d0ebff", "transparent", 1, rounded=False)
        cv.line([(x, y + e / 2), (x, y + h - e / 2)], step)
        cv.line([(x + w, y + e / 2), (x + w, y + h - e / 2)], step)
        cv.ellipse(x, y + h - e, w, e, step, fill or "#d0ebff")
        cv.ellipse(x, y + h * 0.36, w, e, step, "transparent", MUTED, 1)
        cv.ellipse(x, y, w, e, step, "#e7f5ff")
        return (x, y, w, h)
    if k == "laptop":
        w, h = s * 0.9, s * 0.62
        x, y = cx - w / 2, cy - h / 2
        cv.rect(x + w * 0.08, y, w * 0.84, h * 0.72, step, "#343a40", rounded=True)
        cv.rect(x + w * 0.13, y + h * 0.07, w * 0.74, h * 0.56, step, fill or "#d0ebff", sw=1)
        cv.polygon([(x, y + h), (x + w * 0.08, y + h * 0.76), (x + w * 0.92, y + h * 0.76), (x + w, y + h)],
                   step, "#ced4da")
        return (x, y, w, h)
    if k == "phone":
        w, h = s * 0.45, s * 0.85
        x, y = cx - w / 2, cy - h / 2
        cv.rect(x, y, w, h, step, "#343a40")
        cv.rect(x + w * 0.1, y + h * 0.08, w * 0.8, h * 0.76, step, fill or "#d0ebff", sw=1)
        cv.ellipse(cx - s * 0.03, y + h * 0.88, s * 0.06, s * 0.06, step, "#adb5bd", sw=1)
        return (x, y, w, h)
    if k in ("router", "switch"):
        w, h = s * 0.95, s * 0.32
        x, y = cx - w / 2, cy - h / 2 + s * 0.12
        cv.rect(x, y, w, h, step, fill or "#ffd8a8")
        if k == "router":
            for ax in (x + w * 0.2, x + w * 0.8):
                cv.line([(ax, y), (ax + (-s * 0.06 if ax < cx else s * 0.06), y - s * 0.3)], step, INK, 2.5)
            for i in range(4):
                cv.ellipse(x + w * (0.12 + i * 0.1), y + h * 0.4, s * 0.05, s * 0.05, step, "#40c057", sw=1)
        else:
            for i in range(6):
                cv.rect(x + w * (0.08 + i * 0.14), y + h * 0.35, w * 0.09, h * 0.35, step, "#495057", sw=1)
        return (x, y - s * 0.3, w, h + s * 0.3)
    if k == "cloud":
        w, h = s * 1.1, s * 0.7
        pts = []
        bumps = [(0.0, 0.62, 0.3), (0.22, 0.3, 0.3), (0.5, 0.22, 0.36), (0.78, 0.35, 0.28), (1.0, 0.62, 0.26)]
        for bx, by, br in bumps:
            for a in range(200, 341, 35):
                t = math.radians(a)
                pts.append((cx - w / 2 + bx * w + br * w * 0.55 * math.cos(t),
                            cy - h / 2 + by * h + br * h * 0.9 * math.sin(t)))
        pts += [(cx + w / 2, cy + h * 0.3), (cx - w / 2, cy + h * 0.3)]
        pts.append(pts[0])
        _poly(cv, pts, step, INK, fill or "#f1f3f5")
        return (cx - w / 2, cy - h / 2, w, h)
    if k == "attacker":
        draw_device(cv, "user", cx, cy, s, step, "#495057")
        cv.text(cx, cy - s * 0.5, "☠", step, int(s * 0.3), "#e03131", "center", "bottom", math=False)
        return (cx - s * 0.45, cy - s * 0.5, s * 0.9, s)
    if k in ("user", "users"):
        def person(px, py, sc, col):
            cv.ellipse(px - sc * 0.17, py - sc * 0.48, sc * 0.34, sc * 0.34, step, col)
            _poly(cv, [(px - sc * 0.32, py + sc * 0.45), (px - sc * 0.3, py + sc * 0.05),
                       (px, py - sc * 0.08), (px + sc * 0.3, py + sc * 0.05), (px + sc * 0.32, py + sc * 0.45),
                       (px - sc * 0.32, py + sc * 0.45)], step, INK, col)
        if k == "users":
            person(cx - s * 0.2, cy, s * 0.75, "#ffd8a8")
            person(cx + s * 0.2, cy + s * 0.03, s * 0.75, "#d0ebff")
        else:
            person(cx, cy, s, fill or "#ffd8a8")
        return (cx - s * 0.45, cy - s * 0.5, s * 0.9, s)
    if k == "file":
        w, h = s * 0.62, s * 0.85
        x, y = cx - w / 2, cy - h / 2
        f = w * 0.3
        cv.polygon([(x, y), (x + w - f, y), (x + w, y + f), (x + w, y + h), (x, y + h)], step, fill or "#ffffff")
        cv.line([(x + w - f, y), (x + w - f, y + f), (x + w, y + f)], step, INK, 1.5)
        for i in range(4):
            cv.line([(x + w * 0.15, y + h * (0.42 + i * 0.13)), (x + w * 0.85, y + h * (0.42 + i * 0.13))],
                    step, MUTED, 1.5)
        return (x, y, w, h)
    if k == "browser":
        w, h = s * 1.0, s * 0.75
        x, y = cx - w / 2, cy - h / 2
        cv.rect(x, y, w, h, step, fill or "#ffffff")
        cv.line([(x, y + h * 0.18), (x + w, y + h * 0.18)], step)
        for i, c in enumerate(("#fa5252", "#fab005", "#40c057")):
            cv.ellipse(x + w * (0.05 + i * 0.07), y + h * 0.05, s * 0.06, s * 0.06, step, c, sw=1)
        cv.rect(x + w * 0.08, y + h * 0.3, w * 0.5, h * 0.25, step, "#d0ebff", sw=1)
        for i in range(2):
            cv.line([(x + w * 0.08, y + h * (0.68 + i * 0.13)), (x + w * 0.9, y + h * (0.68 + i * 0.13))],
                    step, MUTED, 1.5)
        return (x, y, w, h)
    if k == "queue":
        w, h = s * 1.1, s * 0.4
        x, y = cx - w / 2, cy - h / 2
        for i in range(4):
            cv.rect(x + i * w * 0.25, y, w * 0.22, h, step, fill or "#ffec99")
        cv.arrow(x - s * 0.15, cy, x - 4, cy, step)
        cv.arrow(x + w, cy, x + w + s * 0.15, cy, step)
        return (x, y, w, h)
    if k == "firewall":
        w, h = s * 0.9, s * 0.75
        x, y = cx - w / 2, cy - h / 2
        cv.rect(x, y, w, h, step, fill or "#ffc9c9", rounded=False)
        for r in range(1, 4):
            cv.line([(x, y + h * r / 4), (x + w, y + h * r / 4)], step, "#c92a2a", 1.5)
        for r in range(4):
            off = 0.25 if r % 2 else 0.0
            for c in range(3):
                bx = x + w * (off + c * 0.33 + 0.33)
                if bx < x + w - 4:
                    cv.line([(bx, y + h * r / 4), (bx, y + h * (r + 1) / 4)], step, "#c92a2a", 1.5)
        return (x, y, w, h)
    if k == "internet":
        r = s * 0.42
        cv.ellipse(cx - r, cy - r, 2 * r, 2 * r, step, fill or "#d0ebff")
        cv.ellipse(cx - r * 0.45, cy - r, r * 0.9, 2 * r, step, "transparent", "#1971c2", 1.5)
        cv.line([(cx - r, cy), (cx + r, cy)], step, "#1971c2", 1.5)
        cv.line([(cx - r * 0.85, cy - r * 0.5), (cx + r * 0.85, cy - r * 0.5)], step, "#1971c2", 1)
        cv.line([(cx - r * 0.85, cy + r * 0.5), (cx + r * 0.85, cy + r * 0.5)], step, "#1971c2", 1)
        return (cx - r, cy - r, 2 * r, 2 * r)
    if k == "building":
        w, h = s * 0.7, s * 0.95
        x, y = cx - w / 2, cy - h / 2
        cv.rect(x, y, w, h, step, fill or "#e9ecef", rounded=False)
        for r in range(4):
            for c in range(3):
                cv.rect(x + w * (0.12 + c * 0.28), y + h * (0.08 + r * 0.2), w * 0.18, h * 0.12, step,
                        "#a5d8ff", sw=1, rounded=False)
        cv.rect(cx - w * 0.1, y + h * 0.84, w * 0.2, h * 0.16, step, "#868e96", sw=1, rounded=False)
        return (x, y, w, h)
    if k == "chip":
        w = s * 0.6
        x, y = cx - w / 2, cy - w / 2
        for i in range(4):
            p = x + w * (0.2 + i * 0.2)
            cv.line([(p, y - s * 0.1), (p, y)], step)
            cv.line([(p, y + w), (p, y + w + s * 0.1)], step)
            q = y + w * (0.2 + i * 0.2)
            cv.line([(x - s * 0.1, q), (x, q)], step)
            cv.line([(x + w, q), (x + w + s * 0.1, q)], step)
        cv.rect(x, y, w, w, step, fill or "#495057")
        cv.rect(x + w * 0.25, y + w * 0.25, w * 0.5, w * 0.5, step, "#adb5bd", sw=1)
        return (x, y, w, w)
    w, h = s * 0.9, s * 0.6                      # unknown: a plain box
    cv.rect(cx - w / 2, cy - h / 2, w, h, step, fill or BLUE)
    return (cx - w / 2, cy - h / 2, w, h)


def draw_thing(cv, label, cx, cy, s, step, default="laptop"):
    """A product logo when the label names one (Redis, Terraform, Amazon S3...), otherwise the engine's drawing."""
    try:
        from illustrate import icon_for
        stem = icon_for(label) or ""
    except Exception:
        stem = ""
    if stem.startswith(("aws", "logo")) and cv.icons and cv.icon(label, cx - s * 0.4, cy - s * 0.4, s * 0.8, step):
        return (cx - s * 0.4, cy - s * 0.4, s * 0.8, s * 0.8)
    return draw_device(cv, device_for(label) or default, cx, cy, s, step)


DEVICE_WORDS = [
    (r"attacker|hacker|adversary|eve\b|mallory", "attacker"),
    (r"server|backend|web server|ec2|instance|host\b|compute|vm\b|virtual machine|api\b|service\b", "server"),
    (r"database|\bdb\b|rds|dynamodb|sql|table store|aurora|data store|storage", "database"),
    (r"laptop|computer|pc\b|desktop|workstation|client\b|sender|receiver", "laptop"),
    (r"phone|mobile|smartphone|app\b", "phone"),
    (r"router|gateway|\bnat\b|hop\b", "router"),
    (r"switch|hub\b|load balancer|\balb\b|\bnlb\b", "switch"),
    (r"cloud|aws\b|azure|internet service", "cloud"),
    (r"users|people|customers|clients|team", "users"),
    (r"user|person|customer|admin|student|player|you\b", "user"),
    (r"file|document|report|log|pdf|object|s3\b|bucket", "file"),
    (r"browser|website|web page|webpage|frontend|ui\b", "browser"),
    (r"queue|buffer|sqs|kafka|stream", "queue"),
    (r"firewall|security group|waf|nacl|acl\b", "firewall"),
    (r"internet|web\b|www|global|network\b", "internet"),
    (r"on-?prem|data ?center|office|company|building|headquarters", "building"),
    (r"cpu|processor|chip|gpu|memory|ram\b", "chip"),
    (r"dns|resolver|name server|nameserver|authoritative|tld\b|root server", "server"),
]


def device_for(label):
    low = str(label or "").lower()
    for rx, kind in DEVICE_WORDS:
        if re.search(rx, low):
            return kind
    return None


def build_device(cv, v, box, step):
    x0, y0, x1, y1 = box
    s = min(x1 - x0, y1 - y0) * 0.7
    draw_device(cv, v.get("kind") or device_for(v.get("label")) or "server", (x0 + x1) / 2, (y0 + y1) / 2, s, step)
    if v.get("label"):
        cv.text((x0 + x1) / 2, (y0 + y1) / 2 + s * 0.6, v["label"], step, 22, INK, "center", "top", math=False)
    return step + 1


# ======================================================================== biology: cell

ORGANELLES = ["cell membrane", "cytoplasm", "nucleus", "nucleolus", "mitochondria", "rough er", "smooth er",
              "golgi apparatus", "ribosomes", "lysosome", "vacuole", "chloroplast", "cell wall", "centrioles"]
ALIAS = {"membrane": "cell membrane", "plasma membrane": "cell membrane", "mitochondrion": "mitochondria",
         "er": "rough er", "endoplasmic reticulum": "rough er", "rough endoplasmic reticulum": "rough er",
         "smooth endoplasmic reticulum": "smooth er", "golgi": "golgi apparatus", "golgi body": "golgi apparatus",
         "ribosome": "ribosomes", "lysosomes": "lysosome", "vacuoles": "vacuole", "chloroplasts": "chloroplast",
         "centriole": "centrioles", "centrosome": "centrioles", "peroxisome": "lysosome", "chromatin": "nucleus"}


def _org_name(s):
    s = str(s).lower().strip()
    return ALIAS.get(s, s if s in ORGANELLES else next((o for o in ORGANELLES if o.split()[0] in s), None))


def _mito(cv, cx, cy, w, h, step, ang=0.0):
    cv.ellipse(cx - w / 2, cy - h / 2, w, h, step, "#ffa8a8", "#c92a2a", 2)
    pts = []
    for i in range(9):
        t = i / 8
        pts.append((cx - w * 0.38 + t * w * 0.76, cy + (h * 0.28 if i % 2 else -h * 0.28)))
    cv.line(pts, step, "#c92a2a", 1.5)


def build_cell(cv, v, box, step):
    """{"type":"cell","kind":"animal|plant","parts":[...]} - organelles drawn + labelled one by one."""
    plant = str(v.get("kind", "animal")).lower().startswith("plant")
    parts = [p for p in (_org_name(x) for x in (v.get("parts") or v.get("organelles") or [])) if p]
    if not parts:
        parts = ["cell membrane", "nucleus", "mitochondria", "rough er", "golgi apparatus", "ribosomes"] + \
                (["cell wall", "chloroplast", "vacuole"] if plant else ["lysosome"])
    parts = list(dict.fromkeys(["cell membrane"] + parts))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cw = min(W * 0.62, H * 1.45)
    ch = min(H * 0.92, cw * 0.68)
    cx, cy = x0 + cw / 2 + 10, y0 + H / 2
    rx, ry = cw / 2, ch / 2
    lab_x = cx + rx + 50
    labels = []                                   # (name, point) -> drawn with leader lines

    cur = {"step": step}

    def label(name, px, py):
        labels.append((name, px, py, cur["step"]))

    # outline
    if plant:
        cv.rect(cx - rx, cy - ry, 2 * rx, 2 * ry, step, "#d3f9d8", "#2b8a3e", 4, rounded=False)
        cv.rect(cx - rx + 9, cy - ry + 9, 2 * rx - 18, 2 * ry - 18, step, "#f4fce3", "#5c940d", 2)
        label("cell wall" if "cell wall" in parts else "cell membrane", cx + rx, cy - ry * 0.6)
        if "cell wall" in parts:
            label("cell membrane", cx + rx - 9, cy + ry * 0.7)
    else:
        _poly(cv, _blob(cx, cy, rx, ry, wobble=0.04, seed=7), step, "#e8590c", "#fff4e6", 3)
        label("cell membrane", cx + rx * 0.96, cy - ry * 0.3)
    step += 1
    nx, ny, nr = cx - rx * 0.12, cy - ry * 0.05, min(rx, ry) * 0.36
    for name in parts[1:]:
        cur["step"] = step
        if name == "cytoplasm":
            for i in range(10):
                a = i * 2.4
                px, py = cx + rx * 0.7 * math.cos(a), cy + ry * 0.7 * math.sin(a)
                cv.ellipse(px, py, 4, 4, step, "#ced4da", "#adb5bd", 1)
            label("cytoplasm", cx + rx * 0.5, cy + ry * 0.62)
        elif name == "nucleus":
            cv.ellipse(nx - nr, ny - nr, 2 * nr, 2 * nr, step, "#d0bfff", "#5f3dc4", 3)
            rnd = random.Random(3)
            for k in range(3):
                pts = [(nx - nr * 0.6 + i * nr * 0.17, ny - nr * 0.3 + k * nr * 0.3 + rnd.uniform(-8, 8))
                       for i in range(8)]
                cv.line(pts, step, "#7048e8", 1.5)
            label("nucleus", nx + nr * 0.7, ny - nr * 0.7)
        elif name == "nucleolus":
            cv.ellipse(nx + nr * 0.1, ny + nr * 0.15, nr * 0.38, nr * 0.38, step, "#7048e8", "#5f3dc4", 1)
            label("nucleolus", nx + nr * 0.3, ny + nr * 0.35)
        elif name == "mitochondria":
            for (px, py) in [(cx + rx * 0.5, cy - ry * 0.45), (cx + rx * 0.55, cy + ry * 0.35), (cx - rx * 0.6, cy + ry * 0.45)]:
                _mito(cv, px, py, rx * 0.3, ry * 0.18, step)
            label("mitochondria", cx + rx * 0.62, cy - ry * 0.45)
        elif name == "rough er":
            for k in range(3):
                pts = [(nx + nr + 10 + i * 12, ny - nr * 0.8 + k * 22 + (6 if i % 2 else -6)) for i in range(8)]
                cv.line(pts, step, "#1971c2", 2)
                for i in range(0, 8, 2):
                    cv.ellipse(pts[i][0] - 2.5, pts[i][1] - 9, 5, 5, step, "#1971c2", "#1971c2", 1)
            label("rough er", nx + nr + 60, ny - nr * 0.7)
        elif name == "smooth er":
            for k in range(2):
                pts = [(nx - nr - 80 + i * 11, ny + nr * 0.9 + k * 20 + (6 if i % 2 else -6)) for i in range(7)]
                cv.line(pts, step, "#0c8599", 2)
            label("smooth er", nx - nr - 50, ny + nr * 1.0)
        elif name == "golgi apparatus":
            gx, gy = cx + rx * 0.18, cy + ry * 0.5
            for k in range(4):
                w = rx * (0.36 - k * 0.04)
                cv.line([(gx - w / 2, gy + k * 11), (gx, gy + k * 11 + 10), (gx + w / 2, gy + k * 11)], step,
                        "#e67700", 3)
            cv.ellipse(gx + rx * 0.22, gy - 6, 9, 9, step, "#ffd43b", "#e67700", 1)
            label("golgi apparatus", gx + rx * 0.16, gy + 8)
        elif name == "ribosomes":
            rnd = random.Random(11)
            for i in range(16):
                a, r = rnd.uniform(0, 6.28), rnd.uniform(0.45, 0.85)
                cv.ellipse(cx + rx * r * math.cos(a), cy + ry * r * math.sin(a), 5, 5, step, "#343a40", "#343a40", 1)
            label("ribosomes", cx - rx * 0.3, cy - ry * 0.75)
        elif name == "lysosome":
            lx, ly = cx - rx * 0.62, cy - ry * 0.35
            cv.ellipse(lx, ly, rx * 0.13, rx * 0.13, step, "#fcc2d7", "#c2255c", 2)
            label("lysosome", lx + 6, ly + 4)
        elif name == "vacuole":
            vx, vy = (cx + rx * 0.05, cy + ry * 0.05) if plant else (cx - rx * 0.62, cy + ry * 0.1)
            vw = rx * (0.7 if plant else 0.2)
            cv.ellipse(vx - vw / 2, vy - vw * 0.35, vw, vw * 0.7, step, "#c5f6fa", "#1098ad", 2)
            label("vacuole", vx, vy)
        elif name == "chloroplast":
            for (px, py) in [(cx - rx * 0.6, cy - ry * 0.55), (cx + rx * 0.62, cy + ry * 0.62), (cx - rx * 0.55, cy + ry * 0.6)]:
                cv.ellipse(px - rx * 0.13, py - ry * 0.09, rx * 0.26, ry * 0.18, step, "#8ce99a", "#2b8a3e", 2)
                for i in range(3):
                    cv.line([(px - rx * 0.06 + i * rx * 0.06, py - ry * 0.05), (px - rx * 0.06 + i * rx * 0.06, py + ry * 0.05)],
                            step, "#2b8a3e", 2)
            label("chloroplast", cx - rx * 0.6, cy - ry * 0.55)
        elif name == "centrioles":
            px, py = nx + nr + 20, ny + nr * 0.9
            cv.rect(px, py, 22, 8, step, "#ffe066", sw=1)
            cv.rect(px + 8, py - 8, 8, 22, step, "#ffe066", sw=1)
            label("centrioles", px + 10, py + 4)
        else:
            continue
        step += 1
    # labels on the right, leader lines to the parts, in drawing order
    n = len(labels)
    gap = min(48, (H - 20) / max(1, n))
    top = cy - gap * (n - 1) / 2
    for i, (name, px, py, st) in enumerate(sorted(labels, key=lambda l: l[2])):   # top to bottom: no crossing
        ly = top + i * gap
        cv.line([(lab_x - 8, ly), (px, py)], st, MUTED, 1)
        cv.text(lab_x, ly, name, st, 21, INK, "left", "middle", math=False)
    return step + 1


# ======================================================================== biology: mitosis

PHASES = ["interphase", "prophase", "metaphase", "anaphase", "telophase", "cytokinesis"]
PHASE_NOTE = {
    "interphase": "DNA is copied; chromosomes are loose chromatin",
    "prophase": "chromosomes condense into X shapes; spindle starts to form",
    "metaphase": "chromosomes line up on the middle of the cell",
    "anaphase": "sister chromatids are pulled to opposite poles",
    "telophase": "two new nuclei form; the cell pinches in",
    "cytokinesis": "the cytoplasm splits: two identical daughter cells",
}
CHROM = ["#e03131", "#1971c2", "#2f9e44", "#9c36b5"]


def _xchrom(cv, x, y, s, step, col, angle=0.0):
    """Replicated chromosome: two sister chromatids joined at the centromere."""
    for sgn in (-1, 1):
        a = angle + sgn * 0.35
        dx, dy = math.sin(a) * s, math.cos(a) * s
        cv.line([(x - dx, y - dy), (x, y), (x + dx * 0.9, y + dy * 0.9)], step, col, 5)
    cv.ellipse(x - 3, y - 3, 6, 6, step, INK, INK, 1)


def draw_phase(cv, phase, cx, cy, r, step):
    """Draw one mitosis phase at (cx, cy), cell radius r. Returns the elements' start index."""
    ph = phase.lower()
    t = _Track(cv)
    stretch = {"anaphase": 1.25, "telophase": 1.4}.get(ph, 1.0)
    rx, ry = r * stretch, r * (0.85 if stretch > 1 else 1)
    if ph == "cytokinesis":
        for sx in (-1, 1):
            ccx = cx + sx * r * 0.72
            _poly(cv, _blob(ccx, cy, r * 0.62, r * 0.66, seed=4 + sx), step, "#e8590c", "#fff4e6", 3)
            cv.ellipse(ccx - r * 0.24, cy - r * 0.24, r * 0.48, r * 0.48, step, "#d0bfff", "#5f3dc4", 2)
            for k in range(2):
                cv.line([(ccx - r * 0.14, cy - r * 0.08 + k * r * 0.14), (ccx, cy - r * 0.14 + k * r * 0.14),
                         (ccx + r * 0.14, cy - r * 0.04 + k * r * 0.14)], step, CHROM[k], 3)
        return t
    pinch = 0.55 if ph == "telophase" else 0.0
    _poly(cv, _blob(cx, cy, rx, ry, seed=5, pinch=pinch), step, "#e8590c", "#fff4e6", 3)
    poles = [(cx - rx * 0.88, cy), (cx + rx * 0.88, cy)]
    if ph == "interphase":
        cv.ellipse(cx - r * 0.45, cy - r * 0.45, r * 0.9, r * 0.9, step, "#e5dbff", "#5f3dc4", 3)
        rnd = random.Random(2)
        for k in range(4):
            pts = [(cx - r * 0.32 + i * r * 0.09, cy - r * 0.25 + k * r * 0.16 + rnd.uniform(-6, 6)) for i in range(8)]
            cv.line(pts, step, CHROM[k], 2)
        cv.ellipse(cx + r * 0.05, cy + r * 0.05, r * 0.16, r * 0.16, step, "#7048e8", "#5f3dc4", 1)
        cv.rect(cx + r * 0.55, cy - r * 0.62, 14, 6, step, "#ffe066", sw=1)
        return t
    # centrosomes + spindle for the dividing phases
    cpos = poles if ph != "prophase" else [(cx - r * 0.55, cy - r * 0.55), (cx + r * 0.55, cy + r * 0.55)]
    for (px, py) in cpos:
        cv.rect(px - 7, py - 3, 14, 6, step, "#ffe066", sw=1)
    if ph == "prophase":
        cv.ellipse(cx - r * 0.48, cy - r * 0.48, r * 0.96, r * 0.96, step, "transparent", "#5f3dc4", 2)
        cv.elements[-1]["strokeStyle"] = "dashed"                     # envelope breaking down
        for k, (dx, dy, a) in enumerate([(-0.2, -0.15, 0.3), (0.15, -0.2, -0.4), (-0.12, 0.2, -0.2), (0.22, 0.15, 0.5)]):
            _xchrom(cv, cx + dx * r, cy + dy * r, r * 0.14, step, CHROM[k], a)
        return t
    ys = [cy + (k - 1.5) * r * 0.32 for k in range(4)]
    if ph == "metaphase":
        cv.line([(cx, cy - r * 0.8), (cx, cy + r * 0.8)], step, MUTED, 1.5, dashed=True)
        for k, y in enumerate(ys):
            for (px, py) in poles:
                cv.line([(px, py), (cx, y)], step, "#adb5bd", 1)
            _xchrom(cv, cx, y, r * 0.13, step, CHROM[k], 1.57)
        return t
    if ph == "anaphase":
        for k, y in enumerate(ys):
            for sgn, (px, py) in zip((-1, 1), poles):
                hx = cx + sgn * rx * 0.48
                hy = cy + (y - cy) * 0.6
                cv.line([(px, py), (hx, hy)], step, "#adb5bd", 1)
                cv.line([(hx + sgn * 14, hy - 12), (hx, hy), (hx + sgn * 14, hy + 12)], step, CHROM[k], 5)
        return t
    if ph == "telophase":
        for sgn in (-1, 1):
            ncx = cx + sgn * rx * 0.52
            cv.ellipse(ncx - r * 0.3, cy - r * 0.3, r * 0.6, r * 0.6, step, "#e5dbff", "#5f3dc4", 2)
            for k in range(3):
                cv.line([(ncx - r * 0.16, cy - r * 0.12 + k * r * 0.12), (ncx, cy - r * 0.17 + k * r * 0.12),
                         (ncx + r * 0.16, cy - r * 0.09 + k * r * 0.12)], step, CHROM[k], 3)
        cv.arrow(cx, cy - ry * 0.95, cx, cy - ry * 0.45, step, color="#e8590c")
        cv.arrow(cx, cy + ry * 0.95, cx, cy + ry * 0.45, step, color="#e8590c")
        return t
    return t


def build_mitosis(cv, v, box, step):
    """One big cell going through the phases (each replaces the last), with a phase strip."""
    phases = [p.lower() for p in (v.get("phases") or PHASES) if str(p).lower() in PHASES] or PHASES
    notes = v.get("notes") or {}
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    strip_h = 70
    r = min((H - strip_h - 60) / 2.1, W * 0.2)
    cx, cy = x0 + W * 0.36, y0 + r * 1.05 + 10
    sw_ = (W - 20) / len(phases)
    for i, ph in enumerate(phases):                       # the phase strip, drawn once
        sx = x0 + 10 + i * sw_
        cv.rect(sx + 4, y1 - strip_h, sw_ - 8, strip_h - 14, step, "#f8f9fa", MUTED, 1)
        cv.text(sx + sw_ / 2, y1 - strip_h / 2 - 7, ph.title(), step, 19, INK, "center", "middle", math=False)
    step += 1
    for i, ph in enumerate(phases):
        last = i == len(phases) - 1
        until = None if last else step + 1
        t = _Track(cv)
        draw_phase(cv, ph, cx, cy, r, step)
        cv.text(x0 + W * 0.72, cy - r * 0.5, ph.title(), step, 36, "#e8590c", "left", "top", math=False)
        note, nsz = fit_text(str(notes.get(ph) or PHASE_NOTE[ph]), W * 0.27, r, 22, 15)
        cv.text(x0 + W * 0.72, cy - r * 0.5 + 56, note, step, nsz, INK, "left", "top", math=False)
        sx = x0 + 10 + i * sw_
        cv.rect(sx + 1, y1 - strip_h - 3, sw_ - 2, strip_h - 8, step, "transparent", "#e8590c", 4)
        t.tag(until=until, fade=i > 0)
        step += 1
    return step


# ======================================================================== network: packets

def _envelope(cv, x, y, w, step, label=None, until=None, col="#ffd43b"):
    h = w * 0.62
    t = _Track(cv)
    cv.rect(x - w / 2, y - h / 2, w, h, step, col, INK, 2, rounded=False)
    cv.line([(x - w / 2, y - h / 2), (x, y + h * 0.05), (x + w / 2, y - h / 2)], step, INK, 1.5)
    if label:
        cv.text(x, y - h / 2 - 6, label, step, 18, "#c92a2a", "center", "bottom", math=False)
    t.tag(until=until, fade=True)


def _nodes_row(cv, labels, box, step, size=None):
    x0, y0, x1, y1 = box
    n = len(labels)
    W = x1 - x0
    s = size or min(130, W / n * 0.55)
    cy = y0 + (y1 - y0) * 0.42
    out = []
    for i, lab in enumerate(labels):
        cx = x0 + W * (i + 0.5) / n
        b = draw_thing(cv, lab, cx, cy, s, step, "laptop" if i == 0 else "server")
        cv.text(cx, b[1] + b[3] + 10, lab, step, 20, INK, "center", "top", math=False)
        out.append((cx, cy, b))
    return out, s


def build_packet_flow(cv, v, box, step):
    """{"type":"packet_flow","nodes":["Laptop","Router","DNS server"],
        "hops":[{"from":"Laptop","to":"Router","label":"query"}, ...]}"""
    labels = [n.get("label") if isinstance(n, dict) else str(n) for n in (v.get("nodes") or [])][:6]
    if len(labels) < 2:
        return step
    hops = v.get("hops") or [{"from": labels[i], "to": labels[i + 1], "label": ""} for i in range(len(labels) - 1)]
    pos, s = _nodes_row(cv, labels, box, step)
    step += 1
    idx = {l.lower(): i for i, l in enumerate(labels)}
    for i in range(len(labels) - 1):                      # the links
        a, b = pos[i], pos[i + 1]
        cv.line([(a[2][0] + a[2][2] + 6, a[1]), (b[2][0] - 6, b[1])], step, MUTED, 2, dashed=True)
    step += 1
    x0, y0, x1, y1 = box
    log_y = y0 + (y1 - y0) * 0.78
    for h_i, hop in enumerate(hops[:10]):
        if not isinstance(hop, dict):
            continue
        a = idx.get(str(hop.get("from", "")).lower())
        b = idx.get(str(hop.get("to", "")).lower())
        if a is None or b is None:
            continue
        (ax, ay, _), (bx, by, _) = pos[a], pos[b]
        lift = -s * 0.75 if b > a else -s * 1.15           # replies travel higher, in blue
        for k in range(4):                                 # 4 positions per hop = movement
            f = k / 3
            px, py = ax + (bx - ax) * f, ay + lift + (by - ay) * f
            last_hop = h_i == len(hops) - 1 and k == 3
            _envelope(cv, px, py, s * 0.42, step, hop.get("label") if k in (1, 2) else None,
                      until=None if last_hop else step + 1,
                      col="#ffd43b" if b > a else "#a5d8ff")
            step += 1
        if hop.get("label"):
            cv.text(x0 + 20, log_y + h_i * 26, f"{h_i + 1}. {hop.get('from')} → {hop.get('to')}: {hop['label']}",
                    step - 1, 18, "#495057", "left", "top", math=False)
    return step


def build_sequence(cv, v, box, step):
    """{"type":"sequence","actors":["Client","Server"],
        "messages":[{"from":"Client","to":"Server","label":"SYN"}, ...]}"""
    actors = [a.get("label") if isinstance(a, dict) else str(a) for a in (v.get("actors") or v.get("nodes") or [])][:5]
    msgs = [m for m in (v.get("messages") or v.get("hops") or []) if isinstance(m, dict)][:9]
    if len(actors) < 2 or not msgs:
        return step
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    s = min(95, H * 0.17)
    xs = [x0 + W * (i + 0.5) / len(actors) for i in range(len(actors))]
    for x, a in zip(xs, actors):
        b = draw_thing(cv, a, x, y0 + s * 0.55, s, step)
        cv.text(x, y0 + s * 1.12, a, step, 20, INK, "center", "top", math=False)
        cv.line([(x, y0 + s * 1.5), (x, y1 - 4)], step, MUTED, 1.5, dashed=True)
    step += 1
    idx = {a.lower(): i for i, a in enumerate(actors)}
    top = y0 + s * 1.75
    gap = min(64, (y1 - top - 10) / len(msgs))
    for k, m in enumerate(msgs):
        a, b = idx.get(str(m.get("from", "")).lower()), idx.get(str(m.get("to", "")).lower())
        if a is None or b is None or a == b:
            continue
        y = top + k * gap + gap * 0.5
        col = INKS[k % len(INKS)]
        cv.line([(xs[a], y), (xs[b], y + gap * 0.25)], step, col, 2.5, arrow_end=True)
        mid = (xs[a] + xs[b]) / 2
        cv.text(mid, y + gap * 0.12 - 4, str(m.get("label", "")), step, 19, col, "center", "bottom", math=False)
        if m.get("note"):
            cv.text(max(xs) + 30 if max(xs) + 30 < x1 - 120 else x1 - 220, y, str(m["note"])[:40], step, 16, MUTED,
                    "left", "middle", math=False)
        step += 1
    return step


# ======================================================================== code trace

def _short(val, n=26):
    s = repr(val) if not isinstance(val, str) else repr(val)
    return s if len(s) <= n else s[:n - 1] + "…"


def build_code_trace(cv, v, box, step):
    """{"type":"code_trace","code":"...","steps":[{"line":3,"vars":{"i":0},"array":[...],
        "array_name":"a","pointers":{"lo":0,"hi":6},"note":"..."}]}"""
    code = str(v.get("code") or "").rstrip("\n")
    lines = code.split("\n")[:22]
    steps = [s for s in (v.get("steps") or []) if isinstance(s, dict)][:16]
    if not lines or not steps:
        return step
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cw = W * 0.5
    lh = min(30, (H - 20) / max(len(lines), 1))
    fs = max(12, min(19, lh * 0.66, (cw - 60) / max(len(max(lines, key=len)), 1) / 0.6))
    cv.rect(x0, y0, cw, lh * len(lines) + 24, step, "#f8f9fa", "#495057", 2)
    for i, ln in enumerate(lines):
        el = cv.text(x0 + 14, y0 + 12 + i * lh, f"{i + 1:>2}", step, fs, MUTED, "left", "top", math=False)
        if el:
            el["fontFamily"] = 3
        el = cv.text(x0 + 14 + fs * 2.2, y0 + 12 + i * lh, ln.replace("\t", "    ") or " ", step, fs, INK, "left",
                     "top", math=False)
        if el:
            el["fontFamily"] = 3
    step += 1
    rx = x0 + cw + 40
    rw = x1 - rx
    n = len(steps)
    for k, st in enumerate(steps):
        until = None if k == n - 1 else step + 1
        t = _Track(cv)
        ln = int(st.get("line") or 0)
        if 1 <= ln <= len(lines):
            hl = cv.rect(x0 + 4, y0 + 10 + (ln - 1) * lh, cw - 8, lh, step, "#ffec99", "#f08c00", 1.5)
            hl["opacity"] = 55
            cv.text(x0 + cw + 6, y0 + 10 + (ln - 1) * lh + lh / 2, "◀", step, fs, "#f08c00", "left", "middle",
                    math=False)
        cv.text(rx, y0, f"step {k + 1}/{n}", step, 18, MUTED, "left", "top", math=False)
        yy = y0 + 32
        vars_ = st.get("vars") or {}
        if isinstance(vars_, dict) and vars_:
            cv.text(rx, yy, "variables", step, 18, "#1971c2", "left", "top", math=False)
            yy += 28
            for name, val in list(vars_.items())[:6]:
                cv.rect(rx, yy, rw * 0.35, 34, step, "#e7f5ff", "#1971c2", 1.5)
                cv.text(rx + 10, yy + 17, str(name)[:14], step, 18, INK, "left", "middle", math=False)
                cv.rect(rx + rw * 0.35, yy, rw * 0.6, 34, step, "#ffffff", "#1971c2", 1.5)
                e2 = cv.text(rx + rw * 0.35 + 10, yy + 17, _short(val), step, 17, INK, "left", "middle", math=False)
                if e2:
                    e2["fontFamily"] = 3
                yy += 36
        arr = st.get("array")
        if isinstance(arr, list) and arr:
            yy += 14
            cv.text(rx, yy, str(st.get("array_name") or v.get("array_name") or "array"), step, 18, "#2f9e44",
                    "left", "top", math=False)
            yy += 28
            m = min(len(arr), 12)
            cwid = min(64, rw / m)
            hi = set(st.get("highlight") or [])
            ptr = st.get("pointers") or {}
            for j, val in enumerate(arr[:m]):
                fill = "#b2f2bb" if j in hi else "#ffffff"
                if isinstance(ptr, dict) and ptr and not (min([p for p in ptr.values() if isinstance(p, int)] or [0]) <= j
                                                          <= max([p for p in ptr.values() if isinstance(p, int)] or [m])):
                    fill = "#e9ecef"
                cv.rect(rx + j * cwid, yy, cwid - 4, 44, step, fill, INK, 1.5, rounded=False)
                cv.text(rx + j * cwid + (cwid - 4) / 2, yy + 22, _short(val, 6), step, 17, INK, "center", "middle",
                        math=False)
                cv.text(rx + j * cwid + (cwid - 4) / 2, yy + 48, str(j), step, 13, MUTED, "center", "top", math=False)
            if isinstance(ptr, dict):
                for pi, (pname, pidx) in enumerate(list(ptr.items())[:4]):
                    if isinstance(pidx, int) and 0 <= pidx < m:
                        px = rx + pidx * cwid + (cwid - 4) / 2
                        cv.arrow(px, yy + 100 + pi * 4, px, yy + 66, step, color=INKS[pi % 4])
                        cv.text(px, yy + 104 + pi * 4, str(pname), step, 16, INKS[pi % 4], "center", "top", math=False)
            yy += 130
        if st.get("note"):
            note, nsz = fit_text(str(st["note"]), rw, max(40, y1 - yy), 19, 13)
            cv.text(rx, yy, note, step, nsz, "#c92a2a", "left", "top", math=False)
        if st.get("output"):
            cv.text(x0, y0 + lh * len(lines) + 34, "output: " + str(st["output"])[:60], step, 18, "#2f9e44",
                    "left", "top", math=False)
        t.tag(until=until, fade=True)
        step += 1
    return step


# ======================================================================== register

visuals.BUILDERS.update({"cell": build_cell, "mitosis": build_mitosis, "packet_flow": build_packet_flow,
                         "sequence": build_sequence, "code_trace": build_code_trace, "device": build_device})
for _k in ("sequence",):                       # v3 maps "sequence" to a flow; ours is a protocol diagram
    visuals.ALIASES.pop(_k, None)
visuals.ALIASES.update({"cell_diagram": "cell", "organelles": "cell", "cell_division": "mitosis",
                        "packets": "packet_flow", "packet": "packet_flow", "network_flow": "packet_flow",
                        "sequence_diagram": "sequence", "handshake": "sequence", "trace_code": "code_trace",
                        "code": "code_trace", "runtime_trace": "code_trace"})
