"""
StudyMate icon resolver: concept word -> icon for an Excalidraw scene.

Sources, tried in order (change PREFER to reorder):
  1. excalidraw  - community Excalidraw libraries (hand-drawn, native shapes, MIT)
                   git clone --depth 1 https://github.com/excalidraw/excalidraw-libraries
  2. bioicons    - 2,800+ life-science SVG icons (CC0 / CC-BY / CC-BY-SA, credit needed)
                   git clone --depth 1 https://github.com/duerrsimon/bioicons
  3. iconify     - 200k+ general icons via https://api.iconify.design (needs internet)
  4. emoji       - always available fallback

Usage:
    from icons import IconResolver
    icons = IconResolver(excalidraw_dir="icons/excalidraw-libraries",
                         bioicons_dir="icons/bioicons")
    hit = icons.find("mitochondrion")      # -> IconHit or None
    els, files = hit.to_excalidraw(x, y, size=90)

Matching uses word matching by default. If Ollama is running with
nomic-embed-text, pass use_embeddings=True to also match by meaning
("powerhouse of the cell" -> mitochondrion).
"""
import base64
import copy
import difflib
import json
import os
import re
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

PREFER = ["excalidraw", "bioicons", "iconify", "emoji"]
# libraries whose items are abstract symbols, not pictures (add more as you find them)
BLOCK_LIBRARIES = {"astronomical symbols", "despair", "hearts"}
MIN_SCORE = {"excalidraw": 0.82, "bioicons": 0.82, "iconify": 0.0, "emoji": 0.0}
ICONIFY_SETS = "openmoji,noto,fluent-emoji-flat,streamline-color,mdi"  # colorful sets first
CACHE_DIR = Path(".icon_cache")

# other names the LLM might use -> the name an icon actually has. Extend freely.
ALIASES = {"hard drive": "hard disk", "hdd": "hard disk", "ssd": "disk", "storage": "disk",
           "hash table": "hashing table", "hash map": "hashing table", "hashmap": "hashing table",
           "and gate": "and", "or gate": "or", "not gate": "not", "xor gate": "xor",
           "browser": "web app", "web browser": "web app", "website": "web app",
           "web page": "web app", "webpage": "web app", "pc": "computer", "desktop": "computer",
           "mobile phone": "smartphone", "phone": "smartphone", "access point": "wifi",
           "mitochondria": "mitochondrion", "powerhouse": "mitochondrion"}

EMOJI = {"sun": "☀️", "leaf": "🍃", "plant": "🌱", "tree": "🌳", "water": "💧", "cell": "🦠",
         "bacteria": "🦠", "virus": "🦠", "dna": "🧬", "gene": "🧬", "heart": "❤️",
         "brain": "🧠", "lung": "🫁", "bone": "🦴", "blood": "🩸", "microscope": "🔬",
         "energy": "⚡", "atp": "⚡", "fire": "🔥", "glucose": "🍬", "oxygen": "🫧",
         "server": "🖥️", "computer": "💻", "router": "📡", "database": "🗄️", "cloud": "☁️",
         "lock": "🔒", "key": "🔑", "email": "📧", "phone": "📱", "globe": "🌍",
         "earth": "🌍", "book": "📘", "clock": "⏰", "money": "💰", "factory": "🏭"}


# ---------------- text matching ----------------

def _tokens(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))
    s = re.sub(r"[_\-./]+|\d+", " ", s.lower())
    out = []
    for t in s.split():
        if len(t) > 3 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def _same(a, b):
    """Same word, allowing Latin/English plural endings: mitochondrion ~ mitochondria."""
    if a == b:
        return True
    p = len(os.path.commonprefix([a, b]))
    return p >= 6 and p >= max(len(a), len(b)) - 3


def word_score(query, name):
    """1.0 = same words; ~0.9 = name contains all query words; else fuzzy ratio."""
    q, n = _tokens(query), _tokens(name)
    if not q or not n:
        return 0.0
    if len(q) == len(n) and all(_same(a, b) for a, b in zip(q, n)):
        return 1.0
    if all(any(_same(t, u) for u in n) for t in q):
        return 0.95 - 0.03 * (len(n) - len(q))      # shorter names win
    a, b = " ".join(q), " ".join(n)
    return difflib.SequenceMatcher(None, a, b).ratio() * 0.9


# ---------------- results ----------------

@dataclass
class IconHit:
    source: str            # excalidraw | bioicons | iconify | emoji
    name: str
    score: float
    elements: list = None  # excalidraw library elements
    svg: bytes = None      # svg file contents
    emoji: str = None
    credit: str = ""       # author + license, for the video credits
    extra: dict = field(default_factory=dict)

    def to_excalidraw(self, x, y, size=90):
        """Return (elements, files) placed in a size x size box with top-left (x, y)."""
        if self.elements is not None:
            return place_library_item(self.elements, x, y, size), {}
        if self.svg is not None:
            return svg_image_element(self.svg, x, y, size)
        if self.emoji:
            return [_emoji_text(self.emoji, x, y, size)], {}
        return [], {}


# ---------------- excalidraw library items ----------------

def _load_library_items(root):
    items = []
    for p in Path(root).glob("libraries/**/*.excalidrawlib"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        lib_name = p.stem.replace("-", " ")
        raw = d.get("libraryItems") or d.get("library") or []
        for i, it in enumerate(raw):
            if isinstance(it, dict):
                name, els = it.get("name") or "", it.get("elements") or []
            else:                       # old format: a bare list of elements
                name, els = "", it
            els = [e for e in els if isinstance(e, dict) and not e.get("isDeleted")]
            if not els or any(e.get("type") == "image" for e in els):
                continue                # skip items needing external image files
            if not name:                # unnamed: use any text inside the item
                texts = [e.get("text", "") for e in els if e.get("type") == "text"]
                name = " ".join(texts)[:40]
            if name.strip() and lib_name not in BLOCK_LIBRARIES:
                items.append({"name": name.strip(), "library": lib_name,
                              "path": str(p), "index": i, "elements": els})
    return items


def place_library_item(elements, x, y, size):
    """Copy a library item, give it fresh ids, scale it into a size x size box at (x, y)."""
    els = copy.deepcopy(elements)
    minx = min(e["x"] for e in els)
    miny = min(e["y"] for e in els)
    maxx = max(e["x"] + max(e.get("width", 0), 0) for e in els)
    maxy = max(e["y"] + max(e.get("height", 0), 0) for e in els)
    k = size / max(maxx - minx, maxy - miny, 1)
    ox = x + (size - (maxx - minx) * k) / 2
    oy = y + (size - (maxy - miny) * k) / 2
    new_id = {e["id"]: uuid.uuid4().hex[:16] for e in els}
    new_group = {}
    group = uuid.uuid4().hex[:16]
    for e in els:
        e["id"] = new_id[e["id"]]
        e["x"] = ox + (e["x"] - minx) * k
        e["y"] = oy + (e["y"] - miny) * k
        e["width"] = e.get("width", 0) * k
        e["height"] = e.get("height", 0) * k
        if "points" in e:
            e["points"] = [[px * k, py * k] for px, py in e["points"]]
        if "fontSize" in e:
            e["fontSize"] = e["fontSize"] * k
        e["strokeWidth"] = max(1, e.get("strokeWidth", 1) * min(k, 1.5))
        e["groupIds"] = [new_group.setdefault(g, uuid.uuid4().hex[:16])
                         for g in e.get("groupIds", [])] + [group]
        if e.get("containerId"):
            e["containerId"] = new_id.get(e["containerId"])
        e["boundElements"] = [dict(b, id=new_id[b["id"]]) for b in (e.get("boundElements") or [])
                              if b.get("id") in new_id] or None
        for key in ("startBinding", "endBinding"):
            b = e.get(key)
            e[key] = dict(b, elementId=new_id[b["elementId"]]) if b and b.get("elementId") in new_id else None
        e["frameId"] = None
        e["locked"] = False
        e["version"] = 1
    return els


# ---------------- svg icons ----------------

def _svg_size(svg):
    head = svg[:2000].decode("utf-8", "ignore")
    m = re.search(r'viewBox="\s*[-\d.]+[\s,]+[-\d.]+[\s,]+([\d.]+)[\s,]+([\d.]+)', head)
    if m:
        return float(m.group(1)), float(m.group(2))
    w = re.search(r'\bwidth="([\d.]+)', head)
    h = re.search(r'\bheight="([\d.]+)', head)
    return (float(w.group(1)), float(h.group(1))) if w and h else (1.0, 1.0)


def crop_svg(svg):
    """Tighten the viewBox to the drawn content (many icon SVGs have big empty margins).
    Needs `pip install cairosvg`; returns the svg unchanged if that isn't available."""
    try:
        import io
        import cairosvg
        from PIL import Image
    except ImportError:
        return svg
    head = svg[:3000].decode("utf-8", "ignore")
    m = re.search(r'viewBox="\s*([-\d.]+)[\s,]+([-\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)\s*"', head)
    if not m:
        return svg
    vx, vy, vw, vh = map(float, m.groups())
    try:
        png = cairosvg.svg2png(bytestring=svg, output_width=256)
        img = Image.open(io.BytesIO(png)).convert("RGBA")
        box = img.getchannel("A").getbbox()
    except Exception:
        return svg
    if not box:
        return svg
    sx, sy = vw / img.width, vh / img.height
    pad = 0.04 * max(vw, vh)
    nx, ny = vx + box[0] * sx - pad, vy + box[1] * sy - pad
    nw, nh = (box[2] - box[0]) * sx + 2 * pad, (box[3] - box[1]) * sy + 2 * pad
    new = f'viewBox="{nx:.2f} {ny:.2f} {nw:.2f} {nh:.2f}"'
    text = svg.decode("utf-8", "ignore").replace(m.group(0), new, 1)
    tag = re.search(r"<svg\b[^>]*>", text)
    if tag:                                   # drop fixed width/height so the viewBox rules
        clean = re.sub(r'\s(width|height)="[^"]*"', "", tag.group(0))
        text = text[:tag.start()] + clean + text[tag.end():]
    return text.encode("utf-8")


def svg_image_element(svg, x, y, size):
    svg = crop_svg(svg)
    vw, vh = _svg_size(svg)
    k = size / max(vw, vh, 1e-6)
    w, h = vw * k, vh * k
    file_id = uuid.uuid4().hex
    files = {file_id: {"id": file_id, "mimeType": "image/svg+xml",
                       "dataURL": "data:image/svg+xml;base64," + base64.b64encode(svg).decode(),
                       "created": int(time.time() * 1000)}}
    el = {"id": uuid.uuid4().hex[:16], "type": "image", "fileId": file_id, "status": "saved",
          "scale": [1, 1], "x": x + (size - w) / 2, "y": y + (size - h) / 2,
          "width": w, "height": h, "angle": 0, "strokeColor": "transparent",
          "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": 1,
          "strokeStyle": "solid", "roughness": 0, "opacity": 100, "groupIds": [],
          "frameId": None, "roundness": None, "seed": 1, "version": 1, "versionNonce": 1,
          "isDeleted": False, "boundElements": None, "updated": 1, "link": None, "locked": False}
    return [el], files


def _emoji_text(ch, x, y, size):
    fs = size * 0.8
    return {"id": uuid.uuid4().hex[:16], "type": "text", "x": x, "y": y, "width": size,
            "height": size, "angle": 0, "strokeColor": "#1e1e1e", "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0,
            "opacity": 100, "groupIds": [], "frameId": None, "roundness": None, "seed": 1,
            "version": 1, "versionNonce": 1, "isDeleted": False, "boundElements": None,
            "updated": 1, "link": None, "locked": False, "text": ch, "originalText": ch,
            "fontSize": fs, "fontFamily": 5, "textAlign": "center", "verticalAlign": "middle",
            "containerId": None, "lineHeight": 1.25, "autoResize": True}


def _load_bioicons(root):
    items = []
    for p in Path(root).glob("static/icons/*/*/*/*.svg"):
        lic, category, author = p.parts[-4], p.parts[-3], p.parts[-2]
        items.append({"name": p.stem, "category": category.replace("_", " "),
                      "author": author.replace("-", " "), "license": lic, "path": str(p)})
    return items


# ---------------- optional embeddings ----------------

class _Embedder:
    def __init__(self, model="nomic-embed-text"):
        import numpy as np
        import ollama
        self.np, self.ollama, self.model = np, ollama, model

    def embed(self, texts):
        out = []
        for i in range(0, len(texts), 256):
            out += self.ollama.embed(model=self.model, input=texts[i:i + 256])["embeddings"]
        v = self.np.array(out, dtype="float32")
        return v / self.np.linalg.norm(v, axis=1, keepdims=True)


LIFE_SCIENCE = ("bio", "medic", "anatom", "physiol", "health", "life science", "ecolog",
                "genetic", "microbio", "immun", "zoolog", "botan", "nursing", "neuro", "cell")


def is_life_science(subject):
    if not subject:
        return True                      # unknown subject: allow everything
    s = str(subject).lower()
    return any(k in s for k in LIFE_SCIENCE)


# ---------------- the resolver ----------------

class IconResolver:
    def __init__(self, excalidraw_dir=None, bioicons_dir=None, use_iconify=True,
                 use_embeddings=False, prefer=PREFER):
        self.prefer = prefer
        self.use_iconify = use_iconify
        self.sources = {"excalidraw": _load_library_items(excalidraw_dir) if excalidraw_dir else [],
                        "bioicons": _load_bioicons(bioicons_dir) if bioicons_dir else []}
        self.emb = None
        self.vecs = {}
        if use_embeddings:
            try:
                self.emb = _Embedder()
                CACHE_DIR.mkdir(exist_ok=True)
                for src, items in self.sources.items():
                    self.vecs[src] = self._cached_vectors(src, items)
            except Exception as ex:            # Ollama not running etc.
                print(f"[icons] embeddings off: {ex}")
                self.emb = None
        print("[icons] " + ", ".join(f"{k}: {len(v)}" for k, v in self.sources.items()))

    def _cached_vectors(self, src, items):
        np = self.emb.np
        path = CACHE_DIR / f"{src}_{len(items)}.npy"
        if path.exists():
            return np.load(path)
        v = self.emb.embed([self._search_text(src, it) for it in items])
        np.save(path, v)
        return v

    @staticmethod
    def _search_text(src, it):
        return it["name"] if src == "excalidraw" else f'{it["name"]} {it["category"]}'

    def _best_local(self, src, concept):
        items = self.sources.get(src) or []
        if not items:
            return None, 0.0
        scores = [word_score(concept, it["name"]) for it in items]
        if self.emb is not None and src in self.vecs:
            q = self.emb.embed([concept])[0]
            sem = self.vecs[src] @ q
            scores = [max(w, float(s)) for w, s in zip(scores, sem)]
        best = max(range(len(items)), key=scores.__getitem__)
        return items[best], scores[best]

    def _iconify(self, concept):
        q = urllib.parse.quote(concept)
        url = f"https://api.iconify.design/search?query={q}&limit=5&prefixes={ICONIFY_SETS}"
        try:
            with urllib.request.urlopen(url, timeout=8) as r:
                icons = json.load(r).get("icons", [])
            if not icons:
                return None
            prefix, name = icons[0].split(":", 1)
            with urllib.request.urlopen(f"https://api.iconify.design/{prefix}/{name}.svg?height=128",
                                        timeout=8) as r:
                svg = r.read()
            return IconHit("iconify", icons[0], 0.8, svg=svg, credit=f"Iconify: {icons[0]}")
        except Exception:
            return None

    def find(self, concept, subject=None):
        """Best icon for a concept, following PREFER order. None if nothing fits.
        Pass the lesson subject so life-science icons aren't used for, say, maths
        ("cone" would otherwise match a retina cone cell)."""
        if not concept:
            return None
        concept = str(concept).strip()
        self._skip = set() if is_life_science(subject) else {"bioicons"}
        hit = self._find(concept)
        alias = ALIASES.get(concept.lower())
        if (hit is None or hit.source in ("iconify", "emoji")) and alias:
            better = self._find(alias)
            if better is not None and better.source not in ("iconify", "emoji"):
                return better
        return hit

    def _find(self, concept):
        for src in self.prefer:
            if src in getattr(self, "_skip", ()):
                continue
            if src in ("excalidraw", "bioicons"):
                it, score = self._best_local(src, concept)
                if it is None or score < MIN_SCORE[src]:
                    continue
                if src == "excalidraw":
                    return IconHit(src, it["name"], score, elements=it["elements"],
                                   credit=f'Excalidraw library "{it["library"]}" (MIT)')
                return IconHit(src, it["name"], score, svg=Path(it["path"]).read_bytes(),
                               credit=f'Bioicons: {it["name"]} by {it["author"]} ({it["license"]})')
            if src == "iconify" and self.use_iconify:
                hit = self._iconify(concept)
                if hit:
                    return hit
            if src == "emoji":
                for t in _tokens(concept):
                    if t in EMOJI:
                        return IconHit("emoji", t, 0.5, emoji=EMOJI[t])
        return None
