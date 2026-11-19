"""
One lesson scene -> Excalidraw document + animation timeline.

Scene (whatever the model returned, normalised by planner.normalise_scene):
    {"title": str, "narration": str, "note": str,
     "visuals": [ {...}, ... ]}      # 1-2 visuals drawn side by side

Screen layout (1600 x 900, same as the reference video):
    "Subject · Scene N"  ...  title
    [ visual(s) ...................... ]  [ sticky note ]
    [ dark caption bar: narration, paged so long explanations fit ]
"""
import re

from canvas import Canvas, INK, MUTED, fit_text, text_size, wrap
from mathtext import pretty
from visuals import draw_visual

W, H = 1600, 900
WORDS_PER_SEC = 2.5          # comfortable reading speed for captions
CAPTION_SIZE = 21
CAPTION_LINES = 4            # lines per caption page


def caption_pages(narration):
    """Split narration into pages of <= CAPTION_LINES lines, breaking at sentences."""
    chars = int((W - 300) / (CAPTION_SIZE * 0.55))
    sentences = re.split(r"(?<=[.!?:])\s+", pretty(narration).strip())
    pages, cur = [], ""
    for s in sentences:
        trial = (cur + " " + s).strip()
        if cur and wrap(trial, chars).count("\n") + 1 > CAPTION_LINES:
            pages.append(cur)
            cur = s
        else:
            cur = trial
        while wrap(cur, chars).count("\n") + 1 > CAPTION_LINES:     # one huge sentence
            lines = wrap(cur, chars).split("\n")
            pages.append(" ".join(lines[:CAPTION_LINES]))
            cur = " ".join(lines[CAPTION_LINES:])
    if cur:
        pages.append(cur)
    return [wrap(p, chars) for p in pages if p.strip()], chars


def build_scene(scene, index, subject, icons=None):
    cv = Canvas(seed=index * 7919)
    cv.icons = icons
    cv.subject = subject
    # invisible frame: fixes the exported picture to exactly 1600 x 900
    cv.rect(0, 0, W, H, "frame", "transparent", "transparent", 1, rounded=False)
    cv.text(30, 22, f"{subject} · Scene {index}", "title", 19, MUTED, math=False)
    title, tsz = fit_text(pretty(scene["title"]), W - 200, 70, 42, 26)
    cv.text(W / 2, 64, title, "title", tsz, INK, "center", "middle", math=False)

    note = scene.get("note") or ""
    area_x1 = W - 60
    if note:
        nw = 330
        wrapped, nsz = fit_text(pretty(note), nw - 40, 260, 21, 15)
        _, th = text_size(wrapped, nsz)
        cv.rect(W - nw - 40, 150, nw, th + 40, "note", "#fff3bf", "#e8c547", 1.5,
                angle=0.025)
        cv.text(W - nw - 20, 170, wrapped, "note", nsz, INK, math=False)
        area_x1 = W - nw - 70

    pages, _ = caption_pages(scene.get("narration") or "")
    cap_h = CAPTION_LINES * CAPTION_SIZE * 1.25 + 36
    cap_y = H - 30 - cap_h
    area = (70, 140, area_x1, cap_y - 30)

    visuals = scene.get("visuals") or []
    step = 0
    fallback = scene.get("points") or [s for s in re.split(r"(?<=[.!?])\s+",
                                                           scene.get("narration", ""))
                                       if s.strip()][:5]
    if not visuals:
        visuals = [{"type": "list", "items": fallback}]
    n = min(len(visuals), 3)
    x0, y0, x1, y1 = area
    gap = 40
    stacked = n > 1 and all(_is_wide(v) for v in visuals[:n])
    for i, v in enumerate(visuals[:n]):
        if stacked:      # wide visuals (tables, traces) stack vertically
            h = (y1 - y0 - gap * (n - 1)) / n
            box = (x0, y0 + i * (h + gap), x1, y0 + i * (h + gap) + h)
        else:
            w = (x1 - x0 - gap * (n - 1)) / n
            box = (x0 + i * (w + gap), y0, x0 + i * (w + gap) + w, y1)
        step = draw_visual(cv, v, box, step, fallback)

    if pages:
        cv.rect(110, cap_y, W - 220, cap_h, "bar", "#343a40", "#343a40", 1)
        for k, p in enumerate(pages):
            cv.text(140, cap_y + 18, p, f"page{k}", CAPTION_SIZE, "#ffffff", math=False)

    doc = {"type": "excalidraw", "version": 2, "source": "studymate-v3",
           "elements": cv.elements,
           "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
           "files": cv.files}
    timeline, duration = plan_timeline(cv.elements, pages)
    return doc, timeline, duration, cv.credits


def _is_wide(v):
    t = str((v or {}).get("type", "")).lower() if isinstance(v, dict) else ""
    return t in ("table", "array", "trace", "comparison", "timeline")


def plan_timeline(elements, pages):
    """Start/duration/end (seconds) for every element, in element order."""
    steps = sorted({e["customData"]["step"] for e in elements
                    if isinstance(e["customData"]["step"], int)})
    K = max(1, len(steps))
    page_secs = [max(3.5, len(p.split()) / WORDS_PER_SEC + 0.8) for p in pages]
    read = sum(page_secs)
    T = max(read + 1.4, 1.4 + K * 1.1 + 2.0, 5.0)
    # captions are stretched over the whole scene
    scale = (T - 1.0) / read if read else 1
    starts, t = [], 0.8
    for s in page_secs:
        starts.append(t)
        t += s * scale
    # visual steps are spread over the first ~80% so drawing follows the explanation
    v0 = 1.2
    v_end = max(v0 + K * 1.0, T * 0.8)
    gap = (v_end - v0) / K
    step_pos = {s: i for i, s in enumerate(steps)}
    count = {}
    for e in elements:
        count[e["customData"]["step"]] = count.get(e["customData"]["step"], 0) + 1
    seen = {}
    tl = []
    for e in elements:
        st = e["customData"]["step"]
        j = seen.get(st, 0)
        seen[st] = j + 1
        if st == "frame":
            tl.append({"s": 0, "d": 0.01})
        elif st == "title":
            tl.append({"s": 0.15 + j * 0.25, "d": 0.8})
        elif st == "note":
            tl.append({"s": v0 + gap * min(K, 2) * 0.6 + j * 0.3, "d": 0.8})
        elif st == "bar":
            tl.append({"s": 0.5, "d": 0.35, "fade": True})
        elif isinstance(st, str) and st.startswith("page"):
            k = int(st[4:])
            end = starts[k + 1] if k + 1 < len(starts) else None
            tl.append({"s": starts[k], "d": 0.25, "e": end, "fade": True})
        else:
            i = step_pos[st]
            n = count[st]
            stagger = min(0.18, gap * 0.55 / max(1, n))
            tl.append({"s": v0 + i * gap + j * stagger, "d": min(1.0, max(0.35, gap * 0.7))})
    return tl, round(T + 0.8, 2)
