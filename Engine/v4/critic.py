"""
StudyMate v4 - evaluate -> refine (the AutoFigure idea, without a second model).

Every planned scene is checked by rules a human reviewer would apply; each
failed check is FIXED automatically, and every fix is written to critic.json so
you can see what changed.

Checks
  coverage     every source unit (bullet, label, sentence) appears in the scene
               (narration or visual). Missing ones are appended to the narration.
  overflow     too many list items / table rows / flow steps for one screen
               -> the scene is split into "(1/2)", "(2/2)".
  long labels  a node/step label longer than a phrase -> short label in the box,
               the full sentence goes to the narration (word for word kept).
  empty        a scene with no visual gets a key-point list.
  narration    too long for one scene (> MAX_WORDS) -> split.
  duplicates   two scenes with the same narration -> second removed.
"""
import copy
import json
import re

MAX_LIST, MAX_ROWS, MAX_STEPS, MAX_LABEL, MAX_WORDS = 7, 7, 6, 70, 190
STOP = set("a an the and or of to in on at for by with from as is are be it this that you your can "
           "e g".split())


def _words(s):
    return {w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if w not in STOP}


def covered(unit, scene_words, threshold=0.75):
    w = _words(unit)
    return not w or len(w & scene_words) / len(w) >= threshold


def _scene_words(sc):
    vis = [{k: v for k, v in x.items() if k != "png"} for x in sc.get("visuals", [])]
    return _words(sc.get("title", "") + " " + sc.get("narration", "") + " " +
                  json.dumps(vis, ensure_ascii=False))


def _shorten(label):
    s = re.split(r"\s[–—-]\s|:|\(|,|;|\.\s", label)[0].strip()
    return s if 2 < len(s) <= MAX_LABEL else label[:MAX_LABEL - 1].rsplit(" ", 1)[0] + "…"


def _chunks(seq, n):
    k = (len(seq) + n - 1) // n
    size = (len(seq) + k - 1) // k
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _split_sentences(text, max_words):
    sents = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], ""
    for s in sents:
        if cur and len((cur + " " + s).split()) > max_words:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


def refine_scene(sc, log):
    sc = copy.deepcopy(sc)
    sid = sc.get("id", "?")
    vis = sc.get("visuals") or []
    if not vis:
        items = [u for u in sc.get("units", []) if u][:MAX_LIST] or \
            re.split(r"(?<=[.!?])\s+", sc.get("narration", ""))[:5]
        sc["visuals"] = vis = [{"type": "list", "items": items}]
        log.append({"scene": sid, "check": "empty", "fix": "added key-point list"})
    # long labels
    for v in vis:
        for key in ("steps", "nodes"):
            for i, st in enumerate(v.get(key) or []):
                lab = st.get("label") if isinstance(st, dict) else st
                if isinstance(lab, str) and len(lab) > MAX_LABEL:
                    short = _shorten(lab)
                    if isinstance(st, dict):
                        st["label"] = short
                    else:
                        v[key][i] = short
                    if lab not in sc["narration"]:
                        sc["narration"] += " " + lab
                    log.append({"scene": sid, "check": "long label", "fix": f"{lab[:40]}… -> {short}"})
    # coverage
    have = _scene_words(sc)
    miss = [u for u in sc.get("units", []) if not covered(u, have)]
    if miss:
        sc["narration"] = (sc["narration"] + " " + " ".join(miss)).strip()
        log.append({"scene": sid, "check": "coverage", "fix": f"appended {len(miss)} missing unit(s)",
                    "missing": miss})
    # overflow -> split into parts
    out = [sc]
    for vi, v in enumerate(vis):
        t = v.get("type")
        key, lim = {"list": ("items", MAX_LIST), "table": ("rows", MAX_ROWS),
                    "flow": ("steps", MAX_STEPS)}.get(t, (None, 0))
        if key and len(v.get(key) or []) > lim:
            parts = _chunks(v[key], lim)
            narr = re.split(r"(?<=[.!?])\s+", sc["narration"])
            m = len(parts)
            narr_parts = [narr[round(k * len(narr) / m):round((k + 1) * len(narr) / m)] for k in range(m)]
            out = []
            for k, chunk in enumerate(parts):
                s2 = copy.deepcopy(sc)
                s2["visuals"][vi] = dict(v, **{key: chunk})
                if len(sc["visuals"]) > 1:              # keep the other visual only on part 1
                    s2["visuals"] = [s2["visuals"][vi]] if k else s2["visuals"]
                s2["title"] = f"{sc['title']} ({k + 1}/{len(parts)})"
                s2["narration"] = " ".join(narr_parts[k]) or s2["title"]
                s2["id"] = f"{sid}{'abcdefgh'[k]}"
                out.append(s2)
            log.append({"scene": sid, "check": "overflow", "fix": f"{len(v[key])} {key} -> {len(parts)} scenes"})
            break
    # narration too long
    final = []
    for s in out:
        if len(s["narration"].split()) > MAX_WORDS:
            pieces = _split_sentences(s["narration"], MAX_WORDS)
            for k, p in enumerate(pieces):
                s2 = copy.deepcopy(s)
                s2["narration"] = p
                if k:
                    s2["title"] = s["title"] + " (cont.)"
                    s2["id"] = s["id"] + f"-{k}"
                final.append(s2)
            log.append({"scene": s["id"], "check": "narration length",
                        "fix": f"{len(s['narration'].split())} words -> {len(pieces)} scenes"})
        else:
            final.append(s)
    return final


def refine_lesson(lesson):
    """Returns (refined lesson, report)."""
    log, scenes, seen = [], [], set()
    for sc in lesson["scenes"]:
        for s in refine_scene(sc, log):
            key = s["narration"].strip().lower()
            if key in seen and len(key) > 30:
                log.append({"scene": s["id"], "check": "duplicate", "fix": "removed"})
                continue
            seen.add(key)
            scenes.append(s)
    # final evaluation score: share of source units covered
    units = lesson.get("all_units") or [u for sc in lesson["scenes"] for u in sc.get("units", [])]
    blob = []
    for s in scenes:
        vis = [{k: v for k, v in x.items() if k != "png"} for x in s["visuals"]]
        blob.append(s["narration"] + " " + s["title"] + " " + json.dumps(vis, ensure_ascii=False))
    text = _words(" ".join(blob))
    cov = sum(covered(u, text) for u in units) / max(1, len(units))
    out = dict(lesson, scenes=scenes)
    report = {"scenes_in": len(lesson["scenes"]), "scenes_out": len(scenes),
              "source_units": len(units), "coverage": round(cov, 4), "fixes": log}
    return out, report
