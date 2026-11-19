"""
PDF markdown -> lesson JSON with Ollama, section by section.

Differences from v2 (main.py):
- No strict schema and no 3-scene cap. Ollama only has to return valid JSON
  (format="json"); the model chooses how many scenes and what each visual is.
  Anything it returns is normalised here, and visuals.py draws what it can.
- One call per document section, so long PDFs never overflow the context.
- Coverage check: every sentence of the section must appear in some scene. If
  the model skipped something, it is asked for extra scenes; whatever is still
  missing after that is added as a final "More details" scene. Nothing is lost.
"""
import json
import re
from pathlib import Path

PROMPT_FILE = Path(__file__).parent / "prompt_v3.txt"
STOP = set("""a an the and or of to in on at for by with from as is are was were be been it its
this that these those which who whom so if then than but not no can may might will would could
should into onto over under about only also each every any all more most such other one two three
there their they we you he she his her our your i do does did done has have had use used using""".split())


# ---------------------------------------------------------------- sections


def split_sections(md, max_chars=3500, min_chars=250):
    """Split Docling markdown at headings; merge tiny parts, split huge ones."""
    md = re.sub(r"<!--\s*image\s*-->", "", md)
    parts = re.split(r"\n(?=#{1,3} )", "\n" + md.strip())
    parts = [p.strip() for p in parts if p.strip()]
    merged = []
    for p in parts:
        if merged and (len(merged[-1]) < min_chars):
            merged[-1] += "\n\n" + p
        else:
            merged.append(p)
    out = []
    for p in merged:
        while len(p) > max_chars:
            cut = p.rfind("\n\n", 0, max_chars)
            cut = cut if cut > max_chars // 3 else p.rfind(". ", 0, max_chars) + 1
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


def doc_title(md, fallback):
    m = re.search(r"^#{1,2}\s+(.+)$", md, re.M)
    return m.group(1).strip() if m else fallback


# ---------------------------------------------------------------- coverage


def _words(s):
    return {w for w in re.findall(r"[a-z0-9]+(?:\.\d+)?", str(s).lower()) if w not in STOP}


def source_sentences(section):
    """Sentences / table rows / list items of a section that must be covered."""
    text = re.sub(r"^#+\s*", "", section, flags=re.M)
    units = []
    for line in text.split("\n"):
        line = line.strip(" -*|")
        if not line or set(line) <= set("-|: "):
            continue
        if "|" in line:                                   # table row
            units.append(" ".join(c.strip() for c in line.split("|")))
        else:
            units += re.split(r"(?<=[.!?])\s+(?=[A-Z(])", line)
    units = [u.strip() for u in units if len(_words(u)) >= 3]
    return [u for u in units if not re.search(r"\bPage \d+\s*$", u)]   # page footers


def scene_text(scene):
    return json.dumps(scene, ensure_ascii=False)


def missing_parts(section, scenes, threshold=0.6):
    have = _words(" ".join(scene_text(s) for s in scenes))
    miss = []
    for u in source_sentences(section):
        w = _words(u)
        if w and len(w & have) / len(w) < threshold:
            miss.append(u)
    return miss


# ---------------------------------------------------------------- JSON + normalising


def parse_json(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        chunk = re.sub(r",\s*([}\]])", r"\1", t[a:b + 1])        # trailing commas
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass
    raise ValueError("model did not return JSON")


def _first(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, "", [], {}):
            return d[k]
    return default


def _text(x):
    if isinstance(x, list):
        return " ".join(_text(i) for i in x)
    if isinstance(x, dict):
        return " ".join(_text(v) for v in x.values())
    return "" if x is None else str(x)


def find_scenes(data):
    """The model may wrap scenes in different keys; find the list."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for k in ("scenes", "lesson", "slides", "steps", "parts", "sections"):
        v = data.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
        if isinstance(v, dict):
            inner = find_scenes(v)
            if inner:
                return inner
    if any(k in data for k in ("narration", "visual", "visuals", "title")):
        return [data]
    return []


def normalise_scene(s):
    if not isinstance(s, dict):
        s = {"narration": _text(s)}
    vis = _first(s, "visuals", "visual", "diagram", "figure", "visualization", "graphic",
                 "image", default=[])
    if isinstance(vis, dict):
        vis = [vis]
    if isinstance(vis, str):
        vis = [{"type": "list", "items": [vis]}]
    vis = [v for v in vis if isinstance(v, (dict, list))][:2]
    narration = _text(_first(s, "narration", "explanation", "script", "voiceover", "text",
                             "content", "description", default=""))
    extra = _first(s, "example", "examples")
    if extra and isinstance(extra, (str, list)):
        narration += " " + _text(extra)
    return {"title": _text(_first(s, "title", "heading", "name", default="")).strip()[:90]
            or narration[:40],
            "narration": re.sub(r"\s+", " ", narration).strip(),
            "note": _text(_first(s, "note", "key_fact", "takeaway", "key_point", "sticky",
                                 default="")).strip(),
            "visuals": vis,
            "points": _first(s, "points", "key_points", default=None)}


# ---------------------------------------------------------------- Ollama


class Planner:
    def __init__(self, model="qwen3:14b", num_ctx=16384, temperature=0.4, host=None,
                 verbose=True):
        from ollama import Client
        self.client = Client(host=host) if host else Client()
        self.model = model
        self.opts = {"temperature": temperature, "num_ctx": num_ctx, "num_predict": -1}
        self.system = PROMPT_FILE.read_text(encoding="utf-8")
        self.verbose = verbose

    def _chat(self, messages):
        stream = self.client.chat(model=self.model, messages=messages, format="json",
                                  options=self.opts, think=False, keep_alive="15m",
                                  stream=True)
        out = []
        for chunk in stream:
            piece = chunk["message"]["content"]
            out.append(piece)
            if self.verbose:
                print(piece, end="", flush=True)
        if self.verbose:
            print()
        return "".join(out)

    def _ask(self, messages):
        raw = self._chat(messages)
        try:
            return parse_json(raw)
        except ValueError:
            messages = messages + [{"role": "assistant", "content": raw[:2000]},
                                   {"role": "user", "content":
                                    "That was not valid JSON. Reply with ONLY the JSON object."}]
            return parse_json(self._chat(messages))

    def plan_section(self, section, doc_name, i, n):
        msgs = [{"role": "system", "content": self.system},
                {"role": "user", "content":
                 f"Document: {doc_name}\nSection {i} of {n}. Turn ALL of it into scenes "
                 f"(as many as needed, with explanations and worked examples):\n\n{section}"}]
        data = self._ask(msgs)
        subject = data.get("subject") if isinstance(data, dict) else None
        scenes = [normalise_scene(s) for s in find_scenes(data)]
        missing = missing_parts(section, scenes)
        if missing:
            print(f"[coverage] section {i}: {len(missing)} part(s) missing, asking for more scenes")
            follow = msgs + [
                {"role": "assistant", "content": json.dumps({"scenes": scenes})[:6000]},
                {"role": "user", "content":
                 "These parts of the section are NOT in your scenes yet:\n- " +
                 "\n- ".join(missing) +
                 "\n\nReturn {\"scenes\": [...]} with NEW scenes (same format, with visuals "
                 "and explanations) that cover exactly these parts."}]
            try:
                extra = [normalise_scene(s) for s in find_scenes(self._ask(follow))]
                scenes += extra
            except Exception as ex:
                print(f"[coverage] follow-up failed: {ex}")
            still = missing_parts(section, scenes)
            if still:                               # guarantee: nothing is dropped
                print(f"[coverage] section {i}: adding {len(still)} part(s) as a detail scene")
                scenes.append({"title": "More details", "narration": " ".join(still),
                               "note": "", "visuals": [{"type": "list", "items": still[:6]}]})
        if not scenes:
            scenes = [{"title": doc_name, "narration": section[:900], "note": "",
                       "visuals": []}]
        return subject, scenes

    def plan(self, markdown, fallback_title="Lesson"):
        title = doc_title(markdown, fallback_title)
        sections = split_sections(markdown)
        lesson = {"title": title, "subject": None, "scenes": []}
        for i, sec in enumerate(sections, 1):
            print(f"\n[ollama] section {i}/{len(sections)} ({len(sec):,} chars)")
            subject, scenes = self.plan_section(sec, title, i, len(sections))
            lesson["subject"] = lesson["subject"] or subject
            lesson["scenes"] += scenes
            print(f"[ollama] section {i}: {len(scenes)} scene(s)")
        lesson["subject"] = lesson["subject"] or "Lesson"
        return lesson
