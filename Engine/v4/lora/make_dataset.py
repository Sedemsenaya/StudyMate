"""
StudyMate LoRA dataset builder.

Turns your PDFs (plus the engine's own topic list) into chat-format training examples that teach a
small model to answer like StudyMate's planner + tutor:

    lesson text  ->  {"topic", "title", "visual", "explain", "takeaway"}

Where the answers come from:
  visual   : the engine's exact rules when they recognise the topic (always correct), otherwise the
             teacher model's visual, kept only if the engine can actually draw it
  explain, title, takeaway, topic : a TEACHER model (big model on a GPU through Ollama, e.g. Louis's PC)

Every example is checked: the JSON parses, the visual passes the engine's validator, and it is really
drawn once on a scratch canvas. Bad ones are dropped.

Usage (from the StudyMate folder):
  python Engine\\v4\\lora\\make_dataset.py uploads  --teacher-host http://100.x.y.z:11434 --teacher qwen3:32b
  python Engine\\v4\\lora\\make_dataset.py C:\\courses --teacher qwen3:14b --synthetic 3
Output: Engine\\v4\\lora\\data\\train.jsonl, val.jsonl, stats.json  (resumable: finished chunks are skipped)
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
sys.path[:0] = [str(V4), str(V4.parent / "v3")]

import teach  # noqa: E402  (loads all 7 drawing libraries)
import visuals  # noqa: E402
from extract import read_document  # noqa: E402
from select_visual import _doc_sections  # noqa: E402

# The fine-tuned model gets this tiny prompt: it learns the rest from the examples.
STUDENT_SYSTEM = ("You are StudyMate. Read the lesson text and answer with JSON: topic (2-6 words), "
                  "title (max 6 words), visual (one StudyMate drawing with its fields), explain (a tutor's "
                  "3 short sentences with a concrete example), takeaway (max 12 words).")

TEACHER_SYSTEM = teach.TEACH_SYSTEM_FULL + """
Also give "topic": 2-6 words naming the exact concept. Write explain like a patient tutor: intuition
first, then one concrete example with real numbers or names from the text (3 sentences, max 70 words)."""

TEACHER_SCHEMA = json.loads(json.dumps(teach.TEACH_SCHEMA))
TEACHER_SCHEMA["properties"]["topic"] = {"type": "string"}
TEACHER_SCHEMA["required"] = ["topic", "title", "visual", "explain", "takeaway"]


def chunks_from_pdf(pdf, max_chars=1600):
    md = read_document(pdf, False, None)
    md = md[0] if isinstance(md, tuple) else md
    out, cur, n = [], [], 0
    for sec in _doc_sections(md):
        head, body = str(sec[0] or ""), sec[1]
        body = "\n".join(map(str, body)) if isinstance(body, list) else str(body)
        txt = (head + "\n" + body).strip()
        if len(txt) < 80:
            continue
        if n + len(txt) > max_chars and cur:
            out.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(txt[:max_chars])
        n += len(txt)
    if cur:
        out.append("\n\n".join(cur))
    return out


def synthetic_prompts(per_type):
    """Short lesson-like texts for every drawing type, from the engine's own gallery topics."""
    out = []
    for g in sorted(HERE.glob("topics*.json")):
        try:
            if g.suffix == ".json":
                rows = json.loads(g.read_text(encoding="utf-8"))
            else:
                src = g.read_text(encoding="utf-8")
                rows = eval(src[src.index("T = ") + 4:src.index("only = ")])
            for r in rows:
                subj, title, text = r[0], r[1], r[2]
                for k in range(per_type):
                    out.append(f"{subj}: {title}\n{text}." + ("" if k == 0 else
                               ["\nExplain it with an example.", "\nWhy does it matter?", "\nHow does it work?"][k % 3]))
        except Exception:
            continue
    return out


def ask_teacher(client, model, text, timeout_tokens=700):
    r = client.chat(model=model, messages=[{"role": "system", "content": TEACHER_SYSTEM},
                                           {"role": "user", "content": text}],
                    format=TEACHER_SCHEMA, think=False, keep_alive="30m",
                    options={"temperature": 0.3, "num_predict": timeout_tokens, "num_ctx": 8192})
    txt = re.sub(r"<think>.*?</think>", "", r["message"]["content"], flags=re.S)
    return json.loads(txt[txt.find("{"):txt.rfind("}") + 1])


def drawable(vis):
    """True if the engine really draws it (catches bad fields before they become training data)."""
    try:
        import canvas
        cv = canvas.Canvas()
        cv.icons = None
        b = visuals.BUILDERS.get(vis["type"]) or visuals.BUILDERS.get(visuals.ALIASES.get(vis["type"], ""))
        if not b:
            return False
        b(cv, vis, (0, 0, 1100, 520), 0)
        return len(cv.elements) > 2
    except Exception:
        return False


def build_answer(text, teacher_out):
    rv = teach.rules_visual(text)
    if rv:
        vis, src = rv, "rules"
        tv = (teacher_out or {}).get("visual")
        if isinstance(tv, dict) and tv.get("type") == rv["type"]:      # keep the teacher's numbers if they draw
            merged = {**tv, **rv}                                     # rules' own fields win
            if drawable(merged):
                vis, src = merged, "rules+teacher fields"
    else:
        vis = teach._valid((teacher_out or {}).get("visual"))
        src = "teacher"
        if vis and teacher_out.get("topic"):
            mapped = teach.rules_visual(teacher_out["topic"] + " " + str(teacher_out.get("title", "")))
            if mapped:
                vis, src = mapped, "rules(topic)"
    if not vis or not drawable(vis):
        return None, None
    t = teacher_out or {}
    ans = {"topic": str(t.get("topic") or "")[:60], "title": str(t.get("title") or "")[:60], "visual": vis,
           "explain": str(t.get("explain") or "")[:480], "takeaway": str(t.get("takeaway") or "")[:110]}
    if not (ans["topic"] and ans["explain"]):
        return None, None
    return ans, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="PDF files or folders")
    ap.add_argument("--teacher", default="qwen3:14b", help="big Ollama model that writes the tutor text")
    ap.add_argument("--teacher-host", help="e.g. http://100.x.y.z:11434 (a GPU PC)")
    ap.add_argument("--synthetic", type=int, default=2, help="examples per engine topic (0 = none)")
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--max", type=int, default=0, help="stop after N new examples (testing)")
    args = ap.parse_args()

    from ollama import Client
    client = Client(host=args.teacher_host, timeout=600) if args.teacher_host else Client(timeout=600)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done_f = out / "all.jsonl"
    seen = set()
    if done_f.exists():
        for ln in done_f.read_text(encoding="utf-8").splitlines():
            seen.add(json.loads(ln)["id"])

    texts = []
    for inp in args.inputs:
        p = Path(inp)
        for pdf in ([p] if p.is_file() else sorted(p.rglob("*.pdf"))):
            try:
                cs = chunks_from_pdf(pdf)
                texts += [(f"{pdf.name}", c) for c in cs]
                print(f"[data] {pdf.name}: {len(cs)} chunks")
            except Exception as ex:
                print(f"[data] {pdf.name}: skipped ({type(ex).__name__})")
    if args.synthetic:
        syn = synthetic_prompts(args.synthetic)
        texts += [("engine-topics", s) for s in syn]
        print(f"[data] {len(syn)} synthetic topic prompts")

    stats = {"kept": 0, "dropped": 0, "skipped_done": 0, "by_source": {}, "by_type": {}}
    t0, new = time.perf_counter(), 0
    with done_f.open("a", encoding="utf-8") as f:
        for i, (origin, text) in enumerate(texts, 1):
            key = hashlib.sha1(text.encode()).hexdigest()[:16]
            if key in seen:
                stats["skipped_done"] += 1
                continue
            try:
                tout = ask_teacher(client, args.teacher, text)
            except Exception as ex:
                print(f"[data] teacher failed on chunk {i} ({type(ex).__name__}); skipped")
                stats["dropped"] += 1
                continue
            ans, src = build_answer(text, tout)
            if not ans:
                stats["dropped"] += 1
                continue
            ex = {"id": key, "origin": origin, "source": src, "messages": [
                {"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": text},
                {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)}]}
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            f.flush()
            seen.add(key)
            stats["kept"] += 1
            stats["by_source"][src] = stats["by_source"].get(src, 0) + 1
            stats["by_type"][ans["visual"]["type"]] = stats["by_type"].get(ans["visual"]["type"], 0) + 1
            new += 1
            if new % 10 == 0:
                el = time.perf_counter() - t0
                print(f"[data] {i}/{len(texts)} chunks, {stats['kept']} kept, {el / new:.1f}s per example")
            if args.max and new >= args.max:
                break

    rows = [json.loads(l) for l in done_f.read_text(encoding="utf-8").splitlines()]
    random.Random(7).shuffle(rows)
    nval = max(1, int(len(rows) * args.val)) if len(rows) > 20 else 0
    for name, part in (("val.jsonl", rows[:nval]), ("train.jsonl", rows[nval:])):
        (out / name).write_text("".join(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n"
                                        for r in part), encoding="utf-8")
    stats["total"] = len(rows)
    stats["types_covered"] = len({json.loads(r["messages"][2]["content"])["visual"]["type"] for r in rows})
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"[data] done: {len(rows)} examples ({len(rows) - nval} train, {nval} val), "
          f"{stats['types_covered']} drawing types covered -> {out}")


if __name__ == "__main__":
    main()
