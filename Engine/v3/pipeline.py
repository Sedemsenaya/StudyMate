"""
lesson.json -> one named .excalidraw + .mp4 per scene -> one lesson .mp4

    python pipeline.py lesson.json [out_dir]

Output (never overwrites another lesson, every scene has its own name):
    Output/<Lesson_Title>_<yyyymmdd-HHMM>/
        lesson.json
        scenes/01_what_is_an_algorithm.excalidraw   (+ .timeline.json, .mp4)
        scenes/02_searching_linear_vs_binary_search.excalidraw ...
        <Lesson_Title>_StudyMate.mp4
"""
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from render import Renderer, concat
from scene import build_scene


def slug(s, n=48):
    s = re.sub(r"[^\w\s-]", "", str(s).lower()).strip()
    return re.sub(r"[\s-]+", "_", s)[:n].strip("_") or "scene"


def lesson_dir(base, title):
    stamp = time.strftime("%Y%m%d-%H%M")
    d = Path(base) / f"{slug(title, 60)}_{stamp}"
    k = 2
    while d.exists():
        d = Path(base) / f"{slug(title, 60)}_{stamp}_{k}"
        k += 1
    return d


def build_and_render(lesson, out_dir, icons=None, width=1280, height=720, fps=24,
                     only=None, chromium_path=None):
    out_dir = Path(out_dir)
    scenes_dir = out_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lesson.json").write_text(json.dumps(lesson, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    subject = lesson.get("subject") or "Lesson"
    jobs, credits, used = [], [], set()
    for i, sc in enumerate(lesson["scenes"], 1):
        name = f"{i:02d}_{slug(sc.get('title'))}"
        while name in used:
            name += "_b"
        used.add(name)
        doc, tl, dur, cr = build_scene(sc, i, subject, icons)
        credits += [c for c in cr if c not in credits]
        p = scenes_dir / f"{name}.excalidraw"
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        p.with_suffix(".timeline.json").write_text(
            json.dumps({"timeline": tl, "duration": dur}), encoding="utf-8")
        jobs.append((p, doc, tl, dur))
    if credits:
        (out_dir / "credits.txt").write_text("Icons:\n" + "\n".join(credits), encoding="utf-8")

    async def run():
        mp4s = []
        async with Renderer(width, height, fps, chromium_path=chromium_path) as r:
            for k, (p, doc, tl, dur) in enumerate(jobs, 1):
                if only and k not in only:
                    continue
                t = time.time()
                out = p.with_suffix(".mp4")
                frames, shots = await r.render(doc, tl, dur, out)
                print(f"[{k}/{len(jobs)}] {out.name}  {dur:.0f}s video, "
                      f"{shots}/{frames} frames drawn, {time.time() - t:.0f}s")
                mp4s.append(out)
        return mp4s
    mp4s = asyncio.run(run())
    title = lesson.get("title") or subject
    final = concat(mp4s, out_dir / f"{slug(title, 60).title().replace('_', '_')}_StudyMate.mp4")
    print("lesson video:", final)
    return final


if __name__ == "__main__":
    lesson = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    base = sys.argv[2] if len(sys.argv) > 2 else "Output"
    build_and_render(lesson, lesson_dir(base, lesson.get("title") or "lesson"))
