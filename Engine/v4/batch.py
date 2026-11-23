"""
StudyMate v4 - batch runner: many PDFs -> narrated, animated lesson videos, in parallel.

    python main_v4.py "C:\\Users\\Colli\\Downloads\\AWS Certified Solutions Architect Slides v49 (reduced).pdf"
    python main_v4.py uploads\\                       # every PDF in a folder (recursive)
    python main_v4.py a.pdf b.pdf --pages 194-227     # just part of a deck
    python main_v4.py notes.pdf --planner llm --model qwen3:8b --llm-parallel 2

Stages and what runs in parallel
  1 extract   process pool (--workers): big decks are cut into 50-page chunks so
              every core reads slides at once; docs go to Docling (or pdfium text)
  2 plan      rules planner (slides, instant) or Ollama (docs) with --llm-parallel
              requests at once (set OLLAMA_NUM_PARALLEL to the same number)
  3 critic    evaluate -> refine every scene (coverage, overflow, long labels...)
  4 voice     TTS thread pool (--tts-workers), cached per sentence-set
  5 render    one Chromium, --render-workers tabs drawing scenes at once; each
              scene starts as soon as ITS audio is ready (voice overlaps render)
  6 mux/join  audio muxed in threads, one MP4 per lesson
Logs (with run times): Engine/Output/batch_<stamp>/batch.log, summary.csv,
summary.json, and per lesson: run.log, timings.json, critic.json, ir.json.
"""
import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ENGINE / "v3"))

import figure  # noqa: E402,F401  (registers the "figure" visual)
from critic import refine_lesson                                   # noqa: E402
from extract import is_slide_deck, page_count, read_document, read_slides, split_deck  # noqa: E402
from quiz import looks_like_quiz, read_quiz                         # noqa: E402
from render4 import RenderPool, concat                             # noqa: E402
from scene import build_scene, caption_pages                       # noqa: E402
from select_visual import plan_document_rules, plan_slides, speakable  # noqa: E402

CHUNK = 50


def slug(s, n=60):
    s = re.sub(r"[^\w\s-]", "", str(s).lower()).strip()
    return re.sub(r"[\s-]+", "_", s)[:n].strip("_") or "lesson"


def _pages_for(spec, pdf):
    """--pages for this PDF, clamped to its length (None = all pages)."""
    p = parse_pages(spec)
    if not p:
        return None
    n = page_count(pdf)
    p = [i for i in p if 0 <= i < n]
    return p or None


def parse_pages(spec):
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a) - 1, int(b or a)))
    return out


# ---------------------------------------------------------------- worker functions (processes)

def _read_chunk(pdf, pages):
    t = time.perf_counter()
    sl = read_slides(pdf, pages)
    return sl, time.perf_counter() - t


def _read_doc(pdf, docling, pages=None):
    t = time.perf_counter()
    md, how = read_document(pdf, docling, pages)
    imgs = []
    try:                                    # pictures in the document, by page
        import pypdfium2 as pdfium
        from quiz import _page_images
        doc = pdfium.PdfDocument(str(pdf))
        seen = set()
        for i in (pages or range(len(doc))):
            for im in _page_images(doc, i):
                if im["sha"] not in seen:
                    seen.add(im["sha"])
                    imgs.append(im)
        doc.close()
    except Exception as ex:
        print(f"[extract] pictures skipped ({ex})")
    return md, how, time.perf_counter() - t, imgs


def _read_quiz(pdf, pages=None):
    t = time.perf_counter()
    qs = read_quiz(pdf, pages)
    return qs, time.perf_counter() - t


def add_pictures(ir, images):
    """Document pictures -> a figure scene right after the text of their page."""
    scenes = ir["scenes"]
    for im in sorted(images, key=lambda x: x["page"]):
        k = max([i for i, s in enumerate(scenes) if (s.get("source", {}).get("pages") or [0])[0] <= im["page"]]
                or [len(scenes) - 1])
        scenes.insert(k + 1, {"id": f"{scenes[k]['id'] if scenes else 'S'}-P", "title": f"Picture (page {im['page']})",
                              "idea_kind": "figure", "visual_type": "figure",
                              "visuals": [{"type": "figure", "png": im["png"], "w": im["w"], "h": im["h"],
                                           "caption": f"page {im['page']}"}],
                              "note": "", "narration": f"Here is the picture from page {im['page']}.", "units": [],
                              "source": {"pages": [im["page"]]}, "concept": "picture"})


# ---------------------------------------------------------------- logging

def setup_log(path):
    log = logging.getLogger(str(path))
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def main(argv=None):
    ap = argparse.ArgumentParser(description="StudyMate v4 batch")
    ap.add_argument("inputs", nargs="+", help="PDF files and/or folders")
    ap.add_argument("--out", default=str(ENGINE / "Output"))
    ap.add_argument("--pages", help="only these pages, e.g. 194-227 (decks, documents and practice tests)")
    ap.add_argument("--planner", choices=["auto", "rules", "llm"], default="auto",
                    help="auto: rules for slide decks, Ollama for documents (rules if Ollama is off)")
    ap.add_argument("--model", default="qwen3:8b", help="Ollama text model")
    ap.add_argument("--ollama-host", help="e.g. http://100.x.y.z:11434 (another PC's GPU over Tailscale)")
    ap.add_argument("--ai", choices=["auto", "on", "off"], default="auto",
                    help="practice tests: AI illustration sketches (auto = if Ollama is running)")
    ap.add_argument("--vision", default="auto",
                    help="picture reader: auto (first vision model pulled), a model name, ocr, off")
    ap.add_argument("--ai-timeout", type=int, default=150, help="seconds per AI call, then rules fallback")
    ap.add_argument("--ai-budget", type=int, default=900,
                    help="total seconds of AI tutoring per run (0 = no limit); the rest keeps the engine's drawings")
    ap.add_argument("--questions", help="practice tests: only these question numbers, e.g. 1-6,12")
    ap.add_argument("--sketches", help="JSON of hand-written sketches/captions per question (testing)")
    ap.add_argument("--per-lesson", type=int, default=10, help="practice-test questions per video")
    ap.add_argument("--llm-parallel", type=int, default=int(os.environ.get("OLLAMA_NUM_PARALLEL", 1)),
                    help="Ollama requests at once (1 on a CPU; 2-4 on a GPU with OLLAMA_NUM_PARALLEL set)")
    ap.add_argument("--max-slides", type=int, default=40, help="slides per lesson video")
    ap.add_argument("--max-lessons", type=int, default=0, help="stop after N lessons (testing)")
    ap.add_argument("--tts", choices=["piper", "kokoro", "none"], default="piper")
    ap.add_argument("--voice", help="piper .onnx path or kokoro voice name")
    ap.add_argument("--speed", type=float, default=1.0)
    cpu = os.cpu_count() or 4
    ap.add_argument("--workers", type=int, default=max(1, cpu - 1), help="extract processes")
    ap.add_argument("--tts-workers", type=int, default=max(1, min(4, cpu // 4)))
    ap.add_argument("--render-workers", type=int, default=max(1, min(6, cpu // 3)))
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--no-docling", action="store_true", help="(default now) documents use the fast pdfium reader")
    ap.add_argument("--docling", action="store_true", help="documents: use Docling (better tables, much slower)")
    ap.add_argument("--teach-every", type=int, default=4,
                    help="slides/notes: one AI teaching scene per N scenes (0 = off)")
    ap.add_argument("--vision-max", type=int, default=0, help="describe at most N pictures (0 = all)")
    ap.add_argument("--no-icons", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="stop after planning (ir.json, scene files)")
    ap.add_argument("--chromium", help="path to chrome/chromium (default: playwright's)")
    args = ap.parse_args(argv)

    T0 = time.perf_counter()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.out)
    bdir = out_root / f"batch_{stamp}"
    bdir.mkdir(parents=True, exist_ok=True)
    blog = setup_log(bdir / "batch.log")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    blog.addHandler(console)
    blog.info(f"StudyMate v4 batch {stamp}  cpu={cpu}  workers={args.workers} "
              f"tts={args.tts}x{args.tts_workers} render_tabs={args.render_workers} "
              f"size={args.size}@{args.fps}fps planner={args.planner}")

    pdfs = []
    for inp in args.inputs:
        p = Path(inp)
        pdfs += sorted(p.rglob("*.pdf")) if p.is_dir() else [p]
    blog.info(f"{len(pdfs)} PDF(s)")
    stage = {}

    # ---------------- 1 extract (process pool)
    t = time.perf_counter()
    per_pdf = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for pdf in pdfs:
            if is_slide_deck(pdf):
                pages = _pages_for(args.pages, pdf) or list(range(page_count(pdf)))
                per_pdf[pdf] = {"kind": "slides", "chunks": {}, "cpu_s": 0.0, "pages": len(pages)}
                for i in range(0, len(pages), CHUNK):
                    futs[ex.submit(_read_chunk, str(pdf), pages[i:i + CHUNK])] = (pdf, i)
            elif looks_like_quiz(pdf):
                pages = _pages_for(args.pages, pdf)
                per_pdf[pdf] = {"kind": "quiz", "pages": len(pages) if pages else page_count(pdf)}
                futs[ex.submit(_read_quiz, str(pdf), pages)] = (pdf, "quiz")
            else:
                pages = _pages_for(args.pages, pdf)
                per_pdf[pdf] = {"kind": "doc", "pages": len(pages) if pages else page_count(pdf)}
                futs[ex.submit(_read_doc, str(pdf), args.docling and not args.no_docling, pages)] = (pdf, None)
        for f in as_completed(futs):
            pdf, i = futs[f]
            if i == "quiz":
                qs, secs = f.result()
                per_pdf[pdf].update(questions=qs, reader="quiz reader", cpu_s=secs)
            elif i is None:
                md, how, secs, imgs = f.result()
                per_pdf[pdf].update(md=md, reader=how, cpu_s=secs, images=imgs)
            else:
                sl, secs = f.result()
                per_pdf[pdf]["chunks"][i] = sl
                per_pdf[pdf]["cpu_s"] += secs
    stage["extract"] = time.perf_counter() - t
    for pdf, d in per_pdf.items():
        extra = (f", {len(d['questions'])} questions, "
                 f"{sum(len(q['images']) for q in d['questions'])} pictures") if d["kind"] == "quiz" else ""
        blog.info(f"[extract] {pdf.name}: {d['kind']}, {d['pages']} pages{extra}, "
                  f"{d['cpu_s']:.1f}s cpu ({d.get('reader', 'pdfium')})")
    blog.info(f"[time] extract stage: {stage['extract']:.1f}s wall")

    # ---------------- 2 plan
    t = time.perf_counter()
    lessons = []          # dicts: pdf, ir, plan_s
    use_llm_docs = args.planner == "llm"     # auto: rules + short AI teaching calls (fast)
    llm = None
    if any(d["kind"] == "doc" for d in per_pdf.values()) and (use_llm_docs or args.planner == "llm"):
        try:
            from planner import Planner
            llm = Planner(model=args.model, verbose=False)
            llm.client.list()
        except Exception as ex:
            blog.info(f"[plan] Ollama not reachable ({type(ex).__name__}); using the rules planner")
            llm = None
    li = 0
    llm_jobs = []
    fast = None
    if args.ai != "off" and (args.teach_every or any(d["kind"] == "quiz" for d in per_pdf.values())):
        try:
            from illustrate import FastAI
            fast = FastAI(args.model, args.vision, args.ollama_host, args.ai_timeout, log=blog.info)
            blog.info(f"[ai] {fast.model} (thinking off, short capped calls, "
                      f"{args.ai_timeout}s limit), pictures: {fast.vision or args.vision}")
            small = None
            if any(s in fast.model for s in ("8b", "7b", "14b")):
                names = [m.get("model") or m.get("name") for m in fast._models()]
                small = next((n for w in ("qwen3:4b", "qwen3:1.7b", "qwen2.5:3b", "llama3.2:3b", "gemma3:4b",
                                          "phi4-mini") for n in [fast._pick(w, names)] if n), None)
            fast.big, fast.small = fast.model, small
            secs, gpu = fast.warm_up()
            blog.info(f"[ai] model loaded in {secs:.0f}s, running on {'GPU' if gpu else 'CPU' if gpu is False else '?'}")
            if gpu is False and getattr(fast, "small", None) and args.model == "qwen3:8b":
                fast.unload()                         # free the big one
                fast.model = fast.small               # CPU: a small model tutors 2-4x faster
                secs2, _ = fast.warm_up()
                blog.info(f"[ai] CPU only: using the smaller {fast.model} (loaded in {secs2:.0f}s); "
                          f"pass --model {fast.big} to force the big one")
            elif gpu is False and "8b" in fast.model:
                blog.info("[ai] CPU only: for 2-4x faster tutoring run once: ollama pull qwen3:4b")
            fast.start_budget(args.ai_budget)
            if gpu is False and args.llm_parallel > 1:
                blog.info(f"[ai] CPU only: {args.llm_parallel} calls at once would split the CPU and time out; "
                          "using 1 at a time")
                args.llm_parallel = 1
        except Exception as ex:
            if args.ai == "on":
                raise SystemExit(f"[ai] Ollama not usable: {ex}")
            blog.info(f"[ai] off ({type(ex).__name__}: {str(ex)[:100]}); rules sketches")
    for pdf, d in per_pdf.items():
        if d["kind"] == "quiz":
            from illustrate import plan_quiz
            if args.questions:
                keep = {i + 1 for i in parse_pages(args.questions)}
                d["questions"] = [q for q in d["questions"] if q["number"] in keep]
            t1 = time.perf_counter()
            les, tim = plan_quiz(d["questions"], pdf.name, li + 1, fast, args.per_lesson,
                                 args.vision, blog.info, args.llm_parallel,
                                 json.loads(Path(args.sketches).read_text(encoding="utf-8")) if args.sketches else None)
            secs = time.perf_counter() - t1
            blog.info(f"[plan] {pdf.name}: {len(d['questions'])} questions -> {len(les)} lessons, "
                      f"vision {tim['vision_s']}s, ai {tim['ai_s']}s "
                      f"({tim['ai_sketches']} AI sketches, {tim['rules_sketches']} rules)")
            if fast:
                st = fast.stats
                blog.info(f"[ai] {st['calls']} calls, {st['cached']} cached, {st['fallback']} fell back, "
                          f"{st['tokens']} tokens in {st['seconds']:.0f}s "
                          f"({st['tokens'] / max(1, st['seconds']):.1f} tok/s)")
            for ir in les:
                li += 1
                lessons.append({"pdf": pdf, "ir": ir, "plan_s": secs / max(1, len(les)),
                                "slides": d["pages"]})
            continue
        if d["kind"] == "slides":
            slides = [s for k in sorted(d["chunks"]) for s in d["chunks"][k]]
            for ls in split_deck(slides, pdf.stem, args.max_slides):
                li += 1
                t1 = time.perf_counter()
                ir = plan_slides(ls, pdf.name, li)
                lessons.append({"pdf": pdf, "ir": ir, "plan_s": time.perf_counter() - t1,
                                "slides": len(ls["slides"])})
        else:
            li += 1
            if llm:
                llm_jobs.append((li, pdf, d["md"]))
            else:
                t1 = time.perf_counter()
                ir = plan_document_rules(d["md"], pdf.stem.replace("_", " "), pdf.name, li)
                add_pictures(ir, d.get("images") or [])
                lessons.append({"pdf": pdf, "ir": ir, "plan_s": time.perf_counter() - t1, "slides": d["pages"]})
    if llm_jobs:
        from planner import doc_title, source_sentences, split_sections
        work = []
        for li_, pdf, md in llm_jobs:
            for k, sec in enumerate(split_sections(md), 1):
                work.append((li_, pdf, md, k, sec))
        blog.info(f"[plan] Ollama {args.model}: {len(work)} sections, {args.llm_parallel} at once")
        results = {}
        with ThreadPoolExecutor(args.llm_parallel) as ex:
            futs = {}
            for li_, pdf, md, k, sec in work:
                n = sum(1 for w in work if w[0] == li_)
                futs[ex.submit(lambda a: (time.perf_counter(), llm.plan_section(*a), time.perf_counter()),
                               (sec, pdf.stem, k, n))] = (li_, pdf, md, k, sec)
            for f in as_completed(futs):
                li_, pdf, md, k, sec = futs[f]
                try:
                    a, (subject, scenes), b = f.result()
                except Exception as ex:
                    blog.info(f"[plan] {pdf.name} section {k} failed: {ex}")
                    subject, scenes, a, b = None, [], 0, 0
                results.setdefault(li_, {"pdf": pdf, "md": md, "secs": {}, "time": 0, "subject": None})
                r = results[li_]
                r["secs"][k] = (scenes, sec)
                r["time"] += b - a
                r["subject"] = r["subject"] or subject
                blog.info(f"[plan] {pdf.name} section {k}: {len(scenes)} scenes, {b - a:.1f}s")
        for li_, r in sorted(results.items()):
            scenes, units = [], []
            for k in sorted(r["secs"]):
                sc, sec = r["secs"][k]
                units += source_sentences(sec)
                for s in sc:
                    s.update(source={"pdf": r["pdf"].name, "section": k}, units=[],
                             idea_kind="llm", visual_type=",".join(str(v.get("type")) for v in s.get("visuals", []) if isinstance(v, dict)))
                    scenes.append(s)
            for i, s in enumerate(scenes, 1):
                s["id"] = f"L{li_:02d}-S{i:03d}"
            ir = {"title": doc_title(r["md"], r["pdf"].stem), "subject": r["subject"] or "Lesson",
                  "source_pdf": r["pdf"].name, "scenes": scenes, "all_units": units}
            lessons.append({"pdf": r["pdf"], "ir": ir, "plan_s": r["time"], "slides": per_pdf[r["pdf"]]["pages"]})
    if args.teach_every or fast:
        from teach import caption_figures, teach_lessons
        normal = [L["ir"] for L in lessons if per_pdf[L["pdf"]]["kind"] != "quiz"]
        if normal:
            t1 = time.perf_counter()
            nc = caption_figures(normal, fast, blog.info, args.vision_max)
            t2 = time.perf_counter()
            na = teach_lessons(normal, fast, blog.info, args.teach_every, parallel=args.llm_parallel) \
                if args.teach_every else 0
            st = fast.stats if fast else {"calls": 0, "cached": 0, "fallback": 0, "tokens": 0, "seconds": 0}
            blog.info(f"[ai] pictures described: {nc} in {t2 - t1:.0f}s; teaching scenes added: {na} "
                      f"in {time.perf_counter() - t2:.0f}s; {st['calls']} calls, {st['cached']} cached, "
                      f"{st['fallback']} failed, {st['tokens'] / max(1, st['seconds']):.1f} tok/s")
    if fast:
        fast.unload()                              # give the CPU and RAM back to rendering
    lessons.sort(key=lambda L: L["ir"]["scenes"][0]["id"] if L["ir"]["scenes"] else "")
    if args.max_lessons:
        lessons = lessons[:args.max_lessons]
    stage["plan"] = time.perf_counter() - t
    blog.info(f"[time] plan stage: {stage['plan']:.1f}s, {len(lessons)} lesson(s)")

    # ---------------- 3 critic + 4 build scenes
    t = time.perf_counter()
    icons = None
    if not args.no_icons:
        try:
            sys.path.insert(0, str(ENGINE / "visualization"))
            from icons import IconResolver
            if (ENGINE / "excalidraw-libraries").exists():
                icons = IconResolver(str(ENGINE / "excalidraw-libraries"), str(ENGINE / "bioicons"))
        except Exception as ex:
            blog.info(f"[icons] off ({ex})")
        from illustrate import AwsIcons
        icons = AwsIcons(icons)
    jobs = []
    for L in lessons:
        t1 = time.perf_counter()
        ir, report = refine_lesson(L["ir"])
        L["critic_s"] = time.perf_counter() - t1
        L["ir"], L["report"] = ir, report
        base = f"{slug(Path(L['pdf']).stem, 40)}__{slug(ir['title'], 40)}"
        d = out_root / base
        k = 2
        while d.exists() and not (d / "ir.json").exists():
            d = out_root / f"{base}_{k}"
            k += 1
        (d / "scenes").mkdir(parents=True, exist_ok=True)
        L["dir"] = d
        L["log"] = setup_log(d / "run.log")
        slim = json.loads(json.dumps(ir))
        for s in slim["scenes"]:
            for v in s.get("visuals", []):
                if isinstance(v, dict) and "png" in v:
                    v["png"] = f"<{len(v['png'])} base64 chars>"
        (d / "ir.json").write_text(json.dumps(slim, indent=1, ensure_ascii=False), encoding="utf-8")
        (d / "critic.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        t2 = time.perf_counter()
        L["jobs"] = []
        for i, sc in enumerate(ir["scenes"], 1):
            doc, _, _, credits = build_scene(sc, i, ir.get("subject") or "Lesson", icons)
            pages, _ = caption_pages(sc.get("narration") or "")
            name = f"{i:03d}_{slug(sc['title'], 40)}"
            (d / "scenes" / f"{name}.excalidraw").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            job = {"doc": doc, "pages": pages, "out": d / "scenes" / f"{name}.mp4",
                   "speak": speakable(sc.get("narration") or ""), "lesson": L,
                   "meta": {"id": sc["id"], "title": sc["title"], "visual": sc.get("visual_type")}}
            L["jobs"].append(job)
            jobs.append(job)
        L["build_s"] = time.perf_counter() - t2
        L["log"].info(f"lesson '{ir['title']}' from {Path(L['pdf']).name}: {L['slides']} slides -> "
                      f"{report['scenes_in']} planned -> {report['scenes_out']} scenes, "
                      f"coverage {report['coverage']:.1%}, {len(report['fixes'])} critic fixes")
        L["log"].info(f"[time] plan {L['plan_s']:.2f}s  critic {L['critic_s']:.2f}s  build {L['build_s']:.2f}s")
    stage["critic+build"] = time.perf_counter() - t
    blog.info(f"[time] critic+build stage: {stage['critic+build']:.1f}s, {len(jobs)} scenes")
    if args.plan_only:
        blog.info("plan only: stopping before voice and render")
        return {"lessons": len(lessons), "scenes": len(jobs)}

    # ---------------- 5 voice (threads) + 6 render (tabs), overlapped
    voice = None
    if args.tts != "none":
        from voice import Voice
        voice = Voice(args.tts, args.voice, args.speed)
    tts_pool = ThreadPoolExecutor(max(1, args.tts_workers))

    def speak(text):
        a = time.perf_counter()
        wav, secs, hit = voice.say(text)
        return wav, secs, time.perf_counter() - a, hit

    for j in jobs:
        j["audio"] = tts_pool.submit(speak, j["speak"]) if voice else None
    join_pool = ThreadPoolExecutor(2)
    done_count = {"n": 0}
    t_render = time.perf_counter()

    def finish_lesson(L):
        t1 = time.perf_counter()
        ok = [j["out"] for j in L["jobs"] if "error" not in j["result"]]
        final = concat(ok, L["dir"] / f"{slug(L['ir']['title'], 60)}_StudyMate.mp4") if ok else None
        L["concat_s"] = time.perf_counter() - t1
        L["final"] = final
        L["wall_end"] = time.perf_counter()
        L["log"].info(f"[time] join {L['concat_s']:.1f}s -> {final}")

    def on_done(job):
        L = job["lesson"]
        r = job["result"]
        done_count["n"] += 1
        L["log"].info(f"{job['meta']['id']} {job['meta']['title'][:50]!r} [{job['meta']['visual']}] "
                      f"video {r['video_s']:.1f}s | tts {r['tts_s']:.1f}s{' (cache)' if r['tts_cached'] else ''} "
                      f"| wait {r['wait_audio_s']:.1f}s | render {r.get('render_s', 0):.1f}s "
                      f"({r.get('drawn', 0)}/{r.get('frames', 0)} frames) | mux {r.get('mux_s', 0):.1f}s"
                      f"{' | CACHED' if r.get('cached') else ''}{' | ERROR ' + r['error'] if 'error' in r else ''} "
                      f"| tab {r['worker']}")
        L.setdefault("wall_start", time.perf_counter() - r.get("render_s", 0) - r["wait_audio_s"])
        if done_count["n"] % 10 == 0 or done_count["n"] == len(jobs):
            el = time.perf_counter() - t_render
            blog.info(f"[render] {done_count['n']}/{len(jobs)} scenes, {el:.0f}s elapsed, "
                      f"ETA {el / done_count['n'] * (len(jobs) - done_count['n']):.0f}s")
        if all("result" in j for j in L["jobs"]):
            L["join_future"] = join_pool.submit(finish_lesson, L)

    w, h = map(int, args.size.lower().split("x"))

    async def go():
        async with RenderPool(args.render_workers, w, h, args.fps, args.chromium,
                              log=blog.info) as pool:
            await pool.run(jobs, on_done)
    asyncio.run(go())
    for L in lessons:
        if L.get("join_future"):
            L["join_future"].result()
    tts_pool.shutdown()
    join_pool.shutdown()
    stage["voice+render+join"] = time.perf_counter() - t_render
    total = time.perf_counter() - T0

    # ---------------- logs
    rows = []
    for L in lessons:
        res = [j["result"] for j in L["jobs"]]
        video = sum(r["video_s"] for r in res)
        row = {"pdf": Path(L["pdf"]).name, "lesson": L["ir"]["title"], "slides": L["slides"],
               "scenes": len(res), "coverage": L["report"]["coverage"],
               "critic_fixes": len(L["report"]["fixes"]),
               "errors": sum("error" in r for r in res),
               "video_min": round(video / 60, 2),
               "extract_cpu_s": round(per_pdf[L["pdf"]]["cpu_s"], 2),
               "plan_s": round(L["plan_s"], 2), "critic_s": round(L["critic_s"], 3),
               "build_s": round(L["build_s"], 2),
               "tts_s_sum": round(sum(r["tts_s"] for r in res), 1),
               "render_s_sum": round(sum(r.get("render_s", 0) for r in res), 1),
               "mux_s_sum": round(sum(r.get("mux_s", 0) for r in res), 1),
               "join_s": round(L.get("concat_s", 0), 2),
               "render_speed_x": round(video / max(0.01, sum(r.get("render_s", 0) for r in res)), 2),
               "video": str(L.get("final"))}
        rows.append(row)
        (L["dir"] / "timings.json").write_text(json.dumps({"lesson": row, "scenes": res}, indent=1),
                                               encoding="utf-8")
        L["log"].info("[summary] " + json.dumps(row))
    if rows:
        with open(bdir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    vid = sum(r["video_min"] for r in rows)
    summ = {"stamp": stamp, "pdfs": len(pdfs), "lessons": len(rows), "scenes": len(jobs),
            "video_minutes": round(vid, 1), "stages_s": {k: round(v, 1) for k, v in stage.items()},
            "total_s": round(total, 1), "seconds_per_video_minute": round(total / max(vid, 0.01), 1),
            "settings": vars(args), "lessons_detail": rows}
    (bdir / "summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    blog.info(f"[time] stages: " + ", ".join(f"{k} {v:.1f}s" for k, v in stage.items()))
    blog.info(f"DONE {len(pdfs)} PDF(s), {len(rows)} lesson video(s), {len(jobs)} scenes, "
              f"{vid:.1f} min of video in {total:.0f}s ({total / max(vid, 0.01):.0f}s per video minute)")
    blog.info(f"logs: {bdir}")
    return summ


if __name__ == "__main__":
    main()
