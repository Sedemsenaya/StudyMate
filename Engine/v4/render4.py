"""
StudyMate v4 renderer: many scenes at once.

- One headless Chromium, N tabs ("render workers"). Each tab draws a scene
  stroke by stroke (the v3 web/bundle.js), Python steps the clock, unchanged
  frames are reused, JPEG frames are piped to ffmpeg.
- A scene waits only for ITS narration audio (TTS runs in other threads at the
  same time), then its length = max(audio + 1.4 s, time to draw it).
- Audio is muxed in a thread while the tab moves on to the next scene.
- Already-rendered scenes (same content hash) are skipped: re-runs are cheap.
"""
import asyncio
import functools
import hashlib
import http.server
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

V3 = Path(__file__).resolve().parent.parent / "v3"
sys.path.insert(0, str(V3))
from render import ffmpeg_exe                      # noqa: E402
from scene import WORDS_PER_SEC                    # noqa: E402

WEB = V3 / "web"
SCENE_W, SCENE_H = 1600, 900
LEAD, TAIL = 0.6, 0.9


def plan_timeline(elements, pages, audio_secs=0.0):
    """v3 timeline, but the scene lasts at least as long as its narration audio."""
    steps = sorted({e["customData"]["step"] for e in elements
                    if isinstance(e["customData"]["step"], int)})
    K = max(1, len(steps))
    page_words = [max(1, len(p.split())) for p in pages]
    read = sum(max(3.0, w / WORDS_PER_SEC + 0.6) for w in page_words) if not audio_secs else 0
    T = max(read + 1.4, 1.4 + K * 0.9 + 1.5, 4.0, audio_secs + LEAD + TAIL)
    # captions: pages share the narration time in proportion to their words
    span = (audio_secs if audio_secs else T - 1.2)
    total_w = sum(page_words) or 1
    starts, t = [], LEAD if audio_secs else 0.8
    for w in page_words:
        starts.append(t)
        t += span * w / total_w
    v0 = 1.0
    v_end = max(v0 + K * 0.8, T * 0.75)
    gap = (v_end - v0) / K
    step_pos = {s: i for i, s in enumerate(steps)}
    count, seen, tl = {}, {}, []
    for e in elements:
        count[e["customData"]["step"]] = count.get(e["customData"]["step"], 0) + 1
    for e in elements:
        st = e["customData"]["step"]
        j = seen.get(st, 0)
        seen[st] = j + 1
        if st == "frame":
            tl.append({"s": 0, "d": 0.01})
        elif st == "title":
            tl.append({"s": 0.1 + j * 0.2, "d": 0.7})
        elif st == "note":
            tl.append({"s": v0 + gap * min(K, 2) * 0.6 + j * 0.3, "d": 0.8})
        elif st == "bar":
            tl.append({"s": 0.4, "d": 0.3, "fade": True})
        elif isinstance(st, str) and st.startswith("page"):
            k = int(st[4:])
            end = starts[k + 1] if k + 1 < len(starts) else None
            tl.append({"s": starts[k], "d": 0.25, "e": end, "fade": True})
        else:
            i = step_pos[st]
            n = count[st]
            stagger = min(0.18, gap * 0.55 / max(1, n))
            item = {"s": v0 + i * gap + j * stagger, "d": min(1.0, max(0.35, gap * 0.7))}
            cd = e["customData"]
            if cd.get("fade"):                       # appears at once (moving things, highlights)
                item["d"] = min(item["d"], 0.2)
                item["fade"] = True
            u = cd.get("until")                      # disappears when step `until` starts
            if isinstance(u, int):
                k = step_pos.get(u)
                if k is None:                        # until a step that has no elements of its own
                    later = [x for x in steps if x >= u]
                    k = step_pos[later[0]] if later else None
                item["e"] = v0 + k * gap if k is not None else None
                if item["e"] is None:
                    del item["e"]
            tl.append(item)
    return tl, round(T, 2)


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def content_hash(doc, tl, dur, wav, size, fps):
    h = hashlib.sha1()
    h.update(json.dumps([doc["elements"], tl, dur, size, fps], sort_keys=True).encode())
    h.update(str(wav or "").encode())
    for f in sorted((doc.get("files") or {})):
        h.update(f.encode())
    return h.hexdigest()[:16]


def mux(video, wav, out, dur):
    """Put narration under the video (starts after LEAD). No audio -> silent track."""
    ff = ffmpeg_exe()
    if wav:
        cmd = [ff, "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav),
               "-filter_complex", f"[1:a]adelay={int(LEAD * 1000)}:all=1,apad,atrim=0:{dur:.3f},"
               "aresample=44100,aformat=channel_layouts=stereo[a]",
               "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
               "-t", f"{dur:.3f}", "-movflags", "+faststart", str(out)]
    else:
        cmd = [ff, "-y", "-loglevel", "error", "-i", str(video), "-f", "lavfi",
               "-i", "anullsrc=r=44100:cl=stereo", "-map", "0:v", "-map", "1:a",
               "-c:v", "copy", "-c:a", "aac", "-t", f"{dur:.3f}", str(out)]
    subprocess.run(cmd, check=True)


def concat(mp4s, out_mp4):
    out_mp4 = Path(out_mp4)
    lst = out_mp4.with_suffix(".list.txt")
    lst.write_text("".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in mp4s), encoding="utf-8")
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out_mp4)], check=True)
    lst.unlink(missing_ok=True)
    return out_mp4


class RenderPool:
    """async with RenderPool(workers=3) as pool: await pool.run(jobs)"""

    def __init__(self, workers=2, width=1280, height=720, fps=24, chromium_path=None,
                 jpeg_quality=88, log=print):
        self.n, self.w, self.h, self.fps = workers, width, height, fps
        self.chromium_path = chromium_path
        self.q = jpeg_quality
        self.log = log
        self.ffmpeg = ffmpeg_exe()

    async def __aenter__(self):
        from playwright.async_api import async_playwright
        self.srv = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(_Quiet, directory=str(WEB)))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.pw = await async_playwright().start()
        kw = {"headless": True, "args": ["--disable-gpu-vsync", "--disable-background-timer-throttling",
                                         "--disable-renderer-backgrounding"]}
        if self.chromium_path:
            kw["executable_path"] = self.chromium_path
        self.browser = await self.pw.chromium.launch(**kw)
        url = f"http://127.0.0.1:{self.srv.server_address[1]}/index.html"
        self.pages = []
        for _ in range(self.n):
            for attempt in range(3):               # a busy CPU can make the first load slow
                p = await self.browser.new_page(viewport={"width": self.w, "height": self.h})
                try:
                    await p.goto(url, wait_until="domcontentloaded", timeout=180000)
                    await p.wait_for_function("window.READY === true", timeout=180000)
                    break
                except Exception as ex:
                    self.log(f"[render] tab {len(self.pages) + 1} start attempt {attempt + 1} failed "
                             f"({type(ex).__name__}); retrying")
                    await p.close()
                    if attempt == 2:
                        raise
            self.pages.append(p)
        return self

    async def __aexit__(self, *exc):
        await self.browser.close()
        await self.pw.stop()
        self.srv.shutdown()

    async def _draw(self, page, doc, tl, dur, out_mp4):
        await page.evaluate("([d, t, w, h]) => window.loadScene(d, t, w, h)",
                            [doc, tl, SCENE_W, SCENE_H])
        cmd = [self.ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg",
               "-framerate", str(self.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "21", "-pix_fmt", "yuv420p", "-r", str(self.fps), str(out_mp4)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        frames, shots, last = int(dur * self.fps), 0, None
        loop = asyncio.get_running_loop()
        for f in range(frames):
            changed = await page.evaluate(f"window.seek({f / self.fps})")
            if changed or last is None:
                last = await page.screenshot(type="jpeg", quality=self.q)
                shots += 1
            await loop.run_in_executor(None, proc.stdin.write, last)
        proc.stdin.close()
        if await loop.run_in_executor(None, proc.wait) != 0:
            raise RuntimeError(f"ffmpeg failed for {out_mp4}")
        return frames, shots

    async def run(self, jobs, on_done=None):
        """jobs: list of dict(doc, pages, out, audio=concurrent.futures.Future|None, meta).
        Each finished job gets job["result"] = {...timings...}."""
        queue = asyncio.Queue()
        for j in jobs:
            queue.put_nowait(j)
        loop = asyncio.get_running_loop()

        async def worker(wi, page):
            while not queue.empty():
                job = await queue.get()
                t0 = time.perf_counter()
                wav, a_secs, tts_secs, hit = None, 0.0, 0.0, True
                if job.get("audio") is not None:
                    try:
                        wav, a_secs, tts_secs, hit = await asyncio.wrap_future(job["audio"])
                    except Exception as ex:          # no voice for this scene, keep going
                        self.log(f"[voice] {Path(job['out']).name}: {type(ex).__name__}: {str(ex)[:80]} (silent scene)")
                        wav, a_secs, tts_secs, hit = None, 0.0, 0.0, False
                t_wait = time.perf_counter() - t0
                doc = job["doc"]
                tl, dur = plan_timeline(doc["elements"], job["pages"], a_secs)
                out = Path(job["out"])
                hfile = out.with_suffix(".hash")
                hsh = content_hash(doc, tl, dur, wav, (self.w, self.h), self.fps)
                res = {"id": job["meta"].get("id"), "video_s": dur, "audio_s": round(a_secs, 2),
                       "tts_s": round(tts_secs, 2), "tts_cached": hit, "wait_audio_s": round(t_wait, 2),
                       "worker": wi}
                if out.exists() and hfile.exists() and hfile.read_text() == hsh:
                    res.update(render_s=0.0, mux_s=0.0, cached=True, frames=0, drawn=0)
                else:
                    t1 = time.perf_counter()
                    silent = out.with_suffix(".video.mp4")
                    try:
                        frames, shots = await self._draw(page, doc, tl, dur, silent)
                    except Exception as ex:                 # one bad scene never kills the batch
                        res.update(error=f"{type(ex).__name__}: {ex}")
                        self.log(f"[render] {out.name} FAILED: {ex}")
                        job["result"] = res
                        if on_done:
                            on_done(job)
                        continue
                    t2 = time.perf_counter()
                    await loop.run_in_executor(None, mux, silent, wav, out, dur)
                    silent.unlink(missing_ok=True)
                    (out.with_suffix(".timeline.json")).write_text(json.dumps({"timeline": tl, "duration": dur}))
                    hfile.write_text(hsh)
                    res.update(render_s=round(t2 - t1, 2), mux_s=round(time.perf_counter() - t2, 2),
                               cached=False, frames=frames, drawn=shots)
                job["result"] = res
                if on_done:
                    on_done(job)

        await asyncio.gather(*(worker(i, p) for i, p in enumerate(self.pages)))
