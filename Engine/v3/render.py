"""
Render StudyMate v3 scenes to MP4 locally (no Excalimate website, no MCP).

How: a local Chromium page (Playwright) turns the .excalidraw scene into the
same hand-drawn SVG Excalidraw makes, then draws it stroke by stroke. Python
steps a clock frame by frame, screenshots, and pipes the frames into ffmpeg.
Frames where nothing moves are reused, so long caption pauses cost nothing.

Needs: pip install playwright && playwright install chromium ; ffmpeg on PATH
"""
import asyncio
import functools
import http.server
import json
import shutil
import subprocess
import threading
from pathlib import Path

from playwright.async_api import async_playwright

WEB = Path(__file__).parent / "web"
SCENE_W, SCENE_H = 1600, 900


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def _serve(root):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                          functools.partial(_Quiet, directory=str(root)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg            # pip install imageio-ffmpeg (bundled ffmpeg)
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found: install it (winget install ffmpeg) "
                         "or pip install imageio-ffmpeg")


class Renderer:
    def __init__(self, width=1280, height=720, fps=24, headless=True, chromium_path=None):
        self.w, self.h, self.fps = width, height, fps
        self.headless = headless
        self.chromium_path = chromium_path
        self.ffmpeg = ffmpeg_exe()

    async def __aenter__(self):
        self.srv = _serve(WEB)
        self.pw = await async_playwright().start()
        kw = {"headless": self.headless}
        if self.chromium_path:
            kw["executable_path"] = self.chromium_path
        self.browser = await self.pw.chromium.launch(**kw)
        self.page = await self.browser.new_page(viewport={"width": self.w, "height": self.h})
        self.page.on("console", lambda m: m.type in ("warning", "error") and
                     "Failed to load resource" not in m.text and "Worker" not in m.text and print("[browser]", m.text))
        await self.page.goto(f"http://127.0.0.1:{self.srv.server_address[1]}/index.html")
        await self.page.wait_for_function("window.READY === true", timeout=60000)
        return self

    async def __aexit__(self, *exc):
        await self.browser.close()
        await self.pw.stop()
        self.srv.shutdown()

    async def render(self, doc, timeline, duration, out_mp4):
        info = await self.page.evaluate(
            "([d, t, w, h]) => window.loadScene(d, t, w, h)",
            [doc, timeline, SCENE_W, SCENE_H])
        if not info["matched"]:
            print(f"[render] note: {info}")
        out_mp4 = Path(out_mp4)
        cmd = [self.ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg",
               "-framerate", str(self.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        frames = int(duration * self.fps)
        last = None
        shots = 0
        for f in range(frames):
            changed = await self.page.evaluate(f"window.seek({f / self.fps})")
            if changed or last is None:
                last = await self.page.screenshot(type="jpeg", quality=92)
                shots += 1
            proc.stdin.write(last)
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {out_mp4}")
        return frames, shots


def concat(mp4s, out_mp4):
    """Join scene videos (same size/fps/codec) into one lesson video, no re-encode."""
    out_mp4 = Path(out_mp4)
    lst = out_mp4.with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in mp4s),
                   encoding="utf-8")
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out_mp4)],
                   check=True)
    lst.unlink(missing_ok=True)
    return out_mp4


def render_folder(scene_dir, **kw):
    """Re-render every NN_name.excalidraw (+ .timeline.json) in a folder. Handy after hand edits."""
    async def run():
        scene_dir_p = Path(scene_dir)
        outs = []
        async with Renderer(**kw) as r:
            for ex in sorted(scene_dir_p.glob("*.excalidraw")):
                meta = json.loads(ex.with_suffix(".timeline.json").read_text(encoding="utf-8"))
                doc = json.loads(ex.read_text(encoding="utf-8"))
                out = ex.with_suffix(".mp4")
                await r.render(doc, meta["timeline"], meta["duration"], out)
                print("wrote", out)
                outs.append(out)
        return outs
    return asyncio.run(run())


if __name__ == "__main__":
    import sys
    render_folder(sys.argv[1])
