"""
StudyMate v3: PDF -> animated whiteboard lesson video (fully local).

    python main_v3.py uploads\\cs_B_algorithms_data_structures.pdf
    python main_v3.py uploads\\math_B_geometry_and_trigonometry.pdf --model qwen3:14b
    python main_v3.py --from-json Engine\\Output\\<lesson folder>\\lesson.json   # re-render only

Pipeline:
    Docling (PDF -> markdown)
    -> Ollama, one call per section, free JSON, as many scenes as needed
       (+ coverage check so nothing in the PDF is missing)
    -> Engine/v3/scene.py   one .excalidraw per scene, named after its title
    -> Engine/v3/render.py  local Chromium draws each scene stroke by stroke -> .mp4
    -> one joined lesson .mp4

Output: Engine/Output/<lesson_title>_<date-time>/
            lesson.json, scenes/01_<title>.excalidraw/.mp4 ..., <Lesson_Title>_StudyMate.mp4
Nothing from earlier runs is overwritten.

Setup (once, inside .venv):
    pip install ollama docling playwright sympy
    playwright install chromium
    ffmpeg on PATH (winget install ffmpeg) or: pip install imageio-ffmpeg
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "Engine"
sys.path.insert(0, str(ENGINE / "v3"))

from pipeline import build_and_render, lesson_dir   # noqa: E402


def pdf_to_markdown(pdf, tables=True):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    opts = PdfPipelineOptions()
    opts.do_ocr = True                  # text PDFs; set True for scanned ones
    opts.do_table_structure = tables     # keep tables so no row is lost
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(
        pipeline_options=opts)})
    return conv.convert(str(pdf)).document.export_to_markdown()


def load_icons():
    """Use v2's IconResolver (Excalidraw libraries + bioicons) if it is there."""
    try:
        sys.path.insert(0, str(ENGINE / "visualization"))
        from icons import IconResolver
        return IconResolver(str(ENGINE / "excalidraw-libraries"), str(ENGINE / "bioicons"))
    except Exception as ex:
        print(f"[icons] disabled ({ex})")
        return None


def main():
    ap = argparse.ArgumentParser(description="StudyMate v3")
    ap.add_argument("pdf", nargs="?", help="PDF (or .md/.txt) to turn into a lesson")
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--from-json", help="skip Ollama: render an existing lesson.json")
    ap.add_argument("--out", default=str(ENGINE / "Output"))
    ap.add_argument("--size", default="1280x720", help="video size, e.g. 1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-icons", action="store_true")
    args = ap.parse_args()

    if args.from_json:
        lesson = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
    else:
        if not args.pdf:
            ap.error("give a PDF or --from-json")
        from planner import Planner
        src = Path(args.pdf)
        t = time.time()
        md = src.read_text(encoding="utf-8") if src.suffix.lower() in (".md", ".txt") \
            else pdf_to_markdown(src)
        print(f"[time] document read: {time.time() - t:.1f}s ({len(md):,} chars)")
        t = time.time()
        lesson = Planner(model=args.model).plan(md, fallback_title=src.stem)
        print(f"[time] Ollama: {time.time() - t:.0f}s, {len(lesson['scenes'])} scenes")

    out_dir = lesson_dir(args.out, lesson.get("title") or "lesson")
    w, h = map(int, args.size.lower().split("x"))
    t = time.time()
    final = build_and_render(lesson, out_dir, None if args.no_icons else load_icons(),
                             width=w, height=h, fps=args.fps)
    print(f"[time] render: {time.time() - t:.0f}s")
    print(f"\nDone: {final}")


if __name__ == "__main__":
    main()
