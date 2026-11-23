"""'figure' visual: the slide's own diagram, cropped from the PDF, framed hand-drawn style."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v3"))
import visuals                                    # noqa: E402
from canvas import MUTED                           # noqa: E402


def build_figure(cv, v, box, step):
    png = v.get("png")
    if not png:
        return step
    x0, y0, x1, y1 = box
    cap = 26 if v.get("caption") else 0
    bw, bh = x1 - x0 - 20, y1 - y0 - 20 - cap
    iw, ih = float(v.get("w") or 800), float(v.get("h") or 450)
    s = min(bw / iw, bh / ih)
    w, h = iw * s, ih * s
    x, y = x0 + (x1 - x0 - w) / 2, y0 + 10 + (bh - h) / 2
    fid = hashlib.sha1(png[:4000].encode()).hexdigest()[:24]
    cv.files[fid] = {"mimeType": "image/png", "id": fid, "created": 1,
                     "dataURL": "data:image/png;base64," + png}
    cv.rect(x - 8, y - 8, w + 16, h + 16, step, "transparent", "#495057", 2)
    cv._base("image", x, y, w, h, step + 1, "transparent", "transparent", 1,
             fileId=fid, status="saved", scale=[1, 1], crop=None)
    if cap:
        cv.text(x + w / 2, y + h + 14, v["caption"], step + 1, 16, MUTED, "center", "top", math=False)
    return step + 2


visuals.BUILDERS["figure"] = build_figure
visuals.ALIASES.update({"image": "figure", "slide_figure": "figure", "picture": "figure"})
