"""
StudyMate math visuals: equations and function graphs for Excalidraw scenes.

    pip install matplotlib sympy

- equation_elements("x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}", x, y, height=70)
      LaTeX -> crisp SVG image element (rendered offline by matplotlib mathtext)
- plot_elements(["x**2 - 4", "2*x + 1"], x, y, w, h, x_range=(-4, 4))
      functions -> axes + curves as NATIVE Excalidraw lines, so Excalimate
      can trace each curve on with a draw animation.

The LLM writes math the way students see it ("y = x^2 - 4", "sin(x)");
parse_expr() turns that into something sympy can evaluate.
"""
import io
import math
import re
import uuid

from icons import svg_image_element

INK = "#1e1e1e"
CURVE_COLORS = ["#1971c2", "#e8590c", "#2f9e44", "#9c36b5", "#c92a2a"]


def _el(kind, x, y, w, h, stroke=INK, fill="transparent", sw=2, **kw):
    e = {"id": uuid.uuid4().hex[:16], "type": kind, "x": x, "y": y, "width": w, "height": h,
         "angle": 0, "strokeColor": stroke, "backgroundColor": fill, "fillStyle": "solid",
         "strokeWidth": sw, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
         "groupIds": [], "frameId": None, "roundness": None, "seed": 1, "version": 1,
         "versionNonce": 1, "isDeleted": False, "boundElements": None, "updated": 1,
         "link": None, "locked": False}
    e.update(kw)
    return e


def _line(pts, stroke=INK, sw=2, arrow=False, curved=False, **kw):
    x0, y0 = pts[0]
    rel = [[px - x0, py - y0] for px, py in pts]
    xs = [p[0] for p in rel]
    ys = [p[1] for p in rel]
    return _el("arrow" if arrow else "line", x0, y0, max(xs) - min(xs), max(ys) - min(ys),
               stroke, "transparent", sw, points=rel, lastCommittedPoint=None,
               startBinding=None, endBinding=None, startArrowhead=None,
               endArrowhead="arrow" if arrow else None,
               roundness={"type": 2} if curved else None, **kw)


def _text(x, y, s, size=20, color=INK, align="left"):
    w, h = len(s) * size * 0.55, size * 1.25
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    return _el("text", x, y, w, h, color, "transparent", 1, text=s, originalText=s,
               fontSize=size, fontFamily=5, textAlign="left", verticalAlign="top",
               containerId=None, lineHeight=1.25, autoResize=True)


# ---------------- equations ----------------

def looks_like_latex(s):
    return bool(re.search(r"\\[a-zA-Z]+|\^|_|\{", s or ""))


def latex_svg(latex, color=INK, fontsize=28):
    """Render LaTeX (matplotlib mathtext subset) to SVG bytes. No TeX install needed."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    expr = latex.strip()
    if not (expr.startswith("$") and expr.endswith("$")):
        expr = f"${expr}$"
    fig = Figure(figsize=(0.01, 0.01))
    fig.text(0, 0, expr, fontsize=fontsize, color=color)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.05, transparent=True)
    return buf.getvalue()


def equation_elements(latex, x, y, height=70, color=INK):
    """Equation as an image element whose top-left is (x, y). Returns (elements, files, width),
    or None if the LaTeX can't be rendered (the caller then shows it as plain text)."""
    latex = latex.replace("\\\\", "\\")          # models often double-escape: \\frac -> \frac
    try:
        svg = latex_svg(latex, color)
    except Exception as ex:
        print(f"[mathviz] could not render {latex!r}: {str(ex).strip().splitlines()[-1]}")
        return None
    m = re.search(rb'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    vw, vh = (float(m.group(1)), float(m.group(2))) if m else (4.0, 1.0)
    width = height * vw / vh
    els, files = svg_image_element(svg, x, y, max(width, height))
    # svg_image_element fits into a square; place it exactly instead
    els[0].update(x=x, y=y, width=width, height=height)
    return els, files, width


# ---------------- function graphs ----------------

def parse_expr(s):
    """'y = 2x^2 - 3' / 'f(x)=sin x' -> sympy expression in x."""
    import sympy as sp
    from sympy.parsing.sympy_parser import (parse_expr as _p, standard_transformations,
                                            implicit_multiplication_application, convert_xor)
    s = s.split("=", 1)[-1] if "=" in s else s
    s = s.replace("−", "-").replace("·", "*").replace("π", "pi")
    tr = standard_transformations + (implicit_multiplication_application, convert_xor)
    return _p(s, transformations=tr, local_dict={"x": sp.Symbol("x"), "e": sp.E})


def _samples(expr, x_range, n=160):
    import sympy as sp
    f = sp.lambdify(sp.Symbol("x"), expr, "math")
    a, b = x_range
    out = []
    for i in range(n + 1):
        xv = a + (b - a) * i / n
        try:
            yv = float(f(xv))
            out.append((xv, yv if math.isfinite(yv) else None))
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            out.append((xv, None))
    return out


def _nice_ticks(lo, hi, target=6):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 5, 10)), key=lambda s: abs(s - raw))
    start = math.ceil(lo / step) * step
    ticks, v = [], start
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _pi_label(k):
    """k halves of pi -> 'pi/2', 'pi', '3pi/2', '2pi', '-pi/2' ..."""
    if k % 2 == 0:
        n = k // 2
        return {1: "π", -1: "-π"}.get(n, f"{n}π")
    return {1: "π/2", -1: "-π/2"}.get(k, f"{k}π/2")


def default_x_range(functions):
    """One full period for trig, otherwise -5..5."""
    if any(re.search(r"\b(sin|cos|tan)\b", f) for f in functions):
        return (0, 2 * math.pi)
    return (-5, 5)


def plot_elements(functions, x, y, w, h, x_range=(-5, 5), y_range=None, labels=None):
    """Axes + one native curve per function, inside the box (x, y, w, h)."""
    curves = []
    for fn in functions:
        try:
            curves.append((fn, _samples(parse_expr(fn), x_range)))
        except Exception as ex:                      # unparseable -> skip that curve
            print(f"[mathviz] could not plot {fn!r}: {ex}")
    ys = [v for _, pts in curves for _, v in pts if v is not None]
    if y_range is None:
        if ys:
            lo, hi = min(ys), max(ys)
            lo, hi = min(lo, 0), max(hi, 0)
            pad = (hi - lo) * 0.1 or 1
            y_range = (lo - pad, hi + pad)
        else:
            y_range = (-5, 5)
    (xa, xb), (ya, yb) = x_range, y_range
    sx = lambda v: x + (v - xa) / (xb - xa) * w
    sy = lambda v: y + h - (v - ya) / (yb - ya) * h

    els = []
    ax_y = sy(0) if ya <= 0 <= yb else y + h
    ax_x = sx(0) if xa <= 0 <= xb else x
    els.append(_line([(x - 10, ax_y), (x + w + 15, ax_y)], INK, 2, arrow=True))
    els.append(_line([(ax_x, y + h + 10), (ax_x, y - 15)], INK, 2, arrow=True))
    els.append(_text(x + w + 20, ax_y - 12, "x", 22))
    els.append(_text(ax_x + 8, y - 30, "y", 22))
    trig = any(re.search(r"\b(sin|cos|tan)\b", f) for f in functions)
    if trig:                                  # label the x-axis in multiples of pi/2
        k0, k1 = math.ceil(xa / (math.pi / 2)), math.floor(xb / (math.pi / 2))
        xt = [(k * math.pi / 2, _pi_label(k)) for k in range(k0, k1 + 1)]
    else:
        xt = [(t, f"{t:g}") for t in _nice_ticks(xa, xb)]
    for t, word in xt:
        if abs(t) > 1e-9:
            els.append(_line([(sx(t), ax_y - 5), (sx(t), ax_y + 5)], INK, 1.5))
            els.append(_text(sx(t), ax_y + 8, word, 15, "#495057", "center"))
    for t in _nice_ticks(ya, yb):
        if abs(t) > 1e-9:
            els.append(_line([(ax_x - 5, sy(t)), (ax_x + 5, sy(t))], INK, 1.5))
            els.append(_text(ax_x - 10, sy(t) - 9, f"{t:g}", 15, "#495057", "right"))

    for k, (fn, pts) in enumerate(curves):
        color = CURVE_COLORS[k % len(CURVE_COLORS)]
        seg = []
        for xv, yv in pts + [(None, None)]:
            if yv is not None and ya <= yv <= yb:
                seg.append((sx(xv), sy(yv)))
            elif len(seg) > 1:                       # break at gaps / off-screen parts
                els.append(_line(seg, color, 3, curved=True))
                seg = []
            else:
                seg = []
        name = (labels[k] if labels and k < len(labels) else fn)
        lx, ly = x + w + 30, y + 20 + k * 34
        els.append(_line([(lx, ly + 12), (lx + 30, ly + 12)], color, 3))
        els.append(_text(lx + 40, ly, name, 20, color))
    return els