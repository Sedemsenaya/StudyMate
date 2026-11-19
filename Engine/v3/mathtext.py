"""
Plain-text / LaTeX math -> readable Unicode for Excalidraw text.

    pretty("sin^2(x) + cos^2(x) = 1")   -> "sin²(x) + cos²(x) = 1"
    pretty("\\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}") -> "(-b ± √(b²-4ac)) / (2a)"
    pretty("A = pi x r^2")               -> "A = π × r²"
    pretty("log2(n)")                    -> "log₂(n)"

v2 rendered equations as matplotlib SVG *images*; Excalimate drew those as a
tiny squiggle. v3 keeps every equation as real text, so it is always crisp,
hand-drawn like the rest, and animates like the rest.
"""
import re

SUP = str.maketrans("0123456789+-=()niabcxyk", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵃᵇᶜˣʸᵏ")
SUB = str.maketrans("0123456789+-=()aeoxn", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₙ")
SUP_OK = set("0123456789+-=()niabcxyk")
SUB_OK = set("0123456789+-=()aeoxn")

GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ", "theta": "θ",
         "lambda": "λ", "mu": "μ", "sigma": "σ", "Sigma": "Σ", "omega": "ω", "Omega": "Ω",
         "phi": "φ", "rho": "ρ", "epsilon": "ε", "tau": "τ", "pi": "π"}
LATEX_SYMBOLS = {r"\pm": "±", r"\mp": "∓", r"\times": "×", r"\cdot": "·", r"\div": "÷",
                 r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥", r"\neq": "≠",
                 r"\ne": "≠", r"\approx": "≈", r"\infty": "∞", r"\to": "→",
                 r"\rightarrow": "→", r"\Rightarrow": "⇒", r"\leftarrow": "←", r"\degree": "°",
                 r"\circ": "°", r"\angle": "∠", r"\perp": "⊥", r"\sum": "Σ", r"\int": "∫",
                 r"\partial": "∂", r"\in": "∈", r"\ldots": "…", r"\dots": "…", r"\cdots": "⋯",
                 r"\%": "%", r"\,": " ", r"\;": " ", r"\!": "", r"\quad": "  ", r"\left": "",
                 r"\right": "", r"\log": "log", r"\ln": "ln", r"\sin": "sin", r"\cos": "cos",
                 r"\tan": "tan", r"\lim": "lim", r"\max": "max", r"\min": "min"}
TRIG_LIKE = {"sin", "cos", "tan", "cot", "sec", "csc", "angle", "of", "at", "is", "for",
             "to", "and", "or", "where", "let", "if", "point", "when", "with", "the", "a"}


def _brace(s, i):
    """Return (content, end) of the {...} group starting at s[i] == '{'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    return s[i + 1:], len(s)


def _wrap(x):
    x = x.strip()
    return x if re.fullmatch(r"[\d.]+|[A-Za-zπ]", x) else f"({x})"


def latex_to_plain(s):
    s = s.replace("\\\\", "\\").replace("$", "")
    out, i = "", 0
    while i < len(s):
        if s.startswith(r"\frac", i) or s.startswith(r"\dfrac", i) or s.startswith(r"\tfrac", i):
            j = s.index("{", i) if "{" in s[i:] else len(s)
            num, j = _brace(s, j)
            if j < len(s) and s[j] == "{":
                den, j = _brace(s, j)
            else:
                den = ""
            out += f"{_wrap(latex_to_plain(num))}/{_wrap(latex_to_plain(den))}"
            i = j
        elif s.startswith(r"\sqrt", i):
            j = i + 5
            if j < len(s) and s[j] == "[":
                k = s.index("]", j)
                root = s[j + 1:k]
                j = k + 1
                out += root.translate(SUP) if set(root) <= SUP_OK else root
            if j < len(s) and s[j] == "{":
                inner, j = _brace(s, j)
                out += "√" + _wrap(latex_to_plain(inner))
            else:
                out += "√"
            i = j
        elif s[i] == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", s[i:])
            name = m.group(0)
            if name in LATEX_SYMBOLS:
                out += LATEX_SYMBOLS[name]
            elif m.group(1) in GREEK:
                out += GREEK[m.group(1)]
            elif m.group(1) in ("text", "mathrm", "mathbf", "operatorname", "textbf", "mathit"):
                pass                      # keep the {...} that follows as plain text
            else:
                out += m.group(1)
            i += len(name)
        else:
            out += s[i]
            i += 1
    return out


def _script(s, mark, table, ok):
    def rep(m):
        body = m.group(1) or m.group(2) or m.group(3)
        body = body.strip()
        if body and set(body) <= ok:
            return body.translate(table)
        return mark + ("(" + body + ")" if len(body) > 1 and not body.startswith("(") else body)
    # x^{...} | x^(...) | x^token
    return re.sub(re.escape(mark) + r"(?:\{([^{}]*)\}|\(([^()]*)\)|([A-Za-z0-9]+|[+-]?\d+))", rep, s)


def _times(s):
    """'5 x 3', 'l x w', 'pi x r', '1/2 x base x height' -> use ×. Leave 'sin x', 'angle x' alone."""
    toks = s.split(" ")
    chained = False
    for k in range(1, len(toks) - 1):
        if toks[k] not in ("x", "X", "*"):
            continue
        prev, nxt = toks[k - 1], toks[k + 1]
        p = prev.strip("(),").rstrip("⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ").lower()
        prev_ok = (re.fullmatch(r"[\d.,/]+|π|\)|[a-wyz]|[\d.]+[a-z]{0,2}", prev) is not None
                   or prev.endswith(")") or prev.endswith("²") or chained)
        nxt_ok = nxt not in ("=", ",", "") and not nxt.startswith(("=", ","))
        if p in TRIG_LIKE:
            prev_ok = False
        if prev_ok and nxt_ok or toks[k] == "*":
            toks[k] = "×"
            chained = True
        else:
            chained = False
    return " ".join(toks)


def pretty(s):
    """Best-effort readable math. Safe on ordinary sentences (leaves words alone)."""
    if not s:
        return ""
    s = str(s)
    if "\\" in s or "$" in s:
        s = latex_to_plain(s)
    s = re.sub(r"\bsqrt\s*\(", "√(", s)
    s = re.sub(r"\bsqrt\s+(\w+)", r"√\1", s)
    s = re.sub(r"\bpi\b", "π", s)
    s = re.sub(r"\btheta\b", "θ", s)
    s = re.sub(r"\balpha\b", "α", s)
    s = re.sub(r"\bbeta\b", "β", s)
    s = re.sub(r"\blog2\b", "log₂", s)
    s = re.sub(r"\blog10\b", "log₁₀", s)
    s = re.sub(r"\blog_(\d+)", lambda m: "log" + m.group(1).translate(SUB), s)
    s = re.sub(r"(?<=\d) ?degrees\b", "°", s) if re.search(r"\d ?degrees\b", s) and len(s) < 40 else s
    s = s.replace("**", "^").replace("<=", "≤").replace(">=", "≥").replace("!=", "≠")
    s = s.replace("+/-", "±").replace("->", "→").replace("~=", "≈")
    s = _script(s, "^", SUP, SUP_OK)
    s = _script(s, "_", SUB, SUB_OK)
    s = _times(s)
    s = re.sub(r"(\d) π", r"\1π", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"[ \t]+", " ", s).strip()


if __name__ == "__main__":
    for t in ["sin^2(x) + cos^2(x) = 1", "a^2 + b^2 = c^2", "A = pi x r^2, C = 2 x pi x r",
              "A = 1/2 x base x height", "A = l x w", "sin x = opp / hyp", "about log2(n) steps",
              r"x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}", "sqrt(48^2 + 27^2) = sqrt(3033)",
              "5 x sin(60), about 4.33 m", "O(n^2)", "O(n log n)", "cm^2", "x_1 + x_2",
              "$\\sin^2 x + \\cos^2 x = 1$", "Relative to another angle x, the side across from x"]:
        print(f"{t!r:45} -> {pretty(t)}")
