"""
StudyMate v4 - run a Python snippet and record its real execution, for the code_trace visual.

    steps = trace_python(code, call="binary_search([2, 5, 8, 12, 16, 23, 38], 23)")
    -> [{"line": 4, "vars": {"lo": 0, "hi": 6}, "array": [...], "array_name": "arr",
         "pointers": {"lo": 0, "hi": 6, "mid": 3}, "highlight": [3], "note": "..."}, ...]

The code runs in a separate Python process with a time limit (default 3 s) and no input.
Only the code from YOUR PDFs is run; nothing is downloaded. Use --no-run-code to switch it off.
"""
import json
import re
import subprocess
import sys

RUNNER = r'''
import sys, json
code, call = json.loads(sys.stdin.read())
steps = []
SIMPLE = (int, float, str, bool, type(None))
def clean(v, depth=0):
    if isinstance(v, SIMPLE):
        return v if not isinstance(v, str) else v[:40]
    if isinstance(v, (list, tuple)) and depth == 0 and len(v) <= 16:
        return [clean(x, 1) for x in v if isinstance(x, SIMPLE)]
    if isinstance(v, dict) and depth == 0 and len(v) <= 8:
        return {str(k)[:12]: clean(x, 1) for k, x in v.items() if isinstance(x, SIMPLE)}
    return None
def tracer(frame, event, arg):
    if frame.f_code.co_filename != "<studymate>":
        return tracer
    if event in ("line", "return") and len(steps) < 400:
        loc = {}
        for k, v in frame.f_locals.items():
            if k.startswith("__"):
                continue
            c = clean(v)
            if c is not None:
                loc[k] = c
        steps.append({"line": frame.f_lineno, "event": event, "vars": loc,
                      "ret": clean(arg) if event == "return" else None, "fn": frame.f_code.co_name})
    return tracer
out = []
import io, contextlib
buf = io.StringIO()
g = {"__name__": "__studymate__"}
try:
    with contextlib.redirect_stdout(buf):
        sys.settrace(tracer)
        exec(compile(code + ("\n__result__ = " + call if call else ""), "<studymate>", "exec"), g)
        sys.settrace(None)
except Exception as ex:
    sys.settrace(None)
    steps.append({"line": steps[-1]["line"] if steps else 1, "event": "error", "vars": {}, "error": repr(ex)[:80]})
print(json.dumps({"steps": steps, "stdout": buf.getvalue()[:400],
                  "result": clean(g.get("__result__"))}))
'''

POINTER_NAMES = re.compile(r"^(i|j|k|lo|hi|low|high|left|right|mid|l|r|start|end|top|front|rear|idx|index|p|q)$")


def default_call(code, text=""):
    """A call for the first function, from an example in the text or from its parameter names."""
    m = re.search(r"^def\s+(\w+)\s*\(([^)]*)\)", code, re.M)
    if not m:
        return None
    name, params = m.group(1), [p.split("=")[0].split(":")[0].strip() for p in m.group(2).split(",") if p.strip()]
    ex = re.search(rf"\b{name}\s*\(([^()]*(?:\[[^\]]*\])?[^()]*)\)", text or "")
    if ex and "def " not in text[max(0, ex.start() - 5):ex.start()]:
        return f"{name}({ex.group(1)})"
    args = []
    for p in params:
        pl = p.lower()
        if pl in ("self",):
            continue
        if re.search(r"arr|list|nums|items|a$|data|values|seq|array|xs", pl):
            args.append("[2, 5, 8, 12, 16, 23, 38]")
        elif re.search(r"target|key|x$|value|item|goal", pl):
            args.append("23")
        elif re.search(r"^n$|num|count|size|k$", pl):
            args.append("5")
        elif re.search(r"s$|str|text|word", pl):
            args.append("'level'")
        else:
            args.append("3")
    return f"{name}({', '.join(args)})"


def trace_python(code, call=None, timeout=3, max_steps=14):
    code = code.replace("\t", "    ")
    try:
        compile(code, "<studymate>", "exec")
    except SyntaxError:
        return None
    try:
        r = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=json.dumps([code, call or ""]),
                           capture_output=True, text=True, timeout=timeout)
        data = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None
    raw = [s for s in data["steps"] if s.get("fn") != "<module>" or not call]
    if not raw:
        raw = data["steps"]
    if not raw:
        return None
    # keep the steps where something changes; then sample evenly (first and last kept)
    keep, last = [], None
    for s in raw:
        sig = json.dumps([s["line"], s["event"], s["vars"]], sort_keys=True)
        if sig != last:
            keep.append(s)
            last = sig
    if len(keep) > max_steps:
        idx = sorted({round(i * (len(keep) - 1) / (max_steps - 1)) for i in range(max_steps)})
        keep = [keep[i] for i in idx]
    out = []
    prev = {}
    for s in keep:
        vars_ = dict(s["vars"])
        arr_name = next((k for k, v in vars_.items() if isinstance(v, list) and v), None)
        arr = vars_.pop(arr_name) if arr_name else None
        ptr = {k: v for k, v in vars_.items() if isinstance(v, int) and not isinstance(v, bool) and arr
               and POINTER_NAMES.match(k) and 0 <= v < len(arr)}
        note = ""
        changed = [f"{k}: {prev[k]!r} → {v!r}" if k in prev else f"{k} = {v!r}"
                   for k, v in s["vars"].items() if prev.get(k, object()) != v and not isinstance(v, list)]
        prev = dict(s["vars"])
        note = "; ".join(changed[:3])
        if s["event"] == "return":
            note = f"returns {s.get('ret')!r}"
        elif s["event"] == "error":
            note = s.get("error", "error")
        st = {"line": s["line"], "vars": {k: v for k, v in vars_.items() if not isinstance(v, (list, dict))}}
        if arr:
            st.update(array=arr, array_name=arr_name, pointers=ptr,
                      highlight=[ptr["mid"]] if "mid" in ptr else ([ptr["i"]] if "i" in ptr else []))
        if note:
            st["note"] = note
        out.append(st)
    if data.get("stdout"):
        out[-1]["output"] = data["stdout"].strip().split("\n")[-1]
    return out


def find_code(text):
    """A Python-looking block in a scene's text (lines with def/for/while/if and indentation)."""
    m = re.search(r"((?:^|\n)(?:def |class |for |while |import )[\s\S]{20,1500}?)(?:\n\s*\n[A-Z]|\Z)", text)
    if not m:
        return None
    code = m.group(1).strip("\n")
    if not re.search(r":\s*\n\s+\S", code):
        return None
    return code


if __name__ == "__main__":
    demo = """def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1"""
    for s in trace_python(demo, default_call(demo)):
        print(s)
