"""
StudyMate v4 - drawing library, part 3 (more subjects). Computed for real where it says so.

Computer science
  sort_bars     bubble / insertion / selection sort on bars, every comparison and swap (computed)
  hash_table    keys hashed into buckets, collisions chained (computed)
  bst           binary search tree built by inserting keys, path of each insert (computed)
  heap          min-heap built by inserts: array + tree, sift-up swaps (computed)
  recursion     call stack of factorial / fibonacci growing and returning (computed)
  fsm           state machine (DFA) reading an input string, state by state (computed)
  page_table    virtual -> physical address translation (computed)
  mapreduce     word count: split -> map -> shuffle -> reduce (computed)
  git_graph     commits on main and a feature branch, then the merge
Networks / security
  subnet        CIDR: mask in binary, network, broadcast, host range, host count (computed)
  binary        decimal -> binary by repeated division, and hex (computed)
  blockchain    blocks linked by real SHA-256 hashes; tampering breaks the chain (computed)
  cipher        Caesar shift letter by letter (computed)
  firewall      packets checked against ordered rules: allow / deny (computed)
  load_balancer requests spread round-robin over servers
  ddos          many attackers flooding one server
  layers_ring   defense in depth / CIA triad as rings or a triangle
Data / ML
  convolution   a 3x3 kernel sliding over a grid, each output computed
  confusion     confusion matrix with accuracy, precision, recall, F1 (computed)
  stats         histogram + box plot with mean, median, quartiles (computed)
  bpe           byte-pair encoding merges on a small corpus (computed)
Science
  photosynthesis  chloroplast with light, water, CO2 in and glucose, O2 out
  dna             double helix unzipping, complementary bases filled in (computed)
  central_dogma   DNA -> mRNA -> protein using the real codon table (computed)
  heart           four chambers and the path of blood
  water_cycle     sun, ocean, evaporation, clouds, rain, runoff
  neuron          dendrites, cell body, axon; the signal travels
  states_matter   particles in solid, liquid and gas
  wave            transverse wave with wavelength and amplitude, moving
  refraction      ray through two media, angle from Snell's law (computed)
  ph_scale        0-14 with common substances
  solar_system    the planets on their orbits
Maths / economics
  venn          two or three sets, regions filled with their elements (computed)
  prob_tree     probability tree with products on each path (computed)
  matrix_mult   A x B, each cell of the result computed and shown
  riemann       area under a curve with rectangles, n growing (computed)
  vectors       head-to-tail addition with the resultant (computed)
  supply_demand curves, equilibrium price and quantity (computed)
"""
import hashlib
import math
import random
import re

import visuals
from canvas import INK, MUTED, fit_text, text_size
from drawlib import INKS, _envelope, _poly, _Track, draw_device, draw_thing
from drawlib2 import PAL, STRONG, _box, _table


def _title(cv, x, y, s, step, size=24):
    cv.text(x, y, s, step, size, "#e8590c", math=False)


def _frames(cv, step, n, draw):
    """draw(k, step) for k in 0..n-1, each frame replaces the previous one. Returns next step."""
    for k in range(n):
        t = _Track(cv)
        draw(k, step)
        t.tag(until=step + 1 if k < n - 1 else None, fade=k > 0)
        step += 1
    return step


# ======================================================================== CS

def sort_trace(arr, algo):
    a, out = list(arr), []
    n = len(a)
    if algo == "insertion":
        for i in range(1, n):
            j = i
            while j > 0 and a[j - 1] > a[j]:
                out.append((list(a), (j - 1, j), "swap"))
                a[j - 1], a[j] = a[j], a[j - 1]
                j -= 1
            out.append((list(a), (max(j - 1, 0), j), "ok"))
    elif algo == "selection":
        for i in range(n):
            m = i
            for j in range(i + 1, n):
                if a[j] < a[m]:
                    m = j
            out.append((list(a), (i, m), "swap" if m != i else "ok"))
            a[i], a[m] = a[m], a[i]
    else:
        for i in range(n):
            for j in range(n - 1 - i):
                sw = a[j] > a[j + 1]
                out.append((list(a), (j, j + 1), "swap" if sw else "ok"))
                if sw:
                    a[j], a[j + 1] = a[j + 1], a[j]
    out.append((list(a), (), "sorted"))
    if len(out) > 24:
        idx = sorted({round(i * (len(out) - 1) / 23) for i in range(24)})
        out = [out[i] for i in idx]
    return out


def build_sort_bars(cv, v, box, step):
    arr = [int(x) for x in (v.get("array") or [5, 1, 4, 2, 8, 3])][:10]
    algo = str(v.get("algorithm") or "bubble").lower().split()[0]
    tr = sort_trace(arr, algo)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    bw = W * 0.7 / len(arr)
    mx = max(arr)
    _title(cv, x0, y0, f"{algo.title()} sort", step)
    step += 1

    def draw(k, st):
        a, hl, what = tr[k]
        for i, val in enumerate(a):
            h = (H - 110) * val / mx
            col = "#b2f2bb" if what == "sorted" else ("#ffc9c9" if i in hl and what == "swap" else
                                                        "#ffec99" if i in hl else "#a5d8ff")
            cv.rect(x0 + 20 + i * bw, y1 - 30 - h, bw - 10, h, st, col)
            cv.text(x0 + 20 + i * bw + (bw - 10) / 2, y1 - 26, str(val), st, 20, INK, "center", "top", math=False)
        msg = {"swap": f"compare {a[hl[0]]} and {a[hl[1]]}: swap" if hl else "", "ok": "in order: keep",
               "sorted": "sorted!"}[what] if what in ("swap", "ok", "sorted") else ""
        if what == "swap" and hl:
            msg = f"{a[hl[0]]} > {a[hl[1]]}: swap"
        cv.text(x0 + W * 0.74, y0 + 60, f"step {k + 1}/{len(tr)}\n{msg}", st, 22,
                "#c92a2a" if what == "swap" else "#2f9e44", math=False)
    return _frames(cv, step, len(tr), draw)


def build_hash_table(cv, v, box, step):
    keys = [str(k) for k in (v.get("keys") or ["cat", "dog", "bird", "fish", "lion", "frog"])][:8]
    n = int(v.get("buckets") or 5)
    hv = lambda s: sum(ord(c) for c in s)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    bh = (H - 60) / n
    bx = x0 + W * 0.42
    _title(cv, x0, y0, f"hash(key) = sum of letter codes mod {n}", step, 21)
    for b in range(n):
        _box(cv, bx, y0 + 50 + b * bh, 70, bh - 8, step, str(b), "#e9ecef", 20)
    step += 1
    chains = {b: 0 for b in range(n)}
    for i, k in enumerate(keys):
        b = hv(k) % n
        ky = y0 + 50 + i * min(52, (H - 60) / len(keys))
        _box(cv, x0, ky, 110, 40, step, k, "#ffec99", 19)
        cv.text(x0 + 120, ky + 20, f"{hv(k)} mod {n} = {b}", step, 17, MUTED, "left", "middle", math=False)
        cx = bx + 90 + chains[b] * 120
        cy = y0 + 50 + b * bh + (bh - 8) / 2 - 20
        cv.arrow(bx + 72 + chains[b] * 120, cy + 20, cx - 2, cy + 20, step, color="#1971c2")
        _box(cv, cx, cy, 100, 40, step, k, "#ffc9c9" if chains[b] else "#b2f2bb", 19)
        if chains[b]:
            cv.text(cx + 50, cy + 44, "collision: chained", step, 14, "#c92a2a", "center", "top", math=False)
        chains[b] += 1
        step += 1
    return step


def build_bst(cv, v, box, step):
    keys = [int(k) for k in (v.get("keys") or [50, 30, 70, 20, 40, 60, 80, 35])][:12]
    tree = {}
    pos = {}

    def insert(k):
        path, node, depth, lo, hi = [], None, 0, 0.0, 1.0
        cur = keys[0] if tree else None
        while cur is not None:
            path.append(cur)
            side = "L" if k < cur else "R"
            nxt = tree[cur].get(side)
            mid = (pos[cur][2] + pos[cur][3]) / 2
            lo, hi = (pos[cur][2], mid) if side == "L" else (mid, pos[cur][3])
            depth = pos[cur][1] + 1
            if nxt is None:
                tree[cur][side] = k
                break
            cur = nxt
        tree[k] = {}
        pos[k] = ((lo + hi) / 2, depth, lo, hi)
        return path
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    tw = W * 0.68
    R = 26
    _title(cv, x0, y0, "Binary search tree: insert " + ", ".join(map(str, keys)), step, 20)
    step += 1
    for k in keys:
        path = insert(k)
        fx, d, _, _ = pos[k]
        X, Y = x0 + fx * tw, y0 + 70 + d * min(95, (H - 100) / 4)
        parent = path[-1] if path else None
        if parent is not None:
            px, pd, _, _ = pos[parent]
            cv.line([(x0 + px * tw, y0 + 70 + pd * min(95, (H - 100) / 4) + R), (X, Y - R)], step, INK, 2)
        cv.ellipse(X - R, Y - R, 2 * R, 2 * R, step, "#b2f2bb" if not path else "#ffffff")
        cv.text(X, Y, str(k), step, 20, INK, "center", "middle", math=False)
        desc = f"insert {k}: " + (" → ".join(f"{p} ({'left' if k < p else 'right'})" for p in path) if path else "root")
        d2, s2 = fit_text(desc, W * 0.3, 70, 17, 12)
        t = _Track(cv)
        cv.text(x0 + W * 0.7, y0 + 60, d2, step, s2, "#1971c2", math=False)
        t.tag(until=step + 1 if k != keys[-1] else None, fade=True)
        step += 1
    return step


def build_heap(cv, v, box, step):
    keys = [int(k) for k in (v.get("keys") or [9, 4, 7, 1, 8, 2, 6])][:12]
    h, snaps = [], []
    for k in keys:
        h.append(k)
        i = len(h) - 1
        while i and h[(i - 1) // 2] > h[i]:
            p = (i - 1) // 2
            h[p], h[i] = h[i], h[p]
            i = p
        snaps.append((list(h), k))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    _title(cv, x0, y0, "Min-heap: insert then sift up", step)
    step += 1

    def draw(k, st):
        a, key = snaps[k]
        cw = min(64, W / max(12, len(keys)))
        for i, val in enumerate(a):
            _box(cv, x0 + i * cw, y0 + 46, cw - 4, 44, st, val, "#ffec99" if val == key else "#ffffff", 18)
            cv.text(x0 + i * cw + cw / 2, y0 + 94, str(i), st, 13, MUTED, "center", "top", math=False)
        R = 24
        for i, val in enumerate(a):
            d = int(math.log2(i + 1))
            idx_in = i - (2 ** d - 1)
            X = x0 + W * (idx_in + 0.5) / (2 ** d) * 0.95
            Y = y0 + 150 + d * min(90, (H - 180) / 3.5)
            if i:
                p = (i - 1) // 2
                pd = int(math.log2(p + 1))
                PX = x0 + W * ((p - (2 ** pd - 1)) + 0.5) / (2 ** pd) * 0.95
                PY = y0 + 150 + pd * min(90, (H - 180) / 3.5)
                cv.line([(PX, PY + R), (X, Y - R)], st, MUTED, 1.5)
            cv.ellipse(X - R, Y - R, 2 * R, 2 * R, st, "#ffec99" if val == key else "#d0ebff")
            cv.text(X, Y, str(val), st, 18, INK, "center", "middle", math=False)
        cv.text(x1 - 10, y0 + 46, f"inserted {key}", st, 20, "#c92a2a", "right", "top", math=False)
    return _frames(cv, step, len(snaps), draw)


def build_recursion(cv, v, box, step):
    fn = str(v.get("function") or "factorial").lower()
    n = int(v.get("n") or 4)
    events = []

    def fact(k):
        events.append(("call", f"factorial({k})"))
        r = 1 if k <= 1 else k * fact(k - 1)
        events.append(("ret", f"factorial({k}) = {r}"))
        return r

    def fib(k):
        events.append(("call", f"fib({k})"))
        r = k if k < 2 else fib(k - 1) + fib(k - 2)
        events.append(("ret", f"fib({k}) = {r}"))
        return r
    (fib if fn.startswith("fib") else fact)(min(n, 5 if fn.startswith("fib") else 7))
    events = events[:26]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    _title(cv, x0, y0, "The call stack", step)
    step += 1
    stack = []
    snaps = []
    for kind, s in events:
        if kind == "call":
            stack.append(s)
            snaps.append((list(stack), f"call {s}", None))
        else:
            stack.pop()
            snaps.append((list(stack), f"return {s.split('= ')[1]}", s))

    def draw(k, st):
        frames, msg, ret = snaps[k]
        fh = min(48, (H - 80) / 8)
        for i, f in enumerate(frames):
            _box(cv, x0 + 20, y1 - 20 - (i + 1) * fh, 300, fh - 4, st, f,
                 "#ffec99" if i == len(frames) - 1 else "#e7f5ff", 19, mono=True)
        cv.text(x0 + 360, y0 + 60, msg, st, 24, "#2f9e44" if ret else "#1971c2", math=False)
    step = _frames(cv, step, len(snaps), draw)
    return step


def run_dfa(states, start, accept, trans, word):
    s, path = start, [start]
    for ch in word:
        s = trans.get(f"{s},{ch}") or trans.get((s, ch))
        if s is None:
            break
        path.append(s)
    return path, (s in accept if s else False)


def build_fsm(cv, v, box, step):
    states = v.get("states") or ["q0", "q1", "q2"]
    start = v.get("start") or states[0]
    accept = set(v.get("accept") or ["q2"])
    trans = v.get("transitions") or {"q0,a": "q1", "q0,b": "q0", "q1,a": "q1", "q1,b": "q2", "q2,a": "q1", "q2,b": "q0"}
    word = str(v.get("input") or "aabab")
    path, ok = run_dfa(states, start, accept, trans, word)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    R = 42
    pos = {s: (x0 + W * 0.65 * (i + 0.5) / len(states), y0 + H * 0.5) for i, s in enumerate(states)}
    _title(cv, x0, y0, f'DFA reading "{word}" (ends in "ab")' if not v.get("states") else f'reading "{word}"', step, 21)
    for key, dst in trans.items():
        a, ch = key.split(",") if isinstance(key, str) else key
        (ax, ay), (bx, by) = pos[a], pos[dst]
        if a == dst:
            cv.line([(ax - 14, ay - R), (ax - 30, ay - R - 50), (ax + 30, ay - R - 50), (ax + 14, ay - R)], step, MUTED, 2,
                    arrow_end=True)
            cv.text(ax, ay - R - 56, ch, step, 18, "#1971c2", "center", "bottom", math=False)
        else:
            off = -26 if bx > ax else 26
            cv.line([(ax + (R if bx > ax else -R), ay + off * 0.5), ((ax + bx) / 2, ay + off * 2.2),
                     (bx - (R if bx > ax else -R), by + off * 0.5)], step, MUTED, 2, arrow_end=True)
            cv.text((ax + bx) / 2, ay + off * 2.2 + (-6 if off < 0 else 6), ch, step, 18, "#1971c2", "center",
                    "bottom" if off < 0 else "top", math=False)
    for s, (x, y) in pos.items():
        cv.ellipse(x - R, y - R, 2 * R, 2 * R, step, "#ffffff")
        if s in accept:
            cv.ellipse(x - R + 7, y - R + 7, 2 * R - 14, 2 * R - 14, step, "transparent")
        cv.text(x, y, s, step, 20, INK, "center", "middle", math=False)
    sx, sy = pos[start]
    cv.arrow(sx - R - 60, sy, sx - R - 4, sy, step)
    step += 1

    def draw(k, st):
        x, y = pos[path[k]]
        cv.ellipse(x - R, y - R, 2 * R, 2 * R, st, "#ffec99", "#f08c00", 3)
        cv.text(x, y, path[k], st, 20, INK, "center", "middle", math=False)
        shown = word[:k] + "[" + (word[k] if k < len(word) else "") + "]" + word[k + 1:]
        cv.text(x0 + W * 0.7, y0 + 70, f"input: {shown}", st, 22, INK, math=False)
        if k == len(path) - 1:
            cv.text(x0 + W * 0.7, y0 + 120, "ACCEPT" if ok else "REJECT", st, 30, "#2f9e44" if ok else "#e03131",
                    math=False)
    return _frames(cv, step, len(path), draw)


def build_page_table(cv, v, box, step):
    page = int(v.get("page_size") or 4096)
    va = int(str(v.get("address") or "0x3A7F"), 0)
    table = v.get("table") or {0: 5, 1: 9, 2: 1, 3: 7, 4: 2}
    table = {int(k): int(x) for k, x in table.items()}
    vpn, off = divmod(va, page)
    frame = table.get(vpn)
    pa = frame * page + off if frame is not None else None
    x0, y0, x1, y1 = box
    W = x1 - x0
    _title(cv, x0, y0, f"Virtual address {hex(va)}  (page size {page})", step, 22)
    _box(cv, x0, y0 + 50, 200, 50, step, f"page {vpn}", "#d0ebff", 20)
    _box(cv, x0 + 204, y0 + 50, 200, 50, step, f"offset {off}", "#ffec99", 20)
    cv.text(x0, y0 + 108, f"{va} ÷ {page} = page {vpn}, remainder {off}", step, 18, MUTED, math=False)
    step += 1
    tx = x0 + W * 0.48
    cv.text(tx, y0 + 40, "page table", step, 19, "#e8590c", "left", "bottom", math=False)
    for i, (p, f) in enumerate(sorted(table.items())):
        _box(cv, tx, y0 + 50 + i * 40, 90, 38, step, p, "#e9ecef", 17)
        _box(cv, tx + 92, y0 + 50 + i * 40, 90, 38, step, f"frame {f}", "#ffffff" if p != vpn else "#b2f2bb", 16)
    step += 1
    cv.arrow(x0 + 200, y0 + 102, tx - 4, y0 + 50 + sorted(table).index(vpn) * 40 + 19 if vpn in table else y0 + 70,
             step, "look up", color="#1971c2")
    step += 1
    if pa is not None:
        _box(cv, x0, y0 + 260, 200, 50, step, f"frame {frame}", "#b2f2bb", 20)
        _box(cv, x0 + 204, y0 + 260, 200, 50, step, f"offset {off}", "#ffec99", 20)
        cv.text(x0, y0 + 318, f"physical = {frame} × {page} + {off} = {pa} ({hex(pa)})", step, 20, "#2f9e44",
                math=False)
    else:
        cv.text(x0, y0 + 260, "page not in table: PAGE FAULT", step, 24, "#e03131", math=False)
    return step + 1


def build_mapreduce(cv, v, box, step):
    docs = v.get("documents") or ["the cat sat", "the dog sat", "the cat ran"]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cols = [x0, x0 + W * 0.2, x0 + W * 0.45, x0 + W * 0.72]
    for i, h in enumerate(["input split", "map", "shuffle", "reduce"]):
        _title(cv, cols[i], y0, h, step, 21)
    for i, d in enumerate(docs):
        _box(cv, cols[0], y0 + 50 + i * 90, W * 0.17, 70, step, d, "#e7f5ff", 18)
    step += 1
    pairs = []
    for i, d in enumerate(docs):
        ws = d.split()
        s, sz = fit_text(", ".join(f"({w},1)" for w in ws), W * 0.22, 70, 16, 11)
        _box(cv, cols[1], y0 + 50 + i * 90, W * 0.23, 70, step, s, "#fff3bf", 15)
        cv.arrow(cols[0] + W * 0.17, y0 + 85 + i * 90, cols[1] - 2, y0 + 85 + i * 90, step)
        pairs += [(w, 1) for w in ws]
    step += 1
    groups = {}
    for w, c in pairs:
        groups.setdefault(w, []).append(c)
    gh = min(60, (H - 60) / len(groups))
    for i, (w, cs) in enumerate(sorted(groups.items())):
        _box(cv, cols[2], y0 + 50 + i * gh, W * 0.24, gh - 6, step, f"{w}: {cs}", "#ffd8a8", 16)
    step += 1
    for i, (w, cs) in enumerate(sorted(groups.items())):
        cv.arrow(cols[2] + W * 0.24, y0 + 50 + i * gh + gh / 2 - 3, cols[3] - 2, y0 + 50 + i * gh + gh / 2 - 3, step)
        _box(cv, cols[3], y0 + 50 + i * gh, W * 0.22, gh - 6, step, f"{w}: {sum(cs)}", "#b2f2bb", 17)
        step += 1
    return step


def build_git_graph(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    my, fy = y0 + H * 0.35, y0 + H * 0.7
    xs = [x0 + 60 + i * (W - 120) / 6 for i in range(7)]
    R = 22
    cv.text(x0, my - 50, "main", step, 22, "#1971c2", math=False)
    cv.text(xs[1], fy + 40, "feature/login", step, 22, "#2f9e44", math=False)
    step += 1
    plan = [("main", 0, "init"), ("main", 1, "add README"), ("feat", 2, "login form"), ("main", 3, "fix typo"),
            ("feat", 4, "validate input"), ("merge", 5, "merge feature/login"), ("main", 6, "release v1.0")]
    prev = {"main": xs[0], "feat": None}
    for br, i, msg in plan:
        y = fy if br == "feat" else my
        if br == "feat":
            src = (prev["feat"], fy) if prev["feat"] else (xs[1], my)
            cv.line([(src[0] + R * 0.7, src[1] + (R * 0.7 if src[1] == my else 0)), (xs[i] - R, y)], step, "#2f9e44", 3)
            prev["feat"] = xs[i]
        else:
            if i:
                cv.line([(prev["main"] + R, my), (xs[i] - R, my)], step, "#1971c2", 3)
            if br == "merge":
                cv.line([(prev["feat"] + R * 0.7, fy - R * 0.7), (xs[i] - R * 0.7, my + R * 0.7)], step, "#2f9e44", 3)
            prev["main"] = xs[i]
        cv.ellipse(xs[i] - R, y - R, 2 * R, 2 * R, step, "#b2f2bb" if br == "feat" else "#d0ebff",
                   "#2f9e44" if br == "feat" else "#1971c2", 3)
        h = hashlib.sha1(msg.encode()).hexdigest()[:7]
        cv.text(xs[i], y + (R + 6 if br != "feat" else -R - 6), f"{h}\n{msg}", step, 15, INK, "center",
                "top" if br != "feat" else "bottom", math=False)
        step += 1
    return step


# ======================================================================== networks / security

def subnet_info(cidr):
    ip, _, p = cidr.partition("/")
    p = int(p or 24)
    n = sum(int(o) << (8 * (3 - i)) for i, o in enumerate(ip.split(".")))
    mask = (0xFFFFFFFF << (32 - p)) & 0xFFFFFFFF
    net, bc = n & mask, (n & mask) | (~mask & 0xFFFFFFFF)
    dot = lambda x: ".".join(str((x >> (8 * (3 - i))) & 255) for i in range(4))
    hosts = max(0, 2 ** (32 - p) - 2)
    return {"ip": ip, "prefix": p, "mask": dot(mask), "mask_bin": ".".join(f"{(mask >> (8 * (3 - i))) & 255:08b}" for i in range(4)),
            "ip_bin": ".".join(f"{(n >> (8 * (3 - i))) & 255:08b}" for i in range(4)),
            "network": dot(net), "broadcast": dot(bc), "first": dot(net + 1) if hosts else dot(net),
            "last": dot(bc - 1) if hosts else dot(bc), "hosts": hosts}


def build_subnet(cv, v, box, step):
    s = subnet_info(str(v.get("cidr") or "192.168.10.77/26"))
    x0, y0, x1, y1 = box
    _title(cv, x0, y0, f"{s['ip']}/{s['prefix']}", step, 30)
    step += 1
    rows = [("IP (binary)", s["ip_bin"]), ("mask (binary)", s["mask_bin"]), ("mask", s["mask"]),
            ("network", s["network"]), ("broadcast", s["broadcast"]), ("usable hosts", f"{s['first']} – {s['last']}"),
            ("host count", f"2^{32 - s['prefix']} − 2 = {s['hosts']}")]
    for i, (k, val) in enumerate(rows):
        y = y0 + 60 + i * 52
        cv.text(x0, y, k, step, 21, MUTED, math=False)
        if "binary" in k:
            bits = val.replace(".", "")
            for j, b in enumerate(bits):
                cv.rect(x0 + 200 + j * 26 + (j // 8) * 10, y - 4, 24, 34, step,
                        "#b2f2bb" if j < s["prefix"] else "#ffec99", INK, 1, rounded=False)
                cv.text(x0 + 212 + j * 26 + (j // 8) * 10, y + 13, b, step, 16, INK, "center", "middle", math=False)
        else:
            cv.text(x0 + 200, y, val, step, 24, INK, math=False)
        step += 1
    cv.text(x0 + 200, y0 + 60 + 7 * 52, f"green = network bits ({s['prefix']}), yellow = host bits ({32 - s['prefix']})",
            step, 17, MUTED, math=False)
    return step + 1


def build_binary(cv, v, box, step):
    n = int(v.get("number") or 156)
    x0, y0, x1, y1 = box
    _title(cv, x0, y0, f"{n} in binary and hex", step, 26)
    step += 1
    k, rows = n, []
    while k:
        rows.append((k, k // 2, k % 2))
        k //= 2
    for i, (a, q, r) in enumerate(rows[:10]):
        cv.text(x0, y0 + 50 + i * 38, f"{a} ÷ 2 = {q}  remainder", step, 21, INK, math=False)
        cv.text(x0 + 290, y0 + 50 + i * 38, str(r), step, 24, "#e03131", math=False)
        step += 1
    b = bin(n)[2:]
    cv.arrow(x0 + 340, y0 + 50 + len(rows) * 38, x0 + 340, y0 + 60, step, "read up", color="#e03131")
    cv.text(x0 + 460, y0 + 70, f"{n} = {b}₂", step, 30, "#2f9e44", math=False)
    pad = b.zfill((len(b) + 3) // 4 * 4)
    nib = [pad[i:i + 4] for i in range(0, len(pad), 4)]
    for i, q in enumerate(nib):
        _box(cv, x0 + 460 + i * 130, y0 + 140, 120, 46, step, q, "#d0ebff", 20, mono=True)
        cv.text(x0 + 520 + i * 130, y0 + 194, f"{int(q, 2):X}", step, 26, "#7048e8", "center", "top", math=False)
    cv.text(x0 + 460, y0 + 250, f"= 0x{n:X} in hex", step, 26, "#7048e8", math=False)
    return step + 1


def _sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def build_blockchain(cv, v, box, step):
    data = v.get("blocks") or ["Alice pays Bob 5", "Bob pays Cara 2", "Cara pays Dan 1"]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    bw = (W - 60 * (len(data) - 1)) / len(data)
    prev = "0" * 64
    hashes = []
    for i, d in enumerate(data):
        h = _sha(f"{i}|{d}|{prev}")
        hashes.append((d, prev, h))
        prev = h
    for i, (d, ph, h) in enumerate(hashes):
        x = x0 + i * (bw + 60)
        cv.rect(x, y0 + 40, bw, 220, step, "#f8f9fa", INK, 2.5)
        cv.text(x + 12, y0 + 52, f"block {i}", step, 21, "#e8590c", math=False)
        dd, sz = fit_text(d, bw - 24, 50, 19, 13)
        cv.text(x + 12, y0 + 90, dd, step, sz, INK, math=False)
        cv.text(x + 12, y0 + 150, f"prev: {ph[:10]}…", step, 16, MUTED, math=False)
        cv.text(x + 12, y0 + 180, f"hash: {h[:10]}…", step, 17, "#2f9e44", math=False)
        if i:
            cv.arrow(x - 58, y0 + 160, x - 2, y0 + 160, step, color="#2f9e44")
        step += 1
    # tamper with block 1
    if len(hashes) > 1:
        x = x0 + (bw + 60)
        new = _sha(f"1|{hashes[1][0].replace('2', '200')}|{hashes[1][1]}")
        cv.text(x + 12, y0 + 290, f"change it → hash {new[:10]}…", step, 17, "#e03131", math=False)
        cv.text(x0, y0 + 340, "block 2 still points to the OLD hash: the chain is broken, so tampering is detected",
                step, 20, "#e03131", math=False)
        cv.line([(x + bw + 4, y0 + 150), (x + bw + 56, y0 + 170)], step, "#e03131", 4)
        cv.line([(x + bw + 4, y0 + 170), (x + bw + 56, y0 + 150)], step, "#e03131", 4)
    return step + 1


def build_cipher(cv, v, box, step):
    msg = str(v.get("text") or "HELLO WORLD").upper()[:16]
    k = int(v.get("shift") or 3)
    enc = "".join(chr((ord(c) - 65 + k) % 26 + 65) if c.isalpha() else c for c in msg)
    x0, y0, x1, y1 = box
    W = x1 - x0
    _title(cv, x0, y0, f"Caesar cipher, shift {k}", step)
    cw = min(70, (W - 20) / len(msg))
    for i, c in enumerate(msg):
        _box(cv, x0 + i * cw, y0 + 60, cw - 6, 54, step, c, "#d0ebff", 24, mono=True)
    step += 1
    for i, (a, b) in enumerate(zip(msg, enc)):
        if a != " ":
            cv.arrow(x0 + i * cw + cw / 2 - 3, y0 + 118, x0 + i * cw + cw / 2 - 3, y0 + 178, step, f"+{k}",
                     color="#e8590c")
        _box(cv, x0 + i * cw, y0 + 182, cw - 6, 54, step, b, "#ffc9c9", 24, mono=True)
        step += 1
    cv.text(x0, y0 + 270, f"plaintext {msg}  →  ciphertext {enc}", step, 22, INK, math=False)
    cv.text(x0, y0 + 310, f"only {25} possible keys: easy to brute-force", step, 19, "#e03131", math=False)
    return step + 1


def build_firewall(cv, v, box, step):
    rules = v.get("rules") or [["allow", "tcp", 443], ["allow", "tcp", 22, "10.0.0.0/8"], ["deny", "any", "any"]]
    packets = v.get("packets") or [["203.0.113.5", "tcp", 443], ["198.51.100.9", "tcp", 22], ["10.1.2.3", "tcp", 22],
                                   ["203.0.113.5", "udp", 53]]

    def in_cidr(ip, cidr):
        if not cidr or cidr == "any":
            return True
        s = subnet_info(cidr)
        a = sum(int(o) << (8 * (3 - i)) for i, o in enumerate(ip.split(".")))
        n = sum(int(o) << (8 * (3 - i)) for i, o in enumerate(s["network"].split(".")))
        return (a & int(sum(int(o) << (8 * (3 - i)) for i, o in enumerate(s["mask"].split("."))))) == n
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    fx = x0 + W * 0.42
    draw_device(cv, "firewall", fx, y0 + H * 0.45, 120, step)
    cv.text(x0 + W * 0.62, y0, "rules (first match wins)", step, 19, "#e8590c", math=False)
    for i, r in enumerate(rules):
        src = r[3] if len(r) > 3 else "any"
        cv.text(x0 + W * 0.62, y0 + 34 + i * 32, f"{i + 1}. {r[0].upper()} {r[1]} port {r[2]} from {src}", step, 18,
                "#2f9e44" if r[0] == "allow" else "#e03131", math=False)
    step += 1
    for k, (ip, proto, port) in enumerate(packets):
        hit = None
        for i, r in enumerate(rules):
            src = r[3] if len(r) > 3 else "any"
            if (r[1] in ("any", proto)) and (r[2] in ("any", port)) and in_cidr(ip, src):
                hit = (i, r[0])
                break
        hit = hit or (len(rules) - 1, "deny")
        y = y0 + 40 + k * min(80, (H - 60) / len(packets))
        t = _Track(cv)
        _envelope(cv, x0 + 60, y + 20, 50, step)
        t.tag(fade=True)
        cv.text(x0 + 100, y + 20, f"{ip}\n{proto}/{port}", step, 15, INK, "left", "middle", math=False)
        ok = hit[1] == "allow"
        cv.arrow(x0 + 230, y + 20, fx - 70, y0 + H * 0.45, step, color="#868e96")
        cv.text(fx + 90, y + 20, f"{'✔ ALLOW' if ok else '✘ DENY'} (rule {hit[0] + 1})", step, 19,
                "#2f9e44" if ok else "#e03131", "left", "middle", math=False)
        step += 1
    return step


def build_load_balancer(cv, v, box, step):
    n = int(v.get("servers") or 3)
    reqs = int(v.get("requests") or 6)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    ux, lx, sx = x0 + 70, x0 + W * 0.42, x0 + W * 0.82
    draw_device(cv, "users", ux, y0 + H / 2, 110, step)
    cv.text(ux, y0 + H / 2 + 66, "clients", step, 19, INK, "center", "top", math=False)
    draw_device(cv, "switch", lx, y0 + H / 2, 130, step)
    cv.text(lx, y0 + H / 2 + 50, "load balancer\n(round robin)", step, 19, INK, "center", "top", math=False)
    sy = [y0 + H * (i + 0.5) / n for i in range(n)]
    for i, y in enumerate(sy):
        draw_device(cv, "server", sx, y, 90, step)
        cv.text(sx + 60, y, f"server {i + 1}", step, 18, INK, "left", "middle", math=False)
        cv.line([(lx + 60, y0 + H / 2), (sx - 40, y)], step, MUTED, 1.5, dashed=True)
    step += 1
    counts = [0] * n
    for r in range(reqs):
        i = r % n
        counts[i] += 1
        for k, f in enumerate((0.0, 0.5, 1.0)):
            px = lx + 60 + (sx - 40 - lx - 60) * f
            py = y0 + H / 2 + (sy[i] - y0 - H / 2) * f
            last = r == reqs - 1 and k == 2
            _envelope(cv, px, py - 26, 40, step, f"req {r + 1}" if k == 1 else None, until=None if last else step + 1)
            step += 1
    for i, y in enumerate(sy):
        cv.text(sx + 60, y + 24, f"{counts[i]} request{'s' if counts[i] != 1 else ''}", step, 16, "#2f9e44", "left",
                "top", math=False)
    return step + 1


def build_ddos(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    sx, sy = x0 + W * 0.72, y0 + H / 2
    draw_device(cv, "server", sx, sy, 130, step)
    cv.text(sx, sy + 80, "target server", step, 20, INK, "center", "top", math=False)
    step += 1
    pts = [(x0 + 60 + (i % 3) * 120, y0 + 30 + (i // 3) * (H - 110) / 3) for i in range(12)]
    for i, (x, y) in enumerate(pts):
        draw_device(cv, "attacker" if i == 4 else "laptop", x, y, 60, step)
        cv.line([(x + 30, y), (sx - 50, sy)], step, "#ffa8a8", 1.5, arrow_end=True)
        if i % 3 == 2:
            step += 1
    cv.text(sx, sy - 110, "overloaded: real users can't get in", step, 20, "#e03131", "center", "bottom", math=False)
    cv.text(x0, y1 - 10, "botnet: thousands of hijacked devices", step, 18, MUTED, "left", "bottom", math=False)
    return step + 1


def build_layers_ring(cv, v, box, step):
    layers = v.get("layers") or ["Policies & training", "Perimeter firewall", "Network segmentation", "Host hardening",
                                 "Application security", "Data encryption"]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    if str(v.get("shape")) == "triangle":
        pts = [(x0 + W * 0.3, y0 + 30), (x0 + 60, y1 - 50), (x0 + W * 0.54, y1 - 50)]
        cv.polygon(pts, step, "#fff3bf", INK, 3)
        for (x, y), lab in zip(pts, ["Confidentiality", "Integrity", "Availability"]):
            cv.text(x, y + (-6 if y < y0 + 50 else 10), lab, step, 24, "#e8590c", "center",
                    "bottom" if y < y0 + 50 else "top", math=False)
            step += 1
        notes = v.get("notes") or ["Confidentiality: only the right people see it (encryption, access control)",
                                   "Integrity: it is not changed (hashes, signatures)",
                                   "Availability: it is there when needed (backups, redundancy, DDoS protection)"]
        for i, n in enumerate(notes):
            d, sz = fit_text(n, W * 0.33, 80, 19, 13)
            cv.text(x0 + W * 0.66, y0 + 40 + i * 100, d, step, sz, INK, math=False)
            step += 1
        return step
    cx, cy = x0 + H * 0.5 + 20, (y0 + y1) / 2
    n = len(layers)
    for i, lab in enumerate(layers):
        r = H * 0.48 * (n - i) / n
        cv.ellipse(cx - r, cy - r, 2 * r, 2 * r, step, PAL[i % len(PAL)], STRONG[i % len(STRONG)], 2)
        cv.text(x0 + H + 60, y0 + 30 + i * 46, f"{i + 1}. {lab}", step, 20, STRONG[i % len(STRONG)], math=False)
        step += 1
    cv.text(cx, cy, "data", step, 22, INK, "center", "middle", math=False)
    return step + 1


# ======================================================================== data / ML

def build_convolution(cv, v, box, step):
    img = v.get("image") or [[1, 2, 0, 1, 3], [0, 1, 3, 2, 1], [2, 1, 0, 1, 2], [1, 3, 2, 0, 1], [0, 2, 1, 3, 1]]
    ker = v.get("kernel") or [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
    n, k = len(img), len(ker)
    m = n - k + 1
    out = [[sum(img[i + a][j + b] * ker[a][b] for a in range(k) for b in range(k)) for j in range(m)] for i in range(m)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    c = min(60, (H - 60) / n)
    for i in range(n):
        for j in range(n):
            _box(cv, x0 + j * c, y0 + 40 + i * c, c - 4, c - 4, step, img[i][j], "#ffffff", 18)
    cv.text(x0, y0, "input", step, 19, MUTED, math=False)
    kx = x0 + n * c + 40
    for a in range(k):
        for b in range(k):
            _box(cv, kx + b * c, y0 + 40 + a * c, c - 4, c - 4, step, ker[a][b], "#ffec99", 18)
    cv.text(kx, y0, "kernel", step, 19, MUTED, math=False)
    ox = kx + k * c + 60
    cv.text(ox, y0, "output (feature map)", step, 19, MUTED, math=False)
    step += 1
    for i in range(m):
        for j in range(m):
            t = _Track(cv)
            cv.rect(x0 + j * c - 3, y0 + 37 + i * c, k * c + 2, k * c + 2, step, "transparent", "#f08c00", 4)
            terms = " + ".join(f"{img[i + a][j + b]}·{ker[a][b]}" for a in range(k) for b in range(k))
            tt, sz = fit_text(terms + f" = {out[i][j]}", W - 20, 60, 16, 11)
            cv.text(x0, y1 - 60, tt, step, sz, "#c92a2a", math=False)
            t.tag(until=step + 1, fade=True)
            _box(cv, ox + j * c, y0 + 40 + i * c, c - 4, c - 4, step, out[i][j],
                 "#b2f2bb" if out[i][j] > 0 else "#ffc9c9" if out[i][j] < 0 else "#ffffff", 18)
            step += 1
    return step


def build_confusion(cv, v, box, step):
    tp, fn, fp, tn = (int(v.get(k) or d) for k, d in (("tp", 40), ("fn", 10), ("fp", 5), ("tn", 45)))
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    x0, y0, x1, y1 = box
    c = 140
    cv.text(x0 + 140 + c, y0, "predicted", step, 20, MUTED, "center", "top", math=False)
    cv.text(x0, y0 + 60 + c, "actual", step, 20, MUTED, "left", "middle", math=False)
    for j, h in enumerate(["positive", "negative"]):
        cv.text(x0 + 140 + j * c + c / 2, y0 + 30, h, step, 18, INK, "center", "top", math=False)
        cv.text(x0 + 130, y0 + 60 + j * c + c / 2, h, step, 18, INK, "right", "middle", math=False)
    step += 1
    cells = [((0, 0), "TP", tp, "#b2f2bb"), ((0, 1), "FN", fn, "#ffc9c9"), ((1, 0), "FP", fp, "#ffc9c9"),
             ((1, 1), "TN", tn, "#b2f2bb")]
    for (i, j), name, val, col in cells:
        _box(cv, x0 + 140 + j * c, y0 + 60 + i * c, c - 6, c - 6, step, f"{name}\n{val}", col, 26)
        step += 1
    mx = x0 + 160 + 2 * c + 40
    for i, (n, f, val) in enumerate([("accuracy", "(TP+TN)/all", acc), ("precision", "TP/(TP+FP)", prec),
                                     ("recall", "TP/(TP+FN)", rec), ("F1", "2PR/(P+R)", f1)]):
        cv.text(mx, y0 + 60 + i * 56, f"{n} = {f} = {val:.2f}", step, 23, "#1971c2", math=False)
        step += 1
    return step


def build_stats(cv, v, box, step):
    data = sorted(float(x) for x in (v.get("data") or [12, 15, 15, 18, 19, 21, 22, 22, 22, 24, 25, 27, 30, 31, 38, 45]))
    n = len(data)

    def q(p):
        k = (n - 1) * p
        f = math.floor(k)
        return data[f] + (data[min(f + 1, n - 1)] - data[f]) * (k - f)
    mean, med, q1, q3 = sum(data) / n, q(0.5), q(0.25), q(0.75)
    iqr = q3 - q1
    lo_w = min(x for x in data if x >= q1 - 1.5 * iqr)
    hi_w = max(x for x in data if x <= q3 + 1.5 * iqr)
    outl = [x for x in data if x < lo_w or x > hi_w]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    lo, hi = min(data), max(data)
    bins = int(v.get("bins") or 6)
    bw = (hi - lo) / bins or 1
    counts = [0] * bins
    for x in data:
        counts[min(bins - 1, int((x - lo) / bw))] += 1
    sx = lambda x: x0 + 40 + (x - lo) / (hi - lo or 1) * (W * 0.6)
    hh = H * 0.45
    for i, c in enumerate(counts):
        h = hh * c / max(counts)
        cv.rect(sx(lo + i * bw), y0 + 20 + hh - h, W * 0.6 / bins - 4, h, step, "#a5d8ff")
        cv.text(sx(lo + i * bw) + W * 0.3 / bins, y0 + 16 + hh - h, str(c), step, 16, INK, "center", "bottom", math=False)
        step += 1
    by = y0 + hh + 70
    cv.line([(sx(lo_w), by + 25), (sx(q1), by + 25)], step, INK, 2)
    cv.rect(sx(q1), by, sx(q3) - sx(q1), 50, step, "#ffec99")
    cv.line([(sx(med), by), (sx(med), by + 50)], step, "#e03131", 3)
    cv.line([(sx(q3), by + 25), (sx(hi_w), by + 25)], step, INK, 2)
    for x in outl:
        cv.ellipse(sx(x) - 6, by + 19, 12, 12, step, "#ffc9c9", "#e03131", 1.5)
    for x, lab in ((q1, "Q1"), (med, "median"), (q3, "Q3")):
        cv.text(sx(x), by + 56, f"{lab}\n{x:g}", step, 15, INK, "center", "top", math=False)
    step += 1
    sd = math.sqrt(sum((x - mean) ** 2 for x in data) / (n - 1))
    cv.text(x0 + W * 0.7, y0 + 20, f"n = {n}\nmean = {mean:.2f}\nmedian = {med:g}\nQ1 = {q1:g}, Q3 = {q3:g}\n"
                                   f"IQR = {iqr:g}\nstd dev = {sd:.2f}\noutliers: {', '.join(f'{x:g}' for x in outl) or 'none'}",
            step, 21, INK, math=False)
    return step + 1


def bpe_merges(words, n=5):
    corpus = {tuple(w) + ("_",): c for w, c in words.items()}
    steps = []
    for _ in range(n):
        pairs = {}
        for sym, c in corpus.items():
            for a, b in zip(sym, sym[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + c
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))
        new = {}
        for sym, c in corpus.items():
            out, i = [], 0
            while i < len(sym):
                if i < len(sym) - 1 and (sym[i], sym[i + 1]) == best:
                    out.append(sym[i] + sym[i + 1])
                    i += 2
                else:
                    out.append(sym[i])
                    i += 1
            new[tuple(out)] = c
        corpus = new
        steps.append((best, pairs[best], dict(corpus)))
    return steps


def build_bpe(cv, v, box, step):
    words = v.get("words") or {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    steps = bpe_merges({str(k): int(c) for k, c in words.items()}, int(v.get("merges") or 5))
    x0, y0, x1, y1 = box
    W = x1 - x0
    _title(cv, x0, y0, "Byte-pair encoding: merge the most frequent pair", step, 22)
    step += 1

    def draw(k, st):
        best, cnt, corpus = steps[k]
        cv.text(x0, y0 + 40, f"merge {k + 1}: '{best[0]}' + '{best[1]}' → '{best[0] + best[1]}'  (seen {cnt}×)", st, 22,
                "#c92a2a", math=False)
        for i, (sym, c) in enumerate(corpus.items()):
            x = x0
            for s in sym:
                w = max(40, text_size(s, 20)[0] + 18)
                _box(cv, x, y0 + 90 + i * 60, w, 46, st, s, "#b2f2bb" if s == best[0] + best[1] else "#e7f5ff", 20,
                     mono=True)
                x += w + 4
            cv.text(x + 14, y0 + 113 + i * 60, f"× {c}", st, 19, MUTED, "left", "middle", math=False)
        vocab = [a + b for (a, b), _, _ in steps[:k + 1]]
        cv.text(x0 + W * 0.66, y0 + 90, "new tokens:\n" + "\n".join(vocab), st, 20, "#2f9e44", math=False)
    return _frames(cv, step, len(steps), draw)


# ======================================================================== science

def build_photosynthesis(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cx, cy = x0 + W * 0.42, (y0 + y1) / 2 + 20
    rx, ry = W * 0.2, H * 0.28
    cv.ellipse(cx - rx, cy - ry, 2 * rx, 2 * ry, step, "#d3f9d8", "#2b8a3e", 3)
    for i in range(5):                                # thylakoid stacks
        gx = cx - rx * 0.6 + i * rx * 0.3
        for k in range(4):
            cv.ellipse(gx - 22, cy - 30 + k * 14, 44, 14, step, "#8ce99a", "#2b8a3e", 1.5)
    cv.text(cx, cy + ry + 10, "chloroplast", step, 20, "#2b8a3e", "center", "top", math=False)
    step += 1
    sun = (x0 + 70, y0 + 60)
    cv.ellipse(sun[0] - 40, sun[1] - 40, 80, 80, step, "#ffd43b", "#f08c00", 2)
    for a in range(0, 360, 45):
        r = math.radians(a)
        cv.line([(sun[0] + 50 * math.cos(r), sun[1] + 50 * math.sin(r)), (sun[0] + 66 * math.cos(r), sun[1] + 66 * math.sin(r))],
                step, "#f08c00", 3)
    cv.arrow(sun[0] + 60, sun[1] + 40, cx - rx * 0.7, cy - ry * 0.7, step, "light energy", color="#f08c00")
    step += 1
    ins = [("6 H₂O", x0 + 40, cy + 60, "#1971c2"), ("6 CO₂", x0 + 40, cy + 140, "#495057")]
    outs = [("C₆H₁₂O₆ (glucose)", x1 - 280, cy - 40, "#e8590c"), ("6 O₂", x1 - 280, cy + 80, "#1971c2")]
    for lab, x, y, col in ins:
        cv.text(x, y, lab, step, 26, col, "left", "middle", math=False)
        cv.arrow(x + 90, y, cx - rx - 4, cy + (y - cy) * 0.4, step, color=col)
        step += 1
    for lab, x, y, col in outs:
        cv.arrow(cx + rx + 4, cy + (y - cy) * 0.4, x - 10, y, step, color=col)
        cv.text(x, y, lab, step, 24, col, "left", "middle", math=False)
        step += 1
    cv.text(x0 + W * 0.15, y1 - 10, "6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂", step, 26, INK, "left", "bottom", math=False)
    return step + 1


PAIR = {"A": "T", "T": "A", "G": "C", "C": "G"}
BASE_COL = {"A": "#ff8787", "T": "#74c0fc", "G": "#8ce99a", "C": "#ffd43b", "U": "#da77f2"}


def build_dna(cv, v, box, step):
    seq = re.sub(r"[^ATGC]", "", str(v.get("sequence") or "ATGCGTAC").upper())[:10] or "ATGCGTAC"
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cw = (W * 0.86) / len(seq)
    ya, yb = y0 + H * 0.3, y0 + H * 0.62
    _title(cv, x0, y0, "DNA: A pairs with T, G pairs with C", step, 21)
    # helix backbones (sine) + rungs
    cv.line([(x0 + i * cw * 0.25, ya - 40 * math.sin(i * 0.6)) for i in range(int(len(seq) * 4) + 1)], step, "#495057", 3)
    cv.line([(x0 + i * cw * 0.25, yb + 40 * math.sin(i * 0.6)) for i in range(int(len(seq) * 4) + 1)], step, "#495057", 3)
    step += 1
    for i, b in enumerate(seq):
        x = x0 + i * cw + cw / 2
        _box(cv, x - 22, ya - 2, 44, 40, step, b, BASE_COL[b], 20)
        _box(cv, x - 22, yb - 38, 44, 40, step, PAIR[b], BASE_COL[PAIR[b]], 20)
        cv.line([(x, ya + 38), (x, yb - 38)], step, MUTED, 2, dashed=True)
    step += 1
    # replication: strands separate, new partners added base by base
    cv.text(x0, y1 - 70, "replication: helicase unzips; each old strand is a template for a new one", step, 19, "#e8590c",
            math=False)
    for i, b in enumerate(seq):
        x = x0 + i * cw + cw / 2
        _box(cv, x - 20, ya + 44, 40, 34, step, PAIR[b], "#ffffff", 17)
        _box(cv, x - 20, yb - 78, 40, 34, step, b, "#ffffff", 17)
        if i % 2 == 1:
            step += 1
    return step + 1


_B = "UCAG"
_AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
_NAME = {"F": "Phe", "L": "Leu", "S": "Ser", "Y": "Tyr", "*": "STOP", "C": "Cys", "W": "Trp", "P": "Pro", "H": "His",
         "Q": "Gln", "R": "Arg", "I": "Ile", "M": "Met", "T": "Thr", "N": "Asn", "K": "Lys", "V": "Val", "A": "Ala",
         "D": "Asp", "E": "Glu", "G": "Gly"}
GENETIC_CODE = {a + b + c: _NAME[_AA[16 * i + 4 * j + k]] for i, a in enumerate(_B) for j, b in enumerate(_B)
                for k, c in enumerate(_B)}
CODONS_OLD = {"AUG": "Met", "UUU": "Phe", "UUC": "Phe", "GGC": "Gly", "GGA": "Gly", "GCU": "Ala", "GCC": "Ala", "UCU": "Ser",
          "UGG": "Trp", "CAU": "His", "AAA": "Lys", "GAA": "Glu", "UAC": "Tyr", "CGA": "Arg", "ACG": "Thr", "UAA": "STOP",
          "UAG": "STOP", "UGA": "STOP", "CUG": "Leu", "GUU": "Val", "AAU": "Asn", "GAU": "Asp", "CCC": "Pro",
          "UGU": "Cys", "CAG": "Gln", "AUU": "Ile", "ACU": "Thr", "GCA": "Ala", "CGU": "Arg", "UAU": "Tyr"}


def build_central_dogma(cv, v, box, step):
    template = re.sub(r"[^ATGC]", "", str(v.get("dna") or "TACAAACCGTGGATT").upper())[:18]
    mrna = "".join({"A": "U", "T": "A", "G": "C", "C": "G"}[b] for b in template)
    codons = [mrna[i:i + 3] for i in range(0, len(mrna) - 2, 3)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cw = min(44, (W - 120) / len(template))
    lx = x0 + 120
    cv.text(x0, y0 + 30, "DNA\n(template)", step, 18, MUTED, "left", "middle", math=False)
    for i, b in enumerate(template):
        _box(cv, lx + i * cw, y0 + 10, cw - 3, 40, step, b, BASE_COL[b], 18)
    step += 1
    cv.text(x0, y0 + 120, "mRNA", step, 18, MUTED, "left", "middle", math=False)
    cv.text(lx, y0 + 66, "transcription (nucleus): A→U, T→A, G→C, C→G", step, 16, "#e8590c", math=False)
    for i, b in enumerate(mrna):
        _box(cv, lx + i * cw, y0 + 100, cw - 3, 40, step, b, BASE_COL[b], 18)
        if i % 3 == 2:
            step += 1
    cv.text(x0, y0 + 250, "protein", step, 18, MUTED, "left", "middle", math=False)
    cv.text(lx, y0 + 160, "translation (ribosome): read 3 bases = 1 codon = 1 amino acid", step, 16, "#e8590c", math=False)
    for k, c in enumerate(codons):
        aa = GENETIC_CODE.get(c, "?")
        x = lx + k * 3 * cw
        cv.rect(x - 2, y0 + 97, 3 * cw - 1, 46, step, "transparent", "#f08c00", 3)
        cv.ellipse(x + 3 * cw / 2 - 38, y0 + 220, 76, 60, step, "#ffd8a8" if aa != "STOP" else "#ffc9c9")
        cv.text(x + 3 * cw / 2, y0 + 250, aa, step, 19, INK, "center", "middle", math=False)
        if k:
            cv.line([(x + 3 * cw / 2 - 38 - 3 * cw + 76, y0 + 250), (x + 3 * cw / 2 - 38, y0 + 250)], step, INK, 2)
        step += 1
        if aa == "STOP":
            break
    return step + 1


def build_heart(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cx, cy = x0 + W * 0.38, (y0 + y1) / 2 + 10
    s = min(W * 0.3, H * 0.42)
    outline = [(cx, cy + s), (cx - s * 0.95, cy - s * 0.05), (cx - s * 0.85, cy - s * 0.65), (cx - s * 0.4, cy - s * 0.8),
               (cx, cy - s * 0.5), (cx + s * 0.4, cy - s * 0.8), (cx + s * 0.85, cy - s * 0.65), (cx + s * 0.95, cy - s * 0.05),
               (cx, cy + s)]
    _poly(cv, outline, step, "#c92a2a", "#fff5f5", 3)
    cv.line([(cx, cy - s * 0.5), (cx, cy + s * 0.95)], step, "#c92a2a", 3)
    cv.line([(cx - s * 0.85, cy - s * 0.15), (cx - s * 0.05, cy - s * 0.15)], step, "#c92a2a", 2)
    cv.line([(cx + s * 0.05, cy - s * 0.15), (cx + s * 0.85, cy - s * 0.15)], step, "#c92a2a", 2)
    # viewer's left = heart's right side (blue, deoxygenated)
    labs = [("right atrium", cx - s * 0.5, cy - s * 0.42, "#d0ebff"), ("right ventricle", cx - s * 0.45, cy + s * 0.3, "#d0ebff"),
            ("left atrium", cx + s * 0.5, cy - s * 0.42, "#ffc9c9"), ("left ventricle", cx + s * 0.45, cy + s * 0.3, "#ffc9c9")]
    step += 1
    for lab, x, y, col in labs:
        cv.ellipse(x - s * 0.28, y - s * 0.18, s * 0.56, s * 0.36, step, col, "transparent", 1)
        cv.text(x, y, lab, step, 17, INK, "center", "middle", math=False)
        step += 1
    path = [("body (vena cava)", "right atrium"), ("right atrium", "right ventricle"), ("right ventricle", "lungs"),
            ("lungs", "left atrium"), ("left atrium", "left ventricle"), ("left ventricle", "body (aorta)")]
    tx = x0 + W * 0.72
    for i, (a, b) in enumerate(path):
        col = "#1971c2" if i < 3 else "#e03131"
        cv.text(tx, y0 + 30 + i * 44, f"{i + 1}. {a} → {b}", step, 19, col, math=False)
        step += 1
    cv.text(tx, y0 + 30 + 6 * 44 + 10, "blue: oxygen-poor   red: oxygen-rich", step, 16, MUTED, math=False)
    return step + 1


def build_water_cycle(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    sea_y = y1 - 60
    cv.rect(x0 + W * 0.45, sea_y, W * 0.55, 60, step, "#74c0fc", "#1971c2", 2, rounded=False)
    cv.text(x0 + W * 0.72, sea_y + 30, "ocean", step, 20, "#ffffff", "center", "middle", math=False)
    cv.polygon([(x0, sea_y + 60), (x0 + W * 0.15, y0 + H * 0.35), (x0 + W * 0.3, y0 + H * 0.55), (x0 + W * 0.47, sea_y + 60)],
               step, "#b2f2bb", "#2b8a3e", 2)
    step += 1
    sx, sy = x1 - 70, y0 + 50
    cv.ellipse(sx - 34, sy - 34, 68, 68, step, "#ffd43b", "#f08c00", 2)
    cloud = draw_device(cv, "cloud", x0 + W * 0.3, y0 + 70, 160, step)
    step += 1
    stages = [("evaporation", (x0 + W * 0.75, sea_y - 10), (x0 + W * 0.62, y0 + 120)),
              ("condensation", (x0 + W * 0.58, y0 + 80), (x0 + W * 0.42, y0 + 75)),
              ("precipitation", (x0 + W * 0.25, y0 + 120), (x0 + W * 0.2, y0 + H * 0.4)),
              ("runoff", (x0 + W * 0.3, y0 + H * 0.6), (x0 + W * 0.5, sea_y - 6))]
    for lab, a, b in stages:
        cv.arrow(a[0], a[1], b[0], b[1], step, lab, color="#1971c2")
        if lab == "precipitation":
            for k in range(6):
                cv.line([(x0 + W * (0.22 + k * 0.03), y0 + 110), (x0 + W * (0.21 + k * 0.03), y0 + 135)], step,
                        "#4dabf7", 2)
        step += 1
    return step


def build_neuron(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cx, cy = x0 + 160, (y0 + y1) / 2
    rnd = random.Random(2)
    for a in range(0, 360, 40):
        r = math.radians(a)
        pts = [(cx + 50 * math.cos(r), cy + 50 * math.sin(r)), (cx + 110 * math.cos(r + 0.2), cy + 110 * math.sin(r + 0.2)),
               (cx + 150 * math.cos(r - 0.1), cy + 150 * math.sin(r - 0.1))]
        if 60 < a < 300 or a < 30:
            cv.line(pts, step, "#9c36b5", 3)
    cv.ellipse(cx - 55, cy - 55, 110, 110, step, "#e5dbff", "#7048e8", 3)
    cv.ellipse(cx - 18, cy - 18, 36, 36, step, "#9775fa", "#7048e8", 2)
    ax1 = x1 - 140
    cv.line([(cx + 55, cy), (ax1, cy)], step, "#7048e8", 6)
    for k in range(5):
        mx = cx + 90 + k * (ax1 - cx - 120) / 5
        cv.rect(mx, cy - 14, (ax1 - cx - 120) / 5 - 20, 28, step, "#fff3bf", "#f08c00", 1.5)
    for d in (-1, 0, 1):
        cv.line([(ax1, cy), (ax1 + 60, cy + d * 50)], step, "#7048e8", 3)
        cv.ellipse(ax1 + 54, cy + d * 50 - 8, 16, 16, step, "#9775fa", "#7048e8", 1)
    for lab, x, y in (("dendrites", cx - 150, cy - 140), ("cell body", cx, cy + 70), ("axon (myelin)", (cx + ax1) / 2, cy + 30),
                      ("axon terminals", ax1 + 40, cy + 80)):
        cv.text(x, y, lab, step, 18, INK, "center", "top", math=False)
    step += 1
    n = 8
    for k in range(n + 1):
        x = cx + 55 + (ax1 - cx - 55) * k / n
        tr = _Track(cv)
        cv.text(x, cy - 22, "⚡", step, 28, "#f08c00", "center", "bottom", math=False)
        tr.tag(until=step + 1 if k < n else None, fade=True)
        step += 1
    cv.text(x0, y0, "the action potential jumps along the axon to the terminals", step, 19, "#e8590c", math=False)
    return step + 1


def build_states_matter(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    bw = (W - 60) / 3
    rnd = random.Random(1)
    for i, (name, note) in enumerate((("solid", "fixed places, vibrate"), ("liquid", "close, slide past each other"),
                                      ("gas", "far apart, move fast"))):
        x = x0 + i * (bw + 30)
        cv.rect(x, y0 + 30, bw, H - 110, step, "#f8f9fa", INK, 2.5)
        cv.text(x + bw / 2, y0, name, step, 26, "#e8590c", "center", "top", math=False)
        cv.text(x + bw / 2, y1 - 70, note, step, 17, INK, "center", "top", math=False)
        step += 1
        for f in range(3):
            tr = _Track(cv)
            if i == 0:
                for a in range(6):
                    for b in range(5):
                        cv.ellipse(x + 24 + a * (bw - 40) / 6 + rnd.uniform(-2, 2), y0 + H - 140 - b * 36 + rnd.uniform(-2, 2),
                                   28, 28, step, "#74c0fc", "#1971c2", 1)
            elif i == 1:
                for _ in range(18):
                    cv.ellipse(x + rnd.uniform(10, bw - 40), y0 + H * 0.45 + rnd.uniform(0, H * 0.35), 28, 28, step,
                               "#74c0fc", "#1971c2", 1)
            else:
                for _ in range(8):
                    cv.ellipse(x + rnd.uniform(10, bw - 40), y0 + 40 + rnd.uniform(0, H - 190), 28, 28, step, "#74c0fc",
                               "#1971c2", 1)
            tr.tag(until=step + 1 if f < 2 else None, fade=f > 0)
            step += 1
    return step


def build_wave(cv, v, box, step):
    lam, amp = float(v.get("wavelength") or 2), float(v.get("amplitude") or 1)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cy = (y0 + y1) / 2
    sx = (W * 0.75) / (lam * 2.5)
    sy = H * 0.3 / amp
    cv.line([(x0, cy), (x0 + W * 0.78, cy)], step, MUTED, 1.5, dashed=True)
    step += 1
    for f in range(8):
        ph = f * math.pi / 4
        tr = _Track(cv)
        pts = [(x0 + x * sx, cy - amp * sy * math.sin(2 * math.pi * x / lam - ph)) for x in [i * lam * 2.5 / 80 for i in range(81)]]
        cv.line(pts, step, "#1971c2", 3)
        tr.tag(until=step + 1 if f < 7 else None, fade=f > 0)
        step += 1
    xc = lam * (0.25 + 7 / 8) % lam                    # a crest of the last frame (phase 7π/4)
    a = x0 + xc * sx
    cv.line([(a, cy), (a, cy - amp * sy)], step, "#e03131", 2, arrow_end=True)
    cv.text(a + 8, cy - amp * sy / 2, "amplitude", step, 18, "#e03131", "left", "middle", math=False)
    cv.line([(a, cy - amp * sy - 20), (a + lam * sx, cy - amp * sy - 20)], step, "#2f9e44", 2,
            arrow_end=True, arrow_start=True)
    cv.text(a + lam * sx / 2, cy - amp * sy - 26, "wavelength λ (crest to crest)", step, 18, "#2f9e44", "center",
            "bottom", math=False)
    cv.text(x0 + W * 0.8, y0 + 40, "v = f × λ\ncrest = top\ntrough = bottom", step, 22, INK, math=False)
    return step + 1


def build_refraction(cv, v, box, step):
    n1, n2 = float(v.get("n1") or 1.0), float(v.get("n2") or 1.5)
    th1 = math.radians(float(v.get("angle") or 40))
    s2 = n1 * math.sin(th1) / n2
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cx, cy = x0 + W * 0.35, (y0 + y1) / 2
    cv.rect(x0, cy, W * 0.7, H / 2, step, "#d0ebff", "transparent", 1, rounded=False)
    cv.line([(x0, cy), (x0 + W * 0.7, cy)], step, INK, 2)
    cv.line([(cx, y0), (cx, y1)], step, MUTED, 1.5, dashed=True)
    cv.text(x0 + 10, cy - 30, f"air n₁ = {n1:g}", step, 19, INK, math=False)
    cv.text(x0 + 10, cy + 10, f"glass n₂ = {n2:g}", step, 19, INK, math=False)
    step += 1
    L = H * 0.45
    cv.arrow(cx - L * math.sin(th1), cy - L * math.cos(th1), cx, cy, step, color="#f08c00")
    step += 1
    if s2 <= 1:
        th2 = math.asin(s2)
        cv.arrow(cx, cy, cx + L * math.sin(th2), cy + L * math.cos(th2), step, color="#f08c00")
        res = f"θ₂ = {math.degrees(th2):.1f}° (bends toward the normal)" if n2 > n1 else f"θ₂ = {math.degrees(th2):.1f}°"
    else:
        cv.arrow(cx, cy, cx + L * math.sin(th1), cy - L * math.cos(th1), step, color="#e03131")
        res = "total internal reflection"
    cv.text(x0 + W * 0.73, y0 + 40, f"Snell's law\nn₁ sin θ₁ = n₂ sin θ₂\n{n1:g} × sin {math.degrees(th1):g}° = {n2:g} × sin θ₂\n{res}",
            step, 21, INK, math=False)
    return step + 1


PH = [(0, "battery acid"), (2, "lemon juice"), (3, "vinegar"), (5, "coffee"), (7, "pure water"), (8, "seawater"),
      (9, "baking soda"), (11, "ammonia"), (12, "soapy water"), (13, "bleach"), (14, "drain cleaner")]


def build_ph_scale(cv, v, box, step):
    x0, y0, x1, y1 = box
    W = x1 - x0
    cw = (W - 20) / 15
    cols = ["#e03131", "#f03e3e", "#fd7e14", "#fab005", "#fcc419", "#ffe066", "#d8f5a2", "#8ce99a", "#63e6be", "#3bc9db",
            "#4dabf7", "#5c7cfa", "#7950f2", "#9c36b5", "#862e9c"]
    for i in range(15):
        _box(cv, x0 + i * cw, y0 + 60, cw - 2, 70, step, i, cols[i], 22)
    cv.text(x0, y0 + 20, "acidic", step, 22, "#e03131", math=False)
    cv.text(x0 + 7 * cw, y0 + 20, "neutral", step, 22, "#2f9e44", math=False)
    cv.text(x0 + 14 * cw, y0 + 20, "basic", step, 22, "#862e9c", math=False)
    step += 1
    for k, (p, name) in enumerate(PH):
        x = x0 + p * cw + cw / 2
        y = y0 + 150 + (k % 3) * 50
        cv.line([(x, y0 + 132), (x, y)], step, MUTED, 1)
        cv.text(x, y + 4, name, step, 17, INK, "center", "top", math=False)
        step += 1
    cv.text(x0, y0 + 330, "each step is 10× more H⁺ ions: pH 3 is 10,000× more acidic than pH 7", step, 19, "#1971c2",
            math=False)
    return step + 1


def build_solar_system(cv, v, box, step):
    planets = [("Mercury", 0.39, 6, "#adb5bd"), ("Venus", 0.72, 9, "#ffd8a8"), ("Earth", 1.0, 10, "#4dabf7"),
               ("Mars", 1.52, 8, "#ff8787"), ("Jupiter", 5.2, 24, "#e9a76b"), ("Saturn", 9.5, 20, "#ffe066"),
               ("Uranus", 19.2, 15, "#99e9f2"), ("Neptune", 30.1, 15, "#5c7cfa")]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    sun_r = H * 0.42
    cx, cy = x0 - sun_r * 0.55, (y0 + y1) / 2            # the sun, mostly off the left edge
    cv.ellipse(cx - sun_r, cy - sun_r, 2 * sun_r, 2 * sun_r, step, "#ffd43b", "#f08c00", 2)
    cv.text(x0 + 6, cy, "Sun", step, 22, "#e8590c", "left", "middle", math=False)
    step += 1
    span = W - sun_r * 0.5 - 40
    for i, (name, au, r, col) in enumerate(planets):
        d = sun_r + 30 + span * math.sqrt(au / 30.1) - 30
        arc = [(cx + d * math.cos(t), cy + d * math.sin(t)) for t in [math.radians(a) for a in range(-14, 15, 2)]]
        cv.line(arc, step, "#dee2e6", 1.5)
        px, py = cx + d, cy
        cv.ellipse(px - r, py - r, 2 * r, 2 * r, step, col, INK, 1.5)
        if name == "Saturn":
            cv.ellipse(px - r * 1.8, py - r * 0.45, r * 3.6, r * 0.9, step, "transparent", "#e67700", 2)
        cv.text(px, py + (r + 8) * (1 if i % 2 == 0 else -1), f"{name}\n{au:g} AU", step, 15, INK, "center",
                "top" if i % 2 == 0 else "bottom", math=False)
        step += 1
    cv.text(x1, y1 - 4, "distances on a square-root scale; 1 AU = Earth–Sun distance", step, 15, MUTED, "right",
            "bottom", math=False)
    return step + 1


# ======================================================================== maths / economics

def build_venn(cv, v, box, step):
    sets = v.get("sets") or {"A": [1, 2, 3, 4, 5], "B": [4, 5, 6, 7], "C": [2, 5, 7, 8]}
    names = list(sets)[:3]
    S = {n: set(map(str, sets[n])) for n in names}
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    r = min(W * 0.2, H * 0.3)
    cx, cy = x0 + W * 0.36, (y0 + y1) / 2 - H * 0.04
    centers = [(cx - r * 0.55, cy - r * 0.3), (cx + r * 0.55, cy - r * 0.3), (cx, cy + r * 0.55)][:len(names)]
    cols = ["#a5d8ff", "#ffc9c9", "#b2f2bb"]
    for (x, y), n, c in zip(centers, names, cols):
        e = cv.ellipse(x - r, y - r, 2 * r, 2 * r, step, c, STRONG[names.index(n)], 2)
        e["opacity"] = 60
        lx, ly = (x - r, y - r * 0.9) if x < cx else (x + r, y - r * 0.9) if x > cx else (x, y + r + 16)
        cv.text(lx, ly, n, step, 26, STRONG[names.index(n)], "center", "middle", math=False)
    step += 1
    allx = set().union(*S.values())
    regions = {}
    for e in sorted(allx):
        key = tuple(n for n in names if e in S[n])
        regions.setdefault(key, []).append(e)
    for key, els in regions.items():
        pts = [centers[names.index(n)] for n in key]
        px = sum(p[0] for p in pts) / len(pts)
        py = sum(p[1] for p in pts) / len(pts)
        if len(key) == 1:
            ox, oy = px - cx, py - cy
            L = math.hypot(ox, oy) or 1
            px, py = px + ox / L * r * 0.4, py + oy / L * r * 0.4
        cv.text(px, py, " ".join(els), step, 20, INK, "center", "middle", math=False)
        step += 1
    if len(names) >= 2:
        a, b = names[0], names[1]
        txt = f"{a} ∪ {b} = {{{', '.join(sorted(S[a] | S[b]))}}}\n{a} ∩ {b} = {{{', '.join(sorted(S[a] & S[b]))}}}\n" \
              f"{a} − {b} = {{{', '.join(sorted(S[a] - S[b]))}}}"
        if len(names) == 3:
            txt += f"\n{a} ∩ {b} ∩ {names[2]} = {{{', '.join(sorted(S[a] & S[b] & S[names[2]]))}}}"
        cv.text(x0 + W * 0.68, y0 + 40, txt, step, 20, INK, math=False)
    return step + 1


def build_prob_tree(cv, v, box, step):
    a = float(v.get("p") or 0.3)
    b_given = v.get("given") or {"A": 0.9, "not A": 0.2}
    names = v.get("names") or ["rain", "late"]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    root = (x0 + 30, (y0 + y1) / 2)
    first = [(f"{names[0]}", a, b_given.get("A", 0.9)), (f"no {names[0]}", 1 - a, b_given.get("not A", 0.2))]
    total = 0
    for i, (lab, p, pb) in enumerate(first):
        n1 = (x0 + W * 0.3, y0 + H * (0.25 + 0.5 * i))
        cv.line([root, n1], step, INK, 2)
        cv.text((root[0] + n1[0]) / 2, (root[1] + n1[1]) / 2 - 6, f"{p:g}", step, 19, "#1971c2", "center", "bottom",
                math=False)
        cv.text(n1[0] + 8, n1[1], lab, step, 20, INK, "left", "middle", math=False)
        step += 1
        for j, (lab2, q) in enumerate(((names[1], pb), (f"not {names[1]}", 1 - pb))):
            n2 = (x0 + W * 0.55, n1[1] + (j - 0.5) * H * 0.22)
            cv.line([(n1[0] + 90, n1[1]), n2], step, INK, 2)
            cv.text((n1[0] + 90 + n2[0]) / 2, (n1[1] + n2[1]) / 2 - 6, f"{q:g}", step, 18, "#1971c2", "center", "bottom",
                    math=False)
            prod = p * q
            if j == 0:
                total += prod
            cv.text(n2[0] + 8, n2[1], f"{lab2}:  {p:g} × {q:g} = {prod:.3f}", step, 19, "#2f9e44" if j == 0 else INK,
                    "left", "middle", math=False)
            step += 1
    cv.text(x0 + W * 0.55, y1 - 20, f"P({names[1]}) = {total:.3f}   P({names[0]} | {names[1]}) = {a * first[0][2] / total:.3f}",
            step, 20, "#e8590c", "left", "bottom", math=False)
    return step + 1


def build_matrix_mult(cv, v, box, step):
    A = v.get("A") or [[1, 2], [3, 4]]
    B = v.get("B") or [[5, 6], [7, 8]]
    n, m, p = len(A), len(B), len(B[0])
    C = [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]
    x0, y0, x1, y1 = box
    c = 64
    ax, bx = x0, x0 + len(A[0]) * c + 60
    cx = bx + p * c + 60
    for i in range(n):
        for k in range(m):
            _box(cv, ax + k * c, y0 + 40 + i * c, c - 4, c - 4, step, A[i][k], "#d0ebff", 22)
    for k in range(m):
        for j in range(p):
            _box(cv, bx + j * c, y0 + 40 + k * c, c - 4, c - 4, step, B[k][j], "#ffec99", 22)
    cv.text(ax + len(A[0]) * c + 20, y0 + 40 + c * n / 2, "×", step, 34, INK, "center", "middle", math=False)
    cv.text(bx + p * c + 30, y0 + 40 + c * n / 2, "=", step, 34, INK, "center", "middle", math=False)
    step += 1
    for i in range(n):
        for j in range(p):
            tr = _Track(cv)
            cv.rect(ax - 4, y0 + 36 + i * c, m * c + 4, c + 4, step, "transparent", "#1971c2", 4)
            cv.rect(bx + j * c - 4, y0 + 36, c + 4, m * c + 4, step, "transparent", "#f08c00", 4)
            calc = " + ".join(f"{A[i][k]}·{B[k][j]}" for k in range(m)) + f" = {C[i][j]}"
            cv.text(x0, y0 + 60 + max(n, m) * c, f"row {i + 1} · column {j + 1}:  {calc}", step, 22, "#c92a2a", math=False)
            tr.tag(until=step + 1, fade=True)
            _box(cv, cx + j * c, y0 + 40 + i * c, c - 4, c - 4, step, C[i][j], "#b2f2bb", 22)
            step += 1
    return step


def build_riemann(cv, v, box, step):
    expr = str(v.get("expr") or "x^2")
    a, b = float(v.get("a") or 0), float(v.get("b") or 2)
    f = visuals._make_fn(expr) or (lambda x: x * x)
    exact = None
    try:
        N = 2000
        exact = sum(f(a + (b - a) * (k + 0.5) / N) for k in range(N)) * (b - a) / N
    except Exception:
        pass
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    ymax = max(f(a + (b - a) * k / 50) for k in range(51))
    px = lambda x: x0 + 40 + (x - a) / (b - a) * W * 0.6
    py = lambda y: y1 - 30 - y / (ymax or 1) * (H - 70)
    cv.line([(x0 + 40, y1 - 30), (x0 + 40 + W * 0.62, y1 - 30)], step, INK, 2, arrow_end=True)
    cv.line([(x0 + 40, y1 - 30), (x0 + 40, y0)], step, INK, 2, arrow_end=True)
    cv.line([(px(a + (b - a) * k / 60), py(f(a + (b - a) * k / 60))) for k in range(61)], step, "#1971c2", 3)
    cv.text(px(b), py(f(b)) - 6, f"y = {expr}", step, 19, "#1971c2", "right", "bottom", math=False)
    step += 1
    for k, n in enumerate((2, 4, 8, 16)):
        tr = _Track(cv)
        dx = (b - a) / n
        area = 0
        for i in range(n):
            xm = a + (i + 0.5) * dx
            h = f(xm)
            area += h * dx
            cv.rect(px(a + i * dx), py(h), px(a + dx) - px(a), py(0) - py(h), step, "#ffec99", "#f08c00", 1.5, rounded=False)
        cv.text(x0 + W * 0.68, y0 + 40, f"n = {n} rectangles\narea ≈ {area:.4f}" + (f"\nexact ≈ {exact:.4f}" if exact else ""),
                step, 22, INK, math=False)
        tr.tag(until=step + 1 if k < 3 else None, fade=k > 0)
        step += 1
    return step


def build_vectors(cv, v, box, step):
    vs = v.get("vectors") or [[3, 1], [1, 2], [-1, 1.5]]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    sx = min(W * 0.6 / 6, H / 6)
    ox, oy = x0 + 80, y1 - 60
    cv.line([(ox - 20, oy), (ox + 6.5 * sx, oy)], step, MUTED, 1, arrow_end=True)
    cv.line([(ox, oy + 20), (ox, oy - 5.5 * sx)], step, MUTED, 1, arrow_end=True)
    step += 1
    x = y = 0.0
    for i, (dx, dy) in enumerate(vs):
        cv.line([(ox + x * sx, oy - y * sx), (ox + (x + dx) * sx, oy - (y + dy) * sx)], step, STRONG[i], 3, arrow_end=True)
        cv.text(ox + (x + dx / 2) * sx + 8, oy - (y + dy / 2) * sx, f"({dx:g}, {dy:g})", step, 18, STRONG[i], "left",
                "middle", math=False)
        x, y = x + dx, y + dy
        step += 1
    cv.line([(ox, oy), (ox + x * sx, oy - y * sx)], step, INK, 4, arrow_end=True)
    cv.text(x0 + W * 0.68, y0 + 40, f"resultant = ({x:g}, {y:g})\nlength = √({x:g}² + {y:g}²) = {math.hypot(x, y):.2f}\n"
                                   f"angle = {math.degrees(math.atan2(y, x)):.1f}°", step, 22, INK, math=False)
    return step + 1


def build_supply_demand(cv, v, box, step):
    a, b = float(v.get("demand_a") or 100), float(v.get("demand_b") or 2)     # P = a - b Q
    c, d = float(v.get("supply_c") or 10), float(v.get("supply_d") or 1)      # P = c + d Q
    q = (a - c) / (b + d)
    p = a - b * q
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    qmax = a / b
    sx = W * 0.6 / qmax
    sy = (H - 60) / a
    O = (x0 + 60, y1 - 30)
    cv.line([O, (O[0] + W * 0.62, O[1])], step, INK, 2, arrow_end=True)
    cv.line([O, (O[0], y0)], step, INK, 2, arrow_end=True)
    cv.text(O[0] + W * 0.62, O[1] + 4, "quantity", step, 18, INK, "right", "top", math=False)
    cv.text(O[0] + 6, y0, "price", step, 18, INK, math=False)
    step += 1
    pt = lambda Q, P: (O[0] + Q * sx, O[1] - P * sy)
    cv.line([pt(0, a), pt(qmax, 0)], step, "#1971c2", 3)
    cv.text(*pt(qmax * 0.85, a - b * qmax * 0.85), "demand", step, 20, "#1971c2", "left", "bottom", math=False)
    step += 1
    qs = (a - c) / d if (a - c) / d < qmax else qmax
    cv.line([pt(0, c), pt(qs, c + d * qs)], step, "#e03131", 3)
    cv.text(*pt(qs * 0.9, c + d * qs * 0.9), "supply", step, 20, "#e03131", "right", "bottom", math=False)
    step += 1
    E = pt(q, p)
    cv.line([(O[0], E[1]), E], step, MUTED, 1.5, dashed=True)
    cv.line([E, (E[0], O[1])], step, MUTED, 1.5, dashed=True)
    cv.ellipse(E[0] - 9, E[1] - 9, 18, 18, step, "#2f9e44", "#2f9e44", 1)
    cv.text(x0 + W * 0.68, y0 + 40, f"demand: P = {a:g} − {b:g}Q\nsupply: P = {c:g} + {d:g}Q\nequal where\n"
                                   f"{a:g} − {b:g}Q = {c:g} + {d:g}Q\nQ* = {q:g}, P* = {p:g}", step, 22, INK, math=False)
    return step + 1


# ======================================================================== register

visuals.BUILDERS.update({
    "sort_bars": build_sort_bars, "hash_table": build_hash_table, "bst": build_bst, "heap": build_heap,
    "recursion": build_recursion, "fsm": build_fsm, "page_table": build_page_table, "mapreduce": build_mapreduce,
    "git_graph": build_git_graph, "subnet": build_subnet, "binary": build_binary, "blockchain": build_blockchain,
    "cipher": build_cipher, "firewall": build_firewall, "load_balancer": build_load_balancer, "ddos": build_ddos,
    "layers_ring": build_layers_ring, "convolution": build_convolution, "confusion": build_confusion,
    "stats": build_stats, "bpe": build_bpe, "photosynthesis": build_photosynthesis, "dna": build_dna,
    "central_dogma": build_central_dogma, "heart": build_heart, "water_cycle": build_water_cycle,
    "neuron": build_neuron, "states_matter": build_states_matter, "wave": build_wave, "refraction": build_refraction,
    "ph_scale": build_ph_scale, "solar_system": build_solar_system, "venn": build_venn, "prob_tree": build_prob_tree,
    "matrix_mult": build_matrix_mult, "riemann": build_riemann, "vectors": build_vectors,
    "supply_demand": build_supply_demand,
})
for _k in [k for k in list(visuals.ALIASES) if k in visuals.BUILDERS]:
    visuals.ALIASES.pop(_k)
visuals.ALIASES.update({
    "sorting_bars": "sort_bars", "hashing": "hash_table", "binary_search_tree": "bst", "priority_queue": "heap",
    "call_stack": "recursion", "dfa": "fsm", "automaton": "fsm", "state_machine": "fsm", "virtual_memory": "page_table",
    "git": "git_graph", "cidr": "subnet", "subnetting": "subnet", "number_base": "binary", "hash_chain": "blockchain",
    "caesar": "cipher", "firewall_rules": "firewall", "defense_in_depth": "layers_ring", "cia": "layers_ring",
    "cnn": "convolution", "kernel": "convolution", "confusion_matrix": "confusion", "histogram_box": "stats",
    "boxplot": "stats", "box_plot": "stats", "tokenizer": "bpe", "dna_replication": "dna", "transcription": "central_dogma",
    "translation": "central_dogma", "protein_synthesis": "central_dogma", "circulation": "heart",
    "hydrologic_cycle": "water_cycle", "nerve": "neuron", "matter": "states_matter", "transverse_wave": "wave",
    "snell": "refraction", "ph": "ph_scale", "planets": "solar_system", "sets": "venn", "probability_tree": "prob_tree",
    "tree_diagram": "prob_tree", "matrix_multiplication": "matrix_mult", "integral": "riemann", "area_under_curve": "riemann",
    "vector_addition": "vectors", "economics": "supply_demand", "market": "supply_demand",
})
