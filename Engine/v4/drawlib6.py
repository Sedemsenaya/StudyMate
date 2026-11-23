"""
StudyMate v4 - drawing library, part 6. (computed) = worked out by the engine, not guessed.

Algorithms   kruskal (union-find MST, computed), kmp (prefix table + search, computed), n_queens (backtracking,
             computed), avl (inserts with rotations, computed), big_o (growth curves, computed), b_tree (inserts with
             splits + search path, computed)
Systems      cpu_pipeline (5 stages, load-use stall, computed), producer_consumer (bounded buffer + semaphores,
             computed), go_back_n (lost packet, resend window, computed), ip_header (real IPv4 header + checksum,
             computed), vpn_tunnel (encapsulation), cdn (edge hits / misses, computed), acid_transaction (rollback)
Cloud/DevOps docker_layers, rolling_update (Kubernetes maxSurge 1, computed), terraform_plan (desired vs state diff,
             computed), kafka_partitions (key hashing + consumer group, computed), virtual_dom (React tree diff,
             computed), token_bucket (rate limiter, computed)
Security     diffie_hellman (computed), rsa (small primes, computed)
ML / NLP     linear_regression (least squares + residuals, computed), logistic_regression (gradient descent,
             computed), rnn_unrolled, cross_validation (k folds, computed), tfidf (computed), perceptron (updates,
             computed)
Statistics   boxplot (quartiles, outliers, computed), histogram (Sturges bins, computed), central_limit (sample means,
             computed), hypothesis_test (z-test, p-value, computed)
Maths        taylor_series (computed), fourier_series (computed), newtons_method (computed), eigen_transform (computed),
             number_bases (repeated division, computed), modular_clock (computed)
Physics      spring_shm (computed), lever_torque (computed), buoyancy (computed), electric_field (computed),
             em_spectrum (frequencies computed)
Chemistry    titration (pH curve, computed), limiting_reagent (computed), vsepr (shapes and angles)
Biology      enzyme (lock and key + Michaelis-Menten, computed), action_potential, hardy_weinberg (computed),
             population_growth (exponential vs logistic, computed)
Earth / GIS  idw_interpolation (computed), topographic (contour bands, computed), raster_vector (rasterized, computed),
             spatial_join (point in polygon, computed)
Economics    npv (discounted cash flows, computed), elasticity (midpoint method, computed)
"""
import hashlib
import math
import random

import visuals
from canvas import INK, MUTED
from drawlib import _envelope, _Track, draw_device
from drawlib2 import PAL, STRONG, _box, _table
from drawlib3 import _frames, _title
from drawlib4 import _axes


def _mono(el):
    if el is not None:
        el["fontFamily"] = 3
    return el


def _plot(cv, box, step, xr, yr, xl="", yl="", frac=0.62, top=10, bottom=30, left=50):
    """Axes in the left `frac` of the box. Returns (px, py) mapping data -> canvas."""
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    ax0, ax1, ay0, ay1 = x0 + left, x0 + left + W * frac, y0 + top, y1 - bottom
    _axes(cv, ax0, ay0, ax1 + 10, ay1, step, xl, yl)
    px = lambda x: ax0 + (x - xr[0]) / (xr[1] - xr[0]) * (ax1 - ax0)
    py = lambda y: ay1 - (y - yr[0]) / (yr[1] - yr[0]) * (ay1 - ay0 - 20)
    return px, py


def _note(cv, box, step, s, size=19, frac=0.68, dy=30, color=INK):
    x0, y0, x1, y1 = box
    cv.text(x0 + (x1 - x0) * frac, y0 + dy, s, step, size, color, math=False)


def _pl(cv, pts, step, color, sw=3, dashed=False):
    el = cv.line(pts, step, color, sw, dashed=dashed)
    el["roundness"] = None
    return el


# ======================================================================== algorithms

def build_kruskal(cv, v, box, step):
    pos = {"A": (0.1, 0.2), "B": (0.42, 0.08), "C": (0.75, 0.22), "D": (0.12, 0.75), "E": (0.45, 0.55), "F": (0.78, 0.8)}
    edges = v.get("edges") or [("A", "B", 4), ("A", "D", 3), ("B", "C", 6), ("B", "E", 2), ("A", "E", 5),
                               ("C", "E", 7), ("C", "F", 3), ("D", "E", 4), ("E", "F", 8), ("D", "F", 9)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    P = lambda n: (x0 + W * 0.6 * pos[n][0] + 20, y0 + (H - 40) * pos[n][1] + 20)
    def seg(a, b):
        (ax, ay), (bx, by) = P(a), P(b)
        L = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / L * 24, (by - ay) / L * 24
        return [(ax + ux, ay + uy), (bx - ux, by - uy)]
    for a, b, w in edges:
        (ax, ay), (bx, by) = P(a), P(b)
        cv.line(seg(a, b), step, "#ced4da", 2)
        cv.text((ax + bx) / 2 + 8, (ay + by) / 2 - 8, str(w), step, 16, MUTED, "center", "middle", math=False)
    for n in pos:
        x, y = P(n)
        cv.ellipse(x - 22, y - 22, 44, 44, step, "#d0ebff", INK, 2)
        cv.text(x, y, n, step, 18, INK, "center", "middle", math=False)
    _title(cv, x0 + W * 0.66, y0, "Kruskal: cheapest edges first,\nskip any that make a cycle", step, 19)
    step += 1
    parent = {n: n for n in pos}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    total, used = 0, 0
    for i, (a, b, w) in enumerate(sorted(edges, key=lambda e: e[2])):
        ra, rb = find(a), find(b)
        ok = ra != rb
        (ax, ay), (bx, by) = P(a), P(b)
        if ok:
            parent[ra] = rb
            total += w
            used += 1
            cv.line(seg(a, b), step, "#2f9e44", 5)
        cv.text(x0 + W * 0.66, y0 + 70 + i * 26, f"{a}–{b} ({w}): " + ("add ✔" if ok else "skip, cycle ✘"), step, 17,
                "#2f9e44" if ok else "#e03131", math=False)
        step += 1
        if used == len(pos) - 1:
            break
    cv.text(x0 + W * 0.66, y0 + 90 + (i + 1) * 26, f"minimum spanning tree: {used} edges, total {total}", step, 19,
            "#1971c2", math=False)
    return step + 1


def kmp_table(p):
    lps, k = [0] * len(p), 0
    for i in range(1, len(p)):
        while k and p[i] != p[k]:
            k = lps[k - 1]
        if p[i] == p[k]:
            k += 1
        lps[i] = k
    return lps


def build_kmp(cv, v, box, step):
    pat = str(v.get("pattern") or "ABABC")[:10]
    text = str(v.get("text") or "ABABDABABABC")[:18]
    lps = kmp_table(pat)
    x0, y0, x1, y1 = box
    W = x1 - x0
    c = min(54, W * 0.9 / len(text))
    _title(cv, x0, y0, "KMP: the prefix table tells how far to jump back", step, 20)
    for i, ch in enumerate(pat):
        _box(cv, x0 + i * c, y0 + 50, c - 4, 40, step, ch, "#ffec99", 18)
    cv.text(x0 - 4, y0 + 110, "", step, 14, MUTED, math=False)
    step += 1
    for i, val in enumerate(lps):
        _box(cv, x0 + i * c, y0 + 96, c - 4, 36, step, val, "#d0ebff", 17)
    cv.text(x0 + len(pat) * c + 10, y0 + 114, "← longest prefix that is also a suffix", step, 15, MUTED, "left", "middle",
            math=False)
    step += 1
    for i, ch in enumerate(text):
        _box(cv, x0 + i * c, y0 + 180, c - 4, 40, step, ch, "#ffffff", 18)
    step += 1
    frames, j = [], 0
    for i, ch in enumerate(text):
        while j and ch != pat[j]:
            frames.append((i, j, f"mismatch at text[{i}]: jump pattern back to {lps[j - 1]} (not 0)"))
            j = lps[j - 1]
        if ch == pat[j]:
            j += 1
        if j == len(pat):
            frames.append((i, j, f"match found at index {i - j + 1} ✔"))
            break
    frames = frames[:8]

    def draw(k, st):
        i, jj, msg = frames[k]
        start = i - len(pat) + 1 if "match found" in msg else i - jj
        for q, ch in enumerate(pat):
            _box(cv, x0 + (start + q) * c, y0 + 230, c - 4, 40, st, ch,
                 "#b2f2bb" if "found" in msg else ("#ffc9c9" if q == jj else "#fff3bf"), 18)
        cv.text(x0, y0 + 290, msg, st, 19, "#2f9e44" if "found" in msg else "#e03131", math=False)
    step = _frames(cv, step, len(frames), draw)
    cv.text(x0, y0 + 330, "text pointer never moves back: O(n + m)", step, 18, INK, math=False)
    return step + 1


def build_n_queens(cv, v, box, step):
    n = min(6, int(v.get("n") or 4))
    states = []

    def safe(cols, c):
        r = len(cols)
        return all(c != cc and abs(c - cc) != r - rr for rr, cc in enumerate(cols))

    def solve(cols):
        if len(cols) == n:
            states.append((list(cols), None, "solution ✔"))
            return True
        for c in range(n):
            if safe(cols, c):
                states.append((cols + [c], None, f"place row {len(cols)} at column {c}"))
                if solve(cols + [c]):
                    return True
                states.append((list(cols), (len(cols), c), f"dead end: backtrack from row {len(cols)}, col {c}"))
            else:
                states.append((list(cols), (len(cols), c), f"row {len(cols)}, col {c} is attacked"))
        return False
    solve([])
    if len(states) > 22:
        keep = sorted({round(i * (len(states) - 1) / 21) for i in range(22)})
        states = [states[i] for i in keep]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    c = min(W * 0.55, H - 20) / n
    _title(cv, x0 + W * 0.62, y0, f"{n}-queens by backtracking", step, 21)
    for r in range(n):
        for cc in range(n):
            cv.rect(x0 + cc * c, y0 + r * c, c, c, step, "#f1f3f5" if (r + cc) % 2 else "#ffffff", INK, 1, rounded=False)
    step += 1

    def draw(k, st):
        cols, bad, msg = states[k]
        for r, cc in enumerate(cols):
            cv.text(x0 + cc * c + c / 2, y0 + r * c + c / 2, "♛", st, int(c * 0.6), "#2f9e44" if "solution" in msg else
                    INK, "center", "middle", math=False)
        if bad:
            cv.text(x0 + bad[1] * c + c / 2, y0 + bad[0] * c + c / 2, "✘", st, int(c * 0.5), "#e03131", "center",
                    "middle", math=False)
        cv.text(x0 + W * 0.62, y0 + 60, f"step {k + 1}\n{msg}", st, 20, "#2f9e44" if "solution" in msg else INK,
                math=False)
    step = _frames(cv, step, len(states), draw)
    cv.text(x0 + W * 0.62, y0 + 160, "try a column; if a later row has\nno safe square, undo the last\nchoice and try the next",
            step, 18, INK, math=False)
    return step + 1


class _AVL:
    def __init__(self, k):
        self.k, self.l, self.r, self.h = k, None, None, 1


def _h(n):
    return n.h if n else 0


def _upd(n):
    n.h = 1 + max(_h(n.l), _h(n.r))


def _rot_r(y):
    x = y.l
    y.l, x.r = x.r, y
    _upd(y)
    _upd(x)
    return x


def _rot_l(x):
    y = x.r
    x.r, y.l = y.l, x
    _upd(x)
    _upd(y)
    return y


def avl_insert(n, k, log):
    if not n:
        return _AVL(k)
    if k < n.k:
        n.l = avl_insert(n.l, k, log)
    else:
        n.r = avl_insert(n.r, k, log)
    _upd(n)
    b = _h(n.l) - _h(n.r)
    if b > 1 and k < n.l.k:
        log.append(f"left-left at {n.k}: rotate right")
        return _rot_r(n)
    if b < -1 and k >= n.r.k:
        log.append(f"right-right at {n.k}: rotate left")
        return _rot_l(n)
    if b > 1:
        log.append(f"left-right at {n.k}: rotate left, then right")
        n.l = _rot_l(n.l)
        return _rot_r(n)
    if b < -1:
        log.append(f"right-left at {n.k}: rotate right, then left")
        n.r = _rot_r(n.r)
        return _rot_l(n)
    return n


def _tree_layout(root, kids, key):
    """In-order x, depth y for any binary-ish tree given kids(node) -> (left, right)."""
    out, idx = [], [0]

    def walk(n, d, parent):
        if not n:
            return
        l, r = kids(n)
        walk(l, d + 1, n)
        out.append((key(n), idx[0], d, key(parent) if parent else None, n))
        idx[0] += 1
        walk(r, d + 1, n)
    walk(root, 0, None)
    return out


def build_avl(cv, v, box, step):
    keys = [int(k) for k in (v.get("insert") or [10, 20, 30, 40, 50, 25])][:9]
    snaps, root = [], None
    for k in keys:
        log = []
        root = avl_insert(root, k, log)
        lay = _tree_layout(root, lambda n: (n.l, n.r), lambda n: n.k)
        snaps.append((k, [(kk, i, d, p, _h(nd.l) - _h(nd.r)) for kk, i, d, p, nd in lay], log))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    _title(cv, x0, y0, "AVL tree: rebalance when heights differ by 2", step, 20)
    step += 1

    def draw(k, st):
        new, lay, log = snaps[k]
        n = len(lay)
        P = {kk: (x0 + 30 + (i + 0.5) * W * 0.6 / n, y0 + 70 + d * 80) for kk, i, d, p, b in lay}
        for kk, i, d, p, b in lay:
            if p is not None:
                cv.line([P[p], P[kk]], st, MUTED, 1.5)
        for kk, i, d, p, b in lay:
            x, y = P[kk]
            cv.ellipse(x - 24, y - 22, 48, 44, st, "#ffec99" if kk == new else "#d0ebff", INK, 2)
            cv.text(x, y, str(kk), st, 17, INK, "center", "middle", math=False)
            cv.text(x + 26, y - 26, f"{b:+d}" if b else "0", st, 13, "#e03131" if abs(b) > 1 else MUTED, math=False)
        cv.text(x0 + W * 0.66, y0 + 60, f"insert {new}\n\n" + ("\n".join(log) if log else "still balanced"), st, 20,
                "#e03131" if log else "#2f9e44", math=False)
    step = _frames(cv, step, len(snaps), draw)
    cv.text(x0 + W * 0.66, y0 + 200, "small numbers = balance factor\n(left height − right height)\n\nheight stays O(log n)", step,
            18, INK, math=False)
    return step + 1


def build_big_o(cv, v, box, step):
    fns = [("O(1)", lambda n: 1), ("O(log n)", lambda n: math.log2(n)), ("O(n)", lambda n: n),
           ("O(n log n)", lambda n: n * math.log2(n)), ("O(n²)", lambda n: n * n), ("O(2ⁿ)", lambda n: 2 ** n)]
    N, cap = 20, 100
    px, py = _plot(cv, box, step, (1, N), (0, cap), "input size n", "steps", frac=0.55)
    step += 1
    for i, (name, f) in enumerate(fns):
        pts = []
        for k in range(1, 61):
            n = 1 + (N - 1) * k / 60
            y = f(n)
            if y > cap:
                pts.append((px(n), py(cap)))
                break
            pts.append((px(n), py(y)))
        _pl(cv, pts, step, STRONG[i], 3)
        x, y = pts[-1]
        cv.text(x + 6, y - 4, name, step, 16, STRONG[i], "left", "bottom", math=False)
        step += 1
    x0, y0, x1, y1 = box
    rows = [[name, f"{f(1000):,.0f}" if f(1000) < 1e12 else "≈ 10^301"] for name, f in fns]
    _table(cv, x0 + (x1 - x0) * 0.66, y0 + 20, "steps when n = 1000", ["growth", "steps"], rows, step,
           (x1 - x0) * 0.32, size=15)
    return step + 1


def btree_insert(root, k, t_max=2, log=None):
    """B-tree where a node holds at most t_max keys (order t_max+1). Nodes: {'k': [...], 'c': [...]}"""
    def ins(n):
        if not n["c"]:
            n["k"].append(k)
            n["k"].sort()
        else:
            i = sum(1 for x in n["k"] if k > x)
            split = ins(n["c"][i])
            if split:
                mid, right = split
                n["k"].insert(i, mid)
                n["c"].insert(i + 1, right)
        if len(n["k"]) > t_max:
            m = len(n["k"]) // 2
            mid = n["k"][m]
            right = {"k": n["k"][m + 1:], "c": n["c"][m + 1:]}
            n["k"], n["c"] = n["k"][:m], n["c"][:m + 1]
            if log is not None:
                log.append(f"node full: split, {mid} moves up")
            return mid, right
        return None
    s = ins(root)
    if s:
        mid, right = s
        root = {"k": [mid], "c": [root, right]}
    return root


def build_b_tree(cv, v, box, step):
    keys = [int(x) for x in (v.get("insert") or [10, 20, 5, 6, 12, 30, 7, 17])][:12]
    find = int(v.get("search") or keys[-1])
    root, snaps = {"k": [], "c": []}, []
    for k in keys:
        log = []
        root = btree_insert(root, k, 2, log)
        snaps.append((k, _copy_bt(root), log))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0

    def layout(r):
        levels, q = [], [r]
        while q:
            levels.append(q)
            q = [c for n in q for c in n["c"]]
        return levels

    def draw_tree(r, st, path=()):
        levels = layout(r)
        P = {}
        for d, lvl in enumerate(levels):
            for j, n in enumerate(lvl):
                P[id(n)] = (x0 + W * 0.62 * (j + 0.5) / len(lvl), y0 + 60 + d * 95)
        for lvl in levels:
            for n in lvl:
                for c in n["c"]:
                    cv.line([(P[id(n)][0], P[id(n)][1] + 20), (P[id(c)][0], P[id(c)][1] - 20)], st,
                            "#e8590c" if id(c) in path else MUTED, 3 if id(c) in path else 1.5)
        for lvl in levels:
            for n in lvl:
                x, y = P[id(n)]
                w = 44 * max(1, len(n["k"]))
                _box(cv, x - w / 2, y - 20, w, 40, st, "  ".join(map(str, n["k"])),
                     "#ffd8a8" if id(n) in path else "#e7f5ff", 16)
    _title(cv, x0, y0, "B-tree (at most 2 keys per node)", step, 20)
    step += 1

    def draw(k, st):
        new, r, log = snaps[k]
        draw_tree(r, st)
        cv.text(x0 + W * 0.66, y0 + 60, f"insert {new}\n" + ("\n".join(log) if log else "fits in a leaf"), st, 19,
                "#e03131" if log else "#2f9e44", math=False)
    step = _frames(cv, step, len(snaps), draw)
    path, n, steps = [], root, 0
    while True:
        path.append(id(n))
        steps += 1
        if find in n["k"] or not n["c"]:
            break
        n = n["c"][sum(1 for x in n["k"] if find > x)]
    tr = _Track(cv)
    draw_tree(root, step, tuple(path))
    tr.tag(fade=True)
    cv.text(x0 + W * 0.66, y0 + 200, f"search {find}: {steps} node reads\n(databases use this so each\nnode = one disk page)",
            step, 18, "#e8590c", math=False)
    return step + 1


def _copy_bt(n):
    return {"k": list(n["k"]), "c": [_copy_bt(c) for c in n["c"]]}


# ======================================================================== systems

def pipeline_schedule(instrs):
    """instrs: (text, dest, srcs, is_load). Returns list of (text, [stage per cycle or None])."""
    stages = ["IF", "ID", "EX", "MEM", "WB"]
    rows, start, prev = [], 0, None
    for i, (txt, dest, srcs, load) in enumerate(instrs):
        stall = 1 if prev and prev[3] and prev[1] in srcs else 0
        row = {}
        t = start
        row[t] = "IF"
        t += 1
        for _ in range(stall):
            row[t] = "stall"
            t += 1
        for s in stages[1:]:
            row[t] = s
            t += 1
        rows.append((txt, row, stall))
        start += 1 + stall
        prev = (txt, dest, srcs, load)
    return rows


def build_cpu_pipeline(cv, v, box, step):
    instrs = [("lw  r1, 0(r2)", "r1", ["r2"], True), ("add r3, r1, r4", "r3", ["r1", "r4"], False),
              ("sub r5, r6, r7", "r5", ["r6", "r7"], False), ("and r8, r5, r3", "r8", ["r5", "r3"], False),
              ("or  r9, r1, r2", "r9", ["r1", "r2"], False)]
    rows = pipeline_schedule(instrs)
    ncyc = max(max(r) for _, r, _ in rows) + 1
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    lw, cw = 190, min(70, (W - 200) / ncyc)
    cols = {"IF": "#d0ebff", "ID": "#d3f9d8", "EX": "#fff3bf", "MEM": "#ffd8a8", "WB": "#e5dbff", "stall": "#ffc9c9"}
    for c in range(ncyc):
        cv.text(x0 + lw + c * cw + cw / 2, y0, str(c + 1), step, 15, MUTED, "center", "top", math=False)
    cv.text(x0, y0, "cycle →", step, 15, MUTED, math=False)
    step += 1
    for i, (txt, row, stall) in enumerate(rows):
        y = y0 + 30 + i * 52
        _mono(cv.text(x0, y + 22, txt, step, 17, INK, "left", "middle", math=False))
        for c, s in row.items():
            _box(cv, x0 + lw + c * cw, y, cw - 4, 44, step, s, cols[s], 15)
        if stall:
            cv.text(x0 + lw + ncyc * cw + 8, y + 22, "waits for r1 (load-use)", step, 15, "#e03131", "left", "middle",
                    math=False)
        step += 1
    cv.text(x0, y0 + 50 + len(rows) * 52, f"{len(rows)} instructions in {ncyc} cycles (without a pipeline: {len(rows) * 5}).\n"
                                          "Each stage works on a different instruction at once; forwarding passes\n"
                                          "results early, but a value loaded from memory still costs one stall.",
            step, 17, INK, math=False)
    return step + 1


def build_producer_consumer(cv, v, box, step):
    size = int(v.get("size") or 4)
    events = v.get("events") or list("PPCPPPPCCPCCCCPP")
    buf, items, n = [], [], 0
    frames = []
    for e in events:
        if e == "P":
            if len(buf) == size:
                frames.append((list(buf), "producer blocks: buffer full (empty = 0)", "#e03131"))
                continue
            n += 1
            buf.append(f"item{n}")
            frames.append((list(buf), f"producer adds item{n}", "#2f9e44"))
        else:
            if not buf:
                frames.append((list(buf), "consumer blocks: buffer empty (full = 0)", "#e03131"))
                continue
            it = buf.pop(0)
            frames.append((list(buf), f"consumer takes {it}", "#1971c2"))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cy = y0 + H * 0.35
    draw_device(cv, "server", x0 + 60, cy, 80, step)
    cv.text(x0 + 60, cy + 50, "producer", step, 17, INK, "center", "top", math=False)
    draw_device(cv, "laptop", x0 + W * 0.85, cy, 80, step)
    cv.text(x0 + W * 0.85, cy + 50, "consumer", step, 17, INK, "center", "top", math=False)
    bw = W * 0.5 / size
    bx = x0 + W * 0.18
    for i in range(size):
        cv.rect(bx + i * bw, cy - 30, bw - 4, 60, step, "#ffffff", INK, 2, rounded=False)
    step += 1

    def draw(k, st):
        b, msg, col = frames[k]
        for i, it in enumerate(b):
            _box(cv, bx + i * bw + 4, cy - 24, bw - 12, 48, st, it, "#ffec99", 15)
        cv.text(x0 + W * 0.18, cy + 60, f"semaphores: empty = {size - len(b)}, full = {len(b)}   mutex guards the buffer",
                st, 17, INK, math=False)
        cv.text(x0 + W * 0.18, cy + 100, msg, st, 21, col, math=False)
    return _frames(cv, step, len(frames), draw)


def build_go_back_n(cv, v, box, step):
    total, N, lost = int(v.get("packets") or 8), int(v.get("window") or 4), {int(v.get("lost") or 2)}
    rows, base, expected, first = [], 0, 0, True
    while base < total:
        for s in range(base, min(base + N, total)):
            drop = s in lost and first
            if drop:
                rows.append((s, "lost", None))
                continue
            if s == expected:
                expected += 1
                rows.append((s, "ok", expected - 1))
            else:
                rows.append((s, "discard", expected - 1))
        if expected == base:
            pass
        if lost & set(range(base, base + N)) and first:
            rows.append((None, "timeout", base if expected == base else expected))
            first = False
        base = expected
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    sx, rx = x0 + 80, x0 + W * 0.55
    dy = (H - 60) / len(rows)
    draw_device(cv, "laptop", sx, y0 + 10, 40, step)
    draw_device(cv, "server", rx, y0 + 10, 40, step)
    cv.line([(sx, y0 + 40), (sx, y1)], step, MUTED, 1.5, dashed=True)
    cv.line([(rx, y0 + 40), (rx, y1)], step, MUTED, 1.5, dashed=True)
    cv.text(x0 + W * 0.68, y0 + 10, f"Go-Back-N, window {N}\npacket {min(lost)} is lost\n\nthe receiver only accepts\n"
                                    "packets in order; the sender\nresends everything from the\nlost one after a timeout",
            step, 18, INK, math=False)
    step += 1
    for i, (s, kind, ack) in enumerate(rows):
        y = y0 + 50 + i * dy
        if kind == "timeout":
            cv.text(sx - 70, y, f"⏰ timeout → resend from {ack}", step, 16, "#e03131", math=False)
        elif kind == "lost":
            cv.line([(sx, y), ((sx + rx) / 2, y + dy * 0.4)], step, "#e03131", 2, arrow_end=False)
            cv.text((sx + rx) / 2 + 6, y + dy * 0.4, f"pkt {s} ✘", step, 16, "#e03131", "left", "middle", math=False)
        else:
            cv.arrow(sx, y, rx, y + dy * 0.7, step, f"pkt {s}", color="#1971c2" if kind == "ok" else MUTED)
            lab = f"ACK {ack}" if kind == "ok" else f"out of order: discard, ACK {ack}"
            cv.text(rx + 10, y + dy * 0.7, lab, step, 15, "#2f9e44" if kind == "ok" else "#e8590c", "left", "middle",
                    math=False)
        step += 1
    return step


def ip_checksum(words):
    s = sum(words)
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    return ~s & 0xFFFF


def build_ip_header(cv, v, box, step):
    src, dst = str(v.get("src") or "192.168.1.10"), str(v.get("dst") or "8.8.8.8")
    ttl, proto, length, ident = 64, 17, 60, 0x1c46
    ip = lambda s: [int(o) for o in s.split(".")]
    a, b = ip(src), ip(dst)
    words = [0x4500, length, ident, 0x4000, (ttl << 8) | proto, 0, (a[0] << 8) | a[1], (a[2] << 8) | a[3],
             (b[0] << 8) | b[1], (b[2] << 8) | b[3]]
    ck = ip_checksum(words)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    bw = W * 0.62 / 32
    rows = [[("version 4", 4), ("IHL 5", 4), ("TOS 0", 8), (f"total length {length}", 16)],
            [(f"identification 0x{ident:04x}", 16), ("flags DF", 3), ("fragment offset 0", 13)],
            [(f"TTL {ttl}", 8), (f"protocol {proto} (UDP)", 8), (f"checksum 0x{ck:04x}", 16)],
            [(f"source {src}", 32)], [(f"destination {dst}", 32)]]
    for i in range(0, 33, 8):
        cv.text(x0 + i * bw, y0, str(i), step, 13, MUTED, "center", "top", math=False)
    _title(cv, x0 + W * 0.66, y0, "IPv4 header (20 bytes)", step, 20)
    step += 1
    for r, fields in enumerate(rows):
        x = x0
        for name, bits in fields:
            _box(cv, x, y0 + 26 + r * 56, bits * bw - 3, 50, step, name,
                 "#ffd8a8" if "checksum" in name else "#e7f5ff" if r < 3 else "#d3f9d8", 14)
            x += bits * bw
        step += 1
    s = sum(words)
    _note(cv, box, step, f"checksum (computed):\nadd the ten 16-bit words\n= 0x{s:x}\nfold the carry, flip the bits\n"
                         f"= 0x{ck:04x}\n\nevery router lowers TTL by 1\nand recomputes the checksum", 17, dy=60)
    return step + 1


def build_vpn_tunnel(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cy = y0 + H * 0.3
    pts = [("laptop", "home laptop\n10.8.0.5", 0.06), ("internet", "public internet", 0.38),
           ("firewall", "VPN gateway", 0.66), ("server", "office server\n10.0.0.20", 0.92)]
    for kind, lab, f in pts:
        draw_device(cv, kind, x0 + W * f, cy, 80, step)
        cv.text(x0 + W * f, cy + 50, lab, step, 15, INK, "center", "top", math=False)
    tube = cv.rect(x0 + W * 0.1, cy - 60, W * 0.52, 120, step, "transparent", "#7048e8", 3)
    tube["strokeStyle"] = "dashed"
    cv.text(x0 + W * 0.36, cy - 72, "encrypted tunnel", step, 16, "#7048e8", "center", "bottom", math=False)
    step += 1
    for k, f in enumerate((0.12, 0.25, 0.5, 0.75)):
        tr = _Track(cv)
        _envelope(cv, x0 + W * f, cy - 30, 46, step, "🔒" if f < 0.66 else "")
        tr.tag(until=step + 1 if k < 3 else None, fade=k > 0)
        step += 1
    bx, by = x0 + W * 0.05, y0 + H * 0.62
    _box(cv, bx, by, W * 0.88, 70, step, "", "#e5dbff", 14)
    cv.text(bx + 10, by + 6, "outer IP: home public IP → gateway public IP   |   ESP / WireGuard header", step, 15,
            "#5f3dc4", math=False)
    _box(cv, bx + W * 0.3, by + 28, W * 0.56, 36, step, "inner packet (encrypted): 10.8.0.5 → 10.0.0.20, data", "#ffffff", 15)
    step += 1
    cv.text(bx, by + 86, "the gateway strips the outer header, decrypts, and forwards the inner packet on the office network",
            step, 16, INK, math=False)
    return step + 1


def build_cdn(cv, v, box, step):
    reqs = v.get("requests") or [("US", "logo.png"), ("EU", "logo.png"), ("US", "logo.png"), ("ASIA", "video.mp4"),
                                 ("EU", "logo.png"), ("ASIA", "video.mp4"), ("US", "video.mp4")]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    edges = {"US": 0.12, "EU": 0.38, "ASIA": 0.64}
    ox, oy = x0 + W * 0.38, y0 + 40
    draw_device(cv, "database", ox, oy, 70, step)
    cv.text(ox + 50, oy, "origin server", step, 16, INK, "left", "middle", math=False)
    for r, f in edges.items():
        ex, ey = x0 + W * f, y0 + H * 0.45
        draw_device(cv, "server", ex, ey, 64, step)
        cv.text(ex, ey + 40, f"edge {r}", step, 15, INK, "center", "top", math=False)
        cv.line([(ex, ey - 34), (ox, oy + 36)], step, "#dee2e6", 1.5, dashed=True)
        draw_device(cv, "users", ex, y1 - 50, 54, step)
    step += 1
    cache, hits = {r: set() for r in edges}, 0
    for i, (r, f) in enumerate(reqs):
        ex, ey = x0 + W * edges[r], y0 + H * 0.45
        hit = f in cache[r]
        hits += hit
        cache[r].add(f)
        tr = _Track(cv)
        cv.arrow(ex, y1 - 80, ex, ey + 36, step, color="#1971c2")
        if not hit:
            cv.arrow(ex, ey - 34, ox, oy + 36, step, color="#e03131")
        tr.tag(until=step + 1, fade=True)
        cv.text(x0 + W * 0.74, y0 + 20 + i * 28, f"{r} asks {f}: " + ("HIT (fast)" if hit else "MISS → fetch from origin"),
                step, 16, "#2f9e44" if hit else "#e03131", math=False)
        step += 1
    cv.text(x0 + W * 0.74, y0 + 40 + len(reqs) * 28, f"hit rate {hits}/{len(reqs)} = {hits / len(reqs):.0%}\n"
                                                     "edges keep copies close to\nusers (TTL decides how long)",
            step, 17, INK, math=False)
    return step + 1


def build_acid_transaction(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    frames = [({"A": 100, "B": 50}, "BEGIN; transfer 30 from A to B", INK),
              ({"A": 70, "B": 50}, "UPDATE A = A − 30  (not committed yet)", "#e8590c"),
              ({"A": 70, "B": 50}, "⚡ crash before B is updated!", "#e03131"),
              ({"A": 100, "B": 50}, "ROLLBACK from the log: A back to 100 (atomic: all or nothing)", "#1971c2"),
              ({"A": 100, "B": 50}, "retry: BEGIN", INK), ({"A": 70, "B": 50}, "UPDATE A = A − 30", "#e8590c"),
              ({"A": 70, "B": 80}, "UPDATE B = B + 30", "#e8590c"),
              ({"A": 70, "B": 80}, "COMMIT ✔ written to disk (durable); total still 150 (consistent)", "#2f9e44")]
    draw_device(cv, "database", x0 + 60, y0 + 80, 90, step)
    step += 1

    def draw(k, st):
        acc, msg, col = frames[k]
        for i, (n, bal) in enumerate(acc.items()):
            _box(cv, x0 + 150 + i * 200, y0 + 50, 170, 70, st, f"account {n}\n${bal}", "#e7f5ff", 20)
        cv.text(x0 + 150, y0 + 150, msg, st, 21, col, math=False)
        for j in range(k + 1):
            cv.text(x0 + 150, y0 + 200 + j * 24, f"{j + 1}. {frames[j][1]}", st, 15, MUTED, math=False)
    step = _frames(cv, step, len(frames), draw)
    _note(cv, box, step, "A atomic\nC consistent\nI isolated\nD durable", 20, frac=0.8, dy=40, color="#7048e8")
    return step + 1


# ======================================================================== cloud / devops

def build_docker_layers(cv, v, box, step):
    lines = v.get("dockerfile") or [("FROM python:3.12-slim", 45), ("WORKDIR /app", 0.0), ("COPY requirements.txt .", 0.01),
                                    ("RUN pip install -r requirements.txt", 85), ("COPY . .", 2), ('CMD ["python", "app.py"]', 0)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    for i, (ln, mb) in enumerate(lines):
        _mono(cv.text(x0, y0 + 10 + i * 30, ln, step, 16, INK, math=False))
    step += 1
    lh = (H - 140) / len(lines)
    bx, bw = x0 + W * 0.42, W * 0.3
    for i, (ln, mb) in enumerate(lines):
        y = y1 - 30 - (i + 1) * lh
        _box(cv, bx, y, bw, lh - 6, step, f"layer {i + 1}: {ln.split()[0]}  ({mb:g} MB)", PAL[i % len(PAL)], 15)
        step += 1
    for j in range(2):
        _box(cv, bx + j * (bw / 2 + 4), y1 - 30 - (len(lines) + 1) * lh - 10, bw / 2 - 4, lh, step,
             f"container {j + 1}\nwritable layer", "#ffffff", 14)
    step += 1
    total = sum(mb for _, mb in lines)
    cv.text(x0, y0 + 30 + len(lines) * 30, f"image ≈ {total:g} MB, read-only, shared by\nboth containers\n\n"
                                           "layers are cached: change app code and\nonly 'COPY . .' rebuilds, which is why\n"
                                           "requirements are copied first", step, 17, INK, math=False)
    return step + 1


def build_rolling_update(cv, v, box, step):
    n, surge = int(v.get("replicas") or 4), 1
    pods = ["v1"] * n
    frames = [(list(pods), "4 pods running v1")]
    for i in range(n):
        frames.append((pods + ["v2…"], f"start a new v2 pod (maxSurge {surge})"))
        pods = pods[1:] + ["v2"]
        frames.append((list(pods), "v2 pod ready → remove one v1 pod"))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    draw_device(cv, "router", x0 + W * 0.33, y0 + 30, 60, step)
    cv.text(x0 + W * 0.33 + 40, y0 + 30, "Service (load balancer)", step, 16, INK, "left", "middle", math=False)
    step += 1

    def draw(k, st):
        ps, msg = frames[k]
        for i, p in enumerate(sorted(ps)):
            x = x0 + 20 + i * (W * 0.66 / (n + 1))
            col = "#a5d8ff" if p == "v1" else "#b2f2bb" if p == "v2" else "#fff3bf"
            _box(cv, x, y0 + 140, W * 0.66 / (n + 1) - 14, 90, st, f"pod\n{p}", col, 18)
            if p != "v2…":
                cv.line([(x0 + W * 0.33, y0 + 60), (x + 40, y0 + 138)], st, MUTED, 1.2)
        ready = sum(p != "v2…" for p in ps)
        cv.text(x0 + 20, y0 + 270, f"{msg}\nready pods: {ready} (never below {n})", st, 20, INK, math=False)
    step = _frames(cv, step, len(frames), draw)
    _note(cv, box, step, "kubectl set image ...\n\nzero downtime: traffic only\ngoes to ready pods\n\n"
                         "kubectl rollout undo\nreverses it", 18, frac=0.74, dy=120)
    return step + 1


def build_terraform_plan(cv, v, box, step):
    desired = v.get("desired") or {"aws_instance.web": "t3.small", "aws_s3_bucket.logs": "private",
                                   "aws_db_instance.main": "db.t3.micro"}
    state = v.get("state") or {"aws_instance.web": "t3.micro", "aws_db_instance.main": "db.t3.micro",
                               "aws_security_group.old": "port 22 open"}
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    _table(cv, x0, y0 + 20, "main.tf (desired)", ["resource", "setting"], [[k, v_] for k, v_ in desired.items()], step,
           W * 0.38, size=14)
    _table(cv, x0 + W * 0.42, y0 + 20, "terraform.tfstate (real)", ["resource", "setting"],
           [[k, v_] for k, v_ in state.items()], step, W * 0.38, size=14)
    step += 1
    plan = []
    for k in sorted(set(desired) | set(state)):
        if k not in state:
            plan.append(("+", k, "create", "#2f9e44"))
        elif k not in desired:
            plan.append(("-", k, "destroy", "#e03131"))
        elif desired[k] != state[k]:
            plan.append(("~", k, f"update {state[k]} → {desired[k]}", "#e8590c"))
    for i, (sym, k, what, col) in enumerate(plan):
        _mono(cv.text(x0, y0 + 210 + i * 30, f"  {sym} {k}: {what}", step, 17, col, math=False))
        step += 1
    cnt = lambda s: sum(p[0] == s for p in plan)
    _mono(cv.text(x0, y0 + 230 + len(plan) * 30, f"Plan: {cnt('+')} to add, {cnt('~')} to change, {cnt('-')} to destroy.",
                  step, 18, INK, math=False))
    _note(cv, box, step, "terraform plan compares the\ncode with the state file;\napply makes the cloud match", 17,
          frac=0.84, dy=40)
    return step + 1


def build_kafka_partitions(cv, v, box, step):
    msgs = v.get("messages") or [("user-1", "login"), ("user-2", "click"), ("user-1", "buy"), ("user-3", "login"),
                                 ("user-2", "logout"), ("user-4", "click"), ("user-1", "logout")]
    P = int(v.get("partitions") or 3)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    draw_device(cv, "server", x0 + 40, y0 + H * 0.4, 60, step)
    cv.text(x0 + 40, y0 + H * 0.4 + 40, "producer", step, 15, INK, "center", "top", math=False)
    for p in range(P):
        y = y0 + 30 + p * (H - 60) / P
        cv.text(x0 + 110, y + 20, f"partition {p}", step, 15, STRONG[p], "left", "middle", math=False)
        cv.rect(x0 + 200, y, W * 0.46, 40, step, "#f8f9fa", INK, 1.5, rounded=False)
    consumers = {p: f"consumer {'A' if p < (P + 1) // 2 else 'B'}" for p in range(P)}
    for p in range(P):
        y = y0 + 30 + p * (H - 60) / P
        cv.arrow(x0 + 210 + W * 0.46, y + 20, x0 + W * 0.78, y + 20, step, consumers[p], color=MUTED)
    step += 1
    offs = [0] * P
    for key, val in msgs:
        p = int(hashlib.md5(key.encode()).hexdigest(), 16) % P
        y = y0 + 30 + p * (H - 60) / P
        _box(cv, x0 + 204 + offs[p] * 112, y + 3, 106, 34, step, f"{offs[p]}: {key}", PAL[p], 13)
        offs[p] += 1
        step += 1
    _note(cv, box, step, "hash(key) % partitions:\nsame key → same partition,\nso one user's events stay\nin order\n\n"
                         "each partition goes to one\nconsumer in the group", 17, frac=0.8, dy=40)
    return step + 1


def vdom_diff(a, b, path="root", out=None):
    out = [] if out is None else out
    if a is None:
        out.append(("add", path, b[0]))
        return out
    if b is None:
        out.append(("remove", path, a[0]))
        return out
    if a[0] != b[0]:
        out.append(("replace", path, f"{a[0]} → {b[0]}"))
        return out
    ka, kb = a[1], b[1]
    if isinstance(ka, str) or isinstance(kb, str):
        if ka != kb:
            out.append(("text", path, f'"{ka}" → "{kb}"'))
        return out
    for i in range(max(len(ka), len(kb))):
        vdom_diff(ka[i] if i < len(ka) else None, kb[i] if i < len(kb) else None, f"{path}/{(ka + kb)[i][0] if i < len(ka) else kb[i][0]}{i}", out)
    return out


def build_virtual_dom(cv, v, box, step):
    old = ("div", [("h1", "Cart"), ("p", "2 items"), ("button", "Buy")])
    new = ("div", [("h1", "Cart"), ("p", "3 items"), ("button", "Buy"), ("span", "Free shipping!")])
    patches = vdom_diff(old, new)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0

    def tree(t, ox, st, changed):
        rx, ry = ox + W * 0.16, y0 + 50
        _box(cv, rx - 40, ry - 20, 80, 40, st, f"<{t[0]}>", "#e7f5ff", 16)
        kids = t[1]
        for i, (tag, txt) in enumerate(kids):
            cx, cy = ox + (i + 0.5) * W * 0.33 / len(kids), y0 + 150
            cv.line([(rx, ry + 20), (cx, cy - 22)], st, MUTED, 1.5)
            col = "#ffc9c9" if i in changed.get("text", ()) else "#b2f2bb" if i in changed.get("add", ()) else "#ffffff"
            _box(cv, cx - 50, cy - 22, 100, 44, st, f"<{tag}>", col, 15)
            cv.text(cx, cy + 30, txt, st, 13, MUTED, "center", "top", math=False)
    cv.text(x0, y0, "old virtual DOM", step, 18, "#e8590c", math=False)
    tree(old, x0, step, {})
    step += 1
    cv.text(x0 + W * 0.36, y0, "new virtual DOM (state changed)", step, 18, "#e8590c", math=False)
    ch = {"text": [i for i, (a, b) in enumerate(zip(old[1], new[1])) if a != b], "add": list(range(len(old[1]), len(new[1])))}
    tree(new, x0 + W * 0.36, step, ch)
    step += 1
    for i, (kind, path, what) in enumerate(patches):
        _mono(cv.text(x0, y0 + 250 + i * 30, f"{kind:7s} {path}: {what}", step, 17, "#e03131" if kind != "add" else "#2f9e44",
                      math=False))
        step += 1
    cv.text(x0, y0 + 270 + len(patches) * 30, f"React compares the two trees and touches only {len(patches)} real DOM "
                                              "nodes, not the whole page", step, 17, INK, math=False)
    return step + 1


def build_token_bucket(cv, v, box, step):
    cap, rate = float(v.get("capacity") or 5), float(v.get("refill") or 1)
    arrivals = v.get("arrivals") or [0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 4.0, 4.2, 6.5, 6.6, 6.7]
    tokens, last, res, curve = cap, 0.0, [], [(0, cap)]
    for t in arrivals:
        tokens = min(cap, tokens + (t - last) * rate)
        curve.append((t, tokens))
        ok = tokens >= 1
        if ok:
            tokens -= 1
        curve.append((t, tokens))
        res.append((t, ok))
        last = t
    curve.append((8, min(cap, tokens + (8 - last) * rate)))
    px, py = _plot(cv, box, step, (0, 8), (0, cap + 1), "time (s)", "tokens", frac=0.6, bottom=70)
    step += 1
    _pl(cv, [(px(t), py(k)) for t, k in curve], step, "#1971c2", 3)
    step += 1
    x0, y0, x1, y1 = box
    for t, ok in res:
        cv.text(px(t), y1 - 50, "✔" if ok else "✘", step, 18, "#2f9e44" if ok else "#e03131", "center", "top", math=False)
    step += 1
    _note(cv, box, step, f"bucket holds {cap:g} tokens,\nrefills {rate:g} per second;\neach request spends one\n\n"
                         f"allowed {sum(o for _, o in res)}, rejected (HTTP 429)\n{sum(not o for _, o in res)}\n\n"
                         "bursts up to the capacity are\nfine; a steady flood is not", 18, frac=0.7)
    return step + 1


# ======================================================================== security

def build_diffie_hellman(cv, v, box, step):
    p, g, a, b = (int(v.get(k) or d) for k, d in (("p", 23), ("g", 5), ("a", 6), ("b", 15)))
    A, B = pow(g, a, p), pow(g, b, p)
    sa, sb = pow(B, a, p), pow(A, b, p)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cols = [("Alice", x0 + W * 0.12, "#1971c2"), ("public channel (Eve sees this)", x0 + W * 0.45, MUTED),
            ("Bob", x0 + W * 0.8, "#2f9e44")]
    for name, x, col in cols:
        draw_device(cv, "attacker" if "Eve" in name else "user", x, y0 + 30, 50, step)
        cv.text(x, y0 + 64, name, step, 17, col, "center", "top", math=False)
    step += 1
    rows = [(f"agree p = {p}, g = {g}", f"p = {p}, g = {g}", f"agree p = {p}, g = {g}"),
            (f"secret a = {a}", "", f"secret b = {b}"),
            (f"A = g^a mod p = {A}", f"A = {A}  →   ←  B = {B}", f"B = g^b mod p = {B}"),
            (f"s = B^a mod p = {sa}", "can't get s without a or b\n(discrete log is hard)", f"s = A^b mod p = {sb}")]
    for i, r in enumerate(rows):
        for (name, x, col), txt in zip(cols, r):
            if txt:
                cv.text(x, y0 + 110 + i * 56, txt, step, 18, "#e03131" if "secret" in txt or txt.startswith("s =") and
                        name != cols[1][0] else col, "center", "top", math=False)
        step += 1
    cv.text(x0 + W * 0.45, y1 - 40, f"both get the same shared key {sa} without ever sending it", step, 20, "#e8590c",
            "center", "top", math=False)
    return step + 1


def build_rsa(cv, v, box, step):
    p, q, e, m = (int(v.get(k) or d) for k, d in (("p", 61), ("q", 53), ("e", 17), ("message", 65)))
    n, phi = p * q, (p - 1) * (q - 1)
    d = pow(e, -1, phi)
    c = pow(m, e, n)
    back = pow(c, d, n)
    x0, y0, x1, y1 = box
    W = x1 - x0
    lines = [(f"1. pick primes p = {p}, q = {q}", INK), (f"2. n = p × q = {n}", INK),
             (f"3. φ = (p−1)(q−1) = {phi}", INK), (f"4. public exponent e = {e}", "#2f9e44"),
             (f"5. private d = e⁻¹ mod φ = {d}  (since {e} × {d} mod {phi} = {e * d % phi})", "#e03131"),
             (f"public key (n, e) = ({n}, {e})     private key d = {d}", "#7048e8"),
             (f"encrypt m = {m}:  c = m^e mod n = {c}", "#1971c2"), (f"decrypt:  c^d mod n = {back}  ✔", "#2f9e44")]
    for i, (ln, col) in enumerate(lines):
        cv.text(x0, y0 + 10 + i * 40, ln, step, 21, col, math=False)
        step += 1
    _note(cv, box, step, "safe because factoring n back\ninto p and q is hard when\nthey are ~300 digits long", 18,
          frac=0.72, dy=40)
    return step + 1


# ======================================================================== ML / NLP

def _lsq(pts):
    n = len(pts)
    mx, my = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    b1 = sxy / sxx
    b0 = my - b1 * mx
    ssr = sum((p[1] - b0 - b1 * p[0]) ** 2 for p in pts)
    sst = sum((p[1] - my) ** 2 for p in pts)
    return b0, b1, 1 - ssr / sst


def build_linear_regression(cv, v, box, step):
    rnd = random.Random(7)
    pts = v.get("points") or [(x, 40 + 7 * x + rnd.gauss(0, 6)) for x in [rnd.uniform(0, 8) for _ in range(16)]]
    b0, b1, r2 = _lsq(pts)
    ys = [p[1] for p in pts]
    px, py = _plot(cv, box, step, (0, 8.5), (min(ys) - 10, max(ys) + 10), "hours studied", "score")
    for x, y in pts:
        cv.ellipse(px(x) - 6, py(y) - 6, 12, 12, step, "#a5d8ff", "#1971c2", 1.5)
    step += 1
    _pl(cv, [(px(0), py(b0)), (px(8.5), py(b0 + b1 * 8.5))], step, "#e03131", 3)
    step += 1
    for x, y in pts:
        cv.line([(px(x), py(y)), (px(x), py(b0 + b1 * x))], step, "#ffa8a8", 1.5, dashed=True)
    step += 1
    _note(cv, box, step, f"least squares line:\nscore = {b0:.1f} + {b1:.2f} × hours\n\ndashed = residuals; the line\n"
                         f"minimises their squares\n\nR² = {r2:.2f} ({r2:.0%} of the\nvariation explained)\n\n"
                         f"predict 6 h → {b0 + 6 * b1:.0f}", 19)
    return step + 1


def build_logistic_regression(cv, v, box, step):
    rnd = random.Random(3)
    data = v.get("data") or [(x, 1 if x + rnd.gauss(0, 1.2) > 5 else 0) for x in [rnd.uniform(0, 10) for _ in range(30)]]
    w, b = 0.0, 0.0
    sig = lambda z: 1 / (1 + math.exp(-z))
    snaps = []
    for it in range(1, 2001):
        gw = sum((sig(w * x + b) - y) * x for x, y in data) / len(data)
        gb = sum(sig(w * x + b) - y for x, y in data) / len(data)
        w, b = w - 0.1 * gw, b - 0.1 * gb
        if it in (5, 50, 300, 2000):
            snaps.append((it, w, b))
    px, py = _plot(cv, box, step, (0, 10), (-0.1, 1.1), "hours studied", "P(pass)")
    for x, y in data:
        cv.ellipse(px(x) - 6, py(y) - 6, 12, 12, step, "#b2f2bb" if y else "#ffc9c9", "#2f9e44" if y else "#e03131", 1.5)
    cv.line([(px(0), py(0.5)), (px(10), py(0.5))], step, MUTED, 1, dashed=True)
    step += 1

    def draw(k, st):
        it, ww, bb = snaps[k]
        _pl(cv, [(px(x / 6), py(sig(ww * x / 6 + bb))) for x in range(61)], st, "#1971c2", 3)
        acc = sum((sig(ww * x + bb) > 0.5) == y for x, y in data) / len(data)
        cut = -bb / ww if ww else float("nan")
        cv.text(box[0] + (box[2] - box[0]) * 0.68, box[1] + 30,
                f"gradient descent step {it}\nw = {ww:.2f}, b = {bb:.2f}\n\nP(pass) = 1 / (1 + e^−(wx + b))\n"
                f"decision line at P = 0.5:\nx = {cut:.1f} hours\naccuracy {acc:.0%}", st, 19, INK, math=False)
    return _frames(cv, step, len(snaps), draw)


def build_rnn_unrolled(cv, v, box, step):
    words = v.get("words") or ["the", "cat", "sat", "down"]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    n = len(words)
    cw = W * 0.7 / n
    for i, w in enumerate(words):
        cx = x0 + 40 + i * cw + cw / 2
        _box(cv, cx - 50, y0 + H * 0.65, 100, 40, step, w, "#fff3bf", 17)
        cv.arrow(cx, y0 + H * 0.65 - 2, cx, y0 + H * 0.45 + 32, step, color=MUTED)
        _box(cv, cx - 55, y0 + H * 0.45 - 30, 110, 60, step, "RNN cell\ntanh(Wx + Uh)", "#d0ebff", 14)
        cv.arrow(cx, y0 + H * 0.45 - 32, cx, y0 + H * 0.2 + 20, step, color=MUTED)
        _box(cv, cx - 40, y0 + H * 0.2 - 20, 80, 40, step, f"y{i + 1}", "#d3f9d8", 16)
        if i:
            cv.arrow(cx - cw + 56, y0 + H * 0.45, cx - 57, y0 + H * 0.45, step, f"h{i}", color="#e03131")
        step += 1
    cv.text(x0 + 40, y1 - 30, "the same weights at every step; the hidden state h carries memory forward", step, 17, INK,
            math=False)
    _note(cv, box, step, "long sequences: gradients\nshrink through many steps\n(vanishing gradient) →\nLSTM / GRU gates, then\n"
                         "transformers", 17, frac=0.78, dy=40)
    return step + 1


def build_cross_validation(cv, v, box, step):
    k = int(v.get("folds") or 5)
    rnd = random.Random(9)
    data = [(x, 1 if x + rnd.gauss(0, 1.5) > 5 else 0) for x in [rnd.uniform(0, 10) for _ in range(50)]]
    fold = len(data) // k
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    scores = []
    for f in range(k):
        test = data[f * fold:(f + 1) * fold]
        train = data[:f * fold] + data[(f + 1) * fold:]
        th = max((t / 10 for t in range(101)), key=lambda t: sum((x > t) == y for x, y in train))
        acc = sum((x > th) == y for x, y in test) / len(test)
        scores.append((th, acc))
        y = y0 + 40 + f * (H - 120) / k
        cv.text(x0, y + 16, f"round {f + 1}", step, 16, INK, math=False)
        for j in range(k):
            _box(cv, x0 + 90 + j * W * 0.11, y, W * 0.11 - 6, 34, step, "test" if j == f else "train",
                 "#ffc9c9" if j == f else "#d0ebff", 14)
        cv.text(x0 + 100 + k * W * 0.11, y + 16, f"threshold {th:.1f} → accuracy {acc:.0%}", step, 16, "#1971c2", "left",
                "middle", math=False)
        step += 1
    mean = sum(a for _, a in scores) / k
    sd = math.sqrt(sum((a - mean) ** 2 for _, a in scores) / k)
    cv.text(x0, y1 - 60, f"{k}-fold cross-validation: {mean:.0%} ± {sd:.0%}\nevery point is tested exactly once; a fairer "
                         "estimate than one split", step, 19, "#e8590c", math=False)
    return step + 1


def build_tfidf(cv, v, box, step):
    docs = v.get("docs") or ["the cat sat on the mat", "the dog sat on the log", "cats and dogs are pets"]
    toks = [d.lower().split() for d in docs]
    vocab = sorted({w for t in toks for w in t})
    N = len(docs)
    df = {w: sum(w in t for t in toks) for w in vocab}
    tfidf = [{w: t.count(w) / len(t) * math.log(N / df[w]) for w in vocab} for t in toks]
    show = sorted(vocab, key=lambda w: -max(d[w] for d in tfidf))[:6] + ["the", "sat"]
    show = list(dict.fromkeys(show))[:8]
    x0, y0, x1, y1 = box
    W = x1 - x0
    for i, d in enumerate(docs):
        cv.text(x0, y0 + i * 28, f"doc {i + 1}: {d}", step, 17, INK, math=False)
    step += 1
    rows = [[w, df[w], f"{math.log(N / df[w]):.2f}"] + [f"{d[w]:.3f}" for d in tfidf] for w in show]
    _table(cv, x0, y0 + 100, "", ["word", "docs", "idf"] + [f"doc {i + 1}" for i in range(N)], rows, step, W * 0.62, size=15)
    step += 1
    _note(cv, box, step, "tf = count / words in doc\nidf = log(N / docs with word)\n\n'the' is in 2 of 3 docs → low\n"
                         "a word in every doc → 0\nrare words score highest:\nthey describe the document", 18, dy=100)
    return step + 1


def build_perceptron(cv, v, box, step):
    rnd = random.Random(12)
    pts = []
    while len(pts) < 24:
        x, y = rnd.uniform(-1, 1), rnd.uniform(-1, 1)
        if abs(x + 0.6 * y - 0.1) > 0.12:
            pts.append((x, y, 1 if x + 0.6 * y > 0.1 else -1))
    w, snaps = [0.0, 0.0, 0.0], []
    for ep in range(1, 9):
        errs = 0
        for x, y, t in pts:
            if t * (w[0] * x + w[1] * y + w[2]) <= 0:
                w = [w[0] + 0.2 * t * x, w[1] + 0.2 * t * y, w[2] + 0.2 * t]
                errs += 1
        snaps.append((ep, list(w), errs))
        if not errs:
            break
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    S = min(W * 0.55, H - 20)
    P = lambda x, y: (x0 + (x + 1) / 2 * S, y0 + (1 - (y + 1) / 2) * S)
    cv.rect(x0, y0, S, S, step, "#ffffff", MUTED, 1, rounded=False)
    for x, y, t in pts:
        X, Y = P(x, y)
        cv.text(X, Y, "+" if t > 0 else "−", step, 22, "#2f9e44" if t > 0 else "#e03131", "center", "middle", math=False)
    step += 1

    def draw(k, st):
        ep, ww, errs = snaps[k]
        if abs(ww[1]) > 1e-9:
            line = [P(-1, (-ww[2] + ww[0]) / ww[1]), P(1, (-ww[2] - ww[0]) / ww[1])]
        else:
            xx = -ww[2] / ww[0] if ww[0] else 0
            line = [P(xx, -1), P(xx, 1)]
        (ax, ay), (bx, by) = line
        pts = []
        for t_ in [i / 200 for i in range(201)]:
            X, Y = ax + (bx - ax) * t_, ay + (by - ay) * t_
            if x0 <= X <= x0 + S and y0 <= Y <= y0 + S:
                pts.append((X, Y))
        if len(pts) > 1:
            _pl(cv, [pts[0], pts[-1]], st, "#1971c2", 3)
        cv.text(x0 + S + 30, y0 + 40, f"epoch {ep}: {errs} mistakes\nw = ({ww[0]:.2f}, {ww[1]:.2f}), b = {ww[2]:.2f}",
                st, 20, "#2f9e44" if not errs else INK, math=False)
    step = _frames(cv, step, len(snaps), draw)
    cv.text(x0 + S + 30, y0 + 140, "for each mistake:\nw ← w + η·label·x\n\nstops once every point is on\nthe right side "
                                   "(guaranteed if\nthe data is linearly separable)", step, 18, INK, math=False)
    return step + 1


# ======================================================================== statistics

def _quart(s):
    s = sorted(s)

    def q(p):
        k = (len(s) - 1) * p
        f = math.floor(k)
        return s[f] + (s[min(f + 1, len(s) - 1)] - s[f]) * (k - f)
    return q(0.25), q(0.5), q(0.75)


def build_boxplot(cv, v, box, step):
    data = [float(x) for x in (v.get("data") or [12, 15, 17, 18, 19, 20, 21, 21, 22, 23, 24, 25, 27, 29, 31, 48])]
    q1, med, q3 = _quart(data)
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = [x for x in data if lo_f <= x <= hi_f]
    out = [x for x in data if x not in inside]
    lo, hi = min(data) - 3, max(data) + 3
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    X = lambda val: x0 + 30 + (val - lo) / (hi - lo) * W * 0.62
    cy = y0 + H * 0.35
    for val in data:
        cv.ellipse(X(val) - 5, cy + 90 - 5, 10, 10, step, "#a5d8ff", "#1971c2", 1)
    cv.line([(X(lo), cy + 110), (X(hi), cy + 110)], step, INK, 1.5)
    for t in range(int(lo) // 5 * 5 + 5, int(hi) + 1, 5):
        cv.text(X(t), cy + 116, str(t), step, 13, MUTED, "center", "top", math=False)
    step += 1
    cv.rect(X(q1), cy - 30, X(q3) - X(q1), 60, step, "#ffec99", INK, 2, rounded=False)
    cv.line([(X(med), cy - 30), (X(med), cy + 30)], step, "#e03131", 3)
    step += 1
    cv.line([(X(min(inside)), cy), (X(q1), cy)], step, INK, 2)
    cv.line([(X(q3), cy), (X(max(inside)), cy)], step, INK, 2)
    for e in (min(inside), max(inside)):
        cv.line([(X(e), cy - 15), (X(e), cy + 15)], step, INK, 2)
    for o in out:
        cv.ellipse(X(o) - 7, cy - 7, 14, 14, step, "#ffc9c9", "#e03131", 2)
    step += 1
    _note(cv, box, step, f"Q1 = {q1:g}, median = {med:g}, Q3 = {q3:g}\nIQR = Q3 − Q1 = {iqr:g}\n\nfences: Q1 − 1.5·IQR = {lo_f:g}\n"
                         f"        Q3 + 1.5·IQR = {hi_f:g}\n\noutliers: {', '.join(f'{o:g}' for o in out) or 'none'}", 18)
    return step + 1


def build_histogram(cv, v, box, step):
    rnd = random.Random(21)
    data = v.get("data") or [round(rnd.gauss(70, 10), 1) for _ in range(80)]
    k = math.ceil(math.log2(len(data)) + 1)
    lo, hi = min(data), max(data)
    w = (hi - lo) / k
    counts = [0] * k
    for x in data:
        counts[min(k - 1, int((x - lo) / w))] += 1
    px, py = _plot(cv, box, step, (lo, hi), (0, max(counts) * 1.15), "value", "count")
    step += 1
    for i, c in enumerate(counts):
        a = lo + i * w
        cv.rect(px(a), py(c), px(a + w) - px(a) - 2, py(0) - py(c), step, "#a5d8ff", "#1971c2", 1.5, rounded=False)
        cv.text(px(a + w / 2), py(c) - 4, str(c), step, 13, MUTED, "center", "bottom", math=False)
    step += 1
    mean = sum(data) / len(data)
    cv.line([(px(mean), py(0)), (px(mean), py(max(counts) * 1.1))], step, "#e03131", 2, dashed=True)
    _note(cv, box, step, f"{len(data)} values, {k} bins (Sturges:\n⌈log₂ n⌉ + 1), width {w:.1f}\n\nmean = {mean:.1f} (red)\n\n"
                         "too few bins hide the shape,\ntoo many make it noisy", 18)
    return step + 1


def build_central_limit(cv, v, box, step):
    rnd = random.Random(4)
    n, reps = int(v.get("n") or 30), 400
    means = [sum(rnd.expovariate(1 / 10) for _ in range(n)) / n for _ in range(reps)]
    raw = [rnd.expovariate(1 / 10) for _ in range(reps)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    for k, (vals, name, lo, hi, col) in enumerate(((raw, "population: waiting times (skewed)", 0, 50, "#ffc9c9"),
                                                   (means, f"means of {reps} samples of n = {n}", 4, 16, "#b2f2bb"))):
        bins = 20
        cnt = [0] * bins
        for x in vals:
            if lo <= x < hi:
                cnt[int((x - lo) / (hi - lo) * bins)] += 1
        bx, bw = x0 + k * W * 0.35, W * 0.32
        top = y0 + 30
        cv.text(bx, y0, name, step, 16, INK, math=False)
        for i, c in enumerate(cnt):
            h = (H - 120) * c / max(cnt)
            cv.rect(bx + i * bw / bins, y1 - 60 - h, bw / bins - 1, h, step, col, "transparent", 1, rounded=False)
        cv.line([(bx, y1 - 60), (bx + bw, y1 - 60)], step, INK, 1.5)
        cv.text(bx, y1 - 54, str(lo), step, 13, MUTED, math=False)
        cv.text(bx + bw, y1 - 54, str(hi), step, 13, MUTED, "right", "top", math=False)
        step += 1
    m = sum(means) / reps
    s = math.sqrt(sum((x - m) ** 2 for x in means) / reps)
    _note(cv, box, step, f"the means pile up in a bell\ncurve even though the data\nis skewed\n\nmean of means = {m:.1f}\n"
                         f"spread = {s:.2f} ≈ σ/√n = {10 / math.sqrt(n):.2f}", 18, frac=0.72)
    return step + 1


def _phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def build_hypothesis_test(cv, v, box, step):
    mu0, xbar, sd, n, alpha = (float(v.get(k) or d) for k, d in (("mu0", 50), ("mean", 52.4), ("sd", 8), ("n", 36),
                                                                  ("alpha", 0.05)))
    se = sd / math.sqrt(n)
    z = (xbar - mu0) / se
    p = 2 * (1 - _phi(abs(z)))
    zc = 1.96 if alpha == 0.05 else -_inv(alpha / 2)
    pdf = lambda t: math.exp(-t * t / 2) / math.sqrt(2 * math.pi)
    px, py = _plot(cv, box, step, (-4, 4), (0, 0.45), "z", "")
    _pl(cv, [(px(t / 10), py(pdf(t / 10))) for t in range(-40, 41)], step, INK, 2.5)
    step += 1
    for side in (-1, 1):
        a, b = (zc, 4) if side > 0 else (-4, -zc)
        pts = [(px(a), py(0))] + [(px(a + (b - a) * i / 20), py(pdf(a + (b - a) * i / 20))) for i in range(21)] + [(px(b), py(0))]
        cv.polygon(pts, step, "#ffc9c9", "#e03131", 1)
    cv.text(px(zc), py(0.12), f"reject if |z| > {zc:.2f}", step, 14, "#e03131", "left", "bottom", math=False)
    step += 1
    cv.line([(px(z), py(0)), (px(z), py(0.4))], step, "#1971c2", 3)
    cv.text(px(z) + 6, py(0.4), f"z = {z:.2f}", step, 17, "#1971c2", "left", "top", math=False)
    step += 1
    verdict = "reject H₀" if p < alpha else "fail to reject H₀"
    _note(cv, box, step, f"H₀: μ = {mu0:g}   H₁: μ ≠ {mu0:g}\nsample mean {xbar:g}, n = {n:g}, σ = {sd:g}\n\n"
                         f"SE = σ/√n = {se:.2f}\nz = ({xbar:g} − {mu0:g}) / {se:.2f} = {z:.2f}\np-value = {p:.3f}\n\n"
                         f"α = {alpha:g} → {verdict}", 18)
    return step + 1


def _inv(p):
    lo, hi = -8.0, 8.0
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if _phi(m) < p else (lo, m)
    return lo


# ======================================================================== maths

def build_taylor_series(cv, v, box, step):
    px, py = _plot(cv, box, step, (-2 * math.pi, 2 * math.pi), (-2.2, 2.2), "x", "")
    _pl(cv, [(px(x), py(math.sin(x))) for x in [i * math.pi / 30 for i in range(-60, 61)]], step, INK, 4)
    step += 1

    def T(x, k):
        return sum((-1) ** j * x ** (2 * j + 1) / math.factorial(2 * j + 1) for j in range(k))
    for k in range(1, 6):
        pts = []
        for i in range(-60, 61):
            x = i * math.pi / 30
            y = T(x, k)
            if abs(y) <= 2.2:
                pts.append((px(x), py(y)))
        tr = _Track(cv)
        _pl(cv, pts, step, STRONG[k % len(STRONG)], 2.5)
        terms = " + ".join(["x", "−x³/3!", "+x⁵/5!", "−x⁷/7!", "+x⁹/9!"][:k]).replace("+ −", "− ").replace("+ +", "+ ")
        cv.text(box[0] + (box[2] - box[0]) * 0.68, box[1] + 30, f"sin x ≈ {terms}\n\nerror at x = π: {abs(T(math.pi, k)):.4f}",
                step, 19, STRONG[k % len(STRONG)], math=False)
        tr.tag(until=step + 1 if k < 5 else None, fade=k > 1)
        step += 1
    _note(cv, box, step, "black = sin x\neach extra term hugs the curve\nfurther from 0", 18, dy=160)
    return step + 1


def build_fourier_series(cv, v, box, step):
    px, py = _plot(cv, box, step, (0, 2 * math.pi), (-1.5, 1.5), "t", "")
    sq = lambda t: 1 if (t % (2 * math.pi)) < math.pi else -1
    _pl(cv, [(px(t), py(sq(t))) for t in (0, math.pi - 1e-6, math.pi, 2 * math.pi - 1e-6)], step, MUTED, 2, dashed=True)
    step += 1
    for k, terms in enumerate((1, 2, 3, 10)):
        f = lambda t: 4 / math.pi * sum(math.sin((2 * j + 1) * t) / (2 * j + 1) for j in range(terms))
        tr = _Track(cv)
        _pl(cv, [(px(i * 2 * math.pi / 200), py(f(i * 2 * math.pi / 200))) for i in range(201)], step, "#e03131", 2.5)
        cv.text(box[0] + (box[2] - box[0]) * 0.68, box[1] + 30, f"{terms} sine wave{'s' if terms > 1 else ''}\n"
                f"(4/π)(sin t + sin 3t/3 + …)", step, 20, "#e03131", math=False)
        tr.tag(until=step + 1 if k < 3 else None, fade=k > 0)
        step += 1
    _note(cv, box, step, "any periodic signal = a sum of\nsines; more terms → sharper\ncorners (the overshoot at the\n"
                         "jump is the Gibbs effect)\n\nbasis of audio compression,\nradio and the FFT", 18, dy=130)
    return step + 1


def build_newtons_method(cv, v, box, step):
    f = lambda x: x * x - 2
    df = lambda x: 2 * x
    x, its = float(v.get("start") or 3), []
    for _ in range(4):
        nx = x - f(x) / df(x)
        its.append((x, nx))
        x = nx
    px, py = _plot(cv, box, step, (0, 3.3), (-3, 9), "x", "f(x) = x² − 2")
    _pl(cv, [(px(t / 20), py(f(t / 20))) for t in range(0, 67)], step, INK, 3)
    cv.line([(px(0), py(0)), (px(3.3), py(0))], step, MUTED, 1)
    step += 1
    for i, (a, b) in enumerate(its):
        cv.line([(px(a), py(0)), (px(a), py(f(a)))], step, MUTED, 1, dashed=True)
        _pl(cv, [(px(a), py(f(a))), (px(b), py(0))], step, STRONG[i], 2.5)
        cv.ellipse(px(b) - 5, py(0) - 5, 10, 10, step, STRONG[i], STRONG[i], 1)
        cv.text(box[0] + (box[2] - box[0]) * 0.68, box[1] + 40 + i * 30, f"x{i + 1} = {b:.10f}", step, 18, STRONG[i],
                math=False)
        step += 1
    _note(cv, box, step, "xₙ₊₁ = xₙ − f(xₙ)/f′(xₙ)\nfollow the tangent to zero\n\n√2 = 1.4142135624\n"
                         "correct digits roughly double\neach step", 18, dy=180)
    return step + 1


def build_eigen_transform(cv, v, box, step):
    a, b, c, d = (float(x) for x in (v.get("matrix") or [2, 1, 1, 2]))
    tr_, det = a + d, a * d - b * c
    disc = math.sqrt(max(0, tr_ * tr_ / 4 - det))
    l1, l2 = tr_ / 2 + disc, tr_ / 2 - disc
    vec = lambda l: (b, l - a) if abs(b) > 1e-9 else ((1, 0) if abs(l - a) < 1e-9 else (0, 1))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    s = min(W * 0.6, H) / 8.5
    cx, cy = x0 + W * 0.3, (y0 + y1) / 2
    P = lambda x, y: (cx + x * s, cy - y * s)
    for k in range(-3, 4):
        cv.line([P(k, -3), P(k, 3)], step, "#e9ecef", 1)
        cv.line([P(-3, k), P(3, k)], step, "#e9ecef", 1)
    step += 1
    M = lambda x, y: (a * x + b * y, c * x + d * y)
    for k in range(-1, 2):
        _pl(cv, [P(*M(k, -1)), P(*M(k, 1))], step, "#74c0fc", 2)
        _pl(cv, [P(*M(-1, k)), P(*M(1, k))], step, "#74c0fc", 2)
    step += 1
    for lam, col in ((l1, "#e03131"), (l2, "#2f9e44")):
        vx, vy = vec(lam)
        L = math.hypot(vx, vy)
        vx, vy = vx / L, vy / L
        cv.line([P(0, 0), P(vx, vy)], step, col, 3, arrow_end=True)
        cv.line([P(0, 0), P(lam * vx, lam * vy)], step, col, 1.5, dashed=True, arrow_end=True)
        cv.text(*P(lam * vx + 0.2, lam * vy + 0.2), f"λ = {lam:g}", step, 16, col, math=False)
        step += 1
    other = M(1, 0)
    cv.line([P(0, 0), P(1, 0)], step, "#7048e8", 2.5, arrow_end=True)
    cv.line([P(0, 0), P(*other)], step, "#7048e8", 1.5, dashed=True, arrow_end=True)
    step += 1
    _note(cv, box, step, f"A = [[{a:g}, {b:g}], [{c:g}, {d:g}]]\nblue grid = A applied to the\nsquare grid\n\n"
                         f"eigenvectors only stretch:\nAv = λv, λ = {l1:g} and {l2:g}\n\npurple (1, 0) also turns\n"
                         f"→ ({other[0]:g}, {other[1]:g}): not an eigenvector", 18)
    return step + 1


def build_number_bases(cv, v, box, step):
    n = int(v.get("number") or 156)
    x0, y0, x1, y1 = box
    W = x1 - x0
    _title(cv, x0, y0, f"{n} in binary: divide by 2, keep the remainders", step, 20)
    step += 1
    q, rows = n, []
    while q:
        rows.append((q, q // 2, q % 2))
        q //= 2
    for i, (a, b, r) in enumerate(rows):
        cv.text(x0, y0 + 40 + i * 30, f"{a} ÷ 2 = {b}  remainder {r}", step, 19, INK, math=False)
        cv.text(x0 + 290, y0 + 40 + i * 30, str(r), step, 20, "#e03131", math=False)
        step += 1
    bits = bin(n)[2:]
    cv.line([(x0 + 310, y0 + 40 + len(rows) * 30), (x0 + 310, y0 + 40)], step, "#e03131", 2, arrow_end=True)
    cv.text(x0 + 320, y0 + 50, "read\nupward", step, 15, "#e03131", math=False)
    step += 1
    _note(cv, box, step, f"binary  {bits}\n\ncheck: " + " + ".join(str(2 ** (len(bits) - 1 - i)) for i, ch in enumerate(bits)
                                                         if ch == "1") +
          f" = {n}\n\nhex (groups of 4 bits):\n{' '.join(bits.zfill((len(bits) + 3) // 4 * 4)[i:i + 4] for i in range(0, (len(bits) + 3) // 4 * 4, 4))}"
          f" = 0x{n:X}\n\noctal: {n:o}", 20, frac=0.5, dy=40)
    return step + 1


def build_modular_clock(cv, v, box, step):
    m, a, b = int(v.get("modulus") or 12), int(v.get("a") or 9), int(v.get("b") or 5)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    R = min(W * 0.25, H * 0.4)
    cx, cy = x0 + R + 40, (y0 + y1) / 2
    cv.ellipse(cx - R, cy - R, 2 * R, 2 * R, step, "#f8f9fa", INK, 3)
    P = lambda k, r=R * 0.82: (cx + r * math.sin(2 * math.pi * k / m), cy - r * math.cos(2 * math.pi * k / m))
    for k in range(m):
        cv.text(*P(k), str(k), step, 20, INK, "center", "middle", math=False)
    step += 1
    cv.line([(cx, cy), P(a, R * 0.6)], step, "#1971c2", 4, arrow_end=True)
    step += 1
    for j in range(1, b + 1):
        k = (a + j) % m
        tr = _Track(cv)
        cv.line([(cx, cy), P(k, R * 0.6)], step, "#e03131", 4, arrow_end=True)
        tr.tag(until=step + 1 if j < b else None, fade=True)
        step += 1
    _note(cv, box, step, f"{a} + {b} = {a + b}\n{a + b} mod {m} = {(a + b) % m}\n\nmodular arithmetic wraps\naround, "
                         f"like a clock\n\n{a} × {b} mod {m} = {(a * b) % m}\n2^10 mod 7 = {pow(2, 10, 7)}\n\n"
                         "used in hashing, checksums\nand cryptography (RSA)", 20, frac=0.62)
    return step + 1


# ======================================================================== physics

def build_spring_shm(cv, v, box, step):
    k, m, A = float(v.get("k") or 20), float(v.get("mass") or 0.5), float(v.get("amplitude") or 0.1)
    w = math.sqrt(k / m)
    T = 2 * math.pi / w
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    wall = x0 + 20
    cy = y0 + 70
    cv.line([(wall, cy - 50), (wall, cy + 50)], step, INK, 4)
    step += 1
    gx0, gy = x0 + 40, y0 + H * 0.68
    cv.line([(gx0, gy), (gx0 + W * 0.6, gy)], step, MUTED, 1.5)
    cv.text(gx0 + W * 0.6, gy + 6, "t", step, 15, MUTED, "right", "top", math=False)
    cv.text(gx0 - 6, gy - 70, "x", step, 15, MUTED, "right", "middle", math=False)
    for f in range(9):
        t = f * T / 8
        x = A * math.cos(w * t)
        mx = x0 + W * 0.3 + x / A * W * 0.12
        tr = _Track(cv)
        coils = [(wall, cy)] + [(wall + (mx - 30 - wall) * i / 14, cy + (14 if i % 2 else -14)) for i in range(1, 14)] + \
                [(mx - 30, cy)]
        _pl(cv, coils, step, "#495057", 2)
        cv.rect(mx - 30, cy - 30, 60, 60, step, "#ffd8a8", INK, 2)
        tr.tag(until=step + 1 if f < 8 else None, fade=f > 0)
        if f:
            _pl(cv, [(gx0 + (f - 1 + i / 10) * W * 0.6 / 8, gy - 70 * math.cos(w * (f - 1 + i / 10) * T / 8))
                     for i in range(11)], step, "#1971c2", 2.5)
        step += 1
    _note(cv, box, step, f"Hooke: F = −kx\nk = {k:g} N/m, m = {m:g} kg\n\nω = √(k/m) = {w:.2f} rad/s\n"
                         f"period T = 2π√(m/k) = {T:.2f} s\nmax speed = Aω = {A * w:.2f} m/s\n\n"
                         "amplitude doesn't change\nthe period", 18)
    return step + 1


def build_lever_torque(cv, v, box, step):
    m1, d1, m2 = float(v.get("m1") or 30), float(v.get("d1") or 2), float(v.get("m2") or 20)
    d2 = m1 * d1 / m2
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cx, cy = x0 + W * 0.33, y0 + H * 0.5
    s = W * 0.28 / max(d1, d2)
    cv.polygon([(cx, cy + 4), (cx - 30, cy + 60), (cx + 30, cy + 60)], step, "#ced4da", INK, 2)
    cv.rect(cx - s * max(d1, d2) - 30, cy - 6, 2 * s * max(d1, d2) + 60, 10, step, "#e9ecef", INK, 2, rounded=False)
    step += 1
    for m, d, side, col in ((m1, d1, -1, "#a5d8ff"), (m2, d2, 1, "#ffc9c9")):
        bx = cx + side * d * s
        sz = 30 + m
        cv.rect(bx - sz / 2, cy - 6 - sz, sz, sz, step, col, INK, 2)
        cv.text(bx, cy - 6 - sz / 2, f"{m:g} kg", step, 15, INK, "center", "middle", math=False)
        cv.line([(cx, cy + 20), (bx, cy + 20)], step, MUTED, 1, arrow_end=True)
        cv.text((cx + bx) / 2, cy + 24, f"{d:.2f} m", step, 15, MUTED, "center", "top", math=False)
        step += 1
    _note(cv, box, step, f"balanced when torques match:\nm₁ g d₁ = m₂ g d₂\n{m1:g} × {d1:g} = {m2:g} × d₂\n"
                         f"d₂ = {d2:.2f} m\n\ntorque = force × distance\nfrom the pivot: a longer arm\nneeds less force", 19)
    return step + 1


def build_buoyancy(cv, v, box, step):
    items = v.get("objects") or [("wood", 600), ("ice", 917), ("steel", 7850)]
    rho_w = 1000
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    wy = y0 + H * 0.35
    cv.rect(x0, wy, W * 0.66, H * 0.6, step, "#d0ebff", "#1971c2", 2, rounded=False)
    cv.text(x0 + W * 0.64, y1 - 30, "water 1000 kg/m³", step, 15, "#1971c2", "right", "top", math=False)
    step += 1
    for i, (name, rho) in enumerate(items):
        frac = min(1, rho / rho_w)
        s = 90
        bx = x0 + 40 + i * W * 0.22
        top = wy - s * (1 - frac) if rho < rho_w else y1 - s - 10
        cv.rect(bx, top, s, s, step, PAL[i], INK, 2)
        cv.text(bx + s / 2, top + s / 2, f"{name}\n{rho} kg/m³", step, 14, INK, "center", "middle", math=False)
        cv.text(bx + s / 2, top - 8, f"{frac:.0%} under" if rho < rho_w else "sinks", step, 15,
                "#2f9e44" if rho < rho_w else "#e03131", "center", "bottom", math=False)
        step += 1
    _note(cv, box, step, "Archimedes: the upward push =\nweight of the water pushed\naside\n\n"
                         "floats if less dense than water;\nfraction under = ρ_object / ρ_water\n"
                         "(that's why ~90% of an iceberg\nis hidden)", 18, frac=0.7)
    return step + 1


def build_electric_field(cv, v, box, step):
    charges = v.get("charges") or [(-1.2, 0, 1), (1.2, 0, -1)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    s = min(W * 0.62, H) / 6
    cx, cy = x0 + W * 0.32, (y0 + y1) / 2
    P = lambda x, y: (cx + x * s, cy - y * s)
    for qx, qy, q in charges:
        X, Y = P(qx, qy)
        cv.ellipse(X - 20, Y - 20, 40, 40, step, "#ffc9c9" if q > 0 else "#d0ebff", "#e03131" if q > 0 else "#1971c2", 2)
        cv.text(X, Y, "+" if q > 0 else "−", step, 26, INK, "center", "middle", math=False)
    step += 1
    pts = [(gx / 2, gy / 2) for gx in range(-5, 6) for gy in range(-5, 6)]
    for row in range(3):
        for gx, gy in pts:
            if int((gy * 2 + 5)) % 3 != row:
                continue
            ex = ey = 0
            near = False
            for qx, qy, q in charges:
                dx, dy = gx - qx, gy - qy
                r2 = dx * dx + dy * dy
                if r2 < 0.2:
                    near = True
                ex += q * dx / r2 ** 1.5
                ey += q * dy / r2 ** 1.5
            if near:
                continue
            mag = math.hypot(ex, ey)
            L = 0.32 * min(1, 0.4 + mag)
            a, b = P(gx, gy), P(gx + ex / mag * L, gy + ey / mag * L)
            cv.line([a, b], step, "#7048e8" if mag > 0.5 else "#b197fc", 1.5, arrow_end=True)
        step += 1
    _note(cv, box, step, "arrows show the force on a small\npositive test charge (computed\nfrom Coulomb's law)\n\n"
                         "E = kq / r², summed over\ncharges; lines leave + and\nend on −", 18)
    return step + 1


def build_em_spectrum(cv, v, box, step):
    bands = [("radio", 1, "#ffc9c9", "FM radio, Wi-Fi"), ("microwave", 1e-2, "#ffd8a8", "ovens, radar"),
             ("infrared", 1e-5, "#ffec99", "heat, remotes"), ("visible", 5e-7, "#b2f2bb", "what we see"),
             ("ultraviolet", 1e-8, "#a5d8ff", "sunburn"), ("X-ray", 1e-10, "#d0bfff", "medical images"),
             ("gamma", 1e-12, "#fcc2d7", "radioactive decay")]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    bw = W * 0.95 / len(bands)
    for i, (name, lam, col, use) in enumerate(bands):
        x = x0 + i * bw
        cv.rect(x, y0 + 60, bw - 4, 80, step, col, INK, 1.5, rounded=False)
        cv.text(x + bw / 2, y0 + 100, name, step, 16, INK, "center", "middle", math=False)
        cyc = 2 + i * 2
        _pl(cv, [(x + 6 + (bw - 16) * k / 60, y0 + 30 + 12 * math.sin(2 * math.pi * cyc * k / 60)) for k in range(61)],
            step, STRONG[i % len(STRONG)], 1.5)
        cv.text(x + bw / 2, y0 + 150, f"λ ≈ {lam:.0e} m\nf ≈ {3e8 / lam:.0e} Hz", step, 13, MUTED, "center", "top", math=False)
        cv.text(x + bw / 2, y0 + 200, use, step, 14, INK, "center", "top", math=False)
        step += 1
    cv.text(x0, y0 + 260, "c = f × λ = 3 × 10⁸ m/s: shorter wavelength → higher frequency → more energy (E = hf)",
            step, 18, "#e8590c", math=False)
    return step + 1


# ======================================================================== chemistry

def titration_ph(vb, ca=0.1, va=25.0, cb=0.1):
    na, nb = ca * va / 1000, cb * vb / 1000
    vt = (va + vb) / 1000
    if abs(na - nb) < 1e-12:
        return 7.0
    if na > nb:
        return -math.log10((na - nb) / vt)
    return 14 + math.log10((nb - na) / vt)


def build_titration(cv, v, box, step):
    px, py = _plot(cv, box, step, (0, 50), (0, 14), "NaOH added (mL)", "pH")
    step += 1
    for k in range(5):
        pts = [(px(vb / 4), py(titration_ph(vb / 4))) for vb in range(k * 40, (k + 1) * 40 + 1)]
        _pl(cv, pts, step, "#7048e8", 3)
        step += 1
    cv.ellipse(px(25) - 7, py(7) - 7, 14, 14, step, "#e03131", "#e03131", 1)
    cv.text(px(25) + 10, py(7), "equivalence point\n25.0 mL, pH 7", step, 15, "#e03131", "left", "middle", math=False)
    step += 1
    _note(cv, box, step, "25 mL of 0.1 M HCl titrated\nwith 0.1 M NaOH (computed)\n\n"
                         f"at 10 mL: pH {titration_ph(10):.2f}\nat 24 mL: pH {titration_ph(24):.2f}\n"
                         f"at 26 mL: pH {titration_ph(26):.2f}\n\nthe steep jump is where an\nindicator changes colour", 18)
    return step + 1


def build_limiting_reagent(cv, v, box, step):
    h2, o2 = int(v.get("h2") or 5), int(v.get("o2") or 2)
    rxn = min(h2 // 2, o2)
    water, h_left, o_left = 2 * rxn, h2 - 2 * rxn, o2 - rxn
    lim = "O₂" if o2 * 2 < h2 else "H₂"
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    _title(cv, x0, y0, "2 H₂ + O₂ → 2 H₂O", step, 24)
    step += 1

    def mol(x, y, kind, st):
        if kind == "H2":
            for dx in (0, 22):
                cv.ellipse(x + dx, y, 22, 22, st, "#ffffff", INK, 1.5)
        elif kind == "O2":
            for dx in (0, 30):
                cv.ellipse(x + dx, y - 4, 30, 30, st, "#ff8787", INK, 1.5)
        else:
            cv.ellipse(x + 8, y - 4, 30, 30, st, "#ff8787", INK, 1.5)
            cv.ellipse(x - 6, y + 14, 18, 18, st, "#ffffff", INK, 1.5)
            cv.ellipse(x + 34, y + 14, 18, 18, st, "#ffffff", INK, 1.5)
    cv.text(x0, y0 + 50, f"before: {h2} H₂ and {o2} O₂", step, 18, INK, math=False)
    for i in range(h2):
        mol(x0 + (i % 6) * 60, y0 + 90 + (i // 6) * 40, "H2", step)
    for i in range(o2):
        mol(x0 + (i % 6) * 70, y0 + 180 + (i // 6) * 40, "O2", step)
    step += 1
    cv.arrow(x0 + W * 0.38, y0 + 150, x0 + W * 0.46, y0 + 150, step, color=INK)
    bx = x0 + W * 0.48
    cv.text(bx, y0 + 50, f"after: {water} H₂O, {h_left} H₂ left, {o_left} O₂ left", step, 18, INK, math=False)
    for i in range(water):
        mol(bx + (i % 5) * 64, y0 + 90 + (i // 5) * 50, "H2O", step)
    for i in range(h_left):
        mol(bx + i * 60, y0 + 200, "H2", step)
    for i in range(o_left):
        mol(bx + i * 70, y0 + 240, "O2", step)
    step += 1
    cv.text(x0, y0 + 300, f"need 2 H₂ per O₂: {o2} O₂ needs {2 * o2} H₂ (have {h2})\n→ {lim} runs out first: it is the "
                          f"limiting reagent\n→ it decides the yield: {water} H₂O", step, 19, "#e03131", math=False)
    return step + 1


def build_vsepr(cv, v, box, step):
    shapes = [("CO₂", "linear", 180, 2, 0), ("BF₃", "trigonal planar", 120, 3, 0), ("CH₄", "tetrahedral", 109.5, 4, 0),
              ("NH₃", "trigonal pyramidal", 107, 3, 1), ("H₂O", "bent", 104.5, 2, 2)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cw = W / len(shapes)
    for i, (mol, name, ang, bonds, lone) in enumerate(shapes):
        cx, cy = x0 + i * cw + cw / 2, y0 + H * 0.35
        total = bonds + lone
        if total == 2:
            angs = [180, 0][:bonds] if bonds == 2 else [0]
        if total == 3:
            angs = [90, 210, 330]
        if total == 4:
            angs = [90, 210, 330, 270]
        if mol == "H₂O":
            angs = [232, 308, 60, 120]
        if mol == "NH₃":
            angs = [210, 330, 260, 90]
        for j, a in enumerate(angs[:total]):
            r = 62
            ex, ey = cx + r * math.cos(math.radians(a)), cy - r * math.sin(math.radians(a))
            if j < bonds:
                cv.line([(cx, cy), (ex, ey)], step, INK, 3 if not (total == 4 and j == 3) else 2,
                        dashed=(total == 4 and j == 3 and lone == 0))
                cv.ellipse(ex - 15, ey - 15, 30, 30, step, "#ffffff" if mol[0] in "CBN" and mol != "CO₂" else "#ff8787",
                           INK, 1.5)
            else:
                lx, ly = cx + 40 * math.cos(math.radians(a)), cy - 40 * math.sin(math.radians(a))
                cv.ellipse(lx - 14, ly - 9, 28, 18, step, "#fff3bf", "#f08c00", 1.5)
                cv.text(lx, ly, "••", step, 12, "#f08c00", "center", "middle", math=False)
        cv.ellipse(cx - 20, cy - 20, 40, 40, step, "#868e96", INK, 2)
        cv.text(cx, cy + 100, f"{mol}\n{name}\n{ang:g}°\n{bonds} bonds, {lone} lone pair{'s' if lone != 1 else ''}", step, 15,
                INK, "center", "top", math=False)
        step += 1
    cv.text(x0, y1 - 26, "electron pairs push apart as far as they can; lone pairs push harder, squeezing bond angles",
            step, 17, "#e8590c", math=False)
    return step + 1


# ======================================================================== biology

def build_enzyme(cv, v, box, step):
    vmax, km = float(v.get("vmax") or 10), float(v.get("km") or 2)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    ex, ey = x0 + 100, y0 + 120
    cv.polygon([(ex - 70, ey - 40), (ex - 20, ey - 40), (ex - 20, ey - 10), (ex + 20, ey - 10), (ex + 20, ey - 40),
                (ex + 70, ey - 40), (ex + 70, ey + 50), (ex - 70, ey + 50)], step, "#b2f2bb", "#2b8a3e", 2)
    cv.text(ex, ey + 20, "enzyme", step, 16, INK, "center", "middle", math=False)
    step += 1
    for k, yy in enumerate((ey - 120, ey - 75, ey - 30)):
        tr = _Track(cv)
        cv.rect(ex - 18, yy, 36, 30, step, "#ffc9c9", "#e03131", 2, rounded=False)
        cv.text(ex + 30, yy + 15, "substrate" if k == 0 else "", step, 15, "#e03131", "left", "middle", math=False)
        tr.tag(until=step + 1 if k < 2 else None, fade=k > 0)
        step += 1
    cv.text(ex - 70, ey + 70, "active site fits the substrate\n(lock and key); reaction happens\n"
                              "and products leave; enzyme reused", step, 15, INK, math=False)
    step += 1
    sub = (x0 + W * 0.38, x1 - W * 0.02)
    px, py = _plot(cv, (x0 + W * 0.36, y0, x0 + W * 0.95, y1), step, (0, 20), (0, vmax * 1.15), "[substrate]",
                   "reaction rate", frac=0.6)
    _pl(cv, [(px(s / 3), py(vmax * (s / 3) / (km + s / 3))) for s in range(61)], step, "#7048e8", 3)
    cv.line([(px(0), py(vmax)), (px(20), py(vmax))], step, MUTED, 1, dashed=True)
    cv.line([(px(km), py(0)), (px(km), py(vmax / 2)), (px(0), py(vmax / 2))], step, "#e03131", 1, dashed=True)
    cv.text(px(20), py(vmax) - 4, "Vmax (all enzymes busy)", step, 14, MUTED, "right", "bottom", math=False)
    cv.text(px(km) + 6, py(vmax / 2), f"Km = {km:g}: half speed", step, 14, "#e03131", "left", "bottom", math=False)
    step += 1
    cv.text(px(5), py(vmax * 0.3), "v = Vmax[S] / (Km + [S])", step, 16, "#7048e8", math=False)
    return step + 1


def build_action_potential(cv, v, box, step):
    seg = [(0, -70), (1, -70), (1.5, -55), (2.0, 30), (2.6, -80), (3.6, -70), (5, -70)]
    phases = [(1.25, -55, "threshold −55 mV", "#f08c00"), (0.3, 10, "depolarisation:\nNa⁺ rushes in", "#e03131"),
              (2.55, 10, "repolarisation:\nK⁺ flows out", "#1971c2"), (3.0, -80, "hyperpolarised\n(refractory)", "#7048e8"),
              (4.3, -70, "resting −70 mV", "#2f9e44")]
    px, py = _plot(cv, box, step, (0, 5), (-90, 40), "time (ms)", "membrane voltage (mV)")
    cv.line([(px(0), py(-55)), (px(5), py(-55))], step, "#ffd8a8", 1.5, dashed=True)
    step += 1
    for i in range(len(seg) - 1):
        a, b = seg[i], seg[i + 1]
        pts = [(px(a[0] + (b[0] - a[0]) * t / 10), py(a[1] + (b[1] - a[1]) * (0.5 - 0.5 * math.cos(math.pi * t / 10))))
               for t in range(11)]
        _pl(cv, pts, step, "#c92a2a", 3.5)
        step += 1
    for x, y, lab, col in phases:
        cv.text(px(x) + 8, py(y) - 4, lab, step, 14, col, "left", "bottom", math=False)
    step += 1
    _note(cv, box, step, "all-or-nothing: below threshold\nnothing fires; above it the\nspike is always the same size\n\n"
                         "the Na⁺/K⁺ pump restores the\nresting gradient", 17, frac=0.72)
    return step + 1


def build_hardy_weinberg(cv, v, box, step):
    p = float(v.get("p") or 0.7)
    q = 1 - p
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    S = min(W * 0.5, H - 60)
    sx, sy = x0 + 40, y0 + 40
    cv.text(sx + S * p / 2, sy - 6, f"A  p = {p:g}", step, 16, "#1971c2", "center", "bottom", math=False)
    cv.text(sx + S * p + S * q / 2, sy - 6, f"a  q = {q:g}", step, 16, "#e03131", "center", "bottom", math=False)
    step += 1
    cells = [(0, 0, p, p, "#a5d8ff", f"AA\np² = {p * p:.2f}"), (p, 0, q, p, "#d0bfff", f"Aa\npq = {p * q:.2f}"),
             (0, p, p, q, "#d0bfff", f"Aa\npq = {p * q:.2f}"), (p, p, q, q, "#ffc9c9", f"aa\nq² = {q * q:.2f}")]
    for cx, cy, w, h, col, lab in cells:
        cv.rect(sx + cx * S, sy + cy * S, w * S, h * S, step, col, INK, 2, rounded=False)
        cv.text(sx + (cx + w / 2) * S, sy + (cy + h / 2) * S, lab, step, 17, INK, "center", "middle", math=False)
        step += 1
    _note(cv, box, step, f"p² + 2pq + q² = 1\n{p * p:.2f} + {2 * p * q:.2f} + {q * q:.2f} = 1\n\n"
                         f"in 1000 people:\n{p * p * 1000:.0f} AA, {2 * p * q * 1000:.0f} Aa (carriers),\n{q * q * 1000:.0f} aa\n\n"
                         "holds with no selection,\nmutation, migration or drift\nand random mating", 18, frac=0.62)
    return step + 1


def build_population_growth(cv, v, box, step):
    r, K, N0 = float(v.get("r") or 0.5), float(v.get("K") or 1000), float(v.get("N0") or 20)
    T = 20
    exp_ = lambda t: N0 * math.exp(r * t)
    log_ = lambda t: K / (1 + (K - N0) / N0 * math.exp(-r * t))
    px, py = _plot(cv, box, step, (0, T), (0, K * 1.4), "time", "population")
    cv.line([(px(0), py(K)), (px(T), py(K))], step, MUTED, 1.5, dashed=True)
    cv.text(px(T), py(K) - 4, f"carrying capacity K = {K:g}", step, 14, MUTED, "right", "bottom", math=False)
    step += 1
    pts = []
    for i in range(101):
        t = T * i / 100
        if exp_(t) > K * 1.4:
            break
        pts.append((px(t), py(exp_(t))))
    _pl(cv, pts, step, "#e03131", 3)
    cv.text(pts[-1][0] + 6, pts[-1][1], "exponential (J)", step, 15, "#e03131", "left", "top", math=False)
    step += 1
    _pl(cv, [(px(T * i / 100), py(log_(T * i / 100))) for i in range(101)], step, "#2f9e44", 3)
    cv.text(px(T * 0.7), py(log_(T * 0.7)) + 8, "logistic (S)", step, 15, "#2f9e44", "left", "top", math=False)
    step += 1
    tin = math.log((K - N0) / N0) / r
    _note(cv, box, step, f"dN/dt = rN (1 − N/K)\nr = {r:g}, N₀ = {N0:g}\n\nfastest growth at N = K/2\n(t ≈ {tin:.1f})\n\n"
                         "limits: food, space, disease,\npredators", 18)
    return step + 1


# ======================================================================== earth / GIS

def build_idw_interpolation(cv, v, box, step):
    pts = v.get("samples") or [(0.2, 0.2, 12), (0.8, 0.25, 30), (0.5, 0.6, 20), (0.15, 0.85, 8), (0.85, 0.85, 26)]
    q = (0.55, 0.35)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    S = min(W * 0.55, H - 20)

    def idw(x, y):
        num = den = 0
        for sx, sy, val in pts:
            d = math.hypot(x - sx, y - sy)
            if d < 1e-9:
                return val
            num += val / d ** 2
            den += 1 / d ** 2
        return num / den
    lo, hi = min(p[2] for p in pts), max(p[2] for p in pts)
    ramp = ["#e7f5ff", "#a5d8ff", "#4dabf7", "#1c7ed6", "#1864ab"]
    g = 22
    for i in range(g):
        for j in range(g):
            val = idw((j + 0.5) / g, (i + 0.5) / g)
            k = min(4, int((val - lo) / (hi - lo + 1e-9) * 5))
            cv.rect(x0 + j * S / g, y0 + i * S / g, S / g + 0.5, S / g + 0.5, step, ramp[k], "transparent", 1, rounded=False)
        if i % 6 == 5:
            step += 1
    step += 1
    for sx, sy, val in pts:
        cv.ellipse(x0 + sx * S - 7, y0 + sy * S - 7, 14, 14, step, "#ffffff", INK, 2)
        cv.text(x0 + sx * S + 10, y0 + sy * S, str(val), step, 16, INK, "left", "middle", math=False)
    step += 1
    for sx, sy, val in pts:
        cv.line([(x0 + q[0] * S, y0 + q[1] * S), (x0 + sx * S, y0 + sy * S)], step, "#e03131", 1, dashed=True)
    cv.ellipse(x0 + q[0] * S - 8, y0 + q[1] * S - 8, 16, 16, step, "#e03131", "#e03131", 1)
    step += 1
    _note(cv, box, step, f"inverse distance weighting:\nvalue = Σ(vᵢ/dᵢ²) / Σ(1/dᵢ²)\n\nred point estimate:\n{idw(*q):.1f}\n\n"
                         "nearby samples count most;\nused for rainfall, pollution,\nsoil readings", 18, frac=0.64)
    return step + 1


def build_topographic(cv, v, box, step):
    hills = [(0.3, 0.4, 900, 0.18), (0.7, 0.65, 600, 0.15)]
    elev = lambda x, y: 100 + sum(h * math.exp(-((x - a) ** 2 + (y - b) ** 2) / (2 * s * s)) for a, b, h, s in hills)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    S = min(W * 0.55, H - 20)
    ramp = ["#d3f9d8", "#b2f2bb", "#ffec99", "#ffd8a8", "#e9a76b", "#c08457", "#f1f3f5"]
    g = 30
    interval = 150
    for i in range(g):
        for j in range(g):
            e = elev((j + 0.5) / g, (i + 0.5) / g)
            k = min(len(ramp) - 1, int(e // interval))
            cv.rect(x0 + j * S / g, y0 + i * S / g, S / g + 0.5, S / g + 0.5, step, ramp[k], "transparent", 1, rounded=False)
        if i % 10 == 9:
            step += 1
    for a, b, h, s in hills:
        cv.text(x0 + a * S, y0 + b * S, f"▲ {elev(a, b):.0f} m", step, 16, INK, "center", "middle", math=False)
    step += 1
    _note(cv, box, step, f"each colour band = {interval} m of\nelevation; band edges are the\ncontour lines\n\n"
                         "close lines = steep slope\nwide spacing = gentle slope\n\nthe left hill is steeper\n"
                         "and higher", 18, frac=0.62)
    return step + 1


def build_raster_vector(cv, v, box, step):
    poly = [(0.15, 0.2), (0.55, 0.1), (0.75, 0.45), (0.4, 0.75), (0.1, 0.55)]
    road = [(0.0, 0.9), (0.5, 0.85), (1.0, 0.6)]
    wells = [(0.85, 0.15), (0.3, 0.4)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    S = min(W * 0.42, H - 100)
    cv.text(x0, y0, "vector: points, lines, polygons", step, 17, "#e8590c", math=False)
    P = lambda x, y, ox: (ox + x * S, y0 + 30 + y * S)
    cv.rect(x0, y0 + 30, S, S, step, "#ffffff", MUTED, 1, rounded=False)
    cv.polygon([P(x, y, x0) for x, y in poly], step, "#b2f2bb", "#2b8a3e", 2)
    _pl(cv, [P(x, y, x0) for x, y in road], step, "#495057", 4)
    for x, y in wells:
        X, Y = P(x, y, x0)
        cv.ellipse(X - 7, Y - 7, 14, 14, step, "#1971c2", "#1971c2", 1)
    step += 1

    def inside(x, y):
        c = False
        for i in range(len(poly)):
            (ax, ay), (bx, by) = poly[i], poly[(i + 1) % len(poly)]
            if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
                c = not c
        return c
    ox = x0 + W * 0.48
    cv.text(ox, y0, "raster: a grid of cells", step, 17, "#e8590c", math=False)
    g = 16
    for i in range(g):
        for j in range(g):
            x, y = (j + 0.5) / g, (i + 0.5) / g
            col = "#ffffff"
            if inside(x, y):
                col = "#b2f2bb"
            seg = min(abs(y - (road[k][1] + (road[k + 1][1] - road[k][1]) * (x - road[k][0]) / (road[k + 1][0] - road[k][0])))
                      for k in range(2) if road[k][0] <= x <= road[k + 1][0])
            if seg < 0.5 / g:
                col = "#495057"
            if any(abs(x - wx) < 0.5 / g and abs(y - wy) < 0.5 / g for wx, wy in wells):
                col = "#1971c2"
            cv.rect(ox + j * S / g, y0 + 30 + i * S / g, S / g, S / g, step, col, "#dee2e6", 0.5, rounded=False)
        if i % 4 == 3:
            step += 1
    cv.text(x0, y0 + 50 + S, "vector: exact shapes, small files, good for boundaries and networks.\n"
                             "raster: cell size sets the detail; good for continuous data (elevation, imagery).",
            step, 16, INK, math=False)
    return step + 1


def build_spatial_join(cv, v, box, step):
    zones = v.get("zones") or {"North": [(0, 0), (1, 0), (1, 0.45), (0, 0.5)],
                                "South-west": [(0, 0.5), (0.55, 0.47), (0.5, 1), (0, 1)],
                                "South-east": [(0.55, 0.47), (1, 0.45), (1, 1), (0.5, 1)]}
    rnd = random.Random(14)
    pts = v.get("points") or [(rnd.uniform(0.03, 0.97), rnd.uniform(0.03, 0.97)) for _ in range(26)]

    def inside(x, y, poly):
        c = False
        for i in range(len(poly)):
            (ax, ay), (bx, by) = poly[i], poly[(i + 1) % len(poly)]
            if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
                c = not c
        return c
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    S = min(W * 0.55, H - 20)
    P = lambda x, y: (x0 + x * S, y0 + y * S)
    names = list(zones)
    for i, (n, poly) in enumerate(zones.items()):
        cv.polygon([P(*p) for p in poly], step, PAL[i], STRONG[i], 2)
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        cv.text(*P(cx, cy), n, step, 16, STRONG[i], "center", "middle", math=False)
    step += 1
    counts = {n: 0 for n in names}
    for x, y in pts:
        z = next((n for n in names if inside(x, y, zones[n])), None)
        if z:
            counts[z] += 1
        X, Y = P(x, y)
        cv.ellipse(X - 5, Y - 5, 10, 10, step, STRONG[names.index(z)] if z else INK, INK, 1)
    step += 1
    _table(cv, x0 + S + 40, y0 + 30, "spatial join result", ["zone", "points"], [[n, c] for n, c in counts.items()], step,
           W * 0.3, size=16)
    step += 1
    cv.text(x0 + S + 40, y0 + 220, "each point takes the attributes of\nthe polygon it falls in\n(point-in-polygon, computed)\n\n"
                                   "e.g. 911 calls per district,\nstores per census tract", step, 17, INK, math=False)
    return step + 1


# ======================================================================== economics

def build_npv(cv, v, box, step):
    flows = [float(x) for x in (v.get("cash_flows") or [-1000, 300, 350, 400, 300])]
    r = float(v.get("rate") or 0.08)
    pv = [cf / (1 + r) ** t for t, cf in enumerate(flows)]
    npv = sum(pv)
    lo, hi = min(0, min(flows)), max(flows)
    px, py = _plot(cv, box, step, (-0.6, len(flows) - 0.4), (lo * 1.1, hi * 1.3), "year", "$", frac=0.58)
    cv.line([(px(-0.6), py(0)), (px(len(flows) - 0.4), py(0))], step, INK, 1.5)
    step += 1
    for t, (cf, p) in enumerate(zip(flows, pv)):
        for k, (val, col) in enumerate(((cf, "#dee2e6"), (p, "#2f9e44" if p >= 0 else "#e03131"))):
            x = px(t - 0.3 + k * 0.32)
            cv.rect(x, min(py(val), py(0)), px(0.28) - px(0), abs(py(val) - py(0)), step, col, "transparent", 1, rounded=False)
        cv.text(px(t), py(0) + 6, str(t), step, 14, MUTED, "center", "top", math=False)
        step += 1
    _note(cv, box, step, f"grey = cash flow, colour = today's\nvalue: CF / (1 + {r:g})^t\n\n" +
          "\n".join(f"yr {t}: {cf:,.0f} → {p:,.0f}" for t, (cf, p) in enumerate(zip(flows, pv))) +
          f"\n\nNPV = {npv:,.0f} → " + ("accept (creates value)" if npv > 0 else "reject"), 17, frac=0.66)
    return step + 1


def build_elasticity(cv, v, box, step):
    p1, p2 = float(v.get("p1") or 4), float(v.get("p2") or 5)
    curves = [("elastic (e.g. one brand of cereal)", lambda p: 200 - 30 * p, "#e03131"),
              ("inelastic (e.g. insulin, gasoline)", lambda p: 110 - 4 * p, "#1971c2")]
    px, py = _plot(cv, box, step, (0, 220), (0, 7), "quantity", "price", frac=0.55)
    step += 1
    lines = []
    for name, d, col in curves:
        _pl(cv, [(px(d(p / 10)), py(p / 10)) for p in range(0, 70) if d(p / 10) >= 0], step, col, 3)
        q1, q2 = d(p1), d(p2)
        for p, q in ((p1, q1), (p2, q2)):
            cv.ellipse(px(q) - 5, py(p) - 5, 10, 10, step, col, col, 1)
        e = ((q2 - q1) / ((q1 + q2) / 2)) / ((p2 - p1) / ((p1 + p2) / 2))
        lines.append(f"{name}:\nQ {q1:.0f} → {q2:.0f}, elasticity = {e:.2f}\n" +
                     ("|e| > 1: revenue falls" if abs(e) > 1 else "|e| < 1: revenue rises") +
                     f" ({p1 * q1:.0f} → {p2 * q2:.0f})")
        step += 1
    _note(cv, box, step, f"price rises {p1:g} → {p2:g}\n(midpoint method)\n\n" + "\n\n".join(lines), 16, frac=0.62)
    return step + 1


# ======================================================================== register

visuals.BUILDERS.update({
    "kruskal": build_kruskal, "kmp": build_kmp, "n_queens": build_n_queens, "avl": build_avl, "big_o": build_big_o,
    "b_tree": build_b_tree, "cpu_pipeline": build_cpu_pipeline, "producer_consumer": build_producer_consumer,
    "go_back_n": build_go_back_n, "ip_header": build_ip_header, "vpn_tunnel": build_vpn_tunnel, "cdn": build_cdn,
    "acid_transaction": build_acid_transaction, "docker_layers": build_docker_layers,
    "rolling_update": build_rolling_update, "terraform_plan": build_terraform_plan,
    "kafka_partitions": build_kafka_partitions, "virtual_dom": build_virtual_dom, "token_bucket": build_token_bucket,
    "diffie_hellman": build_diffie_hellman, "rsa": build_rsa, "linear_regression": build_linear_regression,
    "logistic_regression": build_logistic_regression, "rnn_unrolled": build_rnn_unrolled,
    "cross_validation": build_cross_validation, "tfidf": build_tfidf, "perceptron": build_perceptron,
    "boxplot": build_boxplot, "histogram": build_histogram, "central_limit": build_central_limit,
    "hypothesis_test": build_hypothesis_test, "taylor_series": build_taylor_series, "fourier_series": build_fourier_series,
    "newtons_method": build_newtons_method, "eigen_transform": build_eigen_transform, "number_bases": build_number_bases,
    "modular_clock": build_modular_clock, "spring_shm": build_spring_shm, "lever_torque": build_lever_torque,
    "buoyancy": build_buoyancy, "electric_field": build_electric_field, "em_spectrum": build_em_spectrum,
    "titration": build_titration, "limiting_reagent": build_limiting_reagent, "vsepr": build_vsepr,
    "enzyme": build_enzyme, "action_potential": build_action_potential, "hardy_weinberg": build_hardy_weinberg,
    "population_growth": build_population_growth, "idw_interpolation": build_idw_interpolation,
    "topographic": build_topographic, "raster_vector": build_raster_vector, "spatial_join": build_spatial_join,
    "npv": build_npv, "elasticity": build_elasticity,
})
for _k in [k for k in list(visuals.ALIASES) if k in visuals.BUILDERS]:
    visuals.ALIASES.pop(_k)
visuals.ALIASES.update({
    "mst": "kruskal", "minimum_spanning_tree": "kruskal", "union_find": "kruskal", "string_matching": "kmp",
    "backtracking": "n_queens", "self_balancing_tree": "avl", "complexity": "big_o", "time_complexity": "big_o",
    "btree": "b_tree", "database_index": "b_tree", "pipelining": "cpu_pipeline", "hazard": "cpu_pipeline",
    "bounded_buffer": "producer_consumer", "semaphore": "producer_consumer", "arq": "go_back_n",
    "sliding_window_protocol": "go_back_n", "ipv4_header": "ip_header", "checksum": "ip_header", "vpn": "vpn_tunnel",
    "content_delivery": "cdn", "transaction": "acid_transaction", "dockerfile": "docker_layers",
    "deployment": "rolling_update", "infrastructure_as_code": "terraform_plan", "kafka": "kafka_partitions",
    "event_streaming": "kafka_partitions", "react": "virtual_dom", "reconciliation": "virtual_dom",
    "rate_limit": "token_bucket", "key_exchange": "diffie_hellman", "public_key": "rsa",
    "least_squares": "linear_regression", "regression": "linear_regression", "sigmoid": "logistic_regression",
    "classification": "logistic_regression", "rnn": "rnn_unrolled", "lstm": "rnn_unrolled", "kfold": "cross_validation",
    "box_plot": "boxplot", "quartiles": "boxplot", "frequency_distribution": "histogram", "clt": "central_limit",
    "p_value": "hypothesis_test", "z_test": "hypothesis_test", "taylor": "taylor_series", "fourier": "fourier_series",
    "root_finding": "newtons_method", "eigenvalues": "eigen_transform", "linear_transformation": "eigen_transform",
    "binary_conversion": "number_bases", "hexadecimal": "number_bases", "modulo": "modular_clock",
    "simple_harmonic": "spring_shm", "hookes_law": "spring_shm", "torque": "lever_torque", "lever": "lever_torque",
    "archimedes": "buoyancy", "coulomb": "electric_field", "field_lines": "electric_field",
    "electromagnetic_spectrum": "em_spectrum", "acid_base": "titration", "stoichiometry": "limiting_reagent",
    "molecular_geometry": "vsepr", "michaelis_menten": "enzyme", "nerve_impulse": "action_potential",
    "allele_frequencies": "hardy_weinberg", "logistic_growth": "population_growth", "carrying_capacity": "population_growth",
    "interpolation": "idw_interpolation", "contour": "topographic", "elevation": "topographic",
    "data_models": "raster_vector", "point_in_polygon": "spatial_join", "net_present_value": "npv",
    "discounting": "npv", "price_elasticity": "elasticity",
})
