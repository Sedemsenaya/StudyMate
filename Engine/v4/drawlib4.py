"""
StudyMate v4 - drawing library, part 4. Everything marked (computed) is worked out, not guessed.

Algorithms        dp_table (LCS table filled cell by cell, computed), merge_sort (split/merge tree, computed),
                  huffman (tree + codes, computed), trie (words inserted), topo_sort (Kahn's algorithm, computed)
Systems           lru (page replacement hits/faults, computed), cache_sim (direct-mapped hits/misses, computed),
                  tcp_cwnd (slow start + AIMD congestion window, computed), deadlock (wait-for cycle found, computed),
                  race (two threads, lost update, computed), event_loop (JS call stack / Web APIs / queue),
                  twos_complement (computed)
Web / security    jwt (header.payload.signature, real base64 + HMAC-SHA256), sql_injection (the query string built),
                  password_hash (salt + SHA-256, computed)
Cloud / DevOps    cicd (commit -> build -> test -> image -> deploy, logos), autoscaling (load vs instances, computed)
ML / data         decision_tree (Gini split on a toy table, computed), knn (k nearest, vote, computed),
                  normal_dist (68-95-99.7), softmax (logits -> probabilities, computed),
                  overfitting (training vs validation loss), er_diagram (entities, keys, relationships),
                  groupby (rows -> groups -> aggregate, computed)
Biology           punnett (cross, genotype and phenotype ratios, computed), meiosis (crossing over, 4 cells),
                  respiration (glycolysis -> Krebs -> ETC with ATP), osmosis (cells in 3 solutions)
Earth / space     plate_tectonics (divergent, convergent, transform), moon_phases, earth_layers
Physics           lens (thin lens image, computed), pendulum (swinging, period computed),
                  energy_ramp (PE <-> KE bars, computed)
Chemistry         balance (atom counts on both sides, computed), reaction_energy (activation energy, catalyst)
Maths / business  tangent (slope moving along a curve, computed), compound (compound interest, computed),
                  linear_system (two lines, intersection solved), break_even (computed)
GIS               gis_buffer (points, buffers, features inside)
"""
import base64
import hashlib
import hmac
import json
import math
import random
import re

import visuals
from canvas import INK, MUTED, fit_text, text_size
from drawlib import _envelope, _poly, _Track, draw_device, draw_thing
from drawlib2 import PAL, STRONG, _box, _table
from drawlib3 import _frames, _title


def _axes(cv, x0, y0, x1, y1, step, xl="", yl=""):
    cv.line([(x0, y1), (x1, y1)], step, INK, 2, arrow_end=True)
    cv.line([(x0, y1), (x0, y0)], step, INK, 2, arrow_end=True)
    if xl:
        cv.text(x1, y1 + 6, xl, step, 17, INK, "right", "top", math=False)
    if yl:
        cv.text(x0 + 8, y0, yl, step, 17, INK, "left", "top", math=False)


def _curve(fn, a, b, n=60):
    return [(a + (b - a) * i / n, fn(a + (b - a) * i / n)) for i in range(n + 1)]


# ======================================================================== algorithms

def build_dp_table(cv, v, box, step):
    a, b = str(v.get("a") or "ABCBDAB")[:9], str(v.get("b") or "BDCABA")[:9]
    n, m = len(a), len(b)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            L[i][j] = L[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(L[i - 1][j], L[i][j - 1])
    i, j, lcs, path = n, m, [], set()                     # trace back
    while i and j:
        path.add((i, j))
        if a[i - 1] == b[j - 1]:
            lcs.append(a[i - 1])
            i, j = i - 1, j - 1
        elif L[i - 1][j] >= L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    c = min(52, (H - 60) / (n + 2), W * 0.6 / (m + 2))
    _title(cv, x0, y0, f'Longest common subsequence of "{a}" and "{b}"', step, 20)
    for j, ch in enumerate(" " + b):
        cv.text(x0 + (j + 1.5) * c, y0 + 40 + c / 2, ch, step, 20, "#1971c2", "center", "middle", math=False)
    for i, ch in enumerate(" " + a):
        cv.text(x0 + c / 2, y0 + 40 + (i + 1.5) * c, ch, step, 20, "#e8590c", "center", "middle", math=False)
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                _box(cv, x0 + (j + 1) * c, y0 + 40 + (i + 1) * c, c - 3, c - 3, step, 0, "#f1f3f5", 16)
    step += 1
    for i in range(1, n + 1):                              # row by row
        for j in range(1, m + 1):
            match = a[i - 1] == b[j - 1]
            _box(cv, x0 + (j + 1) * c, y0 + 40 + (i + 1) * c, c - 3, c - 3, step, L[i][j],
                 "#b2f2bb" if match else "#ffffff", 16)
        step += 1
    for (i, j) in path:
        cv.rect(x0 + (j + 1) * c - 2, y0 + 38 + (i + 1) * c, c + 1, c + 1, step, "transparent", "#e03131", 2.5)
    rx = x0 + (m + 3) * c
    cv.text(rx, y0 + 60, "if letters match:\n  cell = diagonal + 1\nelse:\n  cell = max(up, left)", step, 19, INK,
            math=False)
    cv.text(rx, y0 + 200, f'LCS = "{"".join(reversed(lcs))}" (length {L[n][m]})', step, 22, "#e03131", math=False)
    return step + 1


def build_merge_sort(cv, v, box, step):
    arr = [int(x) for x in (v.get("array") or [38, 27, 43, 3, 9, 82, 10, 5])][:8]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    levels = int(math.ceil(math.log2(len(arr)))) if len(arr) > 1 else 1
    rh = (H - 30) / (2 * levels + 1)
    cw = min(50, W / len(arr) / 1.3)

    def row(parts, r, st, col):
        total = sum(len(p) for p in parts) * cw + (len(parts) - 1) * 26
        x = x0 + (W - total) / 2
        for p in parts:
            for k, val in enumerate(p):
                _box(cv, x + k * cw, y0 + r * rh, cw - 3, rh * 0.62, st, val, col, 17)
            x += len(p) * cw + 26
    parts = [arr]
    row(parts, 0, step, "#e7f5ff")
    step += 1
    r = 1
    split_levels = [parts]
    while any(len(p) > 1 for p in parts):                  # split
        parts = [q for p in parts for q in ((p[:len(p) // 2], p[len(p) // 2:]) if len(p) > 1 else (p,))]
        row(parts, r, step, "#fff3bf")
        split_levels.append(parts)
        step += 1
        r += 1
    for lv in reversed(split_levels[:-1]):                 # merge back up
        merged = [sorted(p) for p in lv]
        row(merged, r, step, "#b2f2bb")
        step += 1
        r += 1
    cv.text(x0, y0, "split", step, 16, MUTED, math=False)
    return step


def build_huffman(cv, v, box, step):
    text = str(v.get("text") or "ABRACADABRA")
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    import heapq
    heap = [(f, i, {"label": f"{ch}:{f}", "ch": ch}) for i, (ch, f) in enumerate(sorted(freq.items()))]
    heapq.heapify(heap)
    k = len(heap)
    while len(heap) > 1:
        f1, _, a = heapq.heappop(heap)
        f2, _, b = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, k, {"label": str(f1 + f2), "children": [a, b]}))
        k += 1
    root = heap[0][2]
    codes = {}

    def walk(n, code):
        if "ch" in n:
            codes[n["ch"]] = code or "0"
        for bit, ch in zip("01", n.get("children", [])):
            walk(ch, code + bit)
    walk(root, "")

    def label_edges(n):
        out = {"label": n["label"]}
        if n.get("children"):
            out["children"] = [label_edges(c) for c in n["children"]]
        return out
    x0, y0, x1, y1 = box
    W = x1 - x0
    _title(cv, x0, y0, f'Huffman code for "{text}"', step, 21)
    step += 1
    step = visuals.build_tree(cv, {"root": label_edges(root)}, (x0, y0 + 30, x0 + W * 0.62, y1), step)
    bits = sum(freq[c] * len(codes[c]) for c in freq)
    lines = [f"{c}: {codes[c]}  ({freq[c]}×)" for c in sorted(codes, key=lambda c: (len(codes[c]), c))]
    cv.text(x0 + W * 0.66, y0 + 40, "left = 0, right = 1\n" + "\n".join(lines) +
            f"\n\n{bits} bits vs {8 * len(text)} in ASCII", step, 20, INK, math=False)
    return step + 1


def build_trie(cv, v, box, step):
    words = [str(w).lower() for w in (v.get("words") or ["car", "cat", "cart", "dog", "do"])][:6]
    trie, order = {}, []
    for w in words:
        node, path = trie, ""
        for ch in w:
            path += ch
            if ch not in node:
                node[ch] = {}
                order.append((path, w))
            node = node[ch]
        node["$"] = {}
    # layout: depth-first leaf ordering
    pos, leaves = {}, [0]

    def place(node, path, depth):
        kids = [k for k in sorted(node) if k != "$"]
        if not kids:
            pos[path] = (leaves[0], depth)
            leaves[0] += 1
            return pos[path][0]
        xs = [place(node[k], path + k, depth + 1) for k in kids]
        pos[path] = ((min(xs) + max(xs)) / 2, depth)
        return pos[path][0]
    place(trie, "", 0)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    maxd = max(d for _, d in pos.values())
    P = lambda p: (x0 + 30 + pos[p][0] / max(1, leaves[0] - 1) * (W * 0.62), y0 + 30 + pos[p][1] * (H - 60) / max(1, maxd))
    R = 22
    cv.ellipse(P("")[0] - R, P("")[1] - R, 2 * R, 2 * R, step, "#e9ecef")
    cv.text(P("")[0], P("")[1], "root", step, 13, MUTED, "center", "middle", math=False)
    step += 1
    ends = {w for w in words}
    cur = None
    for path, w in order:
        if w != cur and cur is not None:
            step += 1
        cur = w
        (px, py), (x, y) = P(path[:-1]), P(path)
        cv.line([(px, py + R), (x, y - R)], step, MUTED, 1.5)
        cv.ellipse(x - R, y - R, 2 * R, 2 * R, step, "#b2f2bb" if path in ends else "#ffffff",
                   "#2f9e44" if path in ends else INK, 2)
        cv.text(x, y, path[-1], step, 19, INK, "center", "middle", math=False)
    for w in words:                                    # words that end at an inner node
        if w in pos:
            x, y = P(w)
            cv.ellipse(x - R, y - R, 2 * R, 2 * R, step, "transparent", "#2f9e44", 3)
    cv.text(x0 + W * 0.68, y0 + 40, "words: " + ", ".join(words) + "\n\ngreen = a word ends here\nshared prefixes are stored once",
            step, 20, INK, math=False)
    return step + 1


def build_topo_sort(cv, v, box, step):
    nodes = v.get("nodes") or ["intro", "data structures", "algorithms", "OS", "networks", "databases", "capstone"]
    edges = v.get("edges") or [["intro", "data structures"], ["data structures", "algorithms"], ["intro", "OS"],
                               ["OS", "networks"], ["data structures", "databases"], ["algorithms", "capstone"],
                               ["networks", "capstone"], ["databases", "capstone"]]
    indeg = {n: 0 for n in nodes}
    for a, b in edges:
        indeg[b] += 1
    q, order, level = [n for n in nodes if indeg[n] == 0], [], {}
    for n in q:
        level[n] = 0
    deg = dict(indeg)
    while q:
        u = q.pop(0)
        order.append(u)
        for a, b in edges:
            if a == u:
                deg[b] -= 1
                level[b] = max(level.get(b, 0), level[u] + 1)
                if deg[b] == 0:
                    q.append(b)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cols = {}
    for n in nodes:
        cols.setdefault(level.get(n, 0), []).append(n)
    pos = {}
    for c, ms in cols.items():
        for r, n in enumerate(ms):
            pos[n] = (x0 + W * 0.65 * (c + 0.5) / len(cols), y0 + 30 + (H - 60) * (r + 0.5) / len(ms))
    bw, bh = 150, 44
    for a, b in edges:
        (ax, ay), (bx, by) = pos[a], pos[b]
        cv.arrow(ax + bw / 2, ay, bx - bw / 2, by, step, color=MUTED)
    for n in nodes:
        x, y = pos[n]
        _box(cv, x - bw / 2, y - bh / 2, bw, bh, step, n, "#ffffff", 17)
    step += 1
    cv.text(x0 + W * 0.7, y0, "Kahn's algorithm: take a node\nwith no remaining prerequisites", step, 17, MUTED, math=False)
    for k, n in enumerate(order):
        x, y = pos[n]
        _box(cv, x - bw / 2, y - bh / 2, bw, bh, step, n, "#b2f2bb", 17)
        cv.text(x - bw / 2 - 4, y - bh / 2 - 2, str(k + 1), step, 16, "#2f9e44", "right", "bottom", math=False)
        cv.text(x0 + W * 0.7, y0 + 60 + k * 32, f"{k + 1}. {n}", step, 19, INK, math=False)
        step += 1
    return step


# ======================================================================== systems

def lru_run(refs, frames=3, policy="lru"):
    mem, out, last, arrived = [], [], {}, {}
    for t, p in enumerate(refs):
        hit = p in mem
        victim = None
        if not hit:
            if len(mem) >= frames:
                victim = min(mem, key=lambda q: last[q] if policy == "lru" else arrived[q])
                mem[mem.index(victim)] = p
            else:
                mem.append(p)
            arrived[p] = t
        last[p] = t
        out.append((p, list(mem), hit, victim))
    return out


def build_lru(cv, v, box, step):
    refs = [int(x) for x in (v.get("refs") or [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3])][:14]
    frames = int(v.get("frames") or 3)
    policy = str(v.get("policy") or "lru").lower()
    res = lru_run(refs, frames, policy)
    x0, y0, x1, y1 = box
    W = x1 - x0
    c = min(70, (W - 140) / len(refs))
    _title(cv, x0, y0, f"Page replacement: {policy.upper()}, {frames} frames", step, 21)
    cv.text(x0, y0 + 50, "request", step, 17, MUTED, math=False)
    for f in range(frames):
        cv.text(x0, y0 + 100 + f * 50, f"frame {f}", step, 17, MUTED, math=False)
    step += 1
    faults = 0
    for t, (p, mem, hit, victim) in enumerate(res):
        x = x0 + 110 + t * c
        cv.text(x + c / 2, y0 + 50, str(p), step, 22, "#1971c2", "center", "top", math=False)
        for f in range(frames):
            val = mem[f] if f < len(mem) else ""
            _box(cv, x, y0 + 90 + f * 50, c - 6, 44, step, val, "#ffc9c9" if (not hit and val == p) else "#ffffff", 18)
        faults += not hit
        cv.text(x + c / 2, y0 + 100 + frames * 50, "hit" if hit else "fault", step, 15, "#2f9e44" if hit else "#e03131",
                "center", "top", math=False)
        step += 1
    cv.text(x0, y0 + 160 + frames * 50, f"{faults} page faults, {len(refs) - faults} hits "
                                        f"(hit rate {(len(refs) - faults) / len(refs):.0%})", step, 22, INK, math=False)
    return step + 1


def build_cache_sim(cv, v, box, step):
    lines = int(v.get("lines") or 4)
    block = int(v.get("block") or 4)
    addrs = [int(a) for a in (v.get("addresses") or [0, 4, 8, 0, 4, 16, 0, 8, 12, 8])][:10]
    cache = [None] * lines
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    _title(cv, x0, y0, f"Direct-mapped cache: {lines} lines, {block}-byte blocks", step, 21)
    cv.text(x0, y0 + 36, "line = (address ÷ block) mod lines;   tag = (address ÷ block) ÷ lines", step, 17, MUTED,
            math=False)
    step += 1
    hits = 0
    for k, a in enumerate(addrs):
        blk = a // block
        idx, tag = blk % lines, blk // lines
        hit = cache[idx] == tag
        hits += hit
        cache[idx] = tag
        tr = _Track(cv)
        for i in range(lines):
            _box(cv, x0 + W * 0.5, y0 + 80 + i * 56, 90, 48, step, f"line {i}", "#e9ecef", 16)
            _box(cv, x0 + W * 0.5 + 94, y0 + 80 + i * 56, 140, 48, step,
                 "empty" if cache[i] is None else f"tag {cache[i]}",
                 ("#b2f2bb" if hit else "#ffc9c9") if i == idx else "#ffffff", 16)
        cv.text(x0 + W * 0.5 + 250, y0 + 104 + idx * 56, "HIT" if hit else "MISS → load", step, 20,
                "#2f9e44" if hit else "#e03131", "left", "middle", math=False)
        tr.tag(until=step + 1 if k < len(addrs) - 1 else None, fade=k > 0)
        cv.text(x0, y0 + 80 + k * 34, f"addr {a:>3}: block {blk}, line {idx}, tag {tag} → {'hit' if hit else 'miss'}",
                step, 18, "#2f9e44" if hit else "#e03131", math=False)
        step += 1
    cv.text(x0, y0 + 90 + len(addrs) * 34, f"hit rate {hits}/{len(addrs)} = {hits / len(addrs):.0%}", step, 21, INK,
            math=False)
    return step + 1


def build_tcp_cwnd(cv, v, box, step):
    rounds = int(v.get("rounds") or 18)
    ssthresh = float(v.get("ssthresh") or 16)
    losses = set(int(x) for x in (v.get("losses") or [9, 15]))
    cw, xs = 1.0, []
    for r in range(rounds):
        xs.append(cw)
        if r in losses:
            ssthresh = max(cw / 2, 2)
            cw = ssthresh                                  # fast recovery (Reno)
        elif cw < ssthresh:
            cw *= 2
        else:
            cw += 1
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    mx = max(xs) * 1.15
    px = lambda r: x0 + 50 + r * (W * 0.62) / rounds
    py = lambda c: y1 - 30 - c / mx * (H - 60)
    _axes(cv, x0 + 50, y0, x0 + 60 + W * 0.62, y1 - 30, step, "round trip", "cwnd (segments)")
    step += 1
    for r in range(rounds):
        if r:
            cv.line([(px(r - 1), py(xs[r - 1])), (px(r), py(xs[r]))], step, "#1971c2", 3)
        cv.ellipse(px(r) - 5, py(xs[r]) - 5, 10, 10, step, "#e03131" if r in losses else "#1971c2", "#1971c2", 1)
        if r in losses:
            cv.text(px(r), py(xs[r]) - 8, "loss", step, 16, "#e03131", "center", "bottom", math=False)
        step += 1
    cv.text(x0 + W * 0.68, y0 + 30, "slow start: double each round\nuntil ssthresh\n\ncongestion avoidance: +1\n\n"
                                   "on loss: halve (multiplicative\ndecrease), then grow again", step, 19, INK, math=False)
    return step + 1


def build_deadlock(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    P = [(x0 + W * 0.15, y0 + H * 0.25), (x0 + W * 0.5, y0 + H * 0.75)]
    R = [(x0 + W * 0.5, y0 + H * 0.25), (x0 + W * 0.15, y0 + H * 0.75)]
    for i, (x, y) in enumerate(P):
        cv.ellipse(x - 50, y - 50, 100, 100, step, "#d0ebff")
        cv.text(x, y, f"Process {i + 1}", step, 18, INK, "center", "middle", math=False)
    for i, (x, y) in enumerate(R):
        _box(cv, x - 60, y - 40, 120, 80, step, f"Resource {'AB'[i]}", "#ffec99", 18)
    step += 1
    cv.arrow(R[0][0] - 62, R[0][1], P[0][0] + 52, P[0][1], step, "held by", color="#2f9e44")
    step += 1
    cv.arrow(R[1][0] + 62, R[1][1], P[1][0] - 52, P[1][1], step, "held by", color="#2f9e44")
    step += 1
    cv.arrow(P[0][0], P[0][1] + 52, R[1][0], R[1][1] - 42, step, "waits for", color="#e03131")
    step += 1
    cv.arrow(P[1][0], P[1][1] - 52, R[0][0], R[0][1] + 42, step, "waits for", color="#e03131")
    step += 1
    cv.text(x0 + W * 0.7, y0 + 30, "a cycle in the wait-for graph\n= DEADLOCK\n\nall four conditions hold:\n"
                                   "1. mutual exclusion\n2. hold and wait\n3. no preemption\n4. circular wait\n\n"
                                   "fix: lock resources in one\nglobal order (breaks 4)", step, 19, INK, math=False)
    return step + 1


def build_race(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    ops = [("A", "read counter → 5"), ("B", "read counter → 5"), ("A", "add 1 → 6"), ("B", "add 1 → 6"),
           ("A", "write 6"), ("B", "write 6")]
    cv.text(x0 + W * 0.15, y0, "Thread A", step, 22, "#1971c2", "center", "top", math=False)
    cv.text(x0 + W * 0.45, y0, "Thread B", step, 22, "#e8590c", "center", "top", math=False)
    cv.text(x0 + W * 0.75, y0, "counter", step, 22, INK, "center", "top", math=False)
    step += 1
    val = 5
    for k, (t, op) in enumerate(ops):
        y = y0 + 50 + k * (H - 120) / len(ops)
        x = x0 + W * (0.15 if t == "A" else 0.45)
        _box(cv, x - 110, y, 220, 44, step, op, "#d0ebff" if t == "A" else "#ffd8a8", 18)
        if "write" in op:
            val = 6
        tr = _Track(cv)
        _box(cv, x0 + W * 0.75 - 50, y, 100, 44, step, val, "#fff3bf", 20)
        tr.tag(fade=True)
        step += 1
    cv.text(x0, y1 - 40, "two increments, but counter = 6, not 7: a lost update (race condition).\n"
                         "fix: a mutex/lock or an atomic increment makes read-add-write one step.", step, 19, "#e03131",
            math=False)
    return step + 1


def build_event_loop(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    code = ['console.log("A")', 'setTimeout(() => console.log("B"), 0)', 'Promise.resolve().then(() => console.log("C"))',
            'console.log("D")']
    for i, ln in enumerate(code):
        el = cv.text(x0, y0 + i * 30, ln, step, 17, INK, math=False)
        if el:
            el["fontFamily"] = 3
    areas = {"stack": (x0, y0 + 140, W * 0.22), "webapi": (x0 + W * 0.26, y0 + 140, W * 0.22),
             "micro": (x0 + W * 0.52, y0 + 140, W * 0.22), "macro": (x0 + W * 0.52, y0 + 290, W * 0.22)}
    names = {"stack": "call stack", "webapi": "Web APIs (timer)", "micro": "microtask queue", "macro": "task queue"}
    for k, (x, y, w) in areas.items():
        cv.rect(x, y, w, 120, step, "#f8f9fa", INK, 2)
        cv.text(x + 8, y + 6, names[k], step, 17, MUTED, math=False)
    cv.text(x0 + W * 0.78, y0 + 140, "console", step, 18, MUTED, math=False)
    step += 1
    frames = [({"stack": 'log("A")'}, "A"), ({"stack": "setTimeout", "webapi": "timer 0 ms"}, ""),
              ({"stack": "Promise.then", "macro": "() => log(B)"}, ""),
              ({"stack": 'log("D")', "micro": "() => log(C)", "macro": "() => log(B)"}, "D"),
              ({"stack": "() => log(C)", "macro": "() => log(B)"}, "C"), ({"stack": "() => log(B)"}, "B")]
    out = ""
    for k, (state, printed) in enumerate(frames):
        tr = _Track(cv)
        for area, txt in state.items():
            x, y, w = areas[area]
            _box(cv, x + 10, y + 50, w - 20, 50, step, txt, "#ffec99" if area == "stack" else "#d0ebff", 16)
        tr.tag(until=step + 1 if k < len(frames) - 1 else None, fade=k > 0)
        if printed:
            out += printed + " "
            cv.text(x0 + W * 0.78, y0 + 170 + len(out.split()) * 30, printed, step, 24, "#2f9e44", math=False)
        step += 1
    cv.text(x0, y1 - 30, "output: A D C B  (sync code, then microtasks, then tasks)", step, 20, "#e8590c", math=False)
    return step + 1


def build_twos_complement(cv, v, box, step):
    n = int(v.get("number") or -13)
    bits = int(v.get("bits") or 8)
    x0, y0, x1, y1 = box
    mag = abs(n)
    b = format(mag, f"0{bits}b")
    inv = "".join("1" if c == "0" else "0" for c in b)
    res = format((1 << bits) + n if n < 0 else n, f"0{bits}b")
    rows = [(f"+{mag} in binary", b), ("flip every bit", inv), ("add 1", res)] if n < 0 else [(f"{n} in binary", b)]
    _title(cv, x0, y0, f"{n} as an {bits}-bit two's complement number", step, 22)
    step += 1
    for i, (lab, bs) in enumerate(rows):
        y = y0 + 60 + i * 70
        cv.text(x0, y + 22, lab, step, 20, MUTED, "left", "middle", math=False)
        for j, ch in enumerate(bs):
            _box(cv, x0 + 230 + j * 50, y, 46, 46, step, ch,
                 "#b2f2bb" if i == len(rows) - 1 else "#ffffff", 20, mono=True)
        step += 1
    y = y0 + 60 + len(rows) * 70
    weights = [-(1 << (bits - 1))] + [1 << k for k in range(bits - 2, -1, -1)]
    for j, w in enumerate(weights):
        cv.text(x0 + 253 + j * 50, y, str(w), step, 14, "#1971c2", "center", "top", math=False)
    total = " + ".join(str(w) for w, c in zip(weights, res) if c == "1")
    cv.text(x0, y + 40, f"check: {total} = {n}", step, 21, "#2f9e44", math=False)
    return step + 1


# ======================================================================== web / security

def _b64(d):
    return base64.urlsafe_b64encode(d).decode().rstrip("=")


def build_jwt(cv, v, box, step):
    header = {"alg": "HS256", "typ": "JWT"}
    payload = v.get("payload") or {"sub": "42", "name": "Sed", "role": "student", "exp": 1798761600}
    secret = str(v.get("secret") or "my-secret")
    h, p = _b64(json.dumps(header, separators=(",", ":")).encode()), _b64(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64(hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())
    x0, y0, x1, y1 = box
    W = x1 - x0
    parts = [("header", h, json.dumps(header), "#ffc9c9"), ("payload", p, json.dumps(payload), "#d0bfff"),
             ("signature", sig, f'HMAC-SHA256(header.payload, "{secret}")', "#99e9f2")]
    tok, sz = fit_text(f"{h}.{p}.{sig}", W - 20, 90, 15, 10)
    el = cv.text(x0, y0, tok, step, sz, INK, math=False)
    if el:
        el["fontFamily"] = 3
    step += 1
    for i, (name, enc, dec, col) in enumerate(parts):
        y = y0 + 110 + i * 90
        _box(cv, x0, y, 150, 70, step, name, col, 20)
        e, s2 = fit_text(enc[:60] + ("…" if len(enc) > 60 else ""), W * 0.36, 60, 15, 10)
        cv.text(x0 + 170, y + 8, e, step, s2, MUTED, math=False)
        d, s3 = fit_text(dec, W * 0.38, 66, 17, 11)
        cv.text(x0 + W * 0.58, y + 8, ("decodes to " if i < 2 else "") + d, step, s3, INK, math=False)
        step += 1
    cv.text(x0, y0 + 390, "base64 is NOT encryption: anyone can read the payload.\nThe signature only proves it wasn't changed.",
            step, 19, "#e03131", math=False)
    return step + 1


def build_sql_injection(cv, v, box, step):
    user = str(v.get("input") or "' OR '1'='1' --")
    x0, y0, x1, y1 = box
    W = x1 - x0
    code = 'query = "SELECT * FROM users WHERE name = \'" + name + "\' AND pw = \'" + pw + "\'"'
    el = cv.text(x0, y0, code, step, 17, INK, math=False)
    if el:
        el["fontFamily"] = 3
    step += 1
    _box(cv, x0, y0 + 50, 200, 50, step, "name field", "#e9ecef", 18)
    _box(cv, x0 + 210, y0 + 50, 340, 50, step, user, "#ffc9c9", 19, mono=True)
    step += 1
    built = f"SELECT * FROM users WHERE name = '{user}' AND pw = ''"
    b, sz = fit_text(built, W - 20, 60, 19, 12)
    el = cv.text(x0, y0 + 140, b, step, sz, "#c92a2a", math=False)
    if el:
        el["fontFamily"] = 3
    step += 1
    cv.text(x0, y0 + 210, "'1'='1' is always true and -- comments out the password check\n→ the attacker logs in "
                          "as the first user (often admin)", step, 20, INK, math=False)
    step += 1
    fix = 'cursor.execute("SELECT * FROM users WHERE name = ? AND pw = ?", (name, pw))'
    cv.text(x0, y0 + 300, "fix: parameterized query (input is data, never code)", step, 20, "#2f9e44", math=False)
    el = cv.text(x0, y0 + 334, fix, step, 17, "#2f9e44", math=False)
    if el:
        el["fontFamily"] = 3
    return step + 1


def build_password_hash(cv, v, box, step):
    pw = str(v.get("password") or "hunter2")
    salts = v.get("salts") or ["a9f3", "71c2"]
    x0, y0, x1, y1 = box
    W = x1 - x0
    _title(cv, x0, y0, "Store a salted hash, never the password", step, 21)
    step += 1
    for i, s in enumerate(salts[:2]):
        y = y0 + 60 + i * 140
        user = ["alice", "bob"][i]
        cv.text(x0, y, f"{user}: password \"{pw}\"", step, 20, INK, math=False)
        _box(cv, x0, y + 34, 160, 44, step, f"salt {s}", "#ffec99", 18, mono=True)
        cv.arrow(x0 + 164, y + 56, x0 + 240, y + 56, step, "SHA-256", color="#1971c2")
        h = hashlib.sha256((s + pw).encode()).hexdigest()
        el = cv.text(x0 + 250, y + 44, h[:32] + "…", step, 17, "#2f9e44", math=False)
        if el:
            el["fontFamily"] = 3
        step += 1
    cv.text(x0, y0 + 360, "same password, different salts → different hashes:\nrainbow tables and matching hashes "
                          "don't work.\nReal systems use slow hashes: bcrypt, scrypt, Argon2.", step, 19, INK, math=False)
    return step + 1


# ======================================================================== cloud / devops

def build_cicd(cv, v, box, step):
    stages = v.get("stages") or [("git push", "github"), ("build", "github actions"), ("unit tests", "python"),
                                 ("docker image", "docker"), ("push to registry", "docker hub"),
                                 ("deploy", "kubernetes"), ("monitor", "grafana")]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    n = len(stages)
    cw = W / n
    for i, st in enumerate(stages):
        lab, icon = (st if isinstance(st, (list, tuple)) else (st, st))
        cx = x0 + cw * (i + 0.5)
        cy = y0 + H * 0.4
        b = draw_thing(cv, icon, cx, cy, min(90, cw * 0.6), step, "server")
        cv.text(cx, cy + 56, lab, step, 18, INK, "center", "top", math=False)
        if i:
            cv.arrow(cx - cw + min(45, cw * 0.3) + 8, cy, cx - min(45, cw * 0.3) - 8, cy, step)
        cv.text(cx, cy - 70, "✔", step, 26, "#2f9e44", "center", "bottom", math=False)
        step += 1
    cv.text(x0, y1 - 30, "every push runs the same pipeline automatically; a failing test stops the deploy", step, 19,
            "#e8590c", math=False)
    return step + 1


def build_autoscaling(cv, v, box, step):
    load = v.get("load") or [20, 35, 60, 90, 120, 150, 140, 100, 70, 40, 25]
    per = float(v.get("capacity") or 40)
    lo, hi = int(v.get("min") or 1), int(v.get("max") or 5)
    inst, cur = [], lo
    for L in load:                                       # target ~75% utilisation, scale one step at a time
        need = max(lo, min(hi, math.ceil(L / (per * 0.75))))
        cur = cur + (1 if need > cur else -1 if need < cur else 0)
        inst.append(cur)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    n = len(load)
    px = lambda i: x0 + 50 + i * (W * 0.62) / (n - 1)
    mx = max(max(load), hi * per)
    py = lambda y: y1 - 30 - y / mx * (H - 60)
    _axes(cv, x0 + 50, y0, x0 + 60 + W * 0.62, y1 - 30, step, "time", "requests/s")
    step += 1
    cv.line([(px(i), py(L)) for i, L in enumerate(load)], step, "#e03131", 3)
    cv.text(px(3), py(load[3]) - 8, "load", step, 18, "#e03131", "center", "bottom", math=False)
    step += 1
    for i, k in enumerate(inst):
        cap = k * per
        cv.rect(px(i) - 14, py(cap), 28, py(0) - py(cap), step, "#a5d8ff", "#1971c2", 1.5, rounded=False)
        cv.text(px(i), py(cap) - 4, str(k), step, 16, "#1971c2", "center", "bottom", math=False)
        step += 1
    cv.text(x0 + W * 0.68, y0 + 30, f"blue bars = capacity\n(instances × {per:g} req/s)\n\nscale out when busy,\n"
                                   f"scale in when quiet\n(min {lo}, max {hi})", step, 19, INK, math=False)
    return step + 1


# ======================================================================== ML / data

def gini(rows, label):
    n = len(rows)
    if not n:
        return 0
    counts = {}
    for r in rows:
        counts[r[label]] = counts.get(r[label], 0) + 1
    return 1 - sum((c / n) ** 2 for c in counts.values())


def build_decision_tree(cv, v, box, step):
    rows = v.get("rows") or [{"outlook": "sunny", "windy": "no", "play": "yes"}, {"outlook": "sunny", "windy": "yes", "play": "no"},
                             {"outlook": "rain", "windy": "yes", "play": "no"}, {"outlook": "rain", "windy": "no", "play": "yes"},
                             {"outlook": "overcast", "windy": "yes", "play": "yes"}, {"outlook": "overcast", "windy": "no", "play": "yes"},
                             {"outlook": "rain", "windy": "yes", "play": "no"}, {"outlook": "sunny", "windy": "no", "play": "yes"}]
    label = v.get("label") or "play"
    feats = [k for k in rows[0] if k != label]

    def best(rs, fs):
        scores = {}
        for f in fs:
            vals = sorted({r[f] for r in rs})
            w = sum(len([r for r in rs if r[f] == x]) / len(rs) * gini([r for r in rs if r[f] == x], label) for x in vals)
            scores[f] = w
        return min(scores, key=scores.get), scores

    def grow(rs, fs, depth):
        labels = {r[label] for r in rs}
        if len(labels) == 1 or not fs or depth > 2:
            maj = max(sorted(labels), key=lambda l: sum(r[label] == l for r in rs))
            return {"label": f"{label} = {maj}\n({len(rs)} rows)"}
        f, sc = best(rs, fs)
        kids = []
        for x in sorted({r[f] for r in rs}):
            sub = [r for r in rs if r[f] == x]
            k = grow(sub, [g for g in fs if g != f], depth + 1)
            k["label"] = f"{f} = {x}\n" + k["label"] if "children" not in k else f"{f} = {x} → " + k["label"]
            kids.append(k)
        return {"label": f"split on {f}\n(Gini {sc[f]:.2f})", "children": kids}
    root = grow(rows, feats, 0)
    f0, sc0 = best(rows, feats)
    x0, y0, x1, y1 = box
    W = x1 - x0
    cols = feats + [label]
    _table(cv, x0, y0 + 26, "training data", cols, [[r[c] for c in cols] for r in rows], step, W * 0.36, size=14)
    step += 1
    cv.text(x0, y1 - 60, "weighted Gini after a split:\n" + ", ".join(f"{k} {s:.2f}" for k, s in sc0.items()) +
            f"\n→ split on {f0} (lowest)", step, 16, INK, math=False)
    step += 1
    return visuals.build_tree(cv, {"root": root}, (x0 + W * 0.4, y0, x1, y1), step)


def build_knn(cv, v, box, step):
    rnd = random.Random(3)
    a = [(rnd.gauss(2.5, 0.8), rnd.gauss(3, 0.8)) for _ in range(10)]
    b = [(rnd.gauss(5.5, 0.8), rnd.gauss(5.5, 0.8)) for _ in range(10)]
    q = tuple(v.get("query") or (4.2, 4.3))
    k = int(v.get("k") or 5)
    pts = [(p, "A") for p in a] + [(p, "B") for p in b]
    near = sorted(pts, key=lambda pl: math.dist(pl[0], q))[:k]
    votes = {"A": sum(l == "A" for _, l in near), "B": sum(l == "B" for _, l in near)}
    win = max(votes, key=votes.get)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    sx = lambda x: x0 + 30 + x / 8 * W * 0.6
    sy = lambda y: y1 - 20 - y / 8 * (H - 40)
    for (x, y), l in pts:
        if l == "A":
            cv.ellipse(sx(x) - 9, sy(y) - 9, 18, 18, step, "#a5d8ff", "#1971c2", 2)
        else:
            cv.rect(sx(x) - 9, sy(y) - 9, 18, 18, step, "#ffc9c9", "#e03131", 2, rounded=False)
    cv.text(sx(q[0]), sy(q[1]), "?", step, 30, INK, "center", "middle", math=False)
    step += 1
    for i, ((x, y), l) in enumerate(near):
        cv.line([(sx(q[0]), sy(q[1])), (sx(x), sy(y))], step, "#f08c00", 2, dashed=True)
        cv.text(x0 + W * 0.68, y0 + 40 + i * 30, f"{i + 1}. class {l}, distance {math.dist((x, y), q):.2f}", step, 18,
                "#1971c2" if l == "A" else "#e03131", math=False)
        step += 1
    r = math.dist(near[-1][0], q)
    cv.ellipse(sx(q[0]) - r / 8 * W * 0.6, sy(q[1]) - r / 8 * (H - 40), 2 * r / 8 * W * 0.6, 2 * r / 8 * (H - 40), step,
               "transparent", "#f08c00", 1.5)
    cv.text(x0 + W * 0.68, y0 + 60 + k * 30, f"k = {k}: A {votes['A']} votes, B {votes['B']}\n→ predict class {win}", step, 22,
            "#2f9e44", math=False)
    return step + 1


def build_normal_dist(cv, v, box, step):
    mu, sd = float(v.get("mean") or 100), float(v.get("sd") or 15)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    f = lambda z: math.exp(-z * z / 2)
    px = lambda z: x0 + 40 + (z + 3.6) / 7.2 * (W - 80)
    py = lambda y: y1 - 50 - y * (H - 110)
    for k, (band, pct, col) in enumerate(((3, "99.7%", "#e7f5ff"), (2, "95%", "#a5d8ff"), (1, "68%", "#4dabf7"))):
        pts = [(px(z), py(f(z))) for z in [-band + 2 * band * i / 40 for i in range(41)]]
        cv.polygon([(px(-band), py(0))] + pts + [(px(band), py(0))], step, col, "transparent", 1)
        cv.text(px(0), py(f(0)) + 30 + k * 0 - 0, "", step, 10, INK, math=False)
        cv.line([(px(-band), py(0) + 10 + k * 14), (px(band), py(0) + 10 + k * 14)], step, col if k else "#74c0fc", 3)
        cv.text(px(band) + 6, py(0) + 10 + k * 14, pct, step, 15, INK, "left", "middle", math=False)
        step += 1
    cv.line([(px(z), py(f(z))) for z in [-3.6 + 7.2 * i / 80 for i in range(81)]], step, INK, 3)
    for z in range(-3, 4):
        cv.text(px(z), py(0) + 56, f"{mu + z * sd:g}", step, 15, MUTED, "center", "top", math=False)
    cv.text(x0, y0, f"normal distribution, mean {mu:g}, standard deviation {sd:g}", step, 21, "#e8590c", math=False)
    return step + 1


def build_softmax(cv, v, box, step):
    labels = v.get("labels") or ["cat", "dog", "bird", "fish"]
    logits = [float(x) for x in (v.get("logits") or [2.0, 1.0, 0.1, -1.0])]
    ex = [math.exp(z) for z in logits]
    probs = [e / sum(ex) for e in ex]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    n = len(labels)
    rh = (H - 60) / n
    cv.text(x0, y0, "logit", step, 18, MUTED, math=False)
    cv.text(x0 + W * 0.22, y0, "e^logit", step, 18, MUTED, math=False)
    cv.text(x0 + W * 0.42, y0, "probability = e^z / Σ e^z", step, 18, MUTED, math=False)
    step += 1
    for i, (lab, z, e, p) in enumerate(zip(labels, logits, ex, probs)):
        y = y0 + 40 + i * rh
        cv.text(x0, y + rh / 3, f"{lab}: {z:g}", step, 20, INK, "left", "middle", math=False)
        cv.text(x0 + W * 0.22, y + rh / 3, f"{e:.2f}", step, 20, INK, "left", "middle", math=False)
        cv.rect(x0 + W * 0.42, y, (W * 0.45) * p, rh * 0.6, step, PAL[i % len(PAL)], STRONG[i % len(STRONG)], 1.5)
        cv.text(x0 + W * 0.42 + W * 0.45 * p + 8, y + rh * 0.3, f"{p:.1%}", step, 20, INK, "left", "middle", math=False)
        step += 1
    cv.text(x0 + W * 0.42, y1 - 20, f"Σ e^z = {sum(ex):.2f}; probabilities add to 100%", step, 17, "#1971c2", "left",
            "bottom", math=False)
    return step + 1


def build_overfitting(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    train = lambda e: 0.9 * math.exp(-e / 12) + 0.03
    val = lambda e: 0.9 * math.exp(-e / 12) + 0.08 + 0.0012 * max(0, e - 18) ** 1.5
    E = 60
    best = min(range(E), key=val)
    px = lambda e: x0 + 50 + e / E * W * 0.62
    py = lambda y: y1 - 30 - y / 1.05 * (H - 60)
    _axes(cv, x0 + 50, y0, x0 + 60 + W * 0.62, y1 - 30, step, "epochs", "loss")
    step += 1
    cv.line([(px(e), py(train(e))) for e in range(E)], step, "#1971c2", 3)
    cv.text(px(E - 1), py(train(E - 1)) - 6, "training", step, 18, "#1971c2", "right", "bottom", math=False)
    step += 1
    cv.line([(px(e), py(val(e))) for e in range(E)], step, "#e03131", 3)
    cv.text(px(E - 1), py(val(E - 1)) - 6, "validation", step, 18, "#e03131", "right", "bottom", math=False)
    step += 1
    cv.line([(px(best), y0 + 20), (px(best), y1 - 30)], step, "#2f9e44", 2, dashed=True)
    cv.text(px(best) + 6, y0 + 24, f"stop here (epoch {best})", step, 17, "#2f9e44", math=False)
    cv.text(px(best * 0.3), y1 - 40, "underfitting", step, 16, MUTED, "center", "bottom", math=False)
    cv.text(px(best + (E - best) * 0.55), y1 - 40, "overfitting: memorising the training set", step, 16, MUTED,
            "center", "bottom", math=False)
    cv.text(x0 + W * 0.68, y0 + 40, "fixes: early stopping,\nmore data, regularization\n(L2, dropout), simpler model",
            step, 19, INK, math=False)
    return step + 1


def build_er_diagram(cv, v, box, step):
    ents = v.get("entities") or {"Student": ["student_id PK", "name", "email"],
                                 "Enrollment": ["student_id FK", "course_id FK", "grade"],
                                 "Course": ["course_id PK", "title", "units"]}
    rels = v.get("relationships") or [["Student", "Enrollment", "1", "N"], ["Course", "Enrollment", "1", "N"]]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    names = list(ents)
    bw = min(250, W / len(names) - 60)
    pos = {}
    for i, n in enumerate(names):
        x = x0 + i * (W / len(names)) + 20
        y = y0 + 40 + (60 if i % 2 else 0)
        attrs = ents[n]
        h = 50 + 34 * len(attrs)
        cv.rect(x, y, bw, h, step, "#ffffff", INK, 2.5, rounded=False)
        cv.rect(x, y, bw, 44, step, "#d0ebff", INK, 2.5, rounded=False)
        cv.text(x + bw / 2, y + 22, n, step, 21, INK, "center", "middle", math=False)
        for k, a in enumerate(attrs):
            col = "#e03131" if "PK" in a else "#1971c2" if "FK" in a else INK
            cv.text(x + 12, y + 56 + k * 34, a, step, 18, col, math=False)
        pos[n] = (x, y, bw, h)
        step += 1
    for a, b, ca, cb in rels:
        A, B = pos[a], pos[b]
        ax = A[0] + A[2] if A[0] < B[0] else A[0]
        bx = B[0] if A[0] < B[0] else B[0] + B[2]
        ay, by = A[1] + A[3] / 2, B[1] + B[3] / 2
        cv.line([(ax, ay), (bx, by)], step, INK, 2)
        cv.text(ax + (12 if ax < bx else -12), ay - 6, ca, step, 20, "#e8590c", "center", "bottom", math=False)
        cv.text(bx + (-12 if ax < bx else 12), by - 6, cb, step, 20, "#e8590c", "center", "bottom", math=False)
        step += 1
    cv.text(x0, y1 - 30, "PK = primary key (red), FK = foreign key (blue); 1 student → N enrollments; the join table "
                         "makes many-to-many", step, 17, MUTED, math=False)
    return step + 1


def build_groupby(cv, v, box, step):
    rows = v.get("rows") or [["CS", "Ana", 92], ["Math", "Ben", 78], ["CS", "Chen", 85], ["Bio", "Dee", 88],
                             ["Math", "Eli", 95], ["CS", "Fay", 70]]
    cols = v.get("columns") or ["dept", "student", "score"]
    x0, y0, x1, y1 = box
    W = x1 - x0
    el = cv.text(x0, y0, 'df.groupby("dept")["score"].agg(["count", "mean", "max"])', step, 18, "#1971c2", math=False)
    if el:
        el["fontFamily"] = 3
    _table(cv, x0, y0 + 70, "df", cols, rows, step, W * 0.3, size=15)
    step += 1
    groups = {}
    for r in rows:
        groups.setdefault(r[0], []).append(r)
    gy = y0 + 70
    for i, (g, rs) in enumerate(sorted(groups.items())):
        _table(cv, x0 + W * 0.36, gy, f"group {g}", cols, rs, step, W * 0.27, size=14)
        gy += (len(rs) + 1) * 32 + 34
        step += 1
    out = [[g, len(rs), round(sum(r[2] for r in rs) / len(rs), 1), max(r[2] for r in rs)] for g, rs in sorted(groups.items())]
    _table(cv, x0 + W * 0.69, y0 + 70, "result", ["dept", "count", "mean", "max"], out, step, W * 0.3, size=15)
    return step + 1


# ======================================================================== biology

def build_punnett(cv, v, box, step):
    p1, p2 = str(v.get("parent1") or "Aa"), str(v.get("parent2") or "Aa")
    dom = str(v.get("dominant") or "purple flowers")
    rec = str(v.get("recessive") or "white flowers")
    g1, g2 = list(p1), list(p2)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    c = min(120, (H - 80) / 3)
    gx, gy = x0 + 140, y0 + 60
    cv.text(x0, y0, f"cross {p1} × {p2}", step, 24, "#e8590c", math=False)
    for j, a in enumerate(g2):
        cv.text(gx + (j + 0.5) * c, gy - 10, a, step, 28, "#1971c2", "center", "bottom", math=False)
    for i, a in enumerate(g1):
        cv.text(gx - 14, gy + (i + 0.5) * c, a, step, 28, "#e03131", "right", "middle", math=False)
    step += 1
    kids = []
    for i, a in enumerate(g1):
        for j, b in enumerate(g2):
            kid = "".join(sorted(a + b, key=lambda ch: (ch.lower(), ch.islower())))
            kids.append(kid)
            isdom = any(ch.isupper() for ch in kid)
            _box(cv, gx + j * c, gy + i * c, c - 6, c - 6, step, kid, "#d0bfff" if isdom else "#ffffff", 30)
            step += 1
    geno = {}
    for kd in kids:
        geno[kd] = geno.get(kd, 0) + 1
    domc = sum(n for kd, n in geno.items() if any(ch.isupper() for ch in kd))
    cv.text(x0 + W * 0.55, y0 + 60, "genotypes: " + ", ".join(f"{k} {n}/4" for k, n in sorted(geno.items())) +
            f"\n\nphenotypes:\n{dom}: {domc}/4 ({domc / 4:.0%})\n{rec}: {4 - domc}/4 ({(4 - domc) / 4:.0%})\n\n"
            f"ratio {domc}:{4 - domc}", step, 22, INK, math=False)
    return step + 1


def build_meiosis(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    R = min(70, H * 0.17)
    cy = y0 + H * 0.45
    stages = [("parent cell (2n)", [("#e03131", "#e03131"), ("#1971c2", "#1971c2")], 1),
              ("crossing over", [("#e03131", "#1971c2"), ("#1971c2", "#e03131")], 1),
              ("meiosis I: 2 cells (n)", [("#e03131", "#1971c2")], 2),
              ("meiosis II: 4 gametes (n)", [("#e03131",)], 4)]
    sw = W / len(stages)
    for k, (name, chroms, ncell) in enumerate(stages):
        cx = x0 + sw * (k + 0.5)
        for c in range(ncell):
            yy = cy + (c - (ncell - 1) / 2) * (R * 2.2 if ncell > 1 else 0) * (0.55 if ncell == 4 else 1)
            r = R * (0.5 if ncell == 4 else 0.75 if ncell == 2 else 1)
            _poly(cv, [(cx + r * math.cos(t), yy + r * math.sin(t)) for t in [i * math.pi / 14 for i in range(29)]], step,
                  "#e8590c", "#fff4e6", 2.5)
            for j, pair in enumerate(chroms):
                xx = cx - r * 0.3 + j * r * 0.6 / max(1, len(chroms) - 1 or 1)
                if k == 3:
                    col = ["#e03131", "#1971c2", "#e03131", "#1971c2"][c]
                    other = ["#1971c2", "#e03131", "#e03131", "#1971c2"][c]
                    cv.line([(cx, yy - r * 0.5), (cx, yy)], step, col, 5)
                    cv.line([(cx, yy), (cx, yy + r * 0.5)], step, other, 5)
                    continue
                for s, col in enumerate(pair):
                    off = (s - (len(pair) - 1) / 2) * 10
                    cv.line([(xx + off, yy - r * 0.55), (xx + off, yy)], step, col, 5)
                    cv.line([(xx + off, yy), (xx + off, yy + r * 0.55)], step,
                            pair[1 - s] if (k >= 1 and len(pair) == 2) else col, 5)
        cv.text(cx, y1 - 40, name, step, 18, INK, "center", "top", math=False)
        if k:
            cv.arrow(cx - sw + R * 1.1, cy, cx - R * 1.1, cy, step)
        step += 1
    cv.text(x0, y0, "meiosis makes 4 genetically different sex cells with half the chromosomes", step, 19, "#e8590c",
            math=False)
    return step + 1


def build_respiration(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    st = [("Glucose", "C₆H₁₂O₆", "#ffec99"), ("Glycolysis", "cytoplasm\n→ 2 pyruvate\n+2 ATP", "#ffd8a8"),
          ("Krebs cycle", "mitochondrial matrix\nCO₂ released\n+2 ATP, NADH, FADH₂", "#ffc9c9"),
          ("Electron transport chain", "inner membrane\nO₂ + H⁺ → H₂O\n+~34 ATP", "#d0bfff")]
    bw = (W - 60 * 3) / 4
    for i, (name, sub, col) in enumerate(st):
        x = x0 + i * (bw + 60)
        cv.rect(x, y0 + 60, bw, 180, step, col)
        cv.text(x + bw / 2, y0 + 80, name, step, 20, INK, "center", "top", math=False)
        s, sz = fit_text(sub, bw - 16, 110, 17, 12)
        cv.text(x + bw / 2, y0 + 120, s, step, sz, INK, "center", "top", math=False)
        if i:
            cv.arrow(x - 56, y0 + 150, x - 4, y0 + 150, step)
        step += 1
    total = [2, 2, 34]
    for k in range(3):
        x = x0 + (k + 1) * (bw + 60) + bw / 2
        for a in range(min(total[k], 8)):
            cv.text(x - 60 + (a % 4) * 32, y0 + 270 + (a // 4) * 32, "⚡", step, 22, "#f08c00", "left", "top", math=False)
        if total[k] > 8:
            cv.text(x + 70, y0 + 280, f"× {total[k]}", step, 20, "#f08c00", math=False)
        step += 1
    cv.text(x0, y1 - 40, "C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + about 38 ATP (36-38 in textbooks)", step, 22, INK, math=False)
    return step + 1


def build_osmosis(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cases = [("hypertonic\n(salty outside)", 0.6, "out", "cell shrinks"),
             ("isotonic", 1.0, "both", "no net change"), ("hypotonic\n(fresh water)", 1.35, "in", "cell swells / bursts")]
    bw = W / 3
    for i, (name, scale, flow, res) in enumerate(cases):
        cx, cy = x0 + bw * (i + 0.5), y0 + H * 0.42
        cv.rect(cx - bw * 0.42, y0 + 30, bw * 0.84, H * 0.7, step, "#e7f5ff", "#1971c2", 2)
        r = min(bw, H) * 0.16
        _poly(cv, [(cx + r * scale * math.cos(t) * (1 + 0.06 * math.sin(5 * t) * (scale < 1)),
                    cy + r * scale * math.sin(t)) for t in [k * math.pi / 18 for k in range(37)]], step, "#e03131",
              "#ffc9c9", 3)
        rnd = random.Random(i)
        for _ in range({0: 10, 1: 5, 2: 1}[i]):
            cv.text(cx + rnd.uniform(-bw * 0.35, bw * 0.35), y0 + 50 + rnd.uniform(0, H * 0.12), "•", step, 18, "#868e96",
                    math=False)
        for d in range(4):
            a = d * math.pi / 2 + 0.6
            ox, oy = cx + math.cos(a) * r * 2.1, cy + math.sin(a) * r * 2.1
            ix, iy = cx + math.cos(a) * r * 1.2 * scale, cy + math.sin(a) * r * 1.2 * scale
            if flow == "out":
                cv.arrow(ix, iy, ox, oy, step, color="#1971c2")
            elif flow == "in":
                cv.arrow(ox, oy, ix, iy, step, color="#1971c2")
            else:
                cv.line([(ix, iy), (ox, oy)], step, "#1971c2", 2, arrow_end=True, arrow_start=True)
        cv.text(cx, y0 + 30 + H * 0.7 + 8, f"{name}\n{res}", step, 18, INK, "center", "top", math=False)
        step += 1
    cv.text(x0, y0, "water moves across the membrane toward the side with more dissolved solute", step, 19, "#e8590c",
            math=False)
    return step + 1


# ======================================================================== earth / space

def build_plate_tectonics(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    pw = W / 3 - 20
    titles = [("divergent", "plates move apart:\nmid-ocean ridges, rift valleys"),
              ("convergent", "plates collide: one subducts;\ntrenches, volcanoes, mountains"),
              ("transform", "plates slide past:\nearthquakes (San Andreas)")]
    for i, (name, note) in enumerate(titles):
        x = x0 + i * (pw + 30)
        my = y0 + H * 0.4
        cv.rect(x, my + 50, pw, H * 0.3, step, "#ffa94d", "#e8590c", 2, rounded=False)       # mantle
        cv.text(x + pw / 2, my + 50 + H * 0.15, "mantle", step, 16, "#ffffff", "center", "middle", math=False)
        if i == 0:
            cv.rect(x, my, pw / 2 - 10, 50, step, "#ced4da", INK, 2, rounded=False)
            cv.rect(x + pw / 2 + 10, my, pw / 2 - 10, 50, step, "#ced4da", INK, 2, rounded=False)
            cv.arrow(x + pw / 2 - 30, my - 20, x + 20, my - 20, step, color="#e03131")
            cv.arrow(x + pw / 2 + 30, my - 20, x + pw - 20, my - 20, step, color="#e03131")
            cv.line([(x + pw / 2, my + 120), (x + pw / 2, my + 10)], step, "#e03131", 3, arrow_end=True)
        elif i == 1:
            cv.polygon([(x, my), (x + pw / 2 + 20, my), (x + pw / 2 + 60, my + 90), (x + pw / 2 + 20, my + 100),
                        (x + pw / 2 - 20, my + 50), (x, my + 50)], step, "#ced4da", INK, 2)
            cv.polygon([(x + pw / 2 + 25, my - 10), (x + pw, my - 10), (x + pw, my + 50), (x + pw / 2 + 25, my + 50)],
                       step, "#adb5bd", INK, 2)
            cv.polygon([(x + pw * 0.62, my - 10), (x + pw * 0.72, my - 60), (x + pw * 0.82, my - 10)], step, "#e8590c",
                       INK, 2)
            cv.arrow(x + 20, my - 20, x + pw / 2 - 20, my - 20, step, color="#e03131")
            cv.arrow(x + pw - 20, my - 30, x + pw / 2 + 60, my - 30, step, color="#e03131")
        else:
            cv.rect(x + 10, my - 40, pw - 20, 50, step, "#ced4da", INK, 2, rounded=False)
            cv.rect(x + 10, my + 10, pw - 20, 40, step, "#adb5bd", INK, 2, rounded=False)
            cv.arrow(x + 30, my - 15, x + pw - 40, my - 15, step, color="#e03131")
            cv.arrow(x + pw - 30, my + 30, x + 40, my + 30, step, color="#e03131")
        cv.text(x + pw / 2, y0, name, step, 24, "#e8590c", "center", "top", math=False)
        cv.text(x + pw / 2, y1 - 54, note, step, 16, INK, "center", "top", math=False)
        step += 1
    return step


def build_moon_phases(cv, v, box, step):
    names = ["new", "waxing crescent", "first quarter", "waxing gibbous", "full", "waning gibbous", "last quarter",
             "waning crescent"]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    r = min(W / 8 / 2.8, H * 0.12)
    for i, nm in enumerate(names):
        cx, cy = x0 + W * (i + 0.5) / 8, y0 + H * 0.35
        frac = (1 - math.cos(2 * math.pi * i / 8)) / 2           # illuminated fraction (computed)
        cv.ellipse(cx - r, cy - r, 2 * r, 2 * r, step, "#343a40", INK, 1.5)
        k = 1 - 2 * frac                                          # terminator ellipse x-scale
        right = i <= 4                                            # waxing: lit on the right
        pts = []
        for j in range(25):
            t = -math.pi / 2 + math.pi * j / 24
            pts.append((cx + (r if right else -r) * math.cos(t), cy + r * math.sin(t)))
        for j in range(25):
            t = math.pi / 2 - math.pi * j / 24
            pts.append((cx + (r if right else -r) * k * math.cos(t), cy + r * math.sin(t)))
        if frac > 0.01:
            cv.polygon(pts, step, "#fff3bf", "transparent", 1)
        cv.text(cx, cy + r + 10, f"{nm}\n{frac:.0%} lit", step, 15, INK, "center", "top", math=False)
        step += 1
    cv.text(x0, y1 - 30, "the Moon is always half lit by the Sun; the phase is how much of that half faces Earth "
                         "(29.5-day cycle)", step, 17, "#1971c2", math=False)
    return step + 1


def build_earth_layers(cv, v, box, step):
    layers = [("crust", "5-70 km, solid rock", "#8ce99a"), ("mantle", "~2,900 km, hot plastic rock", "#ffa94d"),
              ("outer core", "~2,200 km, liquid iron-nickel", "#ff8787"), ("inner core", "~1,220 km radius, solid iron", "#ffd43b")]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    R = min(W * 0.3, H * 0.47)
    cx, cy = x0 + R + 20, (y0 + y1) / 2
    radii = [1.0, 0.97, 0.55, 0.2]
    for i, (name, note, col) in enumerate(layers):
        r = R * radii[i]
        cv.ellipse(cx - r, cy - r, 2 * r, 2 * r, step, col, INK, 2)
        lx = x0 + 2 * R + 80
        ly = y0 + 40 + i * 70
        cv.line([(cx + r * 0.7, cy - r * 0.7 + (0 if i else 0)), (lx - 8, ly + 12)], step, MUTED, 1)
        cv.text(lx, ly, f"{name}: {note}", step, 20, INK, math=False)
        step += 1
    return step


# ======================================================================== physics

def build_lens(cv, v, box, step):
    f = float(v.get("focal") or 10)
    do = float(v.get("object") or 25)
    ho = float(v.get("height") or 6)
    di = 1 / (1 / f - 1 / do)
    hi = -ho * di / do
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    span = do + max(di, 0) + 10
    s = W * 0.66 / span
    lx, ay = x0 + 20 + (do + 5) * s, (y0 + y1) / 2
    hs = min(s, (H * 0.42) / max(ho, abs(hi)))
    X = lambda d: lx + d * s
    Y = lambda h: ay - h * hs
    cv.line([(x0, ay), (x0 + W * 0.7, ay)], step, MUTED, 1.5)
    cv.ellipse(lx - 12, ay - H * 0.4, 24, H * 0.8, step, "#d0ebff", "#1971c2", 2)
    for d in (-f, f):
        cv.ellipse(X(d) - 5, ay - 5, 10, 10, step, INK, INK, 1)
        cv.text(X(d), ay + 8, "F", step, 16, INK, "center", "top", math=False)
    cv.line([(X(-do), ay), (X(-do), Y(ho))], step, "#2f9e44", 4, arrow_end=True)
    cv.text(X(-do), ay + 8, "object", step, 16, "#2f9e44", "center", "top", math=False)
    step += 1
    cv.line([(X(-do), Y(ho)), (lx, Y(ho)), (X(di), Y(hi))], step, "#f08c00", 2)
    cv.elements[-1]["roundness"] = None
    step += 1
    cv.line([(X(-do), Y(ho)), (X(di), Y(hi))], step, "#e03131", 2)
    step += 1
    cv.line([(X(di), ay), (X(di), Y(hi))], step, "#7048e8", 4, arrow_end=True)
    cv.text(X(di), ay + 8 if hi < 0 else ay - 8, "image", step, 16, "#7048e8", "center", "top" if hi < 0 else "bottom",
            math=False)
    cv.text(x0 + W * 0.72, y0 + 30, f"1/f = 1/dₒ + 1/dᵢ\n1/{f:g} = 1/{do:g} + 1/dᵢ\ndᵢ = {di:.1f} cm\n"
                                   f"magnification = −dᵢ/dₒ = {-di / do:.2f}\n{'real, inverted' if di > 0 else 'virtual, upright'}"
                                   f"{', smaller' if abs(di / do) < 1 else ', larger'}", step, 20, INK, math=False)
    return step + 1


def build_pendulum(cv, v, box, step):
    L = float(v.get("length") or 1.0)
    g = 9.8
    T = 2 * math.pi * math.sqrt(L / g)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    px, py = x0 + W * 0.33, y0 + 20
    Lp = H * 0.75
    cv.line([(px - 80, py), (px + 80, py)], step, INK, 4)
    cv.line([(px, py), (px, py + Lp)], step, MUTED, 1, dashed=True)
    step += 1
    th0 = 0.5
    n = 16
    for k in range(n + 1):
        th = th0 * math.cos(2 * math.pi * k / n)
        bx, by = px + Lp * math.sin(th), py + Lp * math.cos(th)
        tr = _Track(cv)
        cv.line([(px, py), (bx, by)], step, INK, 2)
        cv.ellipse(bx - 22, by - 22, 44, 44, step, "#ff8787", "#c92a2a", 2)
        tr.tag(until=step + 1 if k < n else None, fade=True)
        step += 1
    cv.text(x0 + W * 0.62, y0 + 40, f"period T = 2π √(L/g)\n= 2π √({L:g} / 9.8)\n= {T:.2f} s\n\n"
                                   "doesn't depend on the mass\n(small swings)", step, 22, INK, math=False)
    return step + 1


def build_energy_ramp(cv, v, box, step):
    m, h0 = float(v.get("mass") or 2), float(v.get("height") or 5)
    g = 9.8
    E = m * g * h0
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    top, bot = (x0 + 20, y0 + 40), (x0 + W * 0.5, y1 - 30)
    cv.line([top, (x0 + W * 0.12, y0 + H * 0.55), bot, (x0 + W * 0.6, y1 - 30)], step, INK, 3)
    step += 1
    n = 6
    for k in range(n + 1):
        f = k / n
        xx = top[0] + (bot[0] - top[0]) * f
        hh = h0 * (1 - f) ** 1.5 if f < 1 else 0
        yy = bot[1] - (bot[1] - top[1]) * hh / h0
        pe = m * g * hh
        ke = E - pe
        tr = _Track(cv)
        cv.ellipse(xx - 16, yy - 32, 32, 32, step, "#4dabf7", "#1971c2", 2)
        bx = x0 + W * 0.68
        for j, (lab, val, col) in enumerate((("PE = mgh", pe, "#4dabf7"), ("KE = ½mv²", ke, "#ff8787"))):
            hgt = (H - 120) * val / E
            cv.rect(bx + j * 110, y1 - 60 - hgt, 80, hgt, step, col, INK, 1.5, rounded=False)
            cv.text(bx + j * 110 + 40, y1 - 54, f"{lab}\n{val:.0f} J", step, 15, INK, "center", "top", math=False)
        cv.text(bx + 230, y0 + 60, f"total\n{E:.0f} J", step, 20, "#2f9e44", math=False)
        if k == n:
            cv.text(bx - 20, y0 + 10, f"v at bottom = √(2gh) = {math.sqrt(2 * g * h0):.1f} m/s", step, 18, INK, math=False)
        tr.tag(until=step + 1 if k < n else None, fade=k > 0)
        step += 1
    return step


# ======================================================================== chemistry

def _atoms(formula):
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        out[el] = out.get(el, 0) + int(n or 1)
    return out


def build_balance(cv, v, box, step):
    eq = str(v.get("equation") or "CH4 + 2O2 -> CO2 + 2H2O")
    left, right = [s.strip() for s in re.split(r"->|→|=", eq)][:2]

    def side(s):
        tot = {}
        terms = []
        for t in s.split("+"):
            t = t.strip()
            m = re.match(r"(\d*)(.+)", t)
            k, f = int(m.group(1) or 1), m.group(2)
            terms.append((k, f))
            for el, n in _atoms(f).items():
                tot[el] = tot.get(el, 0) + k * n
        return terms, tot
    lt, la = side(left)
    rt, ra = side(right)
    x0, y0, x1, y1 = box
    W = x1 - x0
    pretty = lambda s: re.sub(r"(?<=[A-Za-z)])(\d+)", lambda m: "".join("₀₁₂₃₄₅₆₇₈₉"[int(d)] for d in m.group(1)), s)
    cv.text(x0, y0, pretty(left) + "  →  " + pretty(right), step, 32, INK, math=False)
    step += 1
    els = sorted(set(la) | set(ra), key=lambda e: "CHON".find(e) if e in "CHON" else 9)
    for i, el in enumerate(els):
        y = y0 + 80 + i * 56
        ok = la.get(el, 0) == ra.get(el, 0)
        cv.text(x0, y, el, step, 26, INK, math=False)
        for side_i, cnt in enumerate((la.get(el, 0), ra.get(el, 0))):
            bx = x0 + 80 + side_i * W * 0.42
            for a in range(min(cnt, 12)):
                cv.ellipse(bx + a * 30, y, 24, 24, step, {"C": "#868e96", "H": "#f1f3f5", "O": "#ff8787", "N": "#74c0fc"}.get(el, "#ffec99"),
                           INK, 1.5)
            cv.text(bx + min(cnt, 12) * 30 + 10, y + 12, str(cnt), step, 20, INK, "left", "middle", math=False)
        cv.text(x0 + W * 0.88, y + 12, "✔" if ok else "✘", step, 28, "#2f9e44" if ok else "#e03131", "left", "middle",
                math=False)
        step += 1
    bal = la == ra
    cv.text(x0, y0 + 90 + len(els) * 56, "balanced: atoms are conserved" if bal else "not balanced yet: change the "
            "coefficients (big numbers), never the subscripts", step, 21, "#2f9e44" if bal else "#e03131", math=False)
    if not bal:                                       # find the smallest whole-number coefficients (computed)
        import itertools
        forms = [f for _, f in lt] + [f for _, f in rt]
        nl = len(lt)
        best = None
        for co in itertools.product(range(1, 9), repeat=len(forms)):
            tl, tr2 = {}, {}
            for k, f in zip(co[:nl], forms[:nl]):
                for el, n in _atoms(f).items():
                    tl[el] = tl.get(el, 0) + k * n
            for k, f in zip(co[nl:], forms[nl:]):
                for el, n in _atoms(f).items():
                    tr2[el] = tr2.get(el, 0) + k * n
            if tl == tr2:
                best = co
                break
        if best:
            fmt = lambda cs, fs: " + ".join((str(c) if c > 1 else "") + f for c, f in zip(cs, fs))
            step += 1
            cv.text(x0, y0 + 140 + len(els) * 56, "balanced:  " + pretty(fmt(best[:nl], forms[:nl])) + "  →  " +
                    pretty(fmt(best[nl:], forms[nl:])), step, 28, "#2f9e44", math=False)
    return step + 1


def build_reaction_energy(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    exo = str(v.get("kind") or "exothermic").startswith("exo")
    r_e, p_e, ea = (0.5, 0.2, 0.95) if exo else (0.25, 0.55, 0.95)
    px = lambda t: x0 + 50 + t * W * 0.62
    py = lambda e: y1 - 30 - e * (H - 60)
    _axes(cv, x0 + 50, y0, x0 + 60 + W * 0.62, y1 - 30, step, "reaction progress", "energy")
    step += 1

    def curve(peak):
        pts = []
        for i in range(61):
            t = i / 60
            base = r_e + (p_e - r_e) * (3 * t * t - 2 * t ** 3)
            pts.append((px(t), py(base + (peak - max(r_e, p_e)) * math.exp(-((t - 0.5) / 0.13) ** 2))))
        return pts
    cv.line(curve(ea), step, "#e03131", 3)
    cv.text(px(0.5), py(ea) - 6, "without catalyst", step, 16, "#e03131", "center", "bottom", math=False)
    step += 1
    cv.line(curve(ea * 0.78), step, "#2f9e44", 3, dashed=True)
    cv.text(px(0.62), py(ea * 0.78) - 4, "with catalyst", step, 16, "#2f9e44", "left", "bottom", math=False)
    step += 1
    cv.line([(px(0.5), py(r_e)), (px(0.5), py(ea))], step, "#1971c2", 2, arrow_end=True, arrow_start=True)
    cv.text(px(0.5) + 6, py((r_e + ea) / 2), "activation\nenergy Eₐ", step, 16, "#1971c2", "left", "middle", math=False)
    cv.line([(px(0.95), py(r_e)), (px(0.95), py(p_e))], step, "#e8590c", 2, arrow_end=True)
    cv.text(px(0.95) + 6, py((r_e + p_e) / 2), "ΔH < 0\nreleases heat" if exo else "ΔH > 0\nabsorbs heat", step, 16,
            "#e8590c", "left", "middle", math=False)
    cv.text(px(0.02), py(r_e) - 6, "reactants", step, 16, INK, "left", "bottom", math=False)
    cv.text(px(0.85), py(p_e) - 6, "products", step, 16, INK, "right", "bottom", math=False)
    return step + 1


# ======================================================================== maths / business

def build_tangent(cv, v, box, step):
    expr = str(v.get("expr") or "x^2/4")
    f = visuals._make_fn(expr) or (lambda x: x * x / 4)
    a, b = float(v.get("a") or -4), float(v.get("b") or 4)
    d = lambda x: (f(x + 1e-4) - f(x - 1e-4)) / 2e-4
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    ys = [f(a + (b - a) * i / 60) for i in range(61)]
    lo, hi = min(ys) - 1, max(ys) + 1
    px = lambda x: x0 + 40 + (x - a) / (b - a) * W * 0.62
    py = lambda y: y1 - 20 - (y - lo) / (hi - lo) * (H - 40)
    cv.line([(px(x), py(y)) for x, y in _curve(f, a, b)], step, "#1971c2", 3)
    cv.text(px(b), py(f(b)), f"y = {expr}", step, 18, "#1971c2", "right", "bottom", math=False)
    step += 1
    for k, x in enumerate([a + (b - a) * i / 8 for i in range(1, 8)]):
        m = d(x)
        tr = _Track(cv)
        L = 1.6
        pts = [(px(x - L), py(f(x) - m * L)), (px(x + L), py(f(x) + m * L))]
        cv.line(pts, step, "#e03131", 2.5)
        cv.ellipse(px(x) - 7, py(f(x)) - 7, 14, 14, step, "#e03131", "#e03131", 1)
        cv.text(x0 + W * 0.68, y0 + 40, f"at x = {x:.1f}\nslope f'(x) = {m:.2f}", step, 24, "#e03131", math=False)
        tr.tag(until=step + 1 if k < 6 else None, fade=k > 0)
        step += 1
    cv.text(x0 + W * 0.68, y0 + 150, "the derivative is the slope\nof the tangent line", step, 19, INK, math=False)
    return step + 1


def build_compound(cv, v, box, step):
    P, r, n, years = float(v.get("principal") or 1000), float(v.get("rate") or 0.07), int(v.get("n") or 12), int(v.get("years") or 30)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    vals = [(t, P * (1 + r / n) ** (n * t), P * (1 + r * t)) for t in range(0, years + 1, max(1, years // 10))]
    mx = vals[-1][1]
    bw = W * 0.62 / len(vals)
    _axes(cv, x0 + 40, y0, x0 + 50 + W * 0.64, y1 - 30, step, "years", "")
    step += 1
    for i, (t, c, s) in enumerate(vals):
        x = x0 + 50 + i * bw
        hc, hs = (H - 70) * c / mx, (H - 70) * s / mx
        cv.rect(x, y1 - 30 - hc, bw * 0.45, hc, step, "#b2f2bb", "#2f9e44", 1.5, rounded=False)
        cv.rect(x + bw * 0.47, y1 - 30 - hs, bw * 0.4, hs, step, "#d0ebff", "#1971c2", 1.5, rounded=False)
        cv.text(x + bw * 0.45, y1 - 26, str(t), step, 14, MUTED, "center", "top", math=False)
        step += 1
    cv.text(x0 + W * 0.7, y0 + 30, f"A = P(1 + r/n)^(nt)\nP = ${P:,.0f}, r = {r:.0%},\nmonthly, {years} years\n\n"
                                   f"compound: ${vals[-1][1]:,.0f}\nsimple: ${vals[-1][2]:,.0f}\n\ngreen = compound\nblue = simple",
            step, 20, INK, math=False)
    return step + 1


def build_linear_system(cv, v, box, step):
    a1, b1, c1 = (float(x) for x in (v.get("eq1") or [1, 1, 6]))      # a x + b y = c
    a2, b2, c2 = (float(x) for x in (v.get("eq2") or [2, -1, 3]))
    det = a1 * b2 - a2 * b1
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    R = 8
    px = lambda x: x0 + 40 + (x + R) / (2 * R) * W * 0.6
    py = lambda y: y1 - 20 - (y + R) / (2 * R) * (H - 40)
    cv.line([(px(-R), py(0)), (px(R), py(0))], step, MUTED, 1)
    cv.line([(px(0), py(-R)), (px(0), py(R))], step, MUTED, 1)
    step += 1
    def clip(a, b, c):
        cand = []
        for x in (-R, R):
            if b and -R <= (c - a * x) / b <= R:
                cand.append((x, (c - a * x) / b))
        for y in (-R, R):
            if a and -R <= (c - b * y) / a <= R:
                cand.append(((c - b * y) / a, y))
        cand = sorted(set((round(x, 6), round(y, 6)) for x, y in cand))
        return [cand[0], cand[-1]] if len(cand) >= 2 else []
    for (a, b, c), col in (((a1, b1, c1), "#1971c2"), ((a2, b2, c2), "#e03131")):
        seg = clip(a, b, c)
        if not seg:
            continue
        pts = [(px(x), py(y)) for x, y in seg]
        cv.line(pts, step, col, 3)
        step += 1
    fmt = lambda a, b, c: f"{a:g}x {'+' if b >= 0 else '−'} {abs(b):g}y = {c:g}"
    if abs(det) > 1e-9:
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        cv.ellipse(px(x) - 9, py(y) - 9, 18, 18, step, "#2f9e44", "#2f9e44", 1)
        cv.text(px(x) + 12, py(y) - 8, f"({x:g}, {y:g})", step, 20, "#2f9e44", "left", "bottom", math=False)
        res = f"solution: x = {x:g}, y = {y:g}\n(where the lines cross)"
    else:
        res = "parallel lines: no solution\n(or the same line: infinitely many)"
    cv.text(x0 + W * 0.66, y0 + 40, f"{fmt(a1, b1, c1)}\n{fmt(a2, b2, c2)}\n\n{res}", step, 22, INK, math=False)
    return step + 1


def build_break_even(cv, v, box, step):
    fixed, var, price = float(v.get("fixed") or 5000), float(v.get("variable") or 15), float(v.get("price") or 40)
    q = fixed / (price - var)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    qmax = q * 2
    mx = price * qmax
    px = lambda x: x0 + 50 + x / qmax * W * 0.62
    py = lambda y: y1 - 30 - y / mx * (H - 60)
    _axes(cv, x0 + 50, y0, x0 + 60 + W * 0.62, y1 - 30, step, "units sold", "$")
    step += 1
    cv.line([(px(0), py(fixed)), (px(qmax), py(fixed + var * qmax))], step, "#e03131", 3)
    cv.text(px(qmax), py(fixed + var * qmax), "total cost", step, 18, "#e03131", "right", "top", math=False)
    step += 1
    cv.line([(px(0), py(0)), (px(qmax), py(price * qmax))], step, "#2f9e44", 3)
    cv.text(px(qmax), py(price * qmax) - 4, "revenue", step, 18, "#2f9e44", "right", "bottom", math=False)
    step += 1
    cv.ellipse(px(q) - 9, py(price * q) - 9, 18, 18, step, "#1971c2", "#1971c2", 1)
    cv.text(px(q) - 10, py(price * q) - 10, "break-even", step, 18, "#1971c2", "right", "bottom", math=False)
    mid = lambda u: py((fixed + var * u + price * u) / 2)
    cv.text(px(q * 0.4), mid(q * 0.4), "loss", step, 18, "#e03131", "center", "middle", math=False)
    cv.text(px(q * 1.6), mid(q * 1.6), "profit", step, 18, "#2f9e44", "center", "middle", math=False)
    cv.text(x0 + W * 0.68, y0 + 40, f"fixed costs ${fixed:,.0f}\ncost per unit ${var:g}\nprice ${price:g}\n\n"
                                   f"break-even = fixed ÷ (price − cost)\n= {fixed:,.0f} ÷ {price - var:g}\n= {q:,.0f} units",
            step, 20, INK, math=False)
    return step + 1


# ======================================================================== GIS

def build_gis_buffer(cv, v, box, step):
    rnd = random.Random(9)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    mw = W * 0.62
    cv.rect(x0, y0, mw, H - 10, step, "#f4fce3", "#5c940d", 2, rounded=False)
    river = [(x0, y0 + H * 0.3), (x0 + mw * 0.3, y0 + H * 0.45), (x0 + mw * 0.6, y0 + H * 0.4), (x0 + mw, y0 + H * 0.6)]
    rl = cv.line(river, step, "#1c7ed6", 6)
    rl["roundness"] = None
    cv.text(x0 + mw * 0.62, y0 + H * 0.36, "river", step, 17, "#1c7ed6", math=False)
    houses = [(x0 + rnd.uniform(20, mw - 20), y0 + rnd.uniform(20, H - 30)) for _ in range(16)]
    for x, y in houses:
        cv.rect(x - 7, y - 7, 14, 14, step, "#ced4da", INK, 1, rounded=False)
    step += 1
    buf = float(v.get("distance") or 0.13) * H

    def dist_to_river(p):
        best = 1e9
        for (ax, ay), (bx, by) in zip(river, river[1:]):
            t = max(0, min(1, ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)))
            best = min(best, math.dist(p, (ax + t * (bx - ax), ay + t * (by - ay))))
        return best
    for i, ((ax, ay), (bx, by)) in enumerate(zip(river, river[1:])):
        L = math.dist((ax, ay), (bx, by))
        nx, ny = -(by - ay) / L * buf, (bx - ax) / L * buf
        e = cv.polygon([(ax + nx, ay + ny), (bx + nx, by + ny), (bx - nx, by - ny), (ax - nx, ay - ny)], step, "#a5d8ff",
                       "transparent", 1)
        e["opacity"] = 45
    for (x, y) in river[1:-1]:
        e = cv.ellipse(x - buf, y - buf, 2 * buf, 2 * buf, step, "#a5d8ff", "transparent", 1)
        e["opacity"] = 45
    step += 1
    inside = [p for p in houses if dist_to_river(p) <= buf]
    for x, y in inside:
        cv.rect(x - 8, y - 8, 16, 16, step, "#ff8787", "#c92a2a", 2, rounded=False)
    step += 1
    cv.text(x0 + mw + 30, y0 + 30, f"1. river (line layer)\n2. buffer {v.get('label') or '100 m'} around it\n"
                                   f"3. select houses that intersect\n\n{len(inside)} of {len(houses)} houses are in the "
                                   f"flood buffer", step, 20, INK, math=False)
    return step + 1


# ======================================================================== register

visuals.BUILDERS.update({
    "dp_table": build_dp_table, "merge_sort": build_merge_sort, "huffman": build_huffman, "trie": build_trie,
    "topo_sort": build_topo_sort, "lru": build_lru, "cache_sim": build_cache_sim, "tcp_cwnd": build_tcp_cwnd,
    "deadlock": build_deadlock, "race": build_race, "event_loop": build_event_loop,
    "twos_complement": build_twos_complement, "jwt": build_jwt, "sql_injection": build_sql_injection,
    "password_hash": build_password_hash, "cicd": build_cicd, "autoscaling": build_autoscaling,
    "decision_tree": build_decision_tree, "knn": build_knn, "normal_dist": build_normal_dist, "softmax": build_softmax,
    "overfitting": build_overfitting, "er_diagram": build_er_diagram, "groupby": build_groupby,
    "punnett": build_punnett, "meiosis": build_meiosis, "respiration": build_respiration, "osmosis": build_osmosis,
    "plate_tectonics": build_plate_tectonics, "moon_phases": build_moon_phases, "earth_layers": build_earth_layers,
    "lens": build_lens, "pendulum": build_pendulum, "energy_ramp": build_energy_ramp, "balance": build_balance,
    "reaction_energy": build_reaction_energy, "tangent": build_tangent, "compound": build_compound,
    "linear_system": build_linear_system, "break_even": build_break_even, "gis_buffer": build_gis_buffer,
})
for _k in [k for k in list(visuals.ALIASES) if k in visuals.BUILDERS]:
    visuals.ALIASES.pop(_k)
visuals.ALIASES.update({
    "lcs": "dp_table", "dynamic_programming": "dp_table", "mergesort": "merge_sort", "divide_and_conquer": "merge_sort",
    "huffman_coding": "huffman", "prefix_tree": "trie", "topological_sort": "topo_sort", "page_replacement": "lru",
    "fifo_paging": "lru", "cache": "cache_sim", "congestion_control": "tcp_cwnd", "race_condition": "race",
    "javascript_event_loop": "event_loop", "twos": "twos_complement", "token": "jwt", "sqli": "sql_injection",
    "hashing_passwords": "password_hash", "pipeline_cicd": "cicd", "devops_pipeline": "cicd", "scaling": "autoscaling",
    "classification_tree": "decision_tree", "nearest_neighbors": "knn", "bell_curve": "normal_dist",
    "gaussian": "normal_dist", "loss_curves": "overfitting", "entity_relationship": "er_diagram", "erd": "er_diagram",
    "aggregate": "groupby", "genetics": "punnett", "cellular_respiration": "respiration", "tonicity": "osmosis",
    "tectonics": "plate_tectonics", "lunar_phases": "moon_phases", "earth_structure": "earth_layers",
    "optics": "lens", "conservation_of_energy": "energy_ramp", "balancing": "balance", "chemical_equation": "balance",
    "activation_energy": "reaction_energy", "derivative": "tangent", "interest": "compound",
    "system_of_equations": "linear_system", "breakeven": "break_even", "buffer": "gis_buffer", "proximity": "gis_buffer",
})
