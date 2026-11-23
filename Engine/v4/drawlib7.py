"""
StudyMate v4 - drawing library, part 7. (computed) = worked out by the engine, not guessed.

Algorithms    a_star (vs BFS, computed), floyd_warshall (all pairs, computed), radix_sort (buckets, computed),
              interval_scheduling (greedy, computed), coin_change (greedy vs DP, computed)
Compilers     lexer (real tokenizer), gc_mark_sweep (reachability, computed)
Architecture  branch_predictor (2-bit counter, computed), amdahl (speedup, computed), ripple_adder (computed),
              karnaugh (Quine-McCluskey groups, computed), endianness (computed), mips_encoding (R/I-type bits,
              computed), memory_hierarchy (latencies scaled to human time, computed)
OS            bankers (safe sequence, computed), memory_alloc (first / best / worst fit, computed)
Networks      distance_vector (Bellman-Ford tables, computed), hamming_code (encode + correct, computed),
              crc (polynomial division, computed), manchester (line coding), acl_rules (first match, computed), vlan
Cloud         aws_vpc, iam_policy (deny > allow > implicit deny, computed), circuit_breaker (states over time,
              computed), raft_election (computed), sharding (range vs hash, computed)
Security      merkle_tree (real SHA-256, computed), port_scan (computed), cert_chain
ML / NLP      activations (computed), naive_bayes (spam, computed), dbscan (computed), dendrogram (single linkage,
              computed), cosine_similarity (computed), temperature_sampling (computed), positional_encoding (computed),
              learning_rate (computed), rag_pipeline (top-k by cosine, computed)
Data          window_functions (real SQLite), standardization (z-scores, computed)
Maths         dot_product, complex_plane, quadratic, sequences, markov_chain (steady state), truth_table, permutations,
              euler_path (degrees + Hierholzer) — all computed
Physics       rc_circuit, standing_waves, double_slit, circular_motion, kepler_orbit (equal areas), heating_curve,
              photoelectric — all computed
Chemistry     maxwell_boltzmann (computed), galvanic_cell (cell voltage computed)
Biology       pcr (copies, computed), gel_electrophoresis (EcoRI digest, computed), phylogenetic_tree (UPGMA, computed),
              feedback_loop (glucose / insulin, computed)
Earth / GIS   ndvi (computed), epicenter (S-P triangulation, computed), slope_hillshade (computed)
Economics     lorenz_gini (computed), prisoners_dilemma (Nash, computed), comparative_advantage (computed)
"""
import fnmatch
import hashlib
import itertools
import math
import random
import re
import sqlite3

import visuals
from canvas import INK, MUTED
from drawlib import _envelope, _Track, draw_device, draw_thing
from drawlib2 import PAL, STRONG, _box, _table
from drawlib3 import _frames, _title
from drawlib6 import _mono, _note, _pl, _plot


def _grid_cells(cv, x, y, c, rows, cols, step, fill_fn, text_fn=None, size=14):
    for r in range(rows):
        for k in range(cols):
            cv.rect(x + k * c, y + r * c, c - 2, c - 2, step, fill_fn(r, k), INK, 1, rounded=False)
            if text_fn:
                s = text_fn(r, k)
                if s != "":
                    cv.text(x + k * c + c / 2 - 1, y + r * c + c / 2 - 1, str(s), step, size, INK, "center", "middle",
                            math=False)


# ======================================================================== algorithms

GRID = ["..............", "..S.....#.....", "........#.....", "........#.....", "........#...G.", "........#.....", "..............", ".............."]


def _search(grid, astar):
    R, C = len(grid), len(grid[0])
    s = next((r, c) for r in range(R) for c in range(C) if grid[r][c] == "S")
    g = next((r, c) for r in range(R) for c in range(C) if grid[r][c] == "G")
    h = (lambda p: abs(p[0] - g[0]) + abs(p[1] - g[1])) if astar else (lambda p: 0)
    import heapq
    pq, dist, prev, order, cnt = [(h(s), 0, s)], {s: 0}, {}, [], 0
    seen = set()
    while pq:
        f, _, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        order.append(u)
        if u == g:
            break
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            w = (u[0] + dr, u[1] + dc)
            if 0 <= w[0] < R and 0 <= w[1] < C and grid[w[0]][w[1]] != "#" and dist[u] + 1 < dist.get(w, 1e9):
                dist[w], prev[w] = dist[u] + 1, u
                cnt += 1
                heapq.heappush(pq, (dist[w] + h(w), cnt, w))
    path, u = [g], g
    while u != s:
        u = prev[u]
        path.append(u)
    return order, path[::-1], dist, h


def build_a_star(cv, v, box, step):
    grid = v.get("grid") or GRID
    R, C = len(grid), len(grid[0])
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    c = min(W * 0.62 / C, (H - 40) / R)
    bfs_order = _search(grid, False)[0]
    order, path, dist, h = _search(grid, True)
    _grid_cells(cv, x0, y0, c, R, C, step, lambda r, k: "#343a40" if grid[r][k] == "#" else "#ffffff",
                lambda r, k: grid[r][k] if grid[r][k] in "SG" else "", 18)
    step += 1
    chunk = max(1, len(order) // 8)
    for i in range(0, len(order), chunk):
        for (r, k) in order[i:i + chunk]:
            if grid[r][k] not in "SG":
                cv.rect(x0 + k * c + 2, y0 + r * c + 2, c - 6, c - 6, step, "#ffec99", "transparent", 1, rounded=False)
                cv.text(x0 + k * c + c / 2, y0 + r * c + c / 2, f"{dist[(r, k)] + h((r, k))}", step, 12, "#e67700",
                        "center", "middle", math=False)
        step += 1
    _pl(cv, [(x0 + k * c + c / 2 - 1, y0 + r * c + c / 2 - 1) for r, k in path], step, "#e03131", 4)
    step += 1
    _note(cv, box, step, f"A* picks the cell with the\nsmallest f = g + h\n(g = steps so far, h = Manhattan\ndistance to G; "
                         f"numbers = f)\n\nA* explored {len(order)} cells\nplain BFS / Dijkstra: {len(bfs_order)}\n"
                         f"both find the shortest path:\n{len(path) - 1} steps", 18)
    return step + 1


def build_floyd_warshall(cv, v, box, step):
    nodes = ["A", "B", "C", "D"]
    E = v.get("edges") or [("A", "B", 3), ("B", "C", 2), ("A", "C", 8), ("C", "D", 1), ("D", "A", 2), ("B", "D", 7)]
    INF = math.inf
    n = len(nodes)
    d = [[0 if i == j else INF for j in range(n)] for i in range(n)]
    for a, b, w in E:
        d[nodes.index(a)][nodes.index(b)] = w
    snaps = [([row[:] for row in d], -1, set())]
    for k in range(n):
        ch = set()
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
                    ch.add((i, j))
        snaps.append(([row[:] for row in d], k, ch))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    pos = {"A": (0.1, 0.15), "B": (0.42, 0.15), "C": (0.42, 0.7), "D": (0.1, 0.7)}
    P = lambda nn: (x0 + W * 0.36 * pos[nn][0] + 20, y0 + (H - 40) * pos[nn][1] + 20)
    for a, b, w in E:
        (ax, ay), (bx, by) = P(a), P(b)
        L = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / L, (by - ay) / L
        off = 6 if (b, a) in [(e[0], e[1]) for e in E] else 0
        cv.line([(ax + ux * 24 - uy * off, ay + uy * 24 + ux * off), (bx - ux * 24 - uy * off, by - uy * 24 + ux * off)],
                step, MUTED, 1.5, arrow_end=True)
        cv.text((ax + bx) / 2 - uy * 14, (ay + by) / 2 + ux * 14, str(w), step, 15, INK, "center", "middle", math=False)
    for nn in nodes:
        x, y = P(nn)
        cv.ellipse(x - 20, y - 20, 40, 40, step, "#d0ebff", INK, 2)
        cv.text(x, y, nn, step, 17, INK, "center", "middle", math=False)
    step += 1
    cw = min(70, W * 0.4 / (n + 1))
    mx = x0 + W * 0.42

    def draw(f, st):
        m, k, ch = snaps[f]
        cv.text(mx, y0, "distances, no middle stop" if k < 0 else f"allow going through {nodes[k]}", st, 19, "#e8590c",
                math=False)
        for i in range(n):
            cv.text(mx + (i + 1) * cw + cw / 2, y0 + 40, nodes[i], st, 16, MUTED, "center", "middle", math=False)
            cv.text(mx + cw / 2, y0 + 70 + i * cw * 0.8, nodes[i], st, 16, MUTED, "center", "middle", math=False)
            for j in range(n):
                val = m[i][j]
                _box(cv, mx + (j + 1) * cw, y0 + 54 + i * cw * 0.8, cw - 4, cw * 0.8 - 4, st, "∞" if val == INF else int(val),
                     "#ffd8a8" if (i, j) in ch else "#ffffff", 16)
        cv.text(mx, y0 + 70 + n * cw * 0.8, f"{len(ch)} shorter paths found" if k >= 0 else "", st, 17, "#e03131",
                math=False)
    step = _frames(cv, step, len(snaps), draw)
    cv.text(mx, y0 + 110 + n * cw * 0.8, "d[i][j] = min(d[i][j], d[i][k] + d[k][j])\nfor every k: O(n³)", step, 17, INK,
            math=False)
    return step + 1


def build_radix_sort(cv, v, box, step):
    arr = [int(x) for x in (v.get("array") or [170, 45, 75, 90, 802, 24, 2, 66])]
    digits = len(str(max(arr)))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    passes, cur = [], list(arr)
    for d in range(digits):
        buckets = [[] for _ in range(10)]
        for x in cur:
            buckets[x // 10 ** d % 10].append(x)
        cur = [x for b in buckets for x in b]
        passes.append((d, buckets, list(cur)))
    _mono(cv.text(x0, y0, "start: " + "  ".join(map(str, arr)), step, 18, INK, math=False))
    step += 1
    bw = W * 0.95 / 10
    name = ["ones", "tens", "hundreds", "thousands"]

    def draw(k, st):
        d, buckets, out = passes[k]
        cv.text(x0, y0 + 36, f"pass {k + 1}: put each number in the bucket of its {name[d]} digit", st, 19, "#e8590c",
                math=False)
        for b in range(10):
            bx = x0 + b * bw
            cv.rect(bx, y0 + 70, bw - 6, H * 0.45, st, "#f8f9fa", MUTED, 1, rounded=False)
            cv.text(bx + bw / 2, y0 + 70 + H * 0.45 + 4, str(b), st, 16, MUTED, "center", "top", math=False)
            for i, x in enumerate(buckets[b]):
                s = str(x).zfill(digits)
                pos = digits - 1 - d
                _box(cv, bx + 4, y0 + 76 + i * 34, bw - 14, 30, st, x, PAL[b % len(PAL)], 15)
        _mono(cv.text(x0, y0 + 110 + H * 0.45, "read buckets in order: " + "  ".join(map(str, out)), st, 18, "#1971c2",
                      math=False))
    step = _frames(cv, step, len(passes), draw)
    cv.text(x0, y1 - 30, f"no comparisons: {digits} passes × n items = O(d·n); each pass must be stable", step, 17, INK,
            math=False)
    return step + 1


def build_interval_scheduling(cv, v, box, step):
    acts = v.get("activities") or [("A", 1, 4), ("B", 3, 5), ("C", 0, 6), ("D", 5, 7), ("E", 3, 9), ("F", 5, 9),
                                   ("G", 6, 10), ("H", 8, 11), ("I", 8, 12), ("J", 2, 14), ("K", 12, 16)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    T = max(e for _, _, e in acts)
    px = lambda t: x0 + 40 + t / T * W * 0.6
    rh = (H - 50) / len(acts)
    srt = sorted(acts, key=lambda a: a[2])
    for t in range(0, T + 1, 2):
        cv.text(px(t), y1 - 20, str(t), step, 13, MUTED, "center", "top", math=False)
    for i, (n, s, e) in enumerate(srt):
        cv.text(x0, y0 + i * rh + rh / 2, n, step, 15, INK, "left", "middle", math=False)
        cv.rect(px(s), y0 + i * rh + 3, px(e) - px(s), rh - 6, step, "#e9ecef", MUTED, 1)
    step += 1
    end, chosen = -1, []
    for i, (n, s, e) in enumerate(srt):
        ok = s >= end
        if ok:
            chosen.append(n)
            end = e
        cv.rect(px(s), y0 + i * rh + 3, px(e) - px(s), rh - 6, step, "#b2f2bb" if ok else "#ffe3e3",
                "#2f9e44" if ok else "#ffa8a8", 1.5)
        cv.text(px(e) + 6, y0 + i * rh + rh / 2, "pick ✔" if ok else "overlaps ✘", step, 13,
                "#2f9e44" if ok else "#e03131", "left", "middle", math=False)
        step += 1
    _note(cv, box, step, f"greedy: sort by finish time,\ntake each activity that starts\nafter the last one ends\n\n"
                         f"chosen: {', '.join(chosen)}\n= {len(chosen)} activities (optimal)", 18, frac=0.76)
    return step + 1


def build_coin_change(cv, v, box, step):
    coins = [int(c) for c in (v.get("coins") or [1, 3, 4])]
    amt = int(v.get("amount") or 6)
    dp, pick = [0] + [math.inf] * amt, [0] * (amt + 1)
    for a in range(1, amt + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a], pick[a] = dp[a - c] + 1, c
    greedy, left = [], amt
    for c in sorted(coins, reverse=True):
        while left >= c:
            greedy.append(c)
            left -= c
    best, a = [], amt
    while a:
        best.append(pick[a])
        a -= pick[a]
    x0, y0, x1, y1 = box
    W = x1 - x0
    _title(cv, x0, y0, f"make {amt} with coins {coins}", step, 21)
    cv.text(x0, y0 + 40, f"greedy (biggest first): {' + '.join(map(str, greedy))} = {len(greedy)} coins", step, 19,
            "#e03131", math=False)
    step += 1
    cw = min(80, W * 0.9 / (amt + 1))
    for a in range(amt + 1):
        cv.text(x0 + a * cw + cw / 2, y0 + 100, str(a), step, 15, MUTED, "center", "middle", math=False)
    step += 1
    for a in range(amt + 1):
        _box(cv, x0 + a * cw, y0 + 116, cw - 4, 44, step, dp[a], "#d0ebff" if a else "#e9ecef", 18)
        cv.text(x0 + a * cw + cw / 2, y0 + 166, f"+{pick[a]}" if a else "", step, 13, "#7048e8", "center", "top", math=False)
        step += 1
    cv.text(x0, y0 + 210, f"DP: best[a] = 1 + min(best[a − coin])\noptimal: {' + '.join(map(str, best))} = {len(best)} coins",
            step, 20, "#2f9e44", math=False)
    return step + 1


# ======================================================================== compilers / runtime

TOKENS = [("NUMBER", r"\d+(\.\d+)?"), ("STRING", r"\"[^\"]*\""), ("KEYWORD", r"\b(if|else|while|for|return|int|def)\b"),
          ("IDENT", r"[A-Za-z_]\w*"), ("OP", r"==|<=|>=|!=|[+\-*/=<>]"), ("PUNCT", r"[(){};,:]"), ("SKIP", r"\s+")]
TOKCOL = {"NUMBER": "#ffec99", "STRING": "#d3f9d8", "KEYWORD": "#e5dbff", "IDENT": "#d0ebff", "OP": "#ffc9c9",
          "PUNCT": "#e9ecef"}


def tokenize(src):
    rx = "|".join(f"(?P<{n}>{p})" for n, p in TOKENS)
    return [(m.lastgroup, m.group()) for m in re.finditer(rx, src) if m.lastgroup != "SKIP"]


def build_lexer(cv, v, box, step):
    src = str(v.get("code") or "if (total >= 100) price = total * 0.9;")
    toks = tokenize(src)
    x0, y0, x1, y1 = box
    W = x1 - x0
    _mono(cv.text(x0, y0, src, step, 22, INK, math=False))
    cv.text(x0, y0 + 40, "the lexer reads characters left to right and groups them into tokens:", step, 17, MUTED,
            math=False)
    step += 1
    x, y = x0, y0 + 80
    for kind, txt in toks:
        w = max(70, 16 * len(txt) + 20)
        if x + w > x0 + W * 0.95:
            x, y = x0, y + 90
        _box(cv, x, y, w, 40, step, txt, TOKCOL[kind], 17, mono=True)
        cv.text(x + w / 2, y + 44, kind, step, 12, MUTED, "center", "top", math=False)
        x += w + 8
        step += 1
    cv.text(x0, y + 100, f"{len(toks)} tokens → the parser builds a syntax tree from them. Whitespace is dropped; "
                         "'total' becomes one IDENT.", step, 17, INK, math=False)
    return step + 1


def build_gc_mark_sweep(cv, v, box, step):
    objs = {"A": (0.25, 0.15), "B": (0.15, 0.5), "C": (0.4, 0.5), "D": (0.28, 0.85), "E": (0.7, 0.3), "F": (0.85, 0.65),
            "G": (0.62, 0.75)}
    refs = [("A", "B"), ("A", "C"), ("C", "D"), ("E", "F"), ("F", "E"), ("G", "F")]
    roots = v.get("roots") or ["A"]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    P = lambda o: (x0 + 80 + W * 0.55 * objs[o][0], y0 + 30 + (H - 70) * objs[o][1])
    cv.text(x0, y0 + 30, "roots\n(stack,\nglobals)", step, 15, "#e8590c", math=False)
    for r in roots:
        cv.arrow(x0 + 60, y0 + 50, P(r)[0] - 30, P(r)[1], step, color="#e8590c")
    for a, b in refs:
        (ax, ay), (bx, by) = P(a), P(b)
        L = math.hypot(bx - ax, by - ay)
        cv.line([(ax + (bx - ax) / L * 28, ay + (by - ay) / L * 28), (bx - (bx - ax) / L * 28, by - (by - ay) / L * 28)],
                step, MUTED, 1.5, arrow_end=True)
    for o in objs:
        x, y = P(o)
        cv.ellipse(x - 26, y - 22, 52, 44, step, "#ffffff", INK, 2)
        cv.text(x, y, o, step, 18, INK, "center", "middle", math=False)
    step += 1
    marked, q = set(roots), list(roots)
    while q:
        u = q.pop(0)
        x, y = P(u)
        cv.ellipse(x - 26, y - 22, 52, 44, step, "#b2f2bb", "#2f9e44", 3)
        cv.text(x, y, u, step, 18, INK, "center", "middle", math=False)
        for a, b in refs:
            if a == u and b not in marked:
                marked.add(b)
                q.append(b)
        step += 1
    for o in objs:
        if o not in marked:
            x, y = P(o)
            cv.ellipse(x - 26, y - 22, 52, 44, step, "#ffc9c9", "#e03131", 3)
            cv.text(x, y, "✘", step, 22, "#e03131", "center", "middle", math=False)
    step += 1
    gone = sorted(set(objs) - marked)
    _note(cv, box, step, f"mark: follow pointers from the\nroots, colour what you reach\n\nsweep: free the rest\n"
                         f"→ {', '.join(gone)} freed\n\nE ↔ F point at each other, so\nreference counting would leak\n"
                         "them; mark-and-sweep doesn't", 18, frac=0.72)
    return step + 1


# ======================================================================== architecture

def build_branch_predictor(cv, v, box, step):
    pattern = (v.get("outcomes") or "TTTN" * 4)
    states = ["strongly\nnot taken", "weakly\nnot taken", "weakly\ntaken", "strongly\ntaken"]
    s, s1, rows = 2, "T", []
    for o in pattern:
        pred = "T" if s >= 2 else "N"
        rows.append((s, pred, o, s1))
        s = min(3, s + 1) if o == "T" else max(0, s - 1)
        s1 = o
    acc2 = sum(p == o for _, p, o, _ in rows) / len(rows)
    acc1 = sum(p1 == o for _, _, o, p1 in rows) / len(rows)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    sw = W * 0.9 / 4
    for i, n in enumerate(states):
        _box(cv, x0 + i * sw + 10, y0, sw - 40, 60, step, n, "#ffc9c9" if i < 2 else "#b2f2bb", 15)
        if i < 3:
            cv.arrow(x0 + i * sw + sw - 28, y0 + 18, x0 + (i + 1) * sw + 8, y0 + 18, step, "T", color="#2f9e44")
            cv.arrow(x0 + (i + 1) * sw + 8, y0 + 44, x0 + i * sw + sw - 28, y0 + 44, step, "N", color="#e03131")
    step += 1
    c = min(46, W * 0.95 / len(rows))
    cv.text(x0, y0 + 100, "actual:", step, 15, MUTED, math=False)
    cv.text(x0, y0 + 150, "2-bit:", step, 15, MUTED, math=False)
    cv.text(x0, y0 + 200, "1-bit:", step, 15, MUTED, math=False)
    for i, (st, p, o, p1) in enumerate(rows):
        x = x0 + 70 + i * c
        _box(cv, x, y0 + 86, c - 4, 32, step, o, "#e9ecef", 15)
        _box(cv, x, y0 + 136, c - 4, 32, step, p, "#b2f2bb" if p == o else "#ffc9c9", 15)
        _box(cv, x, y0 + 186, c - 4, 32, step, p1, "#b2f2bb" if p1 == o else "#ffc9c9", 15)
        if i % 4 == 3:
            step += 1
    cv.text(x0, y0 + 250, f"loop branch: taken 3 times, then not taken (exit)\n2-bit counter accuracy {acc2:.0%}, "
                          f"1-bit (repeat last) {acc1:.0%}\nthe 2-bit counter needs two misses in a row to change its mind",
            step + 1, 18, INK, math=False)
    return step + 2


def build_amdahl(cv, v, box, step):
    fr = [0.5, 0.75, 0.9, 0.95]
    S = lambda p, n: 1 / ((1 - p) + p / n)
    px, py = _plot(cv, box, step, (0, 10), (0, 20), "processors (log₂)", "speedup", frac=0.55)
    for k in range(0, 11, 2):
        cv.text(px(k), py(0) + 4, str(2 ** k), step, 12, MUTED, "center", "top", math=False)
    step += 1
    for i, p in enumerate(fr):
        _pl(cv, [(px(k / 10), py(S(p, 2 ** (k / 10)))) for k in range(101)], step, STRONG[i], 3)
        cv.text(px(10) + 6, py(S(p, 1024)), f"{p:.0%} parallel → max {1 / (1 - p):.0f}×", step, 15, STRONG[i], "left",
                "middle", math=False)
        step += 1
    _note(cv, box, step, "speedup = 1 / ((1 − p) + p / n)\n\nthe serial part caps the gain:\n"
                         f"95% parallel on 1024 cores\n= {S(0.95, 1024):.1f}× (not 1024×)", 18, frac=0.8, dy=200)
    return step + 1


def build_ripple_adder(cv, v, box, step):
    a, b = str(v.get("a") or "1011"), str(v.get("b") or "0110")
    n = max(len(a), len(b))
    a, b = a.zfill(n), b.zfill(n)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    fw = W * 0.8 / n
    carry, sums = 0, [""] * n
    for i in range(n):
        x = x0 + (n - 1 - i) * fw + 20
        _box(cv, x, y0 + 90, fw - 50, 90, step, "full\nadder", "#e7f5ff", 16)
        cv.text(x + (fw - 50) * 0.3, y0 + 60, a[n - 1 - i], step, 20, "#1971c2", "center", "middle", math=False)
        cv.text(x + (fw - 50) * 0.7, y0 + 60, b[n - 1 - i], step, 20, "#e03131", "center", "middle", math=False)
    cv.text(x0, y0 + 20, f"A = {a} ({int(a, 2)})   B = {b} ({int(b, 2)})", step, 19, INK, math=False)
    step += 1
    for i in range(n):
        ai, bi = int(a[n - 1 - i]), int(b[n - 1 - i])
        s = ai ^ bi ^ carry
        co = (ai & bi) | (carry & (ai ^ bi))
        x = x0 + (n - 1 - i) * fw + 20
        cv.text(x + (fw - 50) / 2, y0 + 200, str(s), step, 24, "#2f9e44", "center", "middle", math=False)
        cv.arrow(x - 2, y0 + 135, x - 44, y0 + 135, step, f"c={co}", color="#f08c00")
        cv.text(x + (fw - 50) / 2, y0 + 228, f"{ai}+{bi}+{carry}", step, 13, MUTED, "center", "top", math=False)
        sums[n - 1 - i] = str(s)
        carry = co
        step += 1
    res = str(carry) + "".join(sums)
    cv.text(x0, y0 + 270, f"sum = {res} ({int(res, 2)})  ✔ {int(a, 2)} + {int(b, 2)} = {int(a, 2) + int(b, 2)}\n"
                          "each carry must ripple to the next adder, so delay grows with the number of bits\n"
                          "(carry-lookahead adders fix that)", step, 18, INK, math=False)
    return step + 1


def qm_minimize(minterms, nvars=4):
    """Prime implicants (as strings with '-') and a greedy cover."""
    terms = {format(m, f"0{nvars}b") for m in minterms}
    primes = set()
    while terms:
        nxt, used = set(), set()
        for a, b in itertools.combinations(sorted(terms), 2):
            diff = [i for i in range(nvars) if a[i] != b[i]]
            if len(diff) == 1:
                nxt.add(a[:diff[0]] + "-" + a[diff[0] + 1:])
                used |= {a, b}
        primes |= terms - used
        terms = nxt
    cov = lambda p, m: all(c in "-" + d for c, d in zip(p, format(m, f"0{nvars}b")))
    left, chosen = set(minterms), []
    while left:
        p = max(sorted(primes), key=lambda p: (sum(cov(p, m) for m in left), p.count("-")))
        chosen.append(p)
        left -= {m for m in left if cov(p, m)}
    return chosen, cov


def build_karnaugh(cv, v, box, step):
    mins = [int(m) for m in (v.get("minterms") or [0, 1, 2, 3, 5, 7, 8, 10, 13, 15])]
    chosen, cov = qm_minimize(mins)
    gray = [0, 1, 3, 2]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    c = min(W * 0.4 / 4, (H - 80) / 4)
    gx, gy = x0 + 70, y0 + 60
    for i, g in enumerate(gray):
        cv.text(gx + i * c + c / 2, gy - 10, format(g, "02b"), step, 15, MUTED, "center", "bottom", math=False)
        cv.text(gx - 10, gy + i * c + c / 2, format(g, "02b"), step, 15, MUTED, "right", "middle", math=False)
    cv.text(gx - 60, gy - 40, "AB \\ CD", step, 15, MUTED, math=False)
    for r, ab in enumerate(gray):
        for k, cd in enumerate(gray):
            m = ab * 4 + cd
            cv.rect(gx + k * c, gy + r * c, c, c, step, "#ffffff", INK, 1.5, rounded=False)
            cv.text(gx + k * c + c / 2, gy + r * c + c / 2, "1" if m in mins else "0", step, 22,
                    INK if m in mins else "#ced4da", "center", "middle", math=False)
    step += 1
    names = "ABCD"
    terms = []
    for gi, p in enumerate(chosen):
        col = STRONG[gi % len(STRONG)]
        for r, ab in enumerate(gray):
            for k, cd in enumerate(gray):
                if cov(p, ab * 4 + cd):
                    ins = 4 + gi * 4
                    cv.rect(gx + k * c + ins, gy + r * c + ins, c - 2 * ins, c - 2 * ins, step, "transparent", col, 2.5)
        term = "".join(names[i] + ("'" if ch == "0" else "") for i, ch in enumerate(p) if ch != "-") or "1"
        terms.append(term)
        cv.text(x0 + W * 0.56, y0 + 60 + gi * 34, f"{p} → {term}", step, 19, col, math=False)
        step += 1
    cv.text(x0 + W * 0.56, y0 + 80 + len(chosen) * 34, f"F = {' + '.join(terms)}\n\ngroups of 1, 2, 4 or 8 adjacent 1s\n"
                                                        "(edges wrap around); bigger\ngroups → fewer variables", step, 18,
            INK, math=False)
    return step + 1


def build_endianness(cv, v, box, step):
    val = int(str(v.get("value") or "0x12345678"), 0)
    addr = int(str(v.get("address") or "0x100"), 0)
    big, little = val.to_bytes(4, "big"), val.to_bytes(4, "little")
    x0, y0, x1, y1 = box
    W = x1 - x0
    _title(cv, x0, y0, f"storing 0x{val:08X} at address 0x{addr:X}", step, 22)
    step += 1
    for i in range(4):
        cv.text(x0 + 220 + i * 110 + 50, y0 + 60, f"0x{addr + i:X}", step, 15, MUTED, "center", "middle", math=False)
    for r, (name, bs, note) in enumerate((("big-endian", big, "most significant byte first\n(network byte order)"),
                                          ("little-endian", little, "least significant byte first\n(x86, ARM by default)"))):
        y = y0 + 90 + r * 110
        cv.text(x0, y + 24, name, step, 19, "#e8590c", math=False)
        for i, b_ in enumerate(bs):
            _box(cv, x0 + 220 + i * 110, y, 100, 50, step, f"{b_:02X}", PAL[[0x12, 0x34, 0x56, 0x78].index(b_)
                                                                            if b_ in (0x12, 0x34, 0x56, 0x78) else i], 20,
                 mono=True)
        cv.text(x0 + 220 + 4 * 110 + 20, y + 24, note, step, 15, MUTED, "left", "middle", math=False)
        step += 1
    cv.text(x0, y0 + 330, "it matters when bytes cross machines: files, network packets (use htonl / ntohl)", step, 18,
            INK, math=False)
    return step + 1


REGS = {"$zero": 0, "$t0": 8, "$t1": 9, "$t2": 10, "$s0": 16, "$s1": 17, "$s2": 18, "$sp": 29, "$ra": 31}


def encode_mips(ins):
    op, rest = ins.split(None, 1)
    args = [a.strip() for a in rest.replace("(", ",").replace(")", "").split(",")]
    if op in ("add", "sub", "and", "or", "slt"):
        funct = {"add": 32, "sub": 34, "and": 36, "or": 37, "slt": 42}[op]
        rd, rs, rt = (REGS[a] for a in args)
        f = [("opcode", 0, 6), ("rs", rs, 5), ("rt", rt, 5), ("rd", rd, 5), ("shamt", 0, 5), ("funct", funct, 6)]
        kind = "R-type"
    else:
        opc = {"lw": 35, "sw": 43, "addi": 8, "beq": 4}[op]
        if op in ("lw", "sw"):
            rt, imm, rs = REGS[args[0]], int(args[1]), REGS[args[2]]
        else:
            rt, rs, imm = REGS[args[0]], REGS[args[1]], int(args[2])
        f = [("opcode", opc, 6), ("rs", rs, 5), ("rt", rt, 5), ("immediate", imm & 0xFFFF, 16)]
        kind = "I-type"
    bits = "".join(format(val, f"0{w}b") for _, val, w in f)
    return kind, f, bits


def build_mips_encoding(cv, v, box, step):
    instrs = v.get("instructions") or ["add $t0, $s1, $s2", "lw $t1, 8($s0)"]
    x0, y0, x1, y1 = box
    W = x1 - x0
    bw = W * 0.9 / 32
    for k, ins in enumerate(instrs[:3]):
        kind, f, bits = encode_mips(ins)
        y = y0 + k * 160
        _mono(cv.text(x0, y, f"{ins}    ({kind})", step, 20, "#e8590c", math=False))
        x = x0
        for i, (name, val, w) in enumerate(f):
            _box(cv, x, y + 34, w * bw - 3, 44, step, format(val, f"0{w}b"), PAL[i], 14, mono=True)
            cv.text(x + w * bw / 2, y + 84, f"{name} = {val}", step, 14, INK, "center", "top", math=False)
            x += w * bw
        _mono(cv.text(x0, y + 110, f"= 0x{int(bits, 2):08X}", step, 18, "#1971c2", math=False))
        step += 1
    cv.text(x0, y0 + 160 * len(instrs[:3]), "every MIPS instruction is exactly 32 bits; the opcode tells the CPU how to "
                                             "split the rest", step, 17, INK, math=False)
    return step + 1


def build_memory_hierarchy(cv, v, box, step):
    levels = [("registers", "~1 KB", 0.3e-9), ("L1 cache", "64 KB", 1e-9), ("L2 cache", "1 MB", 4e-9),
              ("L3 cache", "32 MB", 15e-9), ("RAM", "32 GB", 80e-9), ("SSD", "1 TB", 100e-6), ("hard disk", "4 TB", 10e-3)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    lh = (H - 20) / len(levels)
    cx = x0 + W * 0.3

    def human(sec):
        s = sec / 0.3e-9
        for u, d in (("years", 3.15e7), ("days", 86400), ("hours", 3600), ("minutes", 60)):
            if s >= d:
                return f"{s / d:.1f} {u}"
        return f"{s:.0f} s"
    for i, (name, size, t) in enumerate(levels):
        w = W * 0.55 * (i + 1) / len(levels)
        y = y0 + i * lh
        cv.rect(cx - w / 2, y, w, lh - 6, step, PAL[i % len(PAL)], INK, 1.5, rounded=False)
        cv.text(cx, y + lh / 2 - 3, f"{name}  {size}", step, 16, INK, "center", "middle", math=False)
        t_s = f"{t * 1e9:.1f} ns" if t < 1e-6 else f"{t * 1e6:.0f} µs" if t < 1e-3 else f"{t * 1e3:.0f} ms"
        cv.text(x0 + W * 0.62, y + lh / 2 - 3, f"{t_s}   (if a register took 1 s: {human(t)})", step, 15, INK, "left",
                "middle", math=False)
        step += 1
    cv.text(cx, y1 + 2, "smaller + faster + pricier at the top", step, 14, MUTED, "center", "top", math=False)
    return step


# ======================================================================== OS

def bankers(alloc, maxm, avail):
    n = len(alloc)
    need = [[m - a for m, a in zip(maxm[i], alloc[i])] for i in range(n)]
    work, done, seq, log = list(avail), [False] * n, [], []
    while len(seq) < n:
        prog = False
        for i in range(n):
            if not done[i] and all(nd <= w for nd, w in zip(need[i], work)):
                log.append((i, list(work), need[i]))
                work = [w + a for w, a in zip(work, alloc[i])]
                done[i] = True
                seq.append(i)
                prog = True
        if not prog:
            return need, None, log
    return need, seq, log


def build_bankers(cv, v, box, step):
    alloc = v.get("allocation") or [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
    maxm = v.get("max") or [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
    avail = v.get("available") or [3, 3, 2]
    need, seq, log = bankers(alloc, maxm, avail)
    x0, y0, x1, y1 = box
    W = x1 - x0
    rows = [[f"P{i}", " ".join(map(str, alloc[i])), " ".join(map(str, maxm[i])), " ".join(map(str, need[i]))]
            for i in range(len(alloc))]
    _table(cv, x0, y0 + 30, f"available: {' '.join(map(str, avail))}  (resources A B C)",
           ["process", "allocated", "max", "need = max − alloc"], rows, step, W * 0.55, size=15)
    step += 1
    for k, (i, work, nd) in enumerate(log):
        cv.text(x0 + W * 0.6, y0 + 30 + k * 34, f"P{i}: need {nd} ≤ work {work} ✔ → runs, returns {alloc[i]}", step, 16,
                "#2f9e44", math=False)
        step += 1
    cv.text(x0, y0 + 260, ("safe sequence: " + " → ".join(f"P{i}" for i in seq)) if seq else "UNSAFE: no order works",
            step, 22, "#1971c2" if seq else "#e03131", math=False)
    cv.text(x0, y0 + 300, "a request is granted only if the system stays in a safe state → deadlock is avoided", step, 17,
            INK, math=False)
    return step + 1


def allocate(blocks, reqs, how):
    free, out = list(blocks), []
    for r in reqs:
        cand = [i for i, b in enumerate(free) if b >= r]
        if not cand:
            out.append(None)
            continue
        i = cand[0] if how == "first" else min(cand, key=lambda i: free[i]) if how == "best" else max(cand,
                                                                                                         key=lambda i: free[i])
        free[i] -= r
        out.append(i)
    return out, free


def build_memory_alloc(cv, v, box, step):
    blocks = v.get("blocks") or [100, 500, 200, 300, 600]
    reqs = v.get("requests") or [212, 417, 112, 426]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    scale = W * 0.68 / sum(blocks)
    cv.text(x0, y0, f"free blocks (KB): {blocks}    requests: {reqs}", step, 18, INK, math=False)
    step += 1
    for r, how in enumerate(("first", "best", "worst")):
        out, free = allocate(blocks, reqs, how)
        y = y0 + 50 + r * (H - 60) / 3
        cv.text(x0, y + 24, f"{how} fit", step, 18, "#e8590c", math=False)
        x = x0 + 110
        fill = {i: 0 for i in range(len(blocks))}
        for bi, b in enumerate(blocks):
            cv.rect(x, y, b * scale - 4, 50, step, "#f8f9fa", INK, 1.5, rounded=False)
            for j, (req, at) in enumerate(zip(reqs, out)):
                if at == bi:
                    cv.rect(x + fill[bi] * scale, y + 3, req * scale - 6, 44, step, PAL[j], STRONG[j], 1.5, rounded=False)
                    cv.text(x + fill[bi] * scale + req * scale / 2, y + 25, str(req), step, 13, INK, "center", "middle",
                            math=False)
                    fill[bi] += req
            x += b * scale
        failed = [str(q) for q, at in zip(reqs, out) if at is None]
        cv.text(x0 + 110, y + 56, ("couldn't place: " + ", ".join(failed)) if failed else "all placed ✔", step, 15,
                "#e03131" if failed else "#2f9e44", math=False)
        step += 1
    _note(cv, box, step, "first: first hole big enough\nbest: smallest hole that fits\nworst: biggest hole\n\n"
                         "leftover slivers = external\nfragmentation", 17, frac=0.82, dy=50)
    return step + 1


# ======================================================================== networks

def build_distance_vector(cv, v, box, step):
    nodes = ["A", "B", "C", "D"]
    links = v.get("links") or {("A", "B"): 1, ("B", "C"): 2, ("C", "D"): 1, ("A", "D"): 7, ("B", "D"): 5}
    cost = lambda a, b: links.get((a, b), links.get((b, a)))
    tables = {n: {m: (0 if m == n else cost(n, m) or math.inf) for m in nodes} for n in nodes}
    rounds = [{n: dict(t) for n, t in tables.items()}]
    for _ in range(3):
        new = {}
        for n in nodes:
            new[n] = {m: min([tables[n][m]] + [cost(n, k) + tables[k][m] for k in nodes if cost(n, k)]) for m in nodes}
        if new == tables:
            break
        tables = new
        rounds.append({n: dict(t) for n, t in tables.items()})
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    pos = {"A": (0.1, 0.2), "B": (0.5, 0.2), "C": (0.5, 0.75), "D": (0.1, 0.75)}
    P = lambda n: (x0 + W * 0.3 * pos[n][0] + 30, y0 + (H - 60) * pos[n][1] + 30)
    for (a, b), w in links.items():
        cv.line([P(a), P(b)], step, MUTED, 1.5)
        (ax, ay), (bx, by) = P(a), P(b)
        cv.text((ax + bx) / 2 + 8, (ay + by) / 2 - 8, str(w), step, 15, INK, math=False)
    for n in nodes:
        draw_device(cv, "router", *P(n), 54, step)
        cv.text(P(n)[0], P(n)[1] + 26, n, step, 17, INK, "center", "top", math=False)
    step += 1

    def draw(k, st):
        t = rounds[k]
        cv.text(x0 + W * 0.36, y0, f"round {k}: every router tells its neighbours its table" if k else
                "round 0: each router only knows its links", st, 18, "#e8590c", math=False)
        for i, n in enumerate(nodes):
            rows = [[m, "∞" if t[n][m] == math.inf else t[n][m]] for m in nodes]
            hl = [j for j, m in enumerate(nodes) if k and t[n][m] != rounds[k - 1][n][m]]
            _table(cv, x0 + W * 0.36 + (i % 2) * W * 0.31, y0 + 60 + (i // 2) * 200, f"router {n}", ["to", "cost"],
                   rows, st, W * 0.27, hl=hl, size=14)
    step = _frames(cv, step, len(rounds), draw)
    cv.text(x0, y1 - 20, "Bellman-Ford: cost(X→Y) = min over neighbours N of link(X,N) + N's cost to Y   (RIP works this way)",
            step, 15, INK, math=False)
    return step + 1


def hamming74(d):
    d1, d2, d3, d4 = d
    p1, p2, p3 = d1 ^ d2 ^ d4, d1 ^ d3 ^ d4, d2 ^ d3 ^ d4
    return [p1, p2, d1, p3, d2, d3, d4]


def build_hamming_code(cv, v, box, step):
    data = [int(c) for c in str(v.get("data") or "1011")]
    flip = int(v.get("flip") or 5)
    code = hamming74(data)
    recv = list(code)
    recv[flip - 1] ^= 1
    s1 = recv[0] ^ recv[2] ^ recv[4] ^ recv[6]
    s2 = recv[1] ^ recv[2] ^ recv[5] ^ recv[6]
    s3 = recv[3] ^ recv[4] ^ recv[5] ^ recv[6]
    syn = s1 + 2 * s2 + 4 * s3
    x0, y0, x1, y1 = box
    W = x1 - x0
    names = ["p1", "p2", "d1", "p3", "d2", "d3", "d4"]
    c = min(80, W * 0.6 / 7)
    _title(cv, x0, y0, f"Hamming(7,4): data {''.join(map(str, data))}", step, 21)
    step += 1
    for row, (lab, bits, hl) in enumerate((("sent", code, None), ("received", recv, flip - 1))):
        y = y0 + 50 + row * 110
        cv.text(x0, y + 24, lab, step, 17, INK, math=False)
        for i, b in enumerate(bits):
            _box(cv, x0 + 110 + i * c, y, c - 6, 48, step, b, "#ffc9c9" if i == hl else ("#ffec99" if names[i][0] == "p"
                                                                                          else "#d0ebff"), 20)
            cv.text(x0 + 110 + i * c + c / 2, y + 52, f"{i + 1} {names[i]}", step, 12, MUTED, "center", "top", math=False)
        step += 1
    _note(cv, box, step, f"parity checks on the received bits:\np1 covers 1,3,5,7 → {s1}\np2 covers 2,3,6,7 → {s2}\n"
                         f"p3 covers 4,5,6,7 → {s3}\n\nsyndrome = {s3}{s2}{s1} (binary) = {syn}\n→ flip bit {syn} to fix it ✔\n\n"
                         "3 extra bits correct any\nsingle-bit error (used in ECC RAM)", 18, frac=0.68, dy=40)
    return step + 1


def crc_steps(msg, gen):
    pad = msg + "0" * (len(gen) - 1)
    cur, steps = list(pad), []
    for i in range(len(msg)):
        if cur[i] == "1":
            before = "".join(cur)
            for j, g in enumerate(gen):
                cur[i + j] = str(int(cur[i + j]) ^ int(g))
            steps.append((i, before, "".join(cur)))
    return pad, steps, "".join(cur[-(len(gen) - 1):])


def build_crc(cv, v, box, step):
    msg, gen = str(v.get("message") or "11010011101100"), str(v.get("generator") or "1011")
    pad, steps, rem = crc_steps(msg, gen)
    x0, y0, x1, y1 = box
    W = x1 - x0
    cv.text(x0, y0, f"message {msg}, generator {gen} (x³ + x + 1): append {len(gen) - 1} zeros, then XOR-divide", step, 18,
            "#e8590c", math=False)
    step += 1
    cw = 18
    show = steps[:9]
    for k, (i, before, after) in enumerate(show):
        y = y0 + 40 + k * 30
        _mono(cv.text(x0, y, before, step, 17, MUTED, math=False))
        _mono(cv.text(x0 + i * 10.2, y + 14, gen, step, 17, "#e03131", math=False))
        step += 1
    _note(cv, box, step, f"remainder (the CRC) = {rem}\n\nsend {msg}{rem}\n\nthe receiver divides the whole\nthing by {gen}: "
                         "remainder 0\nmeans no error detected\n\n(Ethernet uses a 32-bit CRC)", 19, frac=0.62, dy=50)
    return step + 1


def build_manchester(cv, v, box, step):
    bits = str(v.get("bits") or "10110010")
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    bw = W * 0.85 / len(bits)
    for i, b in enumerate(bits):
        cv.text(x0 + 120 + i * bw + bw / 2, y0, b, step, 22, INK, "center", "top", math=False)
        cv.line([(x0 + 120 + i * bw, y0 + 30), (x0 + 120 + i * bw, y1 - 20)], step, "#dee2e6", 1, dashed=True)
    step += 1
    hi, lo = 0, 50
    for row, (name, fn) in enumerate((("NRZ", lambda b: [(0, b), (1, b)]),
                                      ("Manchester", lambda b: [(0, 1 - b), (0.5, 1 - b), (0.5, b), (1, b)]))):
        base = y0 + 70 + row * 140
        cv.text(x0, base + 25, name, step, 18, "#e8590c", math=False)
        pts = []
        for i, b in enumerate(bits):
            for t, lv in fn(int(b)):
                pts.append((x0 + 120 + (i + t) * bw, base + (lo - 50 * lv)))
        _pl(cv, pts, step, STRONG[row], 3)
        step += 1
    cv.text(x0, y0 + 360, "Manchester (IEEE 802.3): 1 = low→high, 0 = high→low in the middle of each bit.\n"
                          "Every bit has a transition, so the receiver can recover the clock; NRZ loses sync on long runs.",
            step, 17, INK, math=False)
    return step + 1


ACL = [("permit", "tcp", "10.1.1.0/24", "any", 80), ("permit", "tcp", "10.1.1.0/24", "any", 443),
       ("deny", "tcp", "any", "10.2.2.10", 22), ("permit", "ip", "10.1.0.0/16", "any", None)]


def _in_net(ip, net):
    if net == "any":
        return True
    if "/" not in net:
        return ip == net
    n, p = net.split("/")
    to = lambda s: sum(int(o) << (8 * (3 - i)) for i, o in enumerate(s.split(".")))
    m = (0xFFFFFFFF << (32 - int(p))) & 0xFFFFFFFF
    return to(ip) & m == to(n) & m


def build_acl_rules(cv, v, box, step):
    acl = v.get("acl") or ACL
    pkts = v.get("packets") or [("10.1.1.5", "8.8.8.8", "tcp", 443), ("10.1.9.7", "10.2.2.10", "tcp", 22),
                                ("10.1.9.7", "93.184.216.34", "udp", 53), ("172.16.0.4", "10.2.2.10", "tcp", 80)]
    x0, y0, x1, y1 = box
    W = x1 - x0
    for i, (act, pr, src, dst, port) in enumerate(acl):
        _mono(cv.text(x0, y0 + i * 28, f"{10 * (i + 1)} {act} {pr} {src} {dst}" + (f" eq {port}" if port else ""), step, 16,
                      "#2f9e44" if act == "permit" else "#e03131", math=False))
    _mono(cv.text(x0, y0 + len(acl) * 28, "   (implicit) deny any", step, 16, MUTED, math=False))
    step += 1
    for k, (s, d, pr, port) in enumerate(pkts):
        hit = None
        for i, (act, apr, src, dst, aport) in enumerate(acl):
            if (apr == "ip" or apr == pr) and _in_net(s, src) and _in_net(d, dst) and (aport is None or aport == port):
                hit = (i, act)
                break
        res = f"line {10 * (hit[0] + 1)} → {hit[1].upper()}" if hit else "no match → implicit DENY"
        ok = hit and hit[1] == "permit"
        cv.text(x0, y0 + 190 + k * 40, f"{s} → {d} {pr}/{port}:  {res}", step, 18, "#2f9e44" if ok else "#e03131", math=False)
        step += 1
    _note(cv, box, step, "ACLs are read top to bottom;\nthe FIRST match wins and the\nrest are ignored\n\n"
                         "put specific lines above\ngeneral ones", 18, frac=0.7, dy=10)
    return step + 1


def build_vlan(cv, v, box, step):
    ports = v.get("ports") or {1: 10, 2: 10, 3: 20, 4: 20, 5: 10, 6: 20}
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    sx, sy = x0 + W * 0.33, y0 + H * 0.35
    draw_device(cv, "switch", sx, sy, 150, step)
    draw_device(cv, "router", sx, y0 + 20, 70, step)
    cv.line([(sx, y0 + 50), (sx, sy - 30)], step, "#7048e8", 4)
    cv.text(sx + 10, y0 + 70, "trunk (802.1Q tags)", step, 14, "#7048e8", math=False)
    col = {10: "#1971c2", 20: "#e8590c"}
    n = len(ports)
    for i, (p, vl) in enumerate(ports.items()):
        px = x0 + 30 + i * W * 0.6 / (n - 1)
        draw_device(cv, "laptop", px, y1 - 60, 50, step)
        cv.line([(sx - 60 + i * 120 / (n - 1), sy + 20), (px, y1 - 85)], step, col[vl], 2.5)
        cv.text(px, y1 - 28, f"port {p}\nVLAN {vl}", step, 13, col[vl], "center", "top", math=False)
    step += 1
    src = next(p for p, vl in ports.items() if vl == 10)
    for i, (p, vl) in enumerate(ports.items()):
        if vl == 10 and p != src:
            px = x0 + 30 + i * W * 0.6 / (n - 1)
            _envelope(cv, px - 20, y1 - 130, 36, step, "")
    cv.text(x0, y0 + H * 0.55, f"broadcast from port {src} (VLAN 10)\nreaches only VLAN 10 ports", step, 16, "#1971c2",
            math=False)
    step += 1
    _note(cv, box, step, "one switch, two separate\nbroadcast domains\n\nVLAN 10 ↔ 20 traffic must go\nthrough the router "
                         "(router-on-\na-stick) or a layer-3 switch\n\nswitchport mode access\nswitchport access vlan 10", 17,
          frac=0.72)
    return step + 1


# ======================================================================== cloud

def build_aws_vpc(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    vpc = cv.rect(x0 + 10, y0 + 40, W * 0.72, H - 50, step, "transparent", "#2f9e44", 2.5)
    vpc["strokeStyle"] = "dashed"
    cv.text(x0 + 20, y0 + 46, "VPC 10.0.0.0/16", step, 16, "#2f9e44", math=False)
    draw_thing(cv, "Internet gateway", x0 + W * 0.37, y0 + 18, 52, step, "router")
    cv.text(x0 + W * 0.37 + 34, y0 + 14, "internet gateway", step, 14, INK, math=False)
    step += 1
    azw = (W * 0.72 - 30) / 2
    for a in range(2):
        ax = x0 + 20 + a * (azw + 10)
        az = cv.rect(ax, y0 + 74, azw, H - 94, step, "transparent", "#868e96", 1.5)
        az["strokeStyle"] = "dashed"
        cv.text(ax + 6, y0 + 78, f"Availability Zone {'ab'[a]}", step, 13, MUTED, math=False)
        for s, (name, cidr, col, kids) in enumerate((("public subnet", f"10.0.{a + 1}.0/24", "#d3f9d8",
                                                      [("NAT gateway", "router")] if a == 0 else []),
                                                     ("private subnet", f"10.0.{a + 3}.0/24", "#d0ebff",
                                                      [("Amazon EC2", "server")]),
                                                     ("private subnet", f"10.0.{a + 5}.0/24", "#e5dbff",
                                                      [("Amazon RDS", "database")]))):
            sy = y0 + 100 + s * (H - 130) / 3
            cv.rect(ax + 8, sy, azw - 16, (H - 130) / 3 - 10, step, col, "transparent", 1)
            cv.text(ax + 14, sy + 4, f"{name} {cidr}", step, 12, INK, math=False)
            for kname, kind in kids:
                draw_thing(cv, kname, ax + azw / 2, sy + (H - 130) / 6 + 4, 46, step, kind)
                cv.text(ax + azw / 2 + 30, sy + (H - 130) / 6 + 4, kname.replace("Amazon ", ""), step, 13, INK, "left",
                        "middle", math=False)
        step += 1
    cv.rect(x0 + 30, y0 + 100 + 30, W * 0.72 - 60, 26, step, "#fff3bf", "#f08c00", 1.5)
    cv.text(x0 + W * 0.37, y0 + 143, "Application Load Balancer (spans both public subnets)", step, 13, INK, "center",
            "middle", math=False)
    step += 1
    _note(cv, box, step, "route tables:\npublic: 0.0.0.0/0 → IGW\nprivate: 0.0.0.0/0 → NAT\n\nEC2 can reach the internet\n"
                         "for updates (via NAT) but\nnothing can reach it directly\n\nRDS Multi-AZ: standby in AZ b\n"
                         "takes over on failure", 16, frac=0.76, dy=50)
    return step + 1


def build_iam_policy(cv, v, box, step):
    stmts = v.get("statements") or [("Allow", "s3:GetObject", "arn:aws:s3:::reports/*"),
                                    ("Allow", "s3:PutObject", "arn:aws:s3:::reports/uploads/*"),
                                    ("Deny", "s3:*", "arn:aws:s3:::reports/secret/*"),
                                    ("Allow", "ec2:Describe*", "*")]
    reqs = v.get("requests") or [("s3:GetObject", "arn:aws:s3:::reports/q3.pdf"),
                                 ("s3:GetObject", "arn:aws:s3:::reports/secret/keys.txt"),
                                 ("s3:DeleteObject", "arn:aws:s3:::reports/q3.pdf"),
                                 ("ec2:DescribeInstances", "*")]
    x0, y0, x1, y1 = box
    W = x1 - x0
    for i, (eff, act, res) in enumerate(stmts):
        _mono(cv.text(x0, y0 + i * 28, f"{eff:5s} {act:16s} {res}", step, 15, "#2f9e44" if eff == "Allow" else "#e03131",
                      math=False))
    step += 1
    for k, (act, res) in enumerate(reqs):
        m = [eff for eff, a, r in stmts if fnmatch.fnmatchcase(act, a) and fnmatch.fnmatchcase(res, r)]
        verdict = "DENY (explicit deny wins)" if "Deny" in m else "ALLOW" if "Allow" in m else "DENY (nothing allows it)"
        cv.text(x0, y0 + 150 + k * 40, f"{act} on {res.split(':::')[-1]}  →  {verdict}", step, 17,
                "#2f9e44" if verdict == "ALLOW" else "#e03131", math=False)
        step += 1
    cv.text(x0, y0 + 330, "evaluation: 1) any explicit Deny → deny   2) else any Allow → allow   3) else implicit deny",
            step, 17, INK, math=False)
    return step + 1


def build_circuit_breaker(cv, v, box, step):
    calls = v.get("calls") or list("OOXXXOOOOOOOXOO")
    thresh, cool = 3, 4
    state, fails, opened, tl = "closed", 0, None, []
    for t, c in enumerate(calls):
        if state == "open" and t - opened >= cool:
            state = "half-open"
        if state == "open":
            tl.append((state, "fast fail"))
            continue
        ok = c == "O"
        if state == "half-open":
            state = "closed" if ok else "open"
            if not ok:
                opened = t
            fails = 0
            tl.append(("half-open", "trial ✔" if ok else "trial ✘"))
            continue
        fails = 0 if ok else fails + 1
        tl.append(("closed", "✔" if ok else "✘"))
        if fails >= thresh:
            state, opened = "open", t
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    col = {"closed": "#b2f2bb", "open": "#ffc9c9", "half-open": "#fff3bf"}
    for i, s in enumerate(("closed", "open", "half-open")):
        _box(cv, x0 + i * W * 0.3, y0, W * 0.24, 50, step, s, col[s], 18)
    cv.arrow(x0 + W * 0.24 + 4, y0 + 18, x0 + W * 0.3 - 4, y0 + 18, step, f"{thresh} failures", color="#e03131")
    cv.arrow(x0 + W * 0.54 + 4, y0 + 18, x0 + W * 0.6 - 4, y0 + 18, step, "timeout", color="#f08c00")
    cv.arrow(x0 + W * 0.6, y0 + 52, x0 + W * 0.12, y0 + 70, step, "trial succeeds", color="#2f9e44")
    step += 1
    c = min(56, W * 0.95 / len(tl))
    for t, (s, r) in enumerate(tl):
        _box(cv, x0 + t * c, y0 + 130, c - 4, 50, step, r, col[s], 12)
        cv.text(x0 + t * c + c / 2, y0 + 184, str(t), step, 12, MUTED, "center", "top", math=False)
        if t % 3 == 2:
            step += 1
    cv.text(x0, y0 + 220, "after 3 failures the breaker opens: calls fail fast instead of waiting on a dead service;\n"
                          "after a cool-down one trial call is let through; success closes it again", step + 1, 17, INK,
            math=False)
    return step + 2


def build_raft_election(cv, v, box, step):
    rnd = random.Random(int(v.get("seed") or 3))
    n = 5
    timeouts = [rnd.randint(150, 300) for _ in range(n)]
    cand = min(range(n), key=lambda i: timeouts[i])
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    R = min(W * 0.25, H * 0.38)
    cx, cy = x0 + R + 60, (y0 + y1) / 2
    P = lambda i: (cx + R * math.sin(2 * math.pi * i / n), cy - R * math.cos(2 * math.pi * i / n))

    def nodes(st, roles, votes=()):
        for i in range(n):
            x, y = P(i)
            r = roles[i]
            draw_device(cv, "server", x, y, 56, st, "#b2f2bb" if r == "leader" else "#fff3bf" if r == "candidate" else None)
            cv.text(x, y + 34, f"S{i + 1} {r}\ntimeout {timeouts[i]} ms" + (" ✔vote" if i in votes else ""), st, 13,
                    INK, "center", "top", math=False)
    frames = [(["follower"] * n, (), "term 1: leader crashed; followers wait for heartbeats"),
              ([("candidate" if i == cand else "follower") for i in range(n)], (cand,),
               f"S{cand + 1} times out first ({timeouts[cand]} ms) → term 2, votes for itself, asks others"),
              ([("candidate" if i == cand else "follower") for i in range(n)], tuple(range(n)),
               "each follower grants one vote per term → 5 of 5 votes (majority = 3)"),
              ([("leader" if i == cand else "follower") for i in range(n)], (),
               f"S{cand + 1} is leader of term 2 and sends heartbeats")]

    def draw(k, st):
        roles, votes, msg = frames[k]
        nodes(st, roles, votes if k == 2 else ())
        if k in (1, 3):
            for i in range(n):
                if i != cand:
                    cv.arrow(*P(cand), *P(i), st, color="#1971c2" if k == 1 else "#2f9e44")
        cv.text(x0 + W * 0.6, y0 + 40, msg, st, 19, "#e8590c", math=False)
    step = _frames(cv, step, len(frames), draw)
    _note(cv, box, step, "random timeouts make split votes\nrare; a leader needs a majority,\nso a 5-server cluster survives\n"
                         "2 failures\n\n(etcd, Consul, Kubernetes)", 18, frac=0.6, dy=140)
    return step + 1


def build_sharding(cv, v, box, step):
    ids = list(range(1, 1001))
    recent = list(range(901, 1001))
    shards = 3
    rng = lambda i: min(shards - 1, (i - 1) * shards // 1000)
    hsh = lambda i: int(hashlib.md5(str(i).encode()).hexdigest(), 16) % shards
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    for r, (name, fn, desc) in enumerate((("range sharding (ids 1-333, 334-666, 667-1000)", rng,
                                           "new users all land on the last shard → hotspot"),
                                          ("hash sharding (md5(id) % 3)", hsh, "load spreads evenly, but range queries\n"
                                                                              "must ask every shard"))):
        y = y0 + r * (H / 2)
        cv.text(x0, y, name, step, 18, "#e8590c", math=False)
        tot = [sum(fn(i) == s for i in ids) for s in range(shards)]
        hot = [sum(fn(i) == s for i in recent) for s in range(shards)]
        for s in range(shards):
            bx = x0 + s * W * 0.22
            draw_device(cv, "database", bx + 40, y + 70, 60, step)
            cv.text(bx + 80, y + 50, f"shard {s}\n{tot[s]} users\n{hot[s]}% of today's\nsign-ups", step, 14,
                    "#e03131" if hot[s] > 60 else INK, math=False)
        cv.text(x0 + W * 0.68, y + 50, desc, step, 16, INK, math=False)
        step += 1
    return step


# ======================================================================== security

def _h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def build_merkle_tree(cv, v, box, step):
    txs = v.get("transactions") or ["Ana→Ben 5", "Ben→Cy 2", "Cy→Dee 9", "Dee→Ana 1"]
    change = int(v.get("change") or 2)

    def build(tx):
        l0 = [_h(t) for t in tx]
        l1 = [_h(l0[0] + l0[1]), _h(l0[2] + l0[3])]
        return [l0, l1, [_h(l1[0] + l1[1])]]
    a = build(txs)
    t2 = list(txs)
    t2[change] = t2[change].replace("9", "90")
    b = build(t2)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0

    def draw(levels, txl, st, cmp=None):
        for d, lvl in enumerate(levels):
            y = y1 - 120 - d * (H - 160) / 2
            for i, hsh in enumerate(lvl):
                x = x0 + W * 0.62 * (i + 0.5) / len(lvl)
                diff = cmp and cmp[d][i] != hsh
                _box(cv, x - 70, y, 140, 40, st, hsh[:8] + "…", "#ffc9c9" if diff else "#e7f5ff" if d < 2 else "#ffec99",
                     15, mono=True)
                if d:
                    for j in (2 * i, 2 * i + 1):
                        cx = x0 + W * 0.62 * (j + 0.5) / len(levels[d - 1])
                        cv.line([(x, y + 40), (cx, y + (H - 160) / 2)], st, MUTED, 1.2)
        for i, t in enumerate(txl):
            x = x0 + W * 0.62 * (i + 0.5) / 4
            cv.text(x, y1 - 60, t, st, 14, "#e03131" if cmp and t != txs[i] else INK, "center", "top", math=False)
            cv.line([(x, y1 - 64), (x, y1 - 80)], st, MUTED, 1)
    tr = _Track(cv)
    draw(a, txs, step)
    cv.text(x0, y0, "Merkle root = SHA-256 of the two hashes below it, all the way up", step, 17, "#e8590c", math=False)
    tr.tag(until=step + 1)
    step += 1
    draw(b, t2, step, a)
    cv.text(x0, y0, f"change one transaction ({txs[change]} → {t2[change]}): its whole path to the root changes", step, 17,
            "#e03131", math=False)
    _note(cv, box, step, "a block header stores only the\nroot; anyone can prove one\ntransaction is included with\n"
                         "log₂(n) hashes (Bitcoin, Git)", 17, frac=0.68, dy=60)
    return step + 1


def build_port_scan(cv, v, box, step):
    host = v.get("host") or {22: "open", 23: "closed", 80: "open", 443: "open", 3306: "filtered", 3389: "closed",
                             8080: "filtered"}
    svc = {22: "ssh", 23: "telnet", 80: "http", 443: "https", 3306: "mysql", 3389: "rdp", 8080: "http-alt"}
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    draw_device(cv, "attacker", x0 + 40, y0 + H * 0.4, 70, step)
    draw_device(cv, "server", x0 + W * 0.5, y0 + H * 0.4, 80, step)
    cv.text(x0 + W * 0.5, y0 + H * 0.4 + 50, "target 10.0.0.5", step, 15, INK, "center", "top", math=False)
    step += 1
    rows = []
    reply = {"open": "SYN-ACK", "closed": "RST", "filtered": "(no reply)"}
    for k, (p, st) in enumerate(host.items()):
        tr = _Track(cv)
        cv.arrow(x0 + 80, y0 + H * 0.4 - 20, x0 + W * 0.5 - 40, y0 + H * 0.4 - 20, step, f"SYN → port {p}", color="#1971c2")
        if st != "filtered":
            cv.arrow(x0 + W * 0.5 - 40, y0 + H * 0.4 + 10, x0 + 80, y0 + H * 0.4 + 10, step, reply[st],
                     color="#2f9e44" if st == "open" else "#e03131")
        tr.tag(until=step + 1, fade=True)
        rows.append([p, svc.get(p, "?"), reply[st], st.upper()])
        step += 1
    _table(cv, x0 + W * 0.62, y0 + 30, "scan result", ["port", "service", "reply", "state"], rows, step, W * 0.37, size=14)
    cv.text(x0, y1 - 40, "SYN scan: never completes the handshake. Defence: close unused ports, firewall, IDS alerts. "
                         "Only scan hosts you own!", step, 15, INK, math=False)
    return step + 1


def build_cert_chain(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    chain = [("Root CA\n(ISRG Root X1)", "self-signed; ships inside\nyour OS / browser trust store", "#ffec99"),
             ("Intermediate CA\n(R3)", "signed by the root's\nprivate key", "#d0ebff"),
             ("example.com\n(leaf certificate)", "signed by the intermediate;\nholds the site's public key", "#d3f9d8")]
    for i, (name, note, col) in enumerate(chain):
        y = y0 + 20 + i * (H - 60) / 3
        _box(cv, x0 + 40, y, W * 0.3, 80, step, name, col, 17)
        cv.text(x0 + 60 + W * 0.3, y + 40, note, step, 15, INK, "left", "middle", math=False)
        if i:
            cv.arrow(x0 + 40 + W * 0.15, y - 2, x0 + 40 + W * 0.15, y - (H - 60) / 3 + 82, step, "verify signature ↑",
                     color="#7048e8")
        step += 1
    draw_device(cv, "browser", x0 + W * 0.78, y0 + H * 0.5, 90, step)
    cv.text(x0 + W * 0.78, y0 + H * 0.5 + 55, "browser walks up the chain until\nit reaches a root it already trusts;\n"
                                              "also checks the name, dates and\nrevocation", step, 15, INK, "center", "top",
            math=False)
    return step + 1


# ======================================================================== ML / NLP

def build_activations(cv, v, box, step):
    fns = [("ReLU", lambda x: max(0, x)), ("sigmoid", lambda x: 1 / (1 + math.exp(-x))), ("tanh", math.tanh),
           ("GELU", lambda x: 0.5 * x * (1 + math.erf(x / math.sqrt(2))))]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    pw, ph = W * 0.36, (H - 40) / 2
    for i, (name, f) in enumerate(fns):
        bx, by = x0 + (i % 2) * (pw + 40), y0 + (i // 2) * (ph + 20)
        sx = lambda x: bx + (x + 4) / 8 * pw
        sy = lambda y: by + ph * 0.66 - min(2.5, y) / 4 * ph
        cv.line([(bx, sy(0)), (bx + pw, sy(0))], step, MUTED, 1)
        cv.line([(sx(0), by), (sx(0), by + ph)], step, MUTED, 1)
        _pl(cv, [(sx(t / 10), sy(f(t / 10))) for t in range(-40, 41)], step, STRONG[i], 3)
        cv.text(bx + 6, by + 4, name, step, 18, STRONG[i], math=False)
        step += 1
    _note(cv, box, step, "without a non-linear activation,\nstacked layers collapse into\none linear layer\n\n"
                         "ReLU: fast, the default\nsigmoid: 0..1, for outputs\ntanh: −1..1, in RNNs\n"
                         "GELU: smooth ReLU in\ntransformers", 17, frac=0.8)
    return step + 1


def build_naive_bayes(cv, v, box, step):
    train = v.get("train") or [("win money now", "spam"), ("free money offer", "spam"), ("win a free prize", "spam"),
                               ("meeting at noon", "ham"), ("lunch money tomorrow", "ham"), ("project meeting now", "ham")]
    test = str(v.get("message") or "free money now")
    vocab = sorted({w for t, _ in train for w in t.split()})
    cls = ["spam", "ham"]
    words = {c: [w for t, l in train if l == c for w in t.split()] for c in cls}
    prior = {c: sum(l == c for _, l in train) / len(train) for c in cls}
    like = lambda w, c: (words[c].count(w) + 1) / (len(words[c]) + len(vocab))
    score = {c: prior[c] * math.prod(like(w, c) for w in test.split()) for c in cls}
    tot = sum(score.values())
    x0, y0, x1, y1 = box
    W = x1 - x0
    rows = [[t, l] for t, l in train]
    _table(cv, x0, y0 + 30, "training messages", ["text", "label"], rows, step, W * 0.4, size=14)
    step += 1
    rows2 = [[w, f"{like(w, 'spam'):.3f}", f"{like(w, 'ham'):.3f}"] for w in test.split()]
    _table(cv, x0 + W * 0.45, y0 + 30, f'P(word | class) for "{test}" (add-1 smoothing)', ["word", "spam", "ham"], rows2,
           step, W * 0.45, size=14)
    step += 1
    cv.text(x0, y0 + 280, f"P(spam | msg) ∝ {prior['spam']:.2f} × " + " × ".join(f"{like(w, 'spam'):.3f}" for w in test.split())
            + f"\nP(ham | msg) ∝ {prior['ham']:.2f} × " + " × ".join(f"{like(w, 'ham'):.3f}" for w in test.split()) +
            f"\n→ spam {score['spam'] / tot:.0%}, ham {score['ham'] / tot:.0%}: classified as "
            f"{max(score, key=score.get).upper()}", step, 18, "#e8590c", math=False)
    cv.text(x0, y0 + 380, "'naive' = assumes words are independent; still works well for text", step + 1, 16, MUTED,
            math=False)
    return step + 2


def dbscan(pts, eps, minp):
    nb = lambda i: [j for j in range(len(pts)) if math.dist(pts[i], pts[j]) <= eps]
    lab, c = [None] * len(pts), 0
    core = [len(nb(i)) >= minp for i in range(len(pts))]
    for i in range(len(pts)):
        if lab[i] is not None or not core[i]:
            continue
        lab[i] = c
        q = nb(i)
        while q:
            j = q.pop()
            if lab[j] is None:
                lab[j] = c
                if core[j]:
                    q += nb(j)
        c += 1
    return lab, core


def build_dbscan(cv, v, box, step):
    rnd = random.Random(5)
    pts = [(rnd.gauss(2, 0.4), rnd.gauss(2, 0.4)) for _ in range(18)]
    pts += [(4 + 1.6 * math.cos(t), 3 + 1.6 * math.sin(t) * 0.5) for t in [rnd.uniform(0, math.pi) for _ in range(18)]]
    pts += [(rnd.uniform(0, 6), rnd.uniform(0, 5)) for _ in range(6)]
    eps, minp = 0.6, 4
    lab, core = dbscan(pts, eps, minp)
    px, py = _plot(cv, box, step, (0, 6.5), (0, 5.2), "", "", frac=0.58)
    for x, y in pts:
        cv.ellipse(px(x) - 5, py(y) - 5, 10, 10, step, "#dee2e6", MUTED, 1)
    step += 1
    for i, (x, y) in enumerate(pts):
        l = lab[i]
        if l is None:
            cv.text(px(x), py(y), "✕", step, 14, "#868e96", "center", "middle", math=False)
            continue
        r = 8 if core[i] else 5
        cv.ellipse(px(x) - r, py(y) - r, 2 * r, 2 * r, step, PAL[l], STRONG[l], 2 if core[i] else 1)
    step += 1
    i0 = next(i for i in range(len(pts)) if core[i])
    rr = px(eps) - px(0)
    cv.ellipse(px(pts[i0][0]) - rr, py(pts[i0][1]) - rr, 2 * rr, 2 * rr, step, "transparent", "#e8590c", 1.5)
    step += 1
    nc = len(set(l for l in lab if l is not None))
    _note(cv, box, step, f"ε = {eps}, minPts = {minp}\n\nbig dots: core (≥ {minp} neighbours\nwithin ε, orange circle)\n"
                         f"small: border\n✕: noise\n\nfound {nc} clusters and\n{lab.count(None)} noise points; any shape,\n"
                         "no need to pick k", 18)
    return step + 1


def build_dendrogram(cv, v, box, step):
    pts = v.get("points") or {"A": (1, 1), "B": (1.5, 1.2), "C": (5, 5), "D": (5.5, 5.3), "E": (3, 4), "F": (9, 1)}
    clusters = {k: [k] for k in pts}
    hpos = {k: 0 for k in pts}
    order = []
    merges = []
    while len(clusters) > 1:
        best = None
        for a, b in itertools.combinations(sorted(clusters), 2):
            d = min(math.dist(pts[p], pts[q]) for p in clusters[a] for q in clusters[b])
            if best is None or d < best[0]:
                best = (d, a, b)
        d, a, b = best
        new = a + b
        clusters[new] = clusters.pop(a) + clusters.pop(b)
        merges.append((a, b, new, d))
    leaves = clusters[next(iter(clusters))]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    maxd = merges[-1][3]
    X = {k: x0 + 40 + i * W * 0.55 / (len(leaves) - 1) for i, k in enumerate(leaves)}
    Y = {k: y1 - 40 for k in leaves}
    for k in leaves:
        cv.text(X[k], y1 - 34, k, step, 18, INK, "center", "top", math=False)
    step += 1
    for a, b, new, d in merges:
        y = y1 - 40 - d / maxd * (H - 80)
        _pl(cv, [(X[a], Y[a]), (X[a], y), (X[b], y), (X[b], Y[b])], step, "#1971c2", 2.5)
        cv.text((X[a] + X[b]) / 2, y - 4, f"{d:.1f}", step, 13, MUTED, "center", "bottom", math=False)
        X[new], Y[new] = (X[a] + X[b]) / 2, y
        step += 1
    _note(cv, box, step, "single linkage: repeatedly merge\nthe two closest clusters\n(distance = closest pair)\n\n"
                         "height = merge distance; cut\nthe tree at any height to get\nthat many clusters\n\n"
                         "F joins last: an outlier", 17, frac=0.68)
    return step + 1


def build_cosine_similarity(cv, v, box, step):
    vecs = v.get("vectors") or {"dog": (4, 1.2), "puppy": (3.6, 1.6), "cat": (3, 2.4), "car": (0.6, 4)}
    q = str(v.get("query") or "dog")
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    s = min(W * 0.55, H - 40) / 4.5
    O = (x0 + 30, y1 - 30)
    P = lambda a, b: (O[0] + a * s, O[1] - b * s)
    cv.line([O, P(4.5, 0)], step, MUTED, 1)
    cv.line([O, P(0, 4.5)], step, MUTED, 1)
    for i, (w, (a, b)) in enumerate(vecs.items()):
        cv.line([O, P(a, b)], step, STRONG[i], 3, arrow_end=True)
        cv.text(*P(a + 0.1, b + 0.1), w, step, 18, STRONG[i], math=False)
    step += 1
    qa = vecs[q]
    cos = lambda u, w: (u[0] * w[0] + u[1] * w[1]) / (math.hypot(*u) * math.hypot(*w))
    lines = [f"cos({q}, {w}) = {cos(qa, vv):.3f}  (angle {math.degrees(math.acos(min(1, cos(qa, vv)))):.0f}°)"
             for w, vv in vecs.items() if w != q]
    _note(cv, box, step, "cos θ = (a · b) / (|a| |b|)\n\n" + "\n".join(lines) +
          "\n\n1 = same direction (similar\nmeaning), 0 = unrelated\nlength doesn't matter, only angle", 18, frac=0.6)
    return step + 1


def build_temperature_sampling(cv, v, box, step):
    toks = v.get("tokens") or ["Paris", "Lyon", "France", "beautiful", "the"]
    logits = v.get("logits") or [5.0, 3.0, 2.5, 1.5, 1.0]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cv.text(x0, y0, '"The capital of France is ___"   logits: ' + ", ".join(f"{t} {l:g}" for t, l in zip(toks, logits)),
            step, 16, INK, math=False)
    step += 1
    for k, T in enumerate((0.5, 1.0, 2.0)):
        e = [math.exp(l / T) for l in logits]
        p = [x / sum(e) for x in e]
        bx = x0 + k * W * 0.33
        cv.text(bx, y0 + 40, f"temperature {T:g}", step, 18, "#e8590c", math=False)
        bw = (W * 0.3) / len(toks)
        for i, (t, pp) in enumerate(zip(toks, p)):
            h = (H - 150) * pp
            cv.rect(bx + i * bw, y1 - 50 - h, bw - 6, h, step, PAL[i], STRONG[i], 1.5, rounded=False)
            cv.text(bx + i * bw + bw / 2, y1 - 54 - h, f"{pp:.0%}", step, 13, INK, "center", "bottom", math=False)
            cv.text(bx + i * bw + bw / 2, y1 - 46, t, step, 12, INK, "center", "top", math=False)
        step += 1
    cv.text(x0, y1 - 20, "p = softmax(logits / T): low T → confident, repeatable; high T → varied, riskier", step, 16, INK,
            math=False)
    return step + 1


def build_positional_encoding(cv, v, box, step):
    npos, d = 24, 16
    pe = [[math.sin(p / 10000 ** (i / d)) if i % 2 == 0 else math.cos(p / 10000 ** ((i - 1) / d)) for i in range(d)]
          for p in range(npos)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    c = min(W * 0.55 / d, (H - 40) / npos)

    def col(val):
        t = (val + 1) / 2
        r, g, b = int(255 * (1 - t) + 25 * t), int(255 * (1 - t) + 113 * t), int(255 * (1 - t) + 194 * t)
        if val < 0:
            t = -val
            r, g, b = int(255 * (1 - t) + 224 * t), int(255 * (1 - t) + 49 * t), int(255 * (1 - t) + 49 * t)
        return f"#{r:02x}{g:02x}{b:02x}"
    for p in range(npos):
        for i in range(d):
            cv.rect(x0 + 40 + i * c, y0 + 20 + p * c, c, c, step, col(pe[p][i]), "transparent", 1, rounded=False)
        if p % 6 == 5:
            step += 1
    cv.text(x0 + 40, y0 + 4, "dimension →", step, 14, MUTED, math=False)
    cv.text(x0, y0 + 20, "position\n↓", step, 14, MUTED, math=False)
    step += 1
    _note(cv, box, step, "PE(pos, 2i) = sin(pos / 10000^(2i/d))\nPE(pos, 2i+1) = cos(...)\n\nleft columns change fast,\n"
                         "right columns slowly: every\nposition gets a unique pattern\n(like a binary clock)\n\n"
                         "added to the token embeddings\nso attention knows word order", 17, frac=0.62)
    return step + 1


def build_learning_rate(cv, v, box, step):
    rates = [(0.01, "too small: slow"), (0.1, "good"), (0.9, "big: zig-zags"), (1.05, "too big: diverges")]
    T = 30
    px, py = _plot(cv, box, step, (0, T), (0, 10), "step", "loss f(x) = x²", frac=0.55)
    step += 1
    for i, (lr, lab) in enumerate(rates):
        x, pts = 3.0, []
        for t in range(T + 1):
            f = x * x
            if f > 10:
                pts.append((px(t), py(10)))
                break
            pts.append((px(t), py(f)))
            x -= lr * 2 * x
        _pl(cv, pts, step, STRONG[i], 3)
        cv.text(box[0] + (box[2] - box[0]) * 0.66, box[1] + 40 + i * 28, f"lr {lr:g}: {lab}", step, 17, STRONG[i],
                math=False)
        step += 1
    _note(cv, box, step, "x ← x − lr · f′(x), start x = 3\n\nthe learning rate is the most\nimportant hyper-parameter;\n"
                         "schedules start high and decay", 17, frac=0.66, dy=180)
    return step + 1


def build_rag_pipeline(cv, v, box, step):
    chunks = v.get("chunks") or {"refund policy: 30 days with receipt": (0.9, 0.2, 0.1),
                                 "shipping takes 3-5 business days": (0.1, 0.9, 0.2),
                                 "returns need the original box": (0.8, 0.3, 0.2),
                                 "store hours: 9am to 6pm": (0.1, 0.2, 0.9)}
    qv = (0.85, 0.25, 0.15)
    question = str(v.get("question") or "Can I return my order?")
    cos = lambda a, b: sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
    ranked = sorted(chunks, key=lambda c: -cos(qv, chunks[c]))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    steps = [("documents", "file"), ("chunk + embed", "chip"), ("vector database", "database")]
    for i, (n, kind) in enumerate(steps):
        draw_device(cv, kind, x0 + 50 + i * W * 0.17, y0 + 40, 54, step)
        cv.text(x0 + 50 + i * W * 0.17, y0 + 74, n, step, 14, INK, "center", "top", math=False)
        if i:
            cv.arrow(x0 + 80 + (i - 1) * W * 0.17, y0 + 40, x0 + 20 + i * W * 0.17, y0 + 40, step, color=MUTED)
    cv.text(x0 + W * 0.55, y0 + 30, "offline: index once", step, 15, MUTED, math=False)
    step += 1
    draw_device(cv, "user", x0 + 50, y0 + 190, 54, step)
    cv.text(x0 + 90, y0 + 180, f'"{question}" → embed → {qv}', step, 16, "#1971c2", math=False)
    step += 1
    for i, ch in enumerate(ranked):
        cv.text(x0 + 90, y0 + 220 + i * 26, f"{cos(qv, chunks[ch]):.3f}  {ch}", step, 15,
                "#2f9e44" if i < 2 else MUTED, math=False)
    step += 1
    _box(cv, x0 + 40, y0 + 340, W * 0.62, 70, step, "prompt = question + top-2 chunks → LLM → answer grounded in your "
                                                    "documents (with sources)", "#fff3bf", 15)
    step += 1
    _note(cv, box, step, "retrieval-augmented generation:\nthe model looks things up\ninstead of guessing\n\n"
                         "fresh data without retraining;\nfewer hallucinations", 17, frac=0.74, dy=180)
    return step + 1


# ======================================================================== data

def build_window_functions(cv, v, box, step):
    rows = v.get("rows") or [("Ana", "Jan", 120), ("Ana", "Feb", 90), ("Ana", "Mar", 150), ("Ben", "Jan", 80),
                             ("Ben", "Feb", 130), ("Ben", "Mar", 110)]
    q = ("SELECT rep, month, sales,\n  SUM(sales) OVER (PARTITION BY rep ORDER BY m) AS running,\n"
         "  RANK() OVER (PARTITION BY month ORDER BY sales DESC) AS rnk,\n"
         "  sales - LAG(sales) OVER (PARTITION BY rep ORDER BY m) AS change\nFROM s ORDER BY rep, m")
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE s(rep, month, sales, m)")
    order = {"Jan": 1, "Feb": 2, "Mar": 3}
    db.executemany("INSERT INTO s VALUES (?,?,?,?)", [(a, b, c, order[b]) for a, b, c in rows])
    out = db.execute(q).fetchall()
    x0, y0, x1, y1 = box
    W = x1 - x0
    for i, ln in enumerate(q.split("\n")):
        _mono(cv.text(x0, y0 + i * 24, ln, step, 15, "#1971c2", math=False))
    step += 1
    _table(cv, x0, y0 + 160, "result (real SQLite)", ["rep", "month", "sales", "running", "rank", "change"],
           [[r if r is not None else "NULL" for r in row] for row in out], step, W * 0.7, size=14)
    step += 1
    _note(cv, box, step, "window functions compute over\na group of rows but keep every\nrow (GROUP BY would collapse\n"
                         "them)\n\nrunning total, ranking, and\nmonth-over-month change", 17, frac=0.74, dy=160)
    return step + 1


def build_standardization(cv, v, box, step):
    a = v.get("a") or ("math test", [55, 62, 70, 74, 78, 85, 91], 85)
    b = v.get("b") or ("history test", [70, 75, 80, 82, 84, 88, 95], 88)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    res = []
    for r, (name, data, mine) in enumerate((a, b)):
        m = sum(data) / len(data)
        sd = math.sqrt(sum((x - m) ** 2 for x in data) / len(data))
        z = (mine - m) / sd
        res.append((name, mine, z))
        y = y0 + 30 + r * 140
        X = lambda val: x0 + 30 + (val - 50) / 50 * W * 0.6
        cv.line([(X(50), y + 40), (X(100), y + 40)], step, INK, 1.5)
        for t in range(50, 101, 10):
            cv.text(X(t), y + 46, str(t), step, 12, MUTED, "center", "top", math=False)
        for x in data:
            cv.ellipse(X(x) - 6, y + 34, 12, 12, step, "#a5d8ff", "#1971c2", 1)
        cv.line([(X(m), y + 20), (X(m), y + 60)], step, "#2f9e44", 2)
        cv.ellipse(X(mine) - 9, y + 31, 18, 18, step, "#ffc9c9", "#e03131", 2)
        cv.text(x0, y, f"{name}: mean {m:.1f}, sd {sd:.1f}; your {mine} → z = ({mine} − {m:.1f}) / {sd:.1f} = {z:+.2f}",
                step, 16, INK, math=False)
        step += 1
    better = max(res, key=lambda t: t[2])
    cv.text(x0, y0 + 320, f"the raw scores look similar, but the z-score shows the {better[0]} result is further above "
                          "the class\n(z = how many standard deviations from the mean). Standardise features before "
                          "k-NN, SVM, PCA.", step, 17, "#e8590c", math=False)
    return step + 1


# ======================================================================== maths

def build_dot_product(cv, v, box, step):
    a, b = v.get("a") or (4, 1), v.get("b") or (2, 3)
    dot = a[0] * b[0] + a[1] * b[1]
    la, lb = math.hypot(*a), math.hypot(*b)
    ang = math.degrees(math.acos(dot / (la * lb)))
    proj = dot / la
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    s = min(W * 0.55, H - 40) / 5
    O = (x0 + 40, y1 - 40)
    P = lambda x, y: (O[0] + x * s, O[1] - y * s)
    for k in range(6):
        cv.line([P(k, 0), P(k, 4.5)], step, "#f1f3f5", 1)
        cv.line([P(0, k * 0.9), P(5, k * 0.9)], step, "#f1f3f5", 1)
    cv.line([O, P(*a)], step, "#1971c2", 3.5, arrow_end=True)
    cv.text(*P(a[0] + 0.1, a[1]), f"a = {a}", step, 18, "#1971c2", math=False)
    cv.line([O, P(*b)], step, "#e03131", 3.5, arrow_end=True)
    cv.text(*P(b[0] + 0.1, b[1] + 0.1), f"b = {b}", step, 18, "#e03131", math=False)
    step += 1
    ux, uy = a[0] / la, a[1] / la
    foot = (ux * proj, uy * proj)
    cv.line([P(*b), P(*foot)], step, MUTED, 1.5, dashed=True)
    cv.line([O, P(*foot)], step, "#2f9e44", 6)
    step += 1
    _note(cv, box, step, f"a · b = {a[0]}×{b[0]} + {a[1]}×{b[1]} = {dot}\n|a| = {la:.2f}, |b| = {lb:.2f}\n"
                         f"cos θ = {dot}/({la:.2f}×{lb:.2f}) → θ = {ang:.1f}°\n\ngreen = projection of b on a\n= a·b / |a| = "
                         f"{proj:.2f}\n\ndot = 0 ⇔ perpendicular", 19, frac=0.6)
    return step + 1


def build_complex_plane(cv, v, box, step):
    z1, z2 = complex(*(v.get("z1") or (2, 1))), complex(*(v.get("z2") or (1, 2)))
    p = z1 * z2
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    s = min(W * 0.55, H) / 12
    cx, cy = x0 + W * 0.3, (y0 + y1) / 2 + 40
    P = lambda z: (cx + z.real * s, cy - z.imag * s)
    cv.line([(cx - 3 * s, cy), (cx + 6 * s, cy)], step, MUTED, 1.5, arrow_end=True)
    cv.line([(cx, cy + 2 * s), (cx, cy - 6 * s)], step, MUTED, 1.5, arrow_end=True)
    cv.text(cx + 6 * s, cy + 6, "real", step, 14, MUTED, "right", "top", math=False)
    cv.text(cx + 6, cy - 6 * s, "imaginary", step, 14, MUTED, math=False)
    step += 1
    pol = lambda z: (abs(z), math.degrees(math.atan2(z.imag, z.real)))
    for z, col, name in ((z1, "#1971c2", "z₁"), (z2, "#e03131", "z₂"), (p, "#2f9e44", "z₁z₂")):
        r, a = pol(z)
        cv.line([P(0), P(z)], step, col, 3, arrow_end=True)
        cv.text(*P(z + 0.15 + 0.15j), f"{name} = {z.real:g} {'+' if z.imag >= 0 else '−'} {abs(z.imag):g}i\n"
                                      f"(r {r:.2f}, {a:.1f}°)", step, 15, col, math=False)
        step += 1
    r1, a1 = pol(z1)
    r2, a2 = pol(z2)
    _note(cv, box, step, f"multiplying complex numbers:\nmultiply the lengths\n{r1:.2f} × {r2:.2f} = {abs(p):.2f}\n"
                         f"add the angles\n{a1:.1f}° + {a2:.1f}° = {a1 + a2:.1f}°\n\nso × i = rotate 90°\n"
                         "(used in signals, AC circuits,\nFourier transforms)", 18, frac=0.64)
    return step + 1


def build_quadratic(cv, v, box, step):
    a, b, c = (float(x) for x in (v.get("coefficients") or (1, -2, -3)))
    D = b * b - 4 * a * c
    vx = -b / (2 * a)
    vy = a * vx * vx + b * vx + c
    roots = [(-b - math.sqrt(D)) / (2 * a), (-b + math.sqrt(D)) / (2 * a)] if D >= 0 else []
    lo, hi = vx - 4, vx + 4
    f = lambda x: a * x * x + b * x + c
    ys = [f(lo + (hi - lo) * i / 40) for i in range(41)]
    px, py = _plot(cv, box, step, (lo, hi), (min(ys + [0]) - 1, max(ys) + 1), "x", "y", frac=0.55)
    cv.line([(px(lo), py(0)), (px(hi), py(0))], step, MUTED, 1)
    _pl(cv, [(px(lo + (hi - lo) * i / 60), py(f(lo + (hi - lo) * i / 60))) for i in range(61)], step, "#1971c2", 3)
    step += 1
    for r in roots:
        cv.ellipse(px(r) - 7, py(0) - 7, 14, 14, step, "#e03131", "#e03131", 1)
        cv.text(px(r), py(0) + 10, f"x = {r:g}", step, 15, "#e03131", "center", "top", math=False)
    cv.ellipse(px(vx) - 7, py(vy) - 7, 14, 14, step, "#2f9e44", "#2f9e44", 1)
    cv.text(px(vx) + 10, py(vy), f"vertex ({vx:g}, {vy:g})", step, 15, "#2f9e44", "left", "middle", math=False)
    step += 1
    _note(cv, box, step, f"{a:g}x² {'+' if b >= 0 else '−'} {abs(b):g}x {'+' if c >= 0 else '−'} {abs(c):g} = 0\n\n"
                         f"discriminant b² − 4ac = {D:g}\n" + ("> 0: two real roots" if D > 0 else "= 0: one root" if D == 0
                                                              else "< 0: no real roots") +
          "\n\nx = (−b ± √(b² − 4ac)) / 2a\n" + (f"= ({-b:g} ± {math.sqrt(D):g}) / {2 * a:g}" if D >= 0 else ""), 19, frac=0.66)
    return step + 1


def build_sequences(cv, v, box, step):
    a0, d, r = float(v.get("start") or 2), float(v.get("difference") or 3), float(v.get("ratio") or 1.5)
    n = 10
    ar = [a0 + d * k for k in range(n)]
    ge = [a0 * r ** k for k in range(n)]
    mx = max(ar + ge)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    bw = W * 0.6 / n
    for k in range(n):
        for j, (seq, col) in enumerate(((ar, "#a5d8ff"), (ge, "#ffc9c9"))):
            h = (H - 80) * seq[k] / mx
            cv.rect(x0 + 20 + k * bw + j * bw / 2, y1 - 40 - h, bw / 2 - 3, h, step, col, "transparent", 1, rounded=False)
        cv.text(x0 + 20 + k * bw + bw / 2, y1 - 36, str(k + 1), step, 12, MUTED, "center", "top", math=False)
        step += 1
    Sa = n / 2 * (2 * a0 + (n - 1) * d)
    Sg = a0 * (r ** n - 1) / (r - 1)
    _note(cv, box, step, f"blue: arithmetic, +{d:g} each time\naₙ = {a0:g} + {d:g}(n − 1)\nsum of {n} = n/2 (2a + (n−1)d) = {Sa:g}\n\n"
                         f"red: geometric, ×{r:g} each time\naₙ = {a0:g} × {r:g}ⁿ⁻¹\nsum of {n} = a(rⁿ − 1)/(r − 1) = {Sg:.1f}\n\n"
                         "linear vs exponential growth", 17, frac=0.66)
    return step + 1


def build_markov_chain(cv, v, box, step):
    states = ["sunny", "cloudy", "rainy"]
    Pm = v.get("matrix") or [[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.4, 0.4]]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    pos = [(0.2, 0.2), (0.7, 0.2), (0.45, 0.75)]
    P = lambda i: (x0 + W * 0.45 * pos[i][0] + 20, y0 + (H - 60) * pos[i][1] + 20)
    for i in range(3):
        for j in range(3):
            if i == j:
                x, y = P(i)
                cv.text(x, y - 46, f"{Pm[i][i]:g} ↺", step, 14, STRONG[i], "center", "bottom", math=False)
                continue
            (ax, ay), (bx, by) = P(i), P(j)
            L = math.hypot(bx - ax, by - ay)
            ux, uy = (bx - ax) / L, (by - ay) / L
            off = 8
            cv.line([(ax + ux * 42 - uy * off, ay + uy * 42 + ux * off), (bx - ux * 42 - uy * off, by - uy * 42 + ux * off)],
                    step, STRONG[i], 1.5, arrow_end=True)
            cv.text((ax + bx) / 2 - uy * 22, (ay + by) / 2 + ux * 22, f"{Pm[i][j]:g}", step, 13, STRONG[i], "center",
                    "middle", math=False)
    for i, s in enumerate(states):
        x, y = P(i)
        cv.ellipse(x - 40, y - 26, 80, 52, step, PAL[i], STRONG[i], 2)
        cv.text(x, y, s, step, 16, INK, "center", "middle", math=False)
    step += 1
    dist = [1.0, 0, 0]
    rows = []
    for d in range(8):
        rows.append([d] + [f"{p:.3f}" for p in dist])
        dist = [sum(dist[i] * Pm[i][j] for i in range(3)) for j in range(3)]
    _table(cv, x0 + W * 0.52, y0 + 30, "start sunny; chance of each state", ["day"] + states, rows, step, W * 0.46, size=13)
    step += 1
    cv.text(x0 + W * 0.52, y0 + 340, f"it settles to a steady state:\n{', '.join(f'{s} {p:.0%}' for s, p in zip(states, dist))}\n"
                                     "whatever the start (π = πP)", step, 16, "#e8590c", math=False)
    return step + 1


def build_truth_table(cv, v, box, step):
    expr = str(v.get("expression") or "(A and B) or not C")
    vars_ = sorted(set(re.findall(r"\b[A-Z]\b", expr)))
    rows = []
    for vals in itertools.product([0, 1], repeat=len(vars_)):
        env = dict(zip(vars_, map(bool, vals)))
        rows.append(list(vals) + [int(eval(expr, {"__builtins__": {}}, env))])
    x0, y0, x1, y1 = box
    W = x1 - x0
    _mono(cv.text(x0, y0, f"F = {expr}", step, 20, "#e8590c", math=False))
    step += 1
    cw, rh = 70, (y1 - y0 - 60) / (len(rows) + 1)
    for j, h in enumerate(vars_ + ["F"]):
        _box(cv, x0 + j * cw, y0 + 40, cw - 4, rh - 4, step, h, "#e9ecef", 16)
    for i, r in enumerate(rows):
        for j, val in enumerate(r):
            _box(cv, x0 + j * cw, y0 + 40 + (i + 1) * rh, cw - 4, rh - 4, step, val,
                 ("#b2f2bb" if val else "#ffc9c9") if j == len(r) - 1 else "#ffffff", 15)
        step += 1 if i % 2 else 0
    ones = [i for i, r in enumerate(rows) if r[-1]]
    sop = " + ".join("".join(v_ + ("" if r[k] else "'") for k, v_ in enumerate(vars_)) for r in (rows[i] for i in ones))
    _note(cv, box, step, f"{len(rows)} rows = 2^{len(vars_)} input combinations\nF is true in {len(ones)} of them\n\n"
                         f"sum of products (each true row):\n{sop}\n\n(simplify with algebra or a\nKarnaugh map)", 17,
          frac=0.5, dy=60)
    return step + 1


def build_permutations(cv, v, box, step):
    items = list(str(v.get("items") or "ABCDE"))
    k = int(v.get("k") or 3)
    n = len(items)
    perms = list(itertools.permutations(items, k))
    combs = list(itertools.combinations(items, k))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cv.text(x0, y0, f"choose {k} of {n}: {', '.join(items)}", step, 20, "#e8590c", math=False)
    step += 1
    for i, c in enumerate(combs):
        r, col = divmod(i, 5)
        _box(cv, x0 + col * 110, y0 + 50 + r * 46, 100, 38, step, "".join(c), "#d3f9d8", 17, mono=True)
    cv.text(x0, y0 + 50 + (len(combs) // 5 + 1) * 46, f"combinations (order doesn't matter):\nC({n},{k}) = {n}! / ({k}! "
                                                       f"{n - k}!) = {len(combs)}", step, 17, "#2f9e44", math=False)
    step += 1
    first = combs[0]
    orders = list(itertools.permutations(first))
    for i, p in enumerate(orders):
        _box(cv, x0 + W * 0.62 + (i % 3) * 90, y0 + 50 + (i // 3) * 46, 80, 38, step, "".join(p), "#d0ebff", 17, mono=True)
    cv.text(x0 + W * 0.62, y0 + 160, f"each group can be ordered\n{k}! = {len(orders)} ways, so\npermutations P({n},{k})\n"
                                     f"= {len(combs)} × {len(orders)} = {len(perms)}\n= {n}! / {n - k}!", step, 17, "#1971c2",
            math=False)
    step += 1
    cv.text(x0, y1 - 30, "team of 3 from 5 people → combinations; gold/silver/bronze → permutations", step, 17, INK,
            math=False)
    return step + 1


def euler_path(adj):
    deg = {u: len(vs) for u, vs in adj.items()}
    odd = [u for u in adj if deg[u] % 2]
    if len(odd) not in (0, 2):
        return odd, None
    g = {u: list(vs) for u, vs in adj.items()}
    start = odd[0] if odd else next(iter(g))
    stack, path = [start], []
    while stack:
        u = stack[-1]
        if g[u]:
            w = g[u].pop()
            g[w].remove(u)
            stack.append(w)
        else:
            path.append(stack.pop())
    return odd, path[::-1]


def build_euler_path(cv, v, box, step):
    graphs = [("Königsberg bridges", {"A": ["B", "B", "C", "C", "D"], "B": ["A", "A", "D"], "C": ["A", "A", "D"],
                                      "D": ["A", "B", "C"]},
               {"A": (0.5, 0.5), "B": (0.5, 0.1), "C": (0.5, 0.9), "D": (0.9, 0.5)}),
              ("house", {"A": ["B", "C"], "B": ["A", "C", "D", "E"], "C": ["A", "B", "D", "E"], "D": ["B", "C", "E"],
                         "E": ["B", "C", "D"]},
               {"A": (0.5, 0.05), "B": (0.15, 0.4), "C": (0.85, 0.4), "D": (0.15, 0.9), "E": (0.85, 0.9)})]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    for gi, (name, adj, pos) in enumerate(graphs):
        bx = x0 + gi * W * 0.5
        P = lambda n: (bx + 30 + W * 0.3 * pos[n][0], y0 + 40 + (H - 160) * pos[n][1])
        seen = {}
        for u, vs in adj.items():
            for w in vs:
                if u < w:
                    key = (u, w)
                    seen[key] = seen.get(key, 0) + 1
        for (u, w), cnt in seen.items():
            for c in range(cnt):
                off = (c - (cnt - 1) / 2) * 30
                (ax, ay), (bx2, by2) = P(u), P(w)
                mx, my = (ax + bx2) / 2, (ay + by2) / 2
                L = math.hypot(bx2 - ax, by2 - ay)
                nx, ny = -(by2 - ay) / L, (bx2 - ax) / L
                cv.line([(ax, ay), (mx + nx * off, my + ny * off), (bx2, by2)], step, MUTED, 2)
        for n in adj:
            x, y = P(n)
            cv.ellipse(x - 18, y - 18, 36, 36, step, "#d0ebff", INK, 2)
            cv.text(x, y, n, step, 16, INK, "center", "middle", math=False)
            cv.text(x + 22, y - 22, f"deg {len(adj[n])}", step, 12, "#e03131" if len(adj[n]) % 2 else "#2f9e44", math=False)
        odd, path = euler_path(adj)
        if path:
            _pl(cv, [(P(n)[0] + 3, P(n)[1] + 3) for n in path], step + 1, "#2f9e44", 3)
        cv.text(bx + 20, y1 - 100, f"{name}: {len(odd)} odd-degree vertices\n" +
                ("→ no path crosses every edge once" if path is None else f"→ Euler path: {' → '.join(path)}"), step + 1, 16,
                "#e03131" if path is None else "#2f9e44", math=False)
        step += 2
    cv.text(x0, y1 - 30, "Euler: a path using every edge exactly once exists only with 0 or 2 odd-degree vertices",
            step, 17, INK, math=False)
    return step + 1


# ======================================================================== physics

def build_rc_circuit(cv, v, box, step):
    R, C, V = float(v.get("R") or 1000), float(v.get("C") or 0.001), float(v.get("V") or 5)
    tau = R * C
    px, py = _plot(cv, box, step, (0, 5 * tau), (0, V * 1.1), "time (s)", "capacitor voltage", frac=0.55)
    cv.line([(px(0), py(V)), (px(5 * tau), py(V))], step, MUTED, 1, dashed=True)
    step += 1
    _pl(cv, [(px(t * tau / 20), py(V * (1 - math.exp(-t / 20)))) for t in range(101)], step, "#1971c2", 3)
    step += 1
    for k in range(1, 6):
        val = V * (1 - math.exp(-k))
        cv.ellipse(px(k * tau) - 5, py(val) - 5, 10, 10, step, "#e03131", "#e03131", 1)
        cv.text(px(k * tau), py(val) + 8, f"{1 - math.exp(-k):.0%}", step, 13, "#e03131", "center", "top", math=False)
    step += 1
    _note(cv, box, step, f"R = {R:g} Ω, C = {C * 1e6:g} µF, V = {V:g} V\n\ntime constant τ = RC = {tau:g} s\n"
                         "V(t) = V (1 − e^(−t/τ))\n\n63% after 1τ, ~99% after 5τ\n(used in timers, filters,\ndebouncing "
                         "buttons)", 18, frac=0.66)
    return step + 1


def build_standing_waves(cv, v, box, step):
    L, vel = float(v.get("length") or 0.65), float(v.get("speed") or 143)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    rh = (H - 20) / 4
    for n in range(1, 5):
        y = y0 + (n - 1) * rh + rh / 2
        f = n * vel / (2 * L)
        for sgn in (1, -1):
            _pl(cv, [(x0 + 20 + W * 0.55 * i / 80, y - sgn * rh * 0.35 * math.sin(n * math.pi * i / 80)) for i in range(81)],
                step, STRONG[n - 1], 2.5 if sgn > 0 else 1, dashed=sgn < 0)
        for k in range(n + 1):
            cv.ellipse(x0 + 20 + W * 0.55 * k / n - 4, y - 4, 8, 8, step, INK, INK, 1)
        cv.text(x0 + W * 0.6 + 20, y, f"n = {n}: λ = 2L/{n} = {2 * L / n:.2f} m,  f = {f:.0f} Hz", step, 17, STRONG[n - 1],
                "left", "middle", math=False)
        step += 1
    cv.text(x0 + W * 0.6 + 20, y1 - 10, f"string L = {L} m, wave speed {vel:g} m/s\nf = n·v / 2L; dots = nodes", step, 15,
            MUTED, "left", "bottom", math=False)
    return step + 1


def build_double_slit(cv, v, box, step):
    lam, d, Lm = float(v.get("wavelength") or 650e-9), float(v.get("slit_spacing") or 0.2e-3), float(v.get("distance") or 2)
    a = d / 4
    spacing = lam * Lm / d
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    wx = x0 + 60
    cv.line([(wx, y0), (wx, y0 + H / 2 - 30)], step, INK, 4)
    cv.line([(wx, y0 + H / 2 - 18), (wx, y0 + H / 2 + 18)], step, INK, 4)
    cv.line([(wx, y0 + H / 2 + 30), (wx, y1)], step, INK, 4)
    for r in range(1, 6):
        for sy in (y0 + H / 2 - 24, y0 + H / 2 + 24):
            arc = [(wx + r * 40 * math.cos(t / 10), sy + r * 40 * math.sin(t / 10)) for t in range(-12, 13)]
            _pl(cv, arc, step, "#74c0fc", 1.2)
    step += 1
    sx = x0 + W * 0.45
    ys = lambda yy: y0 + H / 2 + yy / (4 * spacing) * (H / 2 - 10)
    pts = []
    for i in range(-200, 201):
        yy = i / 200 * 4 * spacing
        th = yy / Lm
        beta = math.pi * a * th / lam
        inten = math.cos(math.pi * d * th / lam) ** 2 * (1 if abs(beta) < 1e-12 else (math.sin(beta) / beta) ** 2)
        pts.append((sx + inten * W * 0.2, ys(yy)))
        if i % 4 == 0:
            g = int(255 - 225 * inten)
            cv.rect(sx - 30, ys(yy) - 2, 24, 4.5, step, f"#{255:02x}{g:02x}{g:02x}", "transparent", 1, rounded=False)
    _pl(cv, pts, step, "#e03131", 2)
    cv.line([(sx, y0), (sx, y1)], step, INK, 2)
    step += 1
    _note(cv, box, step, f"red laser λ = {lam * 1e9:.0f} nm\nslits {d * 1e3:g} mm apart\nscreen {Lm:g} m away\n\n"
                         f"bright fringes every\nΔy = λL/d = {spacing * 1e3:.1f} mm\n\nwaves add where path\n"
                         "difference = whole λ:\nlight is a wave", 17, frac=0.74)
    return step + 1


def build_circular_motion(cv, v, box, step):
    m, r, vel = float(v.get("mass") or 0.5), float(v.get("radius") or 1.2), float(v.get("speed") or 4)
    ac = vel * vel / r
    F = m * ac
    T = 2 * math.pi * r / vel
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    R = min(W * 0.25, H * 0.4)
    cx, cy = x0 + R + 40, (y0 + y1) / 2
    cv.ellipse(cx - R, cy - R, 2 * R, 2 * R, step, "transparent", MUTED, 1.5)
    cv.ellipse(cx - 5, cy - 5, 10, 10, step, INK, INK, 1)
    step += 1
    for k in range(8):
        a = 2 * math.pi * k / 8
        bx, by = cx + R * math.cos(a), cy - R * math.sin(a)
        tr = _Track(cv)
        cv.line([(cx, cy), (bx, by)], step, MUTED, 1.5)
        cv.ellipse(bx - 14, by - 14, 28, 28, step, "#ffd8a8", INK, 2)
        cv.arrow(bx, by, bx - 60 * math.sin(a), by - 60 * math.cos(a), step, "v", color="#1971c2")
        cv.arrow(bx, by, bx + (cx - bx) * 0.4, by + (cy - by) * 0.4, step, "a", color="#e03131")
        tr.tag(until=step + 1 if k < 7 else None, fade=k > 0)
        step += 1
    _note(cv, box, step, f"m = {m:g} kg, r = {r:g} m, v = {vel:g} m/s\n\nvelocity (blue) is tangent;\n"
                         f"acceleration (red) points to\nthe centre: a = v²/r = {ac:.1f} m/s²\n\ntension F = mv²/r = {F:.1f} N\n"
                         f"period T = 2πr/v = {T:.2f} s\n\ncut the string → it flies off\nalong the tangent", 18, frac=0.62)
    return step + 1


def kepler_positions(e, n):
    out = []
    for k in range(n):
        M = 2 * math.pi * k / n
        E = M
        for _ in range(30):
            E = E - (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
        out.append((math.cos(E) - e, math.sqrt(1 - e * e) * math.sin(E)))
    return out


def build_kepler_orbit(cv, v, box, step):
    e = float(v.get("eccentricity") or 0.6)
    n = 12
    pos = kepler_positions(e, n)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    a = min(W * 0.3, H * 0.45 / math.sqrt(1 - e * e))
    cx, cy = x0 + a * (1 + e) + 20, (y0 + y1) / 2
    P = lambda p: (cx + p[0] * a, cy - p[1] * a)
    cv.ellipse(cx - a * e - a, cy - a * math.sqrt(1 - e * e), 2 * a, 2 * a * math.sqrt(1 - e * e), step, "transparent",
               MUTED, 1.5)
    cv.ellipse(cx - 16, cy - 16, 32, 32, step, "#ffd43b", "#f08c00", 2)
    step += 1
    for k in (0, 5):
        p1, p2 = pos[k], pos[(k + 1) % n]
        cv.polygon([P((0, 0)), P(p1), P(p2)], step, PAL[k % len(PAL)], STRONG[k % len(STRONG)], 1)
    for k, p in enumerate(pos):
        cv.ellipse(P(p)[0] - 6, P(p)[1] - 6, 12, 12, step, "#1971c2", "#1971c2", 1)
    step += 1
    _note(cv, box, step, f"eccentricity {e}\ndots = positions at equal\ntime steps (Kepler's equation,\ncomputed)\n\n"
                         "the two shaded wedges took the\nsame time and have the same\narea: the planet moves fastest\n"
                         "near the Sun (2nd law)", 17, frac=0.66)
    return step + 1


def build_heating_curve(cv, v, box, step):
    m = float(v.get("mass") or 100)
    c_ice, c_w, c_s, Lf, Lv = 2.09, 4.18, 2.01, 334, 2260
    seg = [(-20, 0, m * c_ice * 20, "ice warms"), (0, 0, m * Lf, "melting"), (0, 100, m * c_w * 100, "water warms"),
           (100, 100, m * Lv, "boiling"), (100, 120, m * c_s * 20, "steam warms")]
    tot = sum(s[2] for s in seg)
    px, py = _plot(cv, box, step, (0, tot / 1000), (-30, 130), "heat added (kJ)", "°C", frac=0.58)
    step += 1
    q = 0
    for i, (ta, tb, dq, lab) in enumerate(seg):
        _pl(cv, [(px(q / 1000), py(ta)), (px((q + dq) / 1000), py(tb))], step, STRONG[i], 3.5)
        cv.text(px((q + dq / 2) / 1000), py((ta + tb) / 2) - 10, f"{lab}\n{dq / 1000:.1f} kJ", step, 13, STRONG[i], "center",
                "bottom", math=False)
        q += dq
        step += 1
    _note(cv, box, step, f"{m:g} g of water, −20 → 120 °C\ntotal {tot / 1000:.0f} kJ (computed)\n\nflat parts: energy breaks\n"
                         "bonds instead of raising the\ntemperature (latent heat)\n\nboiling takes 7× the energy\nof melting",
          17, frac=0.66)
    return step + 1


def build_photoelectric(cv, v, box, step):
    h = 4.136e-15
    metals = v.get("metals") or [("sodium", 2.28), ("zinc", 4.3)]
    px, py = _plot(cv, box, step, (0, 2e15), (0, 4), "light frequency (Hz)", "max kinetic energy (eV)", frac=0.55)
    for k in range(5):
        cv.text(px(k * 0.5e15), py(0) + 4, f"{k * 0.5:g}e15", step, 12, MUTED, "center", "top", math=False)
    step += 1
    for i, (name, phi) in enumerate(metals):
        f0 = phi / h
        fe = min(2e15, (4 + phi) / h)
        _pl(cv, [(px(f0), py(0)), (px(fe), py(h * fe - phi))], step, STRONG[i], 3)
        cv.text(px(f0), py(0) - 6, f"{name}: f₀ = {f0:.2e} Hz", step, 14, STRONG[i], "center", "bottom", math=False)
        step += 1
    _note(cv, box, step, "KE = hf − φ (Einstein)\nslope = Planck's constant h\n\nbelow the threshold f₀ no\n"
                         "electrons come out, however\nbright the light: light comes\nin packets (photons)\n\n"
                         f"green light 5.6e14 Hz:\nsodium KE = {h * 5.6e14 - metals[0][1]:.2f} eV, zinc: none", 17, frac=0.66)
    return step + 1


# ======================================================================== chemistry

def build_maxwell_boltzmann(cv, v, box, step):
    M = float(v.get("molar_mass") or 0.028)
    k = 8.314
    Ts = v.get("temperatures") or [300, 600]
    fv = lambda vv, T: 4 * math.pi * (M / (2 * math.pi * k * T)) ** 1.5 * vv * vv * math.exp(-M * vv * vv / (2 * k * T))
    vmax = 1600
    top = max(fv(math.sqrt(2 * k * T / M), T) for T in Ts)
    px, py = _plot(cv, box, step, (0, vmax), (0, top * 1.15), "molecular speed (m/s)", "fraction", frac=0.55)
    thr = 1000
    cv.line([(px(thr), py(0)), (px(thr), py(top))], step, "#e03131", 1.5, dashed=True)
    cv.text(px(thr) + 4, py(top), "speed needed to react", step, 13, "#e03131", math=False)
    step += 1
    fr = []
    for i, T in enumerate(Ts):
        _pl(cv, [(px(vv), py(fv(vv, T))) for vv in range(0, vmax + 1, 20)], step, STRONG[i], 3)
        vp = math.sqrt(2 * k * T / M)
        cv.text(px(vp), py(fv(vp, T)) - 6, f"{T} K", step, 15, STRONG[i], "center", "bottom", math=False)
        fr.append(sum(fv(vv, T) * 5 for vv in range(thr, 4000, 5)))
        step += 1
    _note(cv, box, step, f"N₂ molecules (computed)\n\nfraction faster than {thr} m/s:\n" +
          "\n".join(f"{T} K: {f:.1%}" for T, f in zip(Ts, fr)) +
          f"\n\ndouble the temperature →\n{fr[1] / fr[0]:.1f}× more molecules can react,\nso reactions speed up", 17, frac=0.66)
    return step + 1


def build_galvanic_cell(cv, v, box, step):
    E = {"Zn": -0.76, "Cu": 0.34, "Ag": 0.80, "Fe": -0.44, "Mg": -2.37}
    an, ca = str(v.get("anode") or "Zn"), str(v.get("cathode") or "Cu")
    cell = E[ca] - E[an]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    for i, (m, col, ion) in enumerate(((an, "#ced4da", f"{an}²⁺"), (ca, "#e8a76b", f"{ca}²⁺"))):
        bx = x0 + 30 + i * W * 0.38
        cv.line([(bx, y0 + 120), (bx, y1 - 30), (bx + W * 0.24, y1 - 30), (bx + W * 0.24, y0 + 120)], step, INK, 2.5)
        cv.rect(bx + 4, y0 + 170, W * 0.24 - 8, y1 - y0 - 204, step, "#d0ebff", "transparent", 1, rounded=False)
        cv.rect(bx + W * 0.1, y0 + 90, 30, y1 - y0 - 150, step, col, INK, 2, rounded=False)
        cv.text(bx + W * 0.12, y1 - 60, f"{m}\n{ion} solution", step, 14, INK, "center", "top", math=False)
    step += 1
    sb = [(x0 + 30 + W * 0.2, y0 + 190), (x0 + 30 + W * 0.2, y0 + 130), (x0 + 30 + W * 0.42, y0 + 130),
          (x0 + 30 + W * 0.42, y0 + 190)]
    _pl(cv, sb, step, "#adb5bd", 12)
    cv.text(x0 + 30 + W * 0.31, y0 + 116, "salt bridge", step, 13, MUTED, "center", "bottom", math=False)
    ax, cx_ = x0 + 30 + W * 0.1 + 15, x0 + 30 + W * 0.38 + W * 0.1 + 15
    cv.line([(ax, y0 + 90), (ax, y0 + 40), (cx_, y0 + 40), (cx_, y0 + 90)], step, INK, 2)
    _box(cv, (ax + cx_) / 2 - 50, y0 + 20, 100, 40, step, f"{cell:.2f} V", "#fff3bf", 18)
    step += 1
    cv.arrow(ax + 10, y0 + 32, (ax + cx_) / 2 - 60, y0 + 32, step, "e⁻", color="#1971c2")
    cv.arrow((ax + cx_) / 2 + 60, y0 + 32, cx_ - 10, y0 + 32, step, "e⁻", color="#1971c2")
    step += 1
    _note(cv, box, step, f"anode ({an}): oxidation\n{an} → {an}²⁺ + 2e⁻\n\ncathode ({ca}): reduction\n{ca}²⁺ + 2e⁻ → {ca}\n\n"
                         f"E°cell = E°cathode − E°anode\n= {E[ca]:+.2f} − ({E[an]:+.2f}) = {cell:.2f} V", 17, frac=0.72)
    return step + 1


# ======================================================================== biology

def build_pcr(cv, v, box, step):
    cycles = int(v.get("cycles") or 30)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    stg = [("denature", "95 °C", "strands separate", "#ffc9c9"), ("anneal", "55 °C", "primers stick", "#d0ebff"),
           ("extend", "72 °C", "polymerase copies", "#d3f9d8")]
    for i, (n, t, d, col) in enumerate(stg):
        _box(cv, x0 + i * W * 0.2, y0, W * 0.18, 80, step, f"{n}\n{t}\n{d}", col, 15)
        if i < 2:
            cv.arrow(x0 + i * W * 0.2 + W * 0.18 + 2, y0 + 40, x0 + (i + 1) * W * 0.2 - 2, y0 + 40, step, color=MUTED)
    cv.arrow(x0 + W * 0.5, y0 + 84, x0 + W * 0.09, y0 + 84, step, "repeat", color=MUTED)
    step += 1
    for c in range(4):
        n = 2 ** c
        for k in range(n):
            y = y0 + 130 + c * 45
            _pl(cv, [(x0 + k * 48, y + (k % 2) * 6), (x0 + k * 48 + 40, y + (k % 2) * 6)], step, STRONG[c], 4)
        cv.text(x0 + W * 0.4, y0 + 130 + c * 45, f"after cycle {c}: {n} copies", step, 15, STRONG[c], "left", "middle",
                math=False)
        step += 1
    px, py = _plot(cv, (x0 + W * 0.55, y0 + 120, x1, y1), step, (0, cycles), (0, 10), "cycle", "log₁₀ copies", frac=0.65)
    _pl(cv, [(px(c), py(min(10, c * math.log10(2)))) for c in range(cycles + 1)], step, "#e03131", 3)
    step += 1
    cv.text(x0, y1 - 40, f"each cycle doubles the DNA: after {cycles} cycles 2^{cycles} = {2 ** cycles:,} copies of one "
                         "molecule", step, 17, "#e8590c", math=False)
    return step + 1


def digest(seq, site="GAATTC", cut=1):
    pos, out, last = [m.start() + cut for m in re.finditer(site, seq)], [], 0
    for p in pos:
        out.append(p - last)
        last = p
    out.append(len(seq) - last)
    return out


def build_gel_electrophoresis(cv, v, box, step):
    rnd = random.Random(4)
    samples = v.get("samples") or {}
    if not samples:
        for name in ("sample 1", "sample 2", "sample 3"):
            parts = []
            for _ in range(rnd.randint(1, 3)):
                parts.append("".join(rnd.choice("ACGT") for _ in range(rnd.randint(150, 1800))))
            samples[name] = "GAATTC".join(parts)
    ladder = [100, 200, 300, 500, 700, 1000, 1500, 2000, 3000]
    lanes = [("ladder", ladder)] + [(n, digest(s)) for n, s in samples.items()]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    gw = W * 0.6
    cv.rect(x0, y0 + 20, gw, H - 30, step, "#e5dbff", INK, 2, rounded=False)
    lw = gw / len(lanes)
    D = lambda bp: y0 + 50 + (math.log10(3500) - math.log10(bp)) / (math.log10(3500) - math.log10(80)) * (H - 90)
    for i, (n, sizes) in enumerate(lanes):
        cv.rect(x0 + i * lw + lw * 0.2, y0 + 26, lw * 0.6, 12, step, "#495057", "transparent", 1, rounded=False)
        cv.text(x0 + i * lw + lw / 2, y0 + 4, n, step, 13, INK, "center", "top", math=False)
    cv.text(x0 + gw + 10, y0 + 30, "− (wells)", step, 14, MUTED, math=False)
    cv.text(x0 + gw + 10, y1 - 30, "+", step, 18, "#e03131", math=False)
    step += 1
    for i, (n, sizes) in enumerate(lanes):
        for s in sizes:
            y = D(s)
            cv.rect(x0 + i * lw + lw * 0.2, y, lw * 0.6, 6, step, "#7048e8", "transparent", 1, rounded=False)
            if i == 0:
                cv.text(x0 + 4, y + 3, f"{s}", step, 11, MUTED, "left", "middle", math=False)
    step += 1
    lines = [f"{n}: cut by EcoRI (GAATTC) → {', '.join(map(str, sorted(digest(s), reverse=True)))} bp"
             for n, s in samples.items()]
    _note(cv, box, step, "DNA is negative, so it moves\ntoward +; small fragments\nmove further (distance ∝\n−log size)\n\n"
                         + "\n".join(lines), 15, frac=0.66, dy=40)
    return step + 1


def upgma(names, D):
    clusters = {n: (n, 1, 0.0) for n in names}
    dist = {(a, b): D[a][b] for a in names for b in names if a != b}
    merges = []
    while len(clusters) > 1:
        a, b = min(((a, b) for a in clusters for b in clusters if a < b), key=lambda p: dist[p])
        h = dist[(a, b)] / 2
        new = f"({a},{b})"
        na, nb = clusters[a][1], clusters[b][1]
        merges.append((a, b, new, h))
        for c in clusters:
            if c not in (a, b):
                d = (dist[(min(a, c), max(a, c))] * na + dist[(min(b, c), max(b, c))] * nb) / (na + nb)
                dist[(min(new, c), max(new, c))] = d
                dist[(max(new, c), min(new, c))] = d
        del clusters[a], clusters[b]
        clusters[new] = (new, na + nb, h)
    return merges


def build_phylogenetic_tree(cv, v, box, step):
    names = ["human", "chimp", "gorilla", "mouse", "chicken"]
    M = v.get("distances") or [[0, 2, 4, 30, 60], [2, 0, 4, 30, 60], [4, 4, 0, 30, 60], [30, 30, 30, 0, 60],
                               [60, 60, 60, 60, 0]]
    D = {a: {b: M[i][j] for j, b in enumerate(names)} for i, a in enumerate(names)}
    merges = upgma(names, D)
    order = []

    def leaves(c):
        return re.findall(r"[a-z]+", c)
    order = leaves(merges[-1][2])
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    maxh = merges[-1][3]
    Y = {n: y0 + 30 + i * (H - 60) / (len(order) - 1) for i, n in enumerate(order)}
    X = {n: x0 + W * 0.55 for n in order}
    hx = lambda h: x0 + W * 0.55 - h / maxh * W * 0.5
    for n in order:
        cv.text(x0 + W * 0.56, Y[n], n, step, 18, INK, "left", "middle", math=False)
    step += 1
    for a, b, new, h in merges:
        x = hx(h)
        _pl(cv, [(X[a], Y[a]), (x, Y[a]), (x, Y[b]), (X[b], Y[b])], step, "#2f9e44", 2.5)
        cv.text(x - 4, (Y[a] + Y[b]) / 2, f"{h:g}", step, 12, MUTED, "right", "middle", math=False)
        X[new], Y[new] = x, (Y[a] + Y[b]) / 2
        step += 1
    _note(cv, box, step, "UPGMA from a distance matrix\n(% DNA difference, illustrative)\n\njoin the closest pair, average\n"
                         "their distances to the rest,\nrepeat\n\nbranch point = common\nancestor; closer to the right\n"
                         "= more recently related", 16, frac=0.72)
    return step + 1


def build_feedback_loop(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    loop = [("blood glucose rises\n(after a meal)", "#ffc9c9"), ("pancreas releases\ninsulin", "#d0ebff"),
            ("cells take up glucose;\nliver stores glycogen", "#d3f9d8"), ("glucose falls back\nto normal", "#fff3bf")]
    cx, cy, R = x0 + W * 0.22, y0 + H * 0.4, min(W * 0.18, H * 0.32)
    for i, (t, col) in enumerate(loop):
        a = math.pi / 2 - i * math.pi / 2
        bx, by = cx + R * math.cos(a), cy - R * math.sin(a)
        _box(cv, bx - 85, by - 30, 170, 60, step, t, col, 13)
        a2 = a - math.pi / 2
        nx, ny = cx + R * math.cos(a2), cy - R * math.sin(a2)
        cv.arrow(bx + (nx - bx) * 0.35, by + (ny - by) * 0.35, bx + (nx - bx) * 0.65, by + (ny - by) * 0.65, step, color=MUTED)
        step += 1
    px, py = _plot(cv, (x0 + W * 0.45, y0, x1, y1 - 20), step, (0, 4), (60, 180), "hours after a meal", "glucose (mg/dL)",
                   frac=0.85)
    pts = [(px(t / 20), py(90 + 65 * (t / 20 / 0.75) * math.exp(1 - t / 20 / 0.75))) for t in range(81)]
    cv.rect(px(0), py(110), px(4) - px(0), py(70) - py(110), step, "#d3f9d8", "transparent", 1, rounded=False)
    cv.text(px(4) - 4, py(72), "normal range", step, 13, "#2f9e44", "right", "bottom", math=False)
    _pl(cv, pts, step, "#e03131", 3)
    step += 1
    cv.text(x0, y1 - 30, "negative feedback: the response cancels the change (glucagon does the opposite when glucose "
                         "is low)", step, 16, INK, math=False)
    return step + 1


# ======================================================================== earth / GIS

def build_ndvi(cv, v, box, step):
    rnd = random.Random(8)
    g = 10
    nir = [[0.0] * g for _ in range(g)]
    red = [[0.0] * g for _ in range(g)]
    for r in range(g):
        for c in range(g):
            kind = "water" if c < 2 else "forest" if r < 5 and c > 4 else "city" if r > 6 and c > 5 else "crop"
            nir[r][c], red[r][c] = {"water": (0.03, 0.05), "forest": (0.5, 0.05), "city": (0.25, 0.2),
                                    "crop": (0.35, 0.1)}[kind]
            nir[r][c] += rnd.uniform(-0.02, 0.02)
            red[r][c] += rnd.uniform(-0.01, 0.01)
    nd = [[(nir[r][c] - red[r][c]) / (nir[r][c] + red[r][c]) for c in range(g)] for r in range(g)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cs = min(W * 0.2 / g, (H - 60) / g)

    def ramp(x):
        if x < 0:
            return "#4dabf7"
        t = min(1, x / 0.85)
        r_, g_, b_ = int(230 - 200 * t), int(200 - 50 * t), int(140 - 110 * t)
        return f"#{r_:02x}{g_:02x}{b_:02x}"
    grey = lambda x: "#" + f"{int(255 * min(1, max(0, x) * 1.8)):02x}" * 3
    for k, (name, data, fn) in enumerate((("NIR band", nir, grey), ("red band", red, grey), ("NDVI", nd, ramp))):
        bx = x0 + k * (g * cs + 30)
        cv.text(bx, y0, name, step, 16, "#e8590c", math=False)
        for r in range(g):
            for c in range(g):
                cv.rect(bx + c * cs, y0 + 26 + r * cs, cs, cs, step, fn(data[r][c]), "transparent", 1, rounded=False)
        step += 1
    _note(cv, box, step, "NDVI = (NIR − Red) / (NIR + Red)\n\nhealthy leaves reflect near-\ninfrared and absorb red\n\n"
                         f"forest ≈ {nd[1][7]:.2f}\ncrops ≈ {nd[5][3]:.2f}\ncity ≈ {nd[8][8]:.2f}\nwater ≈ {nd[3][0]:.2f}",
          17, frac=0.74)
    return step + 1


def build_epicenter(cv, v, box, step):
    stations = v.get("stations") or {"A": (40, 60), "B": (150, 50), "C": (90, 150)}
    epi = (100, 90)
    vp, vs = 6.0, 3.5
    k = 1 / (1 / vs - 1 / vp)
    sp = {n: math.dist(p, epi) / k for n, p in stations.items()}
    dist = {n: t * k for n, t in sp.items()}
    best = min(((x / 2, y / 2) for x in range(0, 341) for y in range(0, 341)),
               key=lambda q: sum((math.dist(q, stations[n]) - dist[n]) ** 2 for n in stations))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    s = min(W * 0.55, H - 20) / 200
    P = lambda p: (x0 + (p[0] + 5) * s, y0 + (p[1] + 5) * s)
    cv.rect(x0, y0, 200 * s, 200 * s, step, "#f8f9fa", MUTED, 1, rounded=False)
    for i, (n, p) in enumerate(stations.items()):
        X, Y = P(p)
        cv.polygon([(X, Y - 10), (X - 9, Y + 7), (X + 9, Y + 7)], step, STRONG[i], INK, 1)
        cv.text(X + 12, Y, f"{n}: S−P {sp[n]:.1f} s", step, 14, STRONG[i], "left", "middle", math=False)
    step += 1
    for i, (n, p) in enumerate(stations.items()):
        X, Y = P(p)
        r = dist[n] * s
        cv.ellipse(X - r, Y - r, 2 * r, 2 * r, step, "transparent", STRONG[i], 2)
        step += 1
    X, Y = P(best)
    cv.text(X, Y, "★", step, 28, "#e03131", "center", "middle", math=False)
    step += 1
    _note(cv, box, step, f"P waves ({vp} km/s) arrive before\nS waves ({vs} km/s); the gap gives\nthe distance:\n"
                         f"d = (S−P) × {k:.1f} km\n\n" + "\n".join(f"{n}: {d:.0f} km" for n, d in dist.items()) +
          f"\n\nthree circles meet at the\nepicentre ★ ({best[0]:.0f}, {best[1]:.0f}) km", 17, frac=0.62)
    return step + 1


def build_slope_hillshade(cv, v, box, step):
    g = 16
    z = lambda x, y: 400 * math.exp(-((x - 5) ** 2 + (y - 6) ** 2) / 12) + 250 * math.exp(-((x - 11) ** 2 + (y - 10) ** 2) / 6)
    cell = 100.0
    Z = [[z(c, r) for c in range(g)] for r in range(g)]
    slope, shade = [[0] * g for _ in range(g)], [[0] * g for _ in range(g)]
    az, alt = math.radians(315), math.radians(45)
    for r in range(1, g - 1):
        for c in range(1, g - 1):
            dzdx = (Z[r][c + 1] - Z[r][c - 1]) / (2 * cell)
            dzdy = (Z[r + 1][c] - Z[r - 1][c]) / (2 * cell)
            s = math.atan(math.hypot(dzdx, dzdy))
            a = math.atan2(dzdy, -dzdx)
            slope[r][c] = math.degrees(s)
            shade[r][c] = max(0, math.cos(alt) * math.cos(s) + math.sin(alt) * math.sin(s) * math.cos(az - a))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cs = min(W * 0.3 / g, (H - 50) / g)
    mx = max(max(row) for row in Z)
    for k, (name, fn) in enumerate((("elevation", lambda r, c: f"#{int(120 + 135 * Z[r][c] / mx):02x}{int(200 - 60 * Z[r][c] / mx):02x}{int(120 - 80 * Z[r][c] / mx):02x}"),
                                    ("hillshade (sun from NW, 45°)", lambda r, c: "#" + f"{int(40 + 215 * shade[r][c]):02x}" * 3),
                                    ("slope (red = steep)", lambda r, c: f"#ff{int(240 - 220 * min(1, slope[r][c] / 40)):02x}{int(240 - 220 * min(1, slope[r][c] / 40)):02x}"))):
        bx = x0 + k * (g * cs + 24)
        cv.text(bx, y0, name, step, 15, "#e8590c", math=False)
        for r in range(g):
            for c in range(g):
                cv.rect(bx + c * cs, y0 + 24 + r * cs, cs, cs, step, fn(r, c), "transparent", 1, rounded=False)
        step += 1
    ms = max(max(row) for row in slope)
    cv.text(x0, y1 - 20, f"computed from a {cell:g} m DEM with finite differences: steepest slope {ms:.0f}°. "
                         "Used for landslide risk, solar siting, watershed analysis.", step, 15, INK, math=False)
    return step + 1


# ======================================================================== economics

def build_lorenz_gini(cv, v, box, step):
    shares = v.get("income_shares") or [3, 8, 14, 23, 52]
    tot = sum(shares)
    cum = [0]
    for s in shares:
        cum.append(cum[-1] + s / tot)
    n = len(shares)
    area = sum((cum[i] + cum[i + 1]) / 2 / n for i in range(n))
    gini = 1 - 2 * area
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    S = min(W * 0.5, H - 50)
    P = lambda a, b: (x0 + 50 + a * S, y0 + 10 + (1 - b) * S)
    cv.rect(x0 + 50, y0 + 10, S, S, step, "transparent", MUTED, 1, rounded=False)
    cv.line([P(0, 0), P(1, 1)], step, "#2f9e44", 2, dashed=True)
    cv.text(*P(0.45, 0.55), "perfect equality", step, 14, "#2f9e44", "right", "bottom", math=False)
    step += 1
    pts = [P(i / n, cum[i]) for i in range(n + 1)]
    cv.polygon([P(0, 0), P(1, 1)] + pts[::-1][1:], step, "#ffc9c9", "transparent", 1)
    _pl(cv, pts, step, "#e03131", 3)
    for i in range(1, n):
        cv.text(pts[i][0] + 6, pts[i][1] + 4, f"{cum[i]:.0%}", step, 13, "#e03131", "left", "top", math=False)
    cv.text(*P(0.5, -0.06), "share of households (poorest → richest)", step, 14, MUTED, "center", "top", math=False)
    step += 1
    _note(cv, box, step, f"income share by fifth:\n{', '.join(f'{s}%' for s in shares)}\n\nGini = shaded area ÷ area\n"
                         f"under the diagonal = {gini:.2f}\n\n0 = everyone equal\n1 = one household has it all\n"
                         "(US ≈ 0.39 after taxes)", 17, frac=0.64)
    return step + 1


def build_prisoners_dilemma(cv, v, box, step):
    pay = v.get("payoffs") or {("C", "C"): (-1, -1), ("C", "D"): (-10, 0), ("D", "C"): (0, -10), ("D", "D"): (-5, -5)}
    acts = ["C", "D"]
    nm = {"C": "stay silent", "D": "confess"}
    br1 = {b: max(acts, key=lambda a: pay[(a, b)][0]) for b in acts}
    br2 = {a: max(acts, key=lambda b: pay[(a, b)][1]) for a in acts}
    nash = [(a, b) for a in acts for b in acts if br1[b] == a and br2[a] == b]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    c = min(W * 0.22, (H - 100) / 2)
    gx, gy = x0 + 140, y0 + 60
    for j, b in enumerate(acts):
        cv.text(gx + j * c + c / 2, gy - 8, f"B: {nm[b]}", step, 16, "#e03131", "center", "bottom", math=False)
    for i, a in enumerate(acts):
        cv.text(gx - 8, gy + i * c + c / 2, f"A: {nm[a]}", step, 16, "#1971c2", "right", "middle", math=False)
        for j, b in enumerate(acts):
            pa, pb = pay[(a, b)]
            _box(cv, gx + j * c, gy + i * c, c, c, step, f"A {pa} yrs\nB {pb} yrs", "#ffffff", 17)
    step += 1
    for b in acts:
        a = br1[b]
        cv.text(gx + acts.index(b) * c + 14, gy + acts.index(a) * c + 14, "A✔", step, 13, "#1971c2", math=False)
    for a in acts:
        b = br2[a]
        cv.text(gx + acts.index(b) * c + c - 30, gy + acts.index(a) * c + 14, "B✔", step, 13, "#e03131", math=False)
    step += 1
    for a, b in nash:
        cv.rect(gx + acts.index(b) * c + 3, gy + acts.index(a) * c + 3, c - 6, c - 6, step, "transparent", "#f08c00", 4)
    step += 1
    _note(cv, box, step, "✔ = best response to the other's\nchoice (computed)\n\n"
                         f"Nash equilibrium: both {nm[nash[0][0]]}\n(orange): neither gains by\nswitching alone\n\n"
                         "yet both staying silent is\nbetter for both: individual\nlogic ≠ group logic", 17, frac=0.66)
    return step + 1


def build_comparative_advantage(cv, v, box, step):
    hours = v.get("hours") or {"Ana": {"code": 2, "design": 4}, "Ben": {"code": 6, "design": 3}}
    goods = ["code", "design"]
    people = list(hours)
    oc = {p: {g: hours[p][g] / hours[p][[x for x in goods if x != g][0]] for g in goods} for p in people}
    spec = {g: min(people, key=lambda p: oc[p][g]) for g in goods}
    x0, y0, x1, y1 = box
    W = x1 - x0
    _table(cv, x0, y0 + 30, "hours needed per unit", ["", *goods], [[p] + [hours[p][g] for g in goods] for p in people],
           step, W * 0.4, size=16)
    step += 1
    _table(cv, x0, y0 + 170, "opportunity cost (units of the other given up)", ["", *goods],
           [[p] + [f"{oc[p][g]:.2f}" for g in goods] for p in people], step, W * 0.4, hl=None, size=16)
    step += 1
    abs_ = {g: min(people, key=lambda p: hours[p][g]) for g in goods}
    cv.text(x0 + W * 0.46, y0 + 40, f"absolute advantage (fewer hours):\ncode → {abs_['code']}, design → {abs_['design']}\n\n"
                                   f"comparative advantage (lower\nopportunity cost):\ncode → {spec['code']}, design → "
                                   f"{spec['design']}\n\nspecialise and trade: total output\nrises even if one side is "
                                   "faster\nat both (Ricardo)", step, 18, "#e8590c", math=False)
    return step + 1


# ======================================================================== register

visuals.BUILDERS.update({k[6:]: f for k, f in globals().items() if k.startswith("build_")})
for _k in [k for k in list(visuals.ALIASES) if k in visuals.BUILDERS]:
    visuals.ALIASES.pop(_k)
visuals.ALIASES.update({
    "astar": "a_star", "heuristic_search": "a_star", "all_pairs": "floyd_warshall", "counting_sort": "radix_sort",
    "activity_selection": "interval_scheduling", "greedy": "interval_scheduling", "tokenizer": "lexer",
    "lexical_analysis": "lexer", "garbage_collection": "gc_mark_sweep", "branch_prediction": "branch_predictor",
    "parallel_speedup": "amdahl", "adder": "ripple_adder", "kmap": "karnaugh", "boolean_minimization": "karnaugh",
    "byte_order": "endianness", "instruction_format": "mips_encoding", "cache_hierarchy": "memory_hierarchy",
    "deadlock_avoidance": "bankers", "first_fit": "memory_alloc", "rip": "distance_vector", "bellman_ford": "distance_vector",
    "error_correction": "hamming_code", "error_detection": "crc", "line_coding": "manchester", "access_list": "acl_rules",
    "vpc": "aws_vpc", "iam": "iam_policy", "resilience": "circuit_breaker", "consensus": "raft_election",
    "partitioning": "sharding", "hash_tree": "merkle_tree", "nmap": "port_scan", "pki": "cert_chain",
    "relu": "activations", "spam_filter": "naive_bayes", "density_clustering": "dbscan",
    "hierarchical_clustering": "dendrogram", "similarity": "cosine_similarity", "sampling": "temperature_sampling",
    "rag": "rag_pipeline", "running_total": "window_functions", "z_score": "standardization", "projection": "dot_product",
    "complex_numbers": "complex_plane", "parabola": "quadratic", "series": "sequences", "markov": "markov_chain",
    "boolean_logic": "truth_table", "combinations": "permutations", "graph_theory": "euler_path",
    "capacitor": "rc_circuit", "harmonics": "standing_waves", "interference": "double_slit",
    "centripetal": "circular_motion", "orbit": "kepler_orbit", "phase_change": "heating_curve",
    "photon": "photoelectric", "kinetic_theory": "maxwell_boltzmann", "electrochemistry": "galvanic_cell",
    "dna_amplification": "pcr", "restriction_enzyme": "gel_electrophoresis", "evolutionary_tree": "phylogenetic_tree",
    "homeostasis": "feedback_loop", "vegetation_index": "ndvi", "earthquake": "epicenter", "slope": "slope_hillshade",
    "inequality": "lorenz_gini", "game_theory": "prisoners_dilemma", "trade": "comparative_advantage",
})
