"""
StudyMate v4 - drawing library, part 5. (computed) = worked out by the engine, not guessed.

Algorithms   quick_sort (Lomuto partition on bars, computed), maze_bfs (shortest path on a grid, computed),
             bloom_filter (k hashes into a bit array, false positive shown, computed),
             consistent_hash (keys on a ring move to the next server, computed),
             list_reverse (pointers flipped one by one), sliding_window (best window sum, computed)
Systems      memory_layout (text, data, heap, stack of a running program), fork_tree (processes made by fork,
             computed), disk_sched (FCFS / SSTF / SCAN head movement, computed), raid (0, 1, 5 with real XOR parity),
             cpu_cycle (fetch-decode-execute of a tiny accumulator program, computed), ieee754 (float bits, computed),
             utf8 (text -> code points -> bytes, computed)
Networks     nat_table (private -> public port mapping), routing_lpm (longest prefix match, computed),
             traceroute (TTL expiring hop by hop)
Security     buffer_overflow (stack frame, return address overwritten), brute_force (key space and time, computed)
ML / DL      backprop (computational graph, forward values and gradients, computed), roc_curve (thresholds, AUC,
             computed), grid_world (value iteration, computed), pca (principal axis of 2-D data, computed),
             word_vectors (king - man + woman), transformer_block (encoder layer stack)
Data         sql_joins (inner / left / right / full on two tables, computed), normalization (one table -> 3NF),
             star_schema, moving_average (computed), correlation (scatter + Pearson r, computed)
Maths        sieve (Sieve of Eratosthenes, computed), euclid_gcd (rectangles + steps, computed),
             pascal (triangle, computed), binomial (distribution bars, computed), bayes_test (natural frequencies,
             computed), monte_carlo_pi (computed)
Physics      collision (momentum, elastic / inelastic, computed), incline (force components, computed),
             gas_law (piston, PV = nRT, computed), half_life (decay, computed), doppler (wavefronts)
Chemistry    electron_config (orbitals filled in order, computed)
Biology      food_web, energy_pyramid (10% rule, computed), natural_selection (allele frequency over generations,
             computed)
Earth / GIS  seasons (tilt and orbit), greenhouse, rock_cycle, map_projection (Mercator scale factor, computed),
             voronoi (nearest-site regions, computed), choropleth (equal-interval classes, computed)
Economics    amortization (loan schedule, computed), ppf (production possibilities frontier)
"""
import hashlib
import math
import random
import re
import struct

import visuals
from canvas import INK, MUTED, fit_text, text_size
from drawlib import _envelope, _poly, _Track, draw_device, draw_thing
from drawlib2 import PAL, STRONG, _box, _table
from drawlib3 import _frames, _title
from drawlib4 import _axes


def _mono(el):
    if el is not None:
        el["fontFamily"] = 3
    return el


# ======================================================================== algorithms

def build_quick_sort(cv, v, box, step):
    arr = [int(x) for x in (v.get("array") or [7, 2, 9, 4, 3, 8, 5])][:9]
    a, frames = list(arr), []

    def part(lo, hi):
        p = a[hi]
        i = lo
        for j in range(lo, hi):
            frames.append((list(a), lo, hi, j, i, f"pivot {p}: is {a[j]} < {p}?"))
            if a[j] < p:
                a[i], a[j] = a[j], a[i]
                i += 1
        a[i], a[hi] = a[hi], a[i]
        frames.append((list(a), lo, hi, i, i, f"pivot {p} goes to index {i}"))
        return i

    def qs(lo, hi):
        if lo < hi:
            m = part(lo, hi)
            qs(lo, m - 1)
            qs(m + 1, hi)
    qs(0, len(a) - 1)
    frames.append((list(a), -1, -1, -1, -1, "sorted"))
    if len(frames) > 22:
        idx = sorted({round(i * (len(frames) - 1) / 21) for i in range(22)})
        frames = [frames[i] for i in idx]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    bw = W * 0.66 / len(arr)
    mx = max(arr)
    _title(cv, x0, y0, "Quick sort (pivot = last element)", step)
    step += 1

    def draw(k, st):
        arr_, lo, hi, j, i, msg = frames[k]
        for n, val in enumerate(arr_):
            h = (H - 120) * val / mx
            col = "#b2f2bb" if lo < 0 else "#e9ecef" if not (lo <= n <= hi) else "#ffd43b" if n == hi else \
                "#ffc9c9" if n == j else "#a5d8ff"
            cv.rect(x0 + 10 + n * bw, y1 - 30 - h, bw - 10, h, st, col)
            cv.text(x0 + 10 + n * bw + (bw - 10) / 2, y1 - 26, str(val), st, 19, INK, "center", "top", math=False)
        cv.text(x0 + W * 0.7, y0 + 60, f"{msg}\n\nyellow = pivot\ngrey = outside this partition", st, 20,
                "#c92a2a" if "?" in msg else "#2f9e44", math=False)
    return _frames(cv, step, len(frames), draw)


def build_maze_bfs(cv, v, box, step):
    grid = v.get("grid") or ["S..#......", ".#.#.####.", ".#...#....", ".####.#.#.", "......#.#G"]
    R, C = len(grid), len(grid[0])
    start = next((r, c) for r in range(R) for c in range(C) if grid[r][c] == "S")
    goal = next((r, c) for r in range(R) for c in range(C) if grid[r][c] == "G")
    dist, prev, order, q = {start: 0}, {}, [], [start]
    while q:
        u = q.pop(0)
        order.append(u)
        if u == goal:
            break
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            w = (u[0] + dr, u[1] + dc)
            if 0 <= w[0] < R and 0 <= w[1] < C and grid[w[0]][w[1]] != "#" and w not in dist:
                dist[w], prev[w] = dist[u] + 1, u
                q.append(w)
    path, u = [], goal
    while u in prev or u == start:
        path.append(u)
        if u == start:
            break
        u = prev[u]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    c = min((W * 0.66) / C, (H - 20) / R)
    for r in range(R):
        for cc in range(C):
            ch = grid[r][cc]
            cv.rect(x0 + cc * c, y0 + r * c, c - 2, c - 2, step, "#343a40" if ch == "#" else "#ffffff",
                    INK, 1, rounded=False)
            if ch in "SG":
                cv.text(x0 + cc * c + c / 2, y0 + r * c + c / 2, ch, step, 22, "#e03131", "center", "middle", math=False)
    step += 1
    chunk = max(1, len(order) // 10)
    for i in range(0, len(order), chunk):
        for (r, cc) in order[i:i + chunk]:
            if grid[r][cc] not in "SG":
                cv.rect(x0 + cc * c + 2, y0 + r * c + 2, c - 6, c - 6, step, "#a5d8ff", "transparent", 1, rounded=False)
                cv.text(x0 + cc * c + c / 2, y0 + r * c + c / 2, str(dist[(r, cc)]), step, 14, "#1971c2", "center",
                        "middle", math=False)
        step += 1
    pts = [(x0 + cc * c + c / 2, y0 + r * c + c / 2) for r, cc in reversed(path)]
    line = cv.line(pts, step, "#e03131", 5)
    line["roundness"] = None
    cv.text(x0 + W * 0.7, y0 + 30, f"breadth-first search explores\nring by ring (numbers = steps)\n\n"
                                   f"explored {len(order)} cells\nshortest path: {dist.get(goal, '?')} steps", step, 21, INK,
            math=False)
    return step + 1


def build_bloom_filter(cv, v, box, step):
    m, k = int(v.get("bits") or 16), int(v.get("hashes") or 3)
    adds = v.get("add") or ["cat", "dog", "fish"]
    checks = v.get("check") or ["dog", "bird", "cow"]
    hs = lambda s: [int(hashlib.sha256(f"{i}:{s}".encode()).hexdigest(), 16) % m for i in range(k)]
    bits = [0] * m
    x0, y0, x1, y1 = box
    W = x1 - x0
    cw = min(52, (W - 20) / m)
    _title(cv, x0, y0, f"Bloom filter: {m} bits, {k} hash functions", step, 21)
    for i in range(m):
        cv.text(x0 + i * cw + cw / 2, y0 + 44, str(i), step, 13, MUTED, "center", "top", math=False)
    step += 1

    def draw_bits(st, hl=(), col="#b2f2bb"):
        for i, b in enumerate(bits):
            _box(cv, x0 + i * cw, y0 + 64, cw - 3, 44, st, b, col if i in hl else ("#d0ebff" if b else "#ffffff"), 18)
    for n, w in enumerate(adds):
        h = hs(w)
        for i in h:
            bits[i] = 1
        tr = _Track(cv)
        draw_bits(step, h)
        tr.tag(until=step + 1, fade=n > 0)
        cv.text(x0, y0 + 130 + n * 30, f'add "{w}": bits {sorted(set(h))} set to 1', step, 18, "#2f9e44", math=False)
        step += 1
    draw_bits(step)
    step += 1
    for n, w in enumerate(checks):
        h = hs(w)
        allset = all(bits[i] for i in h)
        truth = w in adds
        verdict = "probably yes" if allset else "definitely no"
        note = "" if allset == truth else "  ← false positive!"
        cv.text(x0, y0 + 150 + (len(adds) + n) * 30, f'check "{w}": bits {sorted(set(h))} → {verdict}{note}', step, 18,
                "#e03131" if note else ("#1971c2" if allset else MUTED), math=False)
        step += 1
    cv.text(x0, y0 + 180 + (len(adds) + len(checks)) * 30, "never a false negative; small chance of a false positive;\n"
                                                           "uses far less memory than storing the items", step, 18, INK,
            math=False)
    return step + 1


def build_consistent_hash(cv, v, box, step):
    servers = v.get("servers") or ["server A", "server B", "server C"]
    keys = v.get("keys") or ["user:1", "user:2", "cart:7", "img:42", "user:9", "cart:3"]
    pos = lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 360
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    R = min(W * 0.28, H * 0.42)
    cx, cy = x0 + R + 40, (y0 + y1) / 2
    cv.ellipse(cx - R, cy - R, 2 * R, 2 * R, step, "transparent", MUTED, 2)
    P = lambda a, r=R: (cx + r * math.sin(math.radians(a)), cy - r * math.cos(math.radians(a)))
    step += 1
    sp = sorted((pos(s), s) for s in servers)
    for i, (a, s) in enumerate(sp):
        x, y = P(a)
        draw_device(cv, "server", x, y, 54, step)
        cv.text(*P(a, R + 46), s, step, 17, STRONG[i], "center", "middle", math=False)
    step += 1

    def owner(a, sp_):
        for b, s in sp_:
            if b >= a:
                return s
        return sp_[0][1]
    col = {s: STRONG[i] for i, (_, s) in enumerate(sp)}
    for kk in keys:
        a = pos(kk)
        x, y = P(a, R - 30)
        cv.ellipse(x - 7, y - 7, 14, 14, step, col[owner(a, sp)], col[owner(a, sp)], 1)
        cv.text(*P(a, R - 62), kk, step, 13, INK, "center", "middle", math=False)
    lines = [f"{kk} → {owner(pos(kk), sp)}" for kk in keys]
    cv.text(x0 + W * 0.62, y0 + 20, "each key goes clockwise to the\nnext server on the ring\n\n" + "\n".join(lines), step,
            18, INK, math=False)
    step += 1
    gone = max((s for _, s in sp), key=lambda s: sum(owner(pos(kk), sp) == s for kk in keys))
    sp2 = [t for t in sp if t[1] != gone]
    moved = [kk for kk in keys if owner(pos(kk), sp) == gone]
    cv.text(x0 + W * 0.62, y0 + 60 + 26 * (len(keys) + 4), f"if {gone} fails, only its keys move:\n" +
            ("\n".join(f"{kk} → {owner(pos(kk), sp2)}" for kk in moved) or "(none)") +
            f"\n({len(moved)} of {len(keys)}, not all of them)", step, 18, "#e03131", math=False)
    return step + 1


def build_list_reverse(cv, v, box, step):
    vals = v.get("values") or [1, 2, 3, 4]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    n = len(vals)
    bw, gap = 100, (W * 0.9 - 100 * n) / max(1, n)
    y = y0 + H * 0.35
    xs = [x0 + 40 + i * (bw + gap) for i in range(n)]
    for i, val in enumerate(vals):
        _box(cv, xs[i], y, bw, 60, step, val, "#d0ebff", 22)
    step += 1
    code = ["prev = None", "while cur:", "    nxt = cur.next", "    cur.next = prev", "    prev, cur = cur, nxt"]
    for i, ln in enumerate(code):
        _mono(cv.text(x0, y0 + H * 0.62 + i * 26, ln, step, 17, INK, math=False))
    for k in range(n + 1):
        tr = _Track(cv)
        for i in range(n - 1):
            if i + 1 < k:          # already reversed: arrow points back
                cv.arrow(xs[i + 1] - 2, y + 44, xs[i] + bw + 2, y + 44, step, color="#e03131")
            elif i >= k - 1 and i + 1 >= k:
                cv.arrow(xs[i] + bw + 2, y + 16, xs[i + 1] - 2, y + 16, step, color="#1971c2")
        if k < n:
            cv.text(xs[k] + bw / 2, y - 8, "cur", step, 17, "#1971c2", "center", "bottom", math=False)
        if k > 0:
            cv.text(xs[k - 1] + bw / 2, y + 66, "prev", step, 17, "#e03131", "center", "top", math=False)
        cv.text(xs[-1] + bw / 2 if k == n else xs[0] - 10, y - 40, "head" if k == n else "", step, 17, "#2f9e44",
                "center", "bottom", math=False)
        tr.tag(until=step + 1 if k < n else None, fade=k > 0)
        step += 1
    return step


def build_sliding_window(cv, v, box, step):
    arr = [int(x) for x in (v.get("array") or [2, 1, 5, 1, 3, 2, 6, 1])]
    k = int(v.get("k") or 3)
    x0, y0, x1, y1 = box
    W = x1 - x0
    c = min(70, (W - 20) / len(arr))
    for i, val in enumerate(arr):
        _box(cv, x0 + i * c, y0 + 60, c - 4, 54, step, val, "#ffffff", 22)
        cv.text(x0 + i * c + c / 2, y0 + 118, str(i), step, 13, MUTED, "center", "top", math=False)
    _title(cv, x0, y0, f"Sliding window: largest sum of {k} in a row", step, 21)
    step += 1
    s = sum(arr[:k])
    best, bi = s, 0
    for i in range(len(arr) - k + 1):
        if i:
            s += arr[i + k - 1] - arr[i - 1]
        if s > best:
            best, bi = s, i
        tr = _Track(cv)
        cv.rect(x0 + i * c - 4, y0 + 54, k * c + 4, 66, step, "transparent", "#f08c00", 4)
        msg = f"window {i}..{i + k - 1}: sum = {s}" + (f"  (− {arr[i - 1]} + {arr[i + k - 1]})" if i else "")
        cv.text(x0, y0 + 160, msg, step, 21, "#c92a2a", math=False)
        cv.text(x0, y0 + 200, f"best so far: {best} at {bi}..{bi + k - 1}", step, 21, "#2f9e44", math=False)
        tr.tag(until=step + 1 if i < len(arr) - k else None, fade=i > 0)
        step += 1
    cv.text(x0, y0 + 260, "each step adds the new element and drops the old one: O(n), not O(n·k)", step, 19, INK,
            math=False)
    return step + 1


# ======================================================================== systems

def build_memory_layout(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    segs = [("stack", "#ffc9c9", "main(): int x = 5\nfactorial(3): n = 3\n↓ grows down"),
            ("(free space)", "#ffffff", ""), ("heap", "#b2f2bb", "↑ grows up\nmalloc / new: node, list"),
            ("data (globals, static)", "#ffec99", "int counter = 0;"), ("text (code)", "#d0ebff", "machine instructions")]
    hs = [0.24, 0.16, 0.22, 0.17, 0.21]
    y = y0
    bx, bw = x0 + 120, W * 0.32
    cv.text(x0, y0, "high address", step, 16, MUTED, math=False)
    for (name, col, note), hf in zip(segs, hs):
        h = (H - 30) * hf
        cv.rect(bx, y, bw, h, step, col, INK, 2, rounded=False)
        cv.text(bx + bw / 2, y + h / 2, name, step, 20, INK, "center", "middle", math=False)
        if note:
            cv.text(bx + bw + 30, y + h / 2, note, step, 18, "#495057", "left", "middle", math=False)
        y += h
        step += 1
    cv.text(x0, y1 - 20, "low address", step, 16, MUTED, math=False)
    return step


def build_fork_tree(cv, v, box, step):
    n = min(4, int(v.get("forks") or 3))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    tw = W * 0.62
    for i in range(n):
        _mono(cv.text(x0 + W * 0.68, y0 + 40 + i * 30, "fork();", step, 20, INK, math=False))
    step += 1
    level = [("P0", 0)]                  # (name, slot) ; slot index among 2^d
    xs = lambda d, s: x0 + tw * (s + 0.5) / 2 ** d
    ys = lambda d: y0 + 40 + d * (H - 80) / n
    cv.ellipse(xs(0, 0) - 26, ys(0) - 22, 52, 44, step, "#d0ebff")
    cv.text(xs(0, 0), ys(0), "P0", step, 16, INK, "center", "middle", math=False)
    step += 1
    count = 1
    for d in range(1, n + 1):
        new = []
        for name, s in level:
            child = f"P{count}"
            count += 1
            for nm, ns, col in ((name, 2 * s, "#e9ecef"), (child, 2 * s + 1, PAL[d % len(PAL)])):
                cv.line([(xs(d - 1, s), ys(d - 1) + 22), (xs(d, ns), ys(d) - 22)], step, MUTED if nm == name else
                        "#e8590c", 1.5)
                cv.ellipse(xs(d, ns) - 24, ys(d) - 20, 48, 40, step, col)
                cv.text(xs(d, ns), ys(d), nm, step, 15, INK, "center", "middle", math=False)
            new += [(name, 2 * s), (child, 2 * s + 1)]
        level = new
        cv.text(x0 + W * 0.66, ys(d), f"after fork #{d}: {2 ** d} processes", step, 18, "#e8590c", "left", "middle",
                math=False)
        step += 1
    cv.text(x0 + W * 0.68, y0 + 40 + n * 30 + 10, "grey = the same process continuing\norange line = a new child", step, 16, MUTED,
            math=False)
    return step + 1


def disk_order(reqs, head, algo, size=200):
    reqs = list(reqs)
    if algo == "sstf":
        out, cur = [], head
        while reqs:
            nxt = min(reqs, key=lambda r: abs(r - cur))
            out.append(nxt)
            reqs.remove(nxt)
            cur = nxt
        return out
    if algo == "scan":
        up = sorted(r for r in reqs if r >= head)
        down = sorted((r for r in reqs if r < head), reverse=True)
        return up + ([size - 1] if down else []) + down
    return reqs


def build_disk_sched(cv, v, box, step):
    reqs = [int(x) for x in (v.get("requests") or [98, 183, 37, 122, 14, 124, 65, 67])]
    head = int(v.get("head") or 53)
    algo = str(v.get("algorithm") or "sstf").lower()
    order = disk_order(reqs, head, algo)
    seq = [head] + order
    total = sum(abs(b - a) for a, b in zip(seq, seq[1:]))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    px = lambda c: x0 + 30 + c / 199 * W * 0.64
    last = -99
    for c in sorted(set(reqs + [head, 0, 199])):
        if px(c) - last > 26:
            cv.text(px(c), y0 + 30, str(c), step, 13, "#e03131" if c == head else MUTED, "center", "bottom", math=False)
            last = px(c)
    cv.line([(px(0), y0 + 34), (px(199), y0 + 34)], step, INK, 2)
    _title(cv, x0, y0 - 6 - 20, f"Disk scheduling: {algo.upper()}, head at {head}", step, 20)
    step += 1
    dy = (H - 80) / len(seq)
    for i in range(1, len(seq)):
        cv.line([(px(seq[i - 1]), y0 + 50 + (i - 1) * dy), (px(seq[i]), y0 + 50 + i * dy)], step, "#1971c2", 2.5,
                arrow_end=True)
        cv.ellipse(px(seq[i]) - 5, y0 + 45 + i * dy, 10, 10, step, "#e03131", "#e03131", 1)
        step += 1
    others = {a: sum(abs(b - c) for c, b in zip([head] + disk_order(reqs, head, a), disk_order(reqs, head, a)))
              for a in ("fcfs", "sstf", "scan")}
    cv.text(x0 + W * 0.7, y0 + 60, f"order: {', '.join(map(str, order))}\n\ntotal head movement: {total} cylinders\n\n" +
            "\n".join(f"{a.upper()}: {t}" for a, t in others.items()), step, 19, INK, math=False)
    return step + 1


def build_raid(cv, v, box, step):
    level = int(v.get("level") or 5)
    blocks = v.get("blocks") or ["A1", "A2", "A3", "B1", "B2", "B3"]
    data = {b: (hash(b) & 0xFF) for b in blocks}
    data = {b: int(hashlib.md5(b.encode()).hexdigest()[:2], 16) for b in blocks}
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    disks = 4 if level == 5 else 2 if level in (0, 1) else 4
    dw = (W * 0.7 - 30 * (disks - 1)) / disks
    for d in range(disks):
        draw_device(cv, "database", x0 + d * (dw + 30) + dw / 2, y0 + 40, 60, step)
        cv.text(x0 + d * (dw + 30) + dw / 2, y0 + 80, f"disk {d}", step, 17, INK, "center", "top", math=False)
    _title(cv, x0, y0 - 26, f"RAID {level}", step, 22)
    step += 1
    rows = []
    if level == 0:
        rows = [blocks[i:i + 2] for i in range(0, len(blocks), 2)]
    elif level == 1:
        rows = [[b, b + "'"] for b in blocks[:4]]
    else:
        for r, i in enumerate(range(0, len(blocks), 3)):
            stripe = blocks[i:i + 3]
            par = 0
            for b in stripe:
                par ^= data[b]
            row = list(stripe)
            row.insert(3 - r % 4, f"P = {par:02X}")
            rows.append(row)
    for r, row in enumerate(rows):
        for d, b in enumerate(row[:disks]):
            par = b.startswith("P")
            lab = b if par or level != 5 else f"{b} ({data[b]:02X})"
            _box(cv, x0 + d * (dw + 30), y0 + 120 + r * 56, dw, 48, step, lab,
                 "#ffd8a8" if par else "#e7f5ff" if "'" not in b else "#d3f9d8", 17, mono=par)
        step += 1
    notes = {0: "striping: fast, full capacity,\nbut ONE failed disk loses everything",
             1: "mirroring: every block twice;\nsurvives a disk failure, 50% capacity",
             5: "striping + parity (XOR) rotated across disks;\nlose any ONE disk and XOR the rest to rebuild it"}
    cv.text(x0 + W * 0.72, y0 + 120, notes.get(level, ""), step, 18, INK, math=False)
    if level == 5 and rows:
        s = rows[0]
        vals = [data[b] for b in blocks[:3]]
        cv.text(x0 + W * 0.72, y0 + 220, f"rebuild {blocks[1]}:\n{vals[0]:02X} XOR {vals[2]:02X} XOR parity\n"
                                         f"= {vals[1]:02X} ✔", step, 18, "#2f9e44", math=False)
    return step + 1


def run_acc(prog, mem):
    """Tiny accumulator machine: LOAD a, ADD a, SUB a, STORE a, JNZ label, HALT."""
    acc, pc, steps = 0, 0, []
    mem = dict(mem)
    for _ in range(40):
        if pc >= len(prog):
            break
        op, arg = prog[pc]
        ir = f"{op} {arg}".strip()
        cur = pc
        steps.append(("fetch", pc, ir, acc, dict(mem), cur))
        if op == "LOAD":
            acc = mem[arg]
        elif op == "ADD":
            acc += mem[arg]
        elif op == "SUB":
            acc -= mem[arg]
        elif op == "STORE":
            mem[arg] = acc
        elif op == "HALT":
            steps.append(("execute", pc, ir, acc, dict(mem), cur))
            break
        pc += 1
        steps.append(("execute", pc, ir, acc, dict(mem), cur))
    return steps


def build_cpu_cycle(cv, v, box, step):
    prog = [("LOAD", "x"), ("ADD", "y"), ("STORE", "z"), ("HALT", "")]
    mem = {"x": 7, "y": 5, "z": 0}
    steps = run_acc(prog, mem)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cv.rect(x0, y0, W * 0.38, H - 20, step, "#f8f9fa", INK, 2)
    cv.text(x0 + 10, y0 + 8, "CPU", step, 22, "#e8590c", math=False)
    mx = x0 + W * 0.5
    cv.rect(mx, y0, W * 0.3, H - 20, step, "#f8f9fa", INK, 2)
    cv.text(mx + 10, y0 + 8, "memory", step, 22, "#e8590c", math=False)
    for i, (op, arg) in enumerate(prog):
        _mono(cv.text(mx + 20, y0 + 50 + i * 34, f"{i}: {op} {arg}", step, 18, INK, math=False))
    cv.arrow(x0 + W * 0.38 + 4, y0 + H / 2, mx - 4, y0 + H / 2, step, "bus", color=MUTED)
    step += 1

    def draw(k, st):
        phase, pc, ir, acc, m, line = steps[k]
        for i, (name, val, col) in enumerate((("PC", pc, "#d0ebff"), ("IR", ir, "#ffec99"), ("ACC", acc, "#b2f2bb"))):
            cv.text(x0 + 20, y0 + 70 + i * 70, name, step=st, size=20, color=INK, math=False)
            _box(cv, x0 + 90, y0 + 56 + i * 70, W * 0.24, 50, st, val, col, 20, mono=True)
        cv.text(x0 + 20, y0 + 290, phase.upper() + (": read instruction at PC" if phase == "fetch" else ": do it, PC + 1"),
                st, 18, "#c92a2a", math=False)
        cv.rect(mx + 10, y0 + 44 + line * 34, W * 0.28, 32, st, "transparent", "#f08c00", 3)
        for i, (name, val) in enumerate(m.items()):
            _mono(cv.text(mx + 20, y0 + 220 + i * 34, f"{name} = {val}", st, 18, "#1971c2", math=False))
    return _frames(cv, step, len(steps), draw)


def build_ieee754(cv, v, box, step):
    num = float(v.get("number") or -6.75)
    bits = format(struct.unpack(">I", struct.pack(">f", num))[0], "032b")
    s, e, f = bits[0], bits[1:9], bits[9:]
    exp = int(e, 2)
    x0, y0, x1, y1 = box
    W = x1 - x0
    _title(cv, x0, y0, f"{num:g} as a 32-bit float (IEEE 754)", step, 22)
    step += 1
    cw = min(34, (W - 40) / 32)
    for i, b in enumerate(bits):
        col = "#ffc9c9" if i == 0 else "#ffec99" if i < 9 else "#d0ebff"
        _box(cv, x0 + i * cw + (6 if i >= 1 else 0) + (6 if i >= 9 else 0), y0 + 60, cw - 2, 44, step, b, col, 16)
    for lab, a, b_, col in (("sign", 0, 1, "#e03131"), ("exponent (8)", 1, 9, "#e67700"), ("fraction (23)", 9, 32, "#1971c2")):
        cv.text(x0 + (a + b_) / 2 * cw + 6, y0 + 112, lab, step, 15, col, "center", "top", math=False)
    step += 1
    mant = 1 + sum(int(c) * 2 ** -(i + 1) for i, c in enumerate(f))
    lines = [f"sign {s} → {'negative' if s == '1' else 'positive'}",
             f"exponent {e} = {exp}, minus bias 127 → {exp - 127}",
             f"fraction → 1.{f.rstrip('0') or '0'} (binary) = {mant:g}",
             f"value = {'−' if s == '1' else ''}{mant:g} × 2^{exp - 127} = {num:g}"]
    for i, ln in enumerate(lines):
        cv.text(x0, y0 + 160 + i * 40, ln, step, 21, INK, math=False)
        step += 1
    cv.text(x0, y0 + 340, "0.1 has no exact binary form, which is why 0.1 + 0.2 != 0.3 in floating point", step, 18,
            "#e03131", math=False)
    return step + 1


def build_utf8(cv, v, box, step):
    text = str(v.get("text") or "Hé€😀")[:6]
    x0, y0, x1, y1 = box
    W = x1 - x0
    _title(cv, x0, y0, f'"{text}" in UTF-8', step, 22)
    step += 1
    x = x0
    total = 0
    for ch in text:
        b = ch.encode("utf-8")
        total += len(b)
        cw = max(110, len(b) * 92)
        _box(cv, x, y0 + 50, cw - 10, 54, step, ch, "#ffec99", 28)
        cv.text(x + (cw - 10) / 2, y0 + 112, f"U+{ord(ch):04X}", step, 16, "#7048e8", "center", "top", math=False)
        for j, by in enumerate(b):
            _box(cv, x + j * 92, y0 + 150, 86, 44, step, f"{by:02X}", "#d0ebff", 18, mono=True)
            _mono(cv.text(x + j * 92 + 43, y0 + 200, f"{by:08b}", step, 12, MUTED, "center", "top", math=False))
        cv.text(x + (cw - 10) / 2, y0 + 228, f"{len(b)} byte{'s' if len(b) > 1 else ''}", step, 16, INK, "center",
                "top", math=False)
        x += cw + 10
        step += 1
    cv.text(x0, y0 + 280, f"{len(text)} characters, {total} bytes. ASCII letters use 1 byte; accents 2; € 3; emoji 4.\n"
                          "Leading bits tell how long each character is: 0xxxxxxx, 110xxxxx, 1110xxxx, 11110xxx.", step, 18,
            INK, math=False)
    return step + 1


# ======================================================================== networks

def build_nat_table(cv, v, box, step):
    public = str(v.get("public") or "203.0.113.7")
    hosts = v.get("hosts") or [("192.168.1.10", 51000), ("192.168.1.11", 51000), ("192.168.1.12", 49152)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    for i, (ip, port) in enumerate(hosts):
        draw_device(cv, "laptop", x0 + 50, y0 + 60 + i * 90, 60, step)
        cv.text(x0 + 90, y0 + 60 + i * 90, f"{ip}:{port}", step, 16, INK, "left", "middle", math=False)
    draw_device(cv, "router", x0 + W * 0.4, y0 + H * 0.35, 90, step)
    cv.text(x0 + W * 0.4, y0 + H * 0.35 + 40, f"NAT router\npublic {public}", step, 16, INK, "center", "top", math=False)
    draw_device(cv, "internet", x0 + W * 0.85, y0 + H * 0.35, 90, step)
    step += 1
    rows = []
    for i, (ip, port) in enumerate(hosts):
        pub_port = 40001 + i
        rows.append([f"{ip}:{port}", f"{public}:{pub_port}"])
        cv.arrow(x0 + 250, y0 + 60 + i * 90, x0 + W * 0.4 - 50, y0 + H * 0.35, step, color="#1971c2")
        cv.arrow(x0 + W * 0.4 + 50, y0 + H * 0.35, x0 + W * 0.85 - 50, y0 + H * 0.35 - 20 + i * 14, step,
                 f"from {public}:{pub_port}" if i == 0 else "", color="#2f9e44")
        step += 1
    _table(cv, x0 + W * 0.3, y0 + H * 0.62, "translation table", ["inside (private)", "outside (public)"], rows, step,
           W * 0.5, size=15)
    return step + 1


def _ip(s):
    return sum(int(o) << (8 * (3 - i)) for i, o in enumerate(s.split(".")))


def build_routing_lpm(cv, v, box, step):
    table = v.get("routes") or [["0.0.0.0/0", "ISP (default)"], ["10.0.0.0/8", "core router"],
                                ["10.1.0.0/16", "campus"], ["10.1.2.0/24", "lab switch"]]
    dests = v.get("destinations") or ["10.1.2.77", "10.1.9.4", "10.200.0.1", "8.8.8.8"]
    x0, y0, x1, y1 = box
    W = x1 - x0
    _table(cv, x0, y0 + 30, "routing table", ["prefix", "next hop"], table, step, W * 0.4, size=16)
    step += 1
    for k, d in enumerate(dests):
        best, bl = None, -1
        matches = []
        for pref, hop in table:
            net, _, p = pref.partition("/")
            p = int(p)
            mask = (0xFFFFFFFF << (32 - p)) & 0xFFFFFFFF if p else 0
            if _ip(d) & mask == _ip(net) & mask:
                matches.append(pref)
                if p > bl:
                    best, bl = hop, p
        cv.text(x0 + W * 0.46, y0 + 40 + k * 70, f"{d}: matches {', '.join(matches)}\n→ longest is /{bl} → {best}", step,
                19, "#2f9e44", math=False)
        step += 1
    cv.text(x0, y0 + 260, "a router picks the MOST specific matching prefix (longest prefix match)", step, 19, INK,
            math=False)
    return step + 1


def build_traceroute(cv, v, box, step):
    hops = v.get("hops") or ["home router", "ISP router", "regional router", "backbone router", "web server"]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    n = len(hops)
    xs = [x0 + 60 + (i + 1) * (W - 120) / (n + 1) for i in range(n)]
    cy = y0 + H * 0.35
    draw_device(cv, "laptop", x0 + 60, cy, 70, step)
    for i, (x, h) in enumerate(zip(xs, hops)):
        draw_device(cv, "server" if i == n - 1 else "router", x, cy, 70, step)
        cv.text(x, cy + 50, h, step, 15, INK, "center", "top", math=False)
    step += 1
    for ttl in range(1, n + 1):
        msg = (f"TTL={ttl}: the packet dies at hop {ttl} ({hops[ttl - 1]}), which replies 'time exceeded'"
               if ttl < n else f"TTL={ttl}: it reaches the {hops[-1]}, which answers: destination reached")
        cv.text(x0, y0 + H * 0.6 + (ttl - 1) * 28, f"{msg} → {10 + ttl * 7} ms", step, 17,
                "#2f9e44" if ttl == n else INK, math=False)
        for k in range(ttl + 1):
            px = x0 + 60 if k == 0 else xs[k - 1]
            last = k == ttl
            tr = _Track(cv)
            _envelope(cv, px, cy - 60, 40, step, f"TTL {ttl - k}" if not last else "✘" if ttl < n else "✔")
            tr.tag(until=step + 1, fade=True)
            step += 1
    return step


# ======================================================================== security

def build_buffer_overflow(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    code = ["void login() {", "    char name[8];", "    gets(name);   // no length check!", "}"]
    for i, ln in enumerate(code):
        _mono(cv.text(x0, y0 + i * 26, ln, step, 17, INK, math=False))
    step += 1
    sx, sw = x0 + W * 0.45, W * 0.3
    cells = [("return address", "#ffc9c9"), ("saved frame pointer", "#ffd8a8")] + [(f"name[{i}]", "#ffffff") for i in range(7, -1, -1)]
    ch = (H - 40) / len(cells)
    for i, (lab, col) in enumerate(cells):
        _box(cv, sx, y0 + i * ch, sw, ch - 2, step, lab, col, 15)
    cv.text(sx + sw + 10, y0 + 10, "higher addresses", step, 14, MUTED, math=False)
    cv.text(sx + sw + 10, y1 - 50, "name starts here,\nwrites go UP ↑", step, 14, MUTED, math=False)
    step += 1
    payload = "AAAAAAAAAAAAAAAA\\xde\\xad\\xbe\\xef"
    cv.text(x0, y0 + 140, "attacker types 20 bytes:", step, 18, "#e03131", math=False)
    _mono(cv.text(x0, y0 + 170, payload, step, 17, "#e03131", math=False))
    step += 1
    fill = ["A"] * 16 + ["0xdeadbeef"]
    order = list(range(len(cells) - 1, -1, -1))
    for k, i in enumerate(order):
        lab = "0xdeadbeef ← hijacked!" if i == 0 else "AAAAAAAA" if i == 1 else "'A'"
        _box(cv, sx, y0 + i * ch, sw, ch - 2, step, lab, "#e03131" if i == 0 else "#ffe3e3", 15)
        if k % 2:
            step += 1
    cv.text(x0, y0 + 230, "when login() returns, the CPU jumps\nto the attacker's address.\n\nfixes: fgets / bounds checks,\n"
                          "stack canaries, ASLR, NX bit,\nmemory-safe languages", step + 1, 18, INK, math=False)
    return step + 2


def build_brute_force(cv, v, box, step):
    rate = float(v.get("guesses_per_second") or 1e10)
    rows = v.get("passwords") or [("digits", 10, 6), ("lowercase", 26, 8), ("upper+lower+digits", 62, 8),
                                  ("all 95 symbols", 95, 12), ("4 random words (7776 list)", 7776, 4)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0

    def human(sec):
        for unit, s in (("years", 3.15e7), ("days", 86400), ("hours", 3600), ("minutes", 60)):
            if sec >= s:
                val = sec / s
                return f"{val:,.0f} {unit}" if val < 1e15 else f"{val:.1e} {unit}"
        return f"{sec:.1f} s" if sec >= 1 else "instantly (under 1 s)"
    _title(cv, x0, y0, f"Brute force at {rate:.0e} guesses per second", step, 21)
    step += 1
    mxlog = max(math.log10(c ** n) for _, c, n in rows)
    for i, (name, c, n) in enumerate(rows):
        space = c ** n
        bits = n * math.log2(c)
        t = space / rate / 2
        y = y0 + 50 + i * (H - 80) / len(rows)
        cv.text(x0, y + 4, f"{name}, length {n}", step, 18, INK, math=False)
        w = (W * 0.45) * math.log10(space) / mxlog
        cv.rect(x0 + W * 0.3, y, w, 32, step, PAL[i % len(PAL)], STRONG[i % len(STRONG)], 1.5)
        cv.text(x0 + W * 0.3 + w + 10, y + 16, f"{c}^{n} = {space:.1e}  ({bits:.0f} bits) → {human(t)}", step, 16,
                "#2f9e44" if t > 3.15e9 else "#e03131", "left", "middle", math=False)
        step += 1
    return step


# ======================================================================== ML / DL

def build_backprop(cv, v, box, step):
    xv, wv, bv, yt = (float(v.get(k) or d) for k, d in (("x", 2), ("w", 3), ("b", 1), ("target", 4)))
    z = wv * xv + bv
    loss = (z - yt) ** 2
    dL_dz = 2 * (z - yt)
    dL_dw, dL_db, dL_dx = dL_dz * xv, dL_dz, dL_dz * wv
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    nodes = {"x": (0.05, 0.25), "w": (0.05, 0.65), "mul": (0.28, 0.45), "b": (0.28, 0.85), "add": (0.5, 0.6),
             "loss": (0.75, 0.6)}
    P = lambda k: (x0 + W * nodes[k][0] + 40, y0 + (H - 40) * nodes[k][1])
    labels = {"x": f"x = {xv:g}", "w": f"w = {wv:g}", "b": f"b = {bv:g}", "mul": "×", "add": "+",
              "loss": f"(z − {yt:g})²"}
    for k, lab in labels.items():
        x, y = P(k)
        if k in ("mul", "add"):
            cv.ellipse(x - 30, y - 30, 60, 60, step, "#ffec99")
        else:
            _box(cv, x - 60, y - 24, 120, 48, step, lab, "#e7f5ff" if k != "loss" else "#ffc9c9", 17)
            continue
        cv.text(x, y, lab, step, 24, INK, "center", "middle", math=False)
    for a, b in (("x", "mul"), ("w", "mul"), ("mul", "add"), ("b", "add"), ("add", "loss")):
        (ax, ay), (bx, by) = P(a), P(b)
        cv.arrow(ax + 62 if a not in ("mul", "add") else ax + 32, ay, bx - 32 if b in ("mul", "add") else bx - 62, by,
                 step, color=MUTED)
    step += 1
    fw = [("mul", f"wx = {wv * xv:g}"), ("add", f"z = {z:g}"), ("loss", f"L = {loss:g}")]
    for k, txt in fw:
        x, y = P(k)
        cv.text(x, y - 40, txt, step, 18, "#1971c2", "center", "bottom", math=False)
        step += 1
    bw = [("loss", f"dL/dz = 2(z−t) = {dL_dz:g}"), ("b", f"dL/db = {dL_db:g}"), ("w", f"dL/dw = dL/dz · x = {dL_dw:g}"),
          ("x", f"dL/dx = dL/dz · w = {dL_dx:g}")]
    for k, txt in bw:
        x, y = P(k)
        cv.text(x, y + 30, txt, step, 17, "#e03131", "center", "top", math=False)
        step += 1
    cv.text(x0 + W * 0.62, y0 + 10, "blue: forward pass (values)\nred: backward pass (gradients,\nchain rule)\n\n"
                                   f"update: w ← w − 0.01 · {dL_dw:g}\n= {wv - 0.01 * dL_dw:g}", step, 18, INK, math=False)
    return step + 1


def build_roc_curve(cv, v, box, step):
    rnd = random.Random(5)
    pos = v.get("positives") or [rnd.gauss(0.68, 0.15) for _ in range(40)]
    neg = v.get("negatives") or [rnd.gauss(0.4, 0.15) for _ in range(40)]
    ths = sorted(set([1.1] + [round(t, 3) for t in pos + neg] + [-0.1]), reverse=True)
    pts = [(sum(n >= t for n in neg) / len(neg), sum(p >= t for p in pos) / len(pos)) for t in ths]
    auc = sum((pts[i + 1][0] - pts[i][0]) * (pts[i + 1][1] + pts[i][1]) / 2 for i in range(len(pts) - 1))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    S = min(W * 0.55, H - 40)
    px = lambda f: x0 + 50 + f * S
    py = lambda t: y0 + 10 + (1 - t) * S
    _axes(cv, x0 + 50, y0, x0 + 60 + S, y0 + 10 + S, step, "false positive rate", "true positive rate")
    cv.line([(px(0), py(0)), (px(1), py(1))], step, MUTED, 1.5, dashed=True)
    step += 1
    chunk = max(1, len(pts) // 8)
    for i in range(0, len(pts) - 1, chunk):
        seg = pts[i:i + chunk + 1]
        line = cv.line([(px(a), py(b)) for a, b in seg], step, "#e03131", 3)
        line["roundness"] = None
        step += 1
    cv.text(x0 + W * 0.68, y0 + 30, f"each point = one threshold\n\nAUC = {auc:.2f}\n(1.0 perfect, 0.5 = guessing,\n"
                                   "the dashed line)", step, 21, INK, math=False)
    return step + 1


def value_iteration(R, C, goal, pit, walls, gamma=0.9, iters=12):
    V = {(r, c): 0.0 for r in range(R) for c in range(C)}
    hist = []
    for _ in range(iters):
        new = {}
        for (r, c) in V:
            if (r, c) in walls:
                new[(r, c)] = 0
                continue
            if (r, c) == goal:
                new[(r, c)] = 1.0
                continue
            if (r, c) == pit:
                new[(r, c)] = -1.0
                continue
            best = -1e9
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < R and 0 <= nc < C) or (nr, nc) in walls:
                    nr, nc = r, c
                best = max(best, -0.04 + gamma * V[(nr, nc)])
            new[(r, c)] = best
        V = new
        hist.append(dict(V))
    return hist


def build_grid_world(cv, v, box, step):
    R, C = 3, 4
    goal, pit, walls = (0, 3), (1, 3), {(1, 1)}
    hist = value_iteration(R, C, goal, pit, walls)
    show = [hist[i] for i in (0, 1, 2, 4, 8, len(hist) - 1)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    c = min(W * 0.55 / C, (H - 60) / R)
    _title(cv, x0, y0, "Value iteration (γ = 0.9, −0.04 per step)", step, 20)
    step += 1

    def draw(k, st):
        V = show[k]
        for (r, cc), val in V.items():
            if (r, cc) in walls:
                col = "#343a40"
            else:
                g = max(-1, min(1, val))
                col = "#b2f2bb" if g > 0.5 else "#d3f9d8" if g > 0 else "#ffc9c9" if g < 0 else "#ffffff"
            cv.rect(x0 + cc * c, y0 + 40 + r * c, c - 4, c - 4, st, col, INK, 1.5, rounded=False)
            if (r, cc) not in walls:
                cv.text(x0 + cc * c + c / 2, y0 + 40 + r * c + c / 2, f"{val:+.2f}", st, 18, INK, "center", "middle",
                        math=False)
        cv.text(x0 + W * 0.6, y0 + 60, f"sweep {[1, 2, 3, 5, 9, len(hist)][k]}", st, 24, "#e8590c", math=False)
    step = _frames(cv, step, len(show), draw)
    V = show[-1]
    arrows = {(0, 1): "→", (1, 0): "↓", (0, -1): "←", (-1, 0): "↑"}
    for (r, cc) in V:
        if (r, cc) in walls or (r, cc) in (goal, pit):
            continue
        best = max(arrows, key=lambda d: V.get((r + d[0], cc + d[1]), -9) if (r + d[0], cc + d[1]) not in walls else -9)
        cv.text(x0 + cc * c + c - 18, y0 + 40 + r * c + 16, arrows[best], step, 22, "#1971c2", "center", "middle",
                math=False)
    cv.text(x0 + W * 0.6, y0 + 120, "V(s) = max over actions of\nreward + γ · V(next state)\n\narrows = the best action\n"
                                   "(the learned policy)", step, 19, INK, math=False)
    return step + 1


def build_pca(cv, v, box, step):
    rnd = random.Random(2)
    pts = v.get("points") or [(t + rnd.gauss(0, 0.6), 0.6 * t + rnd.gauss(0, 0.5)) for t in [rnd.uniform(-3, 3) for _ in range(40)]]
    n = len(pts)
    mx, my = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts) / (n - 1)
    syy = sum((p[1] - my) ** 2 for p in pts) / (n - 1)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts) / (n - 1)
    tr, det = sxx + syy, sxx * syy - sxy ** 2
    l1 = tr / 2 + math.sqrt(tr * tr / 4 - det)
    l2 = tr / 2 - math.sqrt(tr * tr / 4 - det)
    vx, vy = (l1 - syy, sxy) if abs(sxy) > 1e-12 else (1, 0)
    L = math.hypot(vx, vy)
    vx, vy = vx / L, vy / L
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    s = min(W * 0.62, H) / 9
    cx, cy = x0 + W * 0.32, (y0 + y1) / 2
    P = lambda x, y: (cx + (x - mx) * s, cy - (y - my) * s)
    for x, y in pts:
        X, Y = P(x, y)
        cv.ellipse(X - 6, Y - 6, 12, 12, step, "#a5d8ff", "#1971c2", 1)
    step += 1
    for sign_v, lam, col, nm in (((vx, vy), l1, "#e03131", "PC1"), ((-vy, vx), l2, "#2f9e44", "PC2")):
        r = 2 * math.sqrt(lam)
        a, b = P(mx - sign_v[0] * r, my - sign_v[1] * r), P(mx + sign_v[0] * r, my + sign_v[1] * r)
        cv.line([a, b], step, col, 4, arrow_end=True)
        cv.text(b[0] + 6, b[1], nm, step, 20, col, "left", "middle", math=False)
        step += 1
    for x, y in pts[::2]:                            # projections onto PC1
        t = (x - mx) * vx + (y - my) * vy
        cv.line([P(x, y), P(mx + t * vx, my + t * vy)], step, "#ffa8a8", 1, dashed=True)
    step += 1
    cv.text(x0 + W * 0.66, y0 + 40, f"covariance → eigenvectors\n\nPC1 explains {l1 / (l1 + l2):.0%} of the variance\n"
                                   f"PC2 explains {l2 / (l1 + l2):.0%}\n\nkeep PC1 only: 2-D → 1-D\nwith little information lost",
            step, 20, INK, math=False)
    return step + 1


def build_word_vectors(cv, v, box, step):
    vec = {"man": (1, 1), "woman": (1, 3), "king": (4, 1.4), "queen": (4, 3.4), "boy": (0.3, 0.9), "girl": (0.3, 2.9)}
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    s = min(W * 0.6 / 5, (H - 40) / 4.2)
    P = lambda x, y: (x0 + 40 + x * s, y1 - 30 - y * s)
    for w, (x, y) in vec.items():
        X, Y = P(x, y)
        cv.ellipse(X - 7, Y - 7, 14, 14, step, "#1971c2", "#1971c2", 1)
        cv.text(X + 10, Y, w, step, 20, INK, "left", "middle", math=False)
    step += 1
    for a, b, col in (("man", "woman", "#e03131"), ("king", "queen", "#e03131"), ("man", "king", "#2f9e44"),
                      ("woman", "queen", "#2f9e44")):
        cv.line([P(*vec[a]), P(*vec[b])], step, col, 2.5, arrow_end=True)
        step += 1
    res = tuple(vec["king"][i] - vec["man"][i] + vec["woman"][i] for i in range(2))
    near = min((w for w in vec if w not in ("king", "man", "woman")), key=lambda w: math.dist(vec[w], res))
    cv.text(x0 + W * 0.66, y0 + 40, f"king − man + woman\n= ({res[0]:g}, {res[1]:g})\nnearest word: {near}\n\n"
                                   "red arrows = 'gender' direction\ngreen arrows = 'royalty' direction\n\n"
                                   "(real embeddings have\nhundreds of dimensions)", step, 20, INK, math=False)
    return step + 1


def build_transformer_block(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    layers = [("input tokens", "#e9ecef"), ("token embedding + positional encoding", "#d0ebff"),
              ("multi-head self-attention", "#ffc9c9"), ("add & layer norm", "#fff3bf"),
              ("feed-forward network", "#d3f9d8"), ("add & layer norm", "#fff3bf"), ("× N layers → output / next token", "#e5dbff")]
    bw = W * 0.42
    bx = x0 + W * 0.08
    lh = (H - 30) / len(layers)
    for i, (name, col) in enumerate(reversed(layers)):
        y = y0 + i * lh
        _box(cv, bx, y, bw, lh - 14, step, name, col, 18)
    order = list(range(len(layers)))
    step += 1
    for i in range(len(layers) - 1):
        ya = y0 + (len(layers) - 1 - i) * lh
        cv.arrow(bx + bw / 2, ya - 2, bx + bw / 2, ya - 12, step, color=INK)
    cv.line([(bx - 20, y0 + 4.5 * lh), (bx - 40, y0 + 4.5 * lh), (bx - 40, y0 + 3.5 * lh), (bx - 2, y0 + 3.5 * lh)],
            step, "#1971c2", 2, arrow_end=True)
    cv.text(bx - 44, y0 + 4 * lh, "residual", step, 14, "#1971c2", "right", "middle", math=False)
    step += 1
    cv.text(x0 + W * 0.56, y0 + 30, "attention: every token looks at\nevery other token, weighted by\nsoftmax(QKᵀ/√d)·V\n\n"
                                   "feed-forward: the same small MLP\napplied to each token\n\nresiduals + layer norm keep\n"
                                   "deep stacks trainable", step, 19, INK, math=False)
    return step + 1


# ======================================================================== data

def build_sql_joins(cv, v, box, step):
    left = v.get("left") or [[1, "Ana"], [2, "Ben"], [3, "Chen"]]
    right = v.get("right") or [[1, "CS 224"], [3, "CS 226"], [4, "CS 242"]]
    kind = str(v.get("join") or "all").lower()
    L = {r[0]: r for r in left}
    Rr = {r[0]: r for r in right}
    joins = {"INNER": [k for k in L if k in Rr], "LEFT": list(L), "RIGHT": list(Rr),
             "FULL": sorted(set(L) | set(Rr))}
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    _table(cv, x0, y0 + 30, "students", ["id", "name"], left, step, W * 0.18, size=15)
    _table(cv, x0, y0 + 200, "enrollments", ["id", "course"], right, step, W * 0.18, size=15)
    step += 1
    names = list(joins) if kind == "all" else [kind.upper()]
    for i, j in enumerate(names):
        rows = [[k, L[k][1] if k in L else "NULL", Rr[k][1] if k in Rr else "NULL"] for k in joins[j]]
        cx = x0 + W * 0.22 + (i % 2) * W * 0.39
        cy = y0 + (i // 2) * (H / 2) + 10
        # mini venn
        for d, col in ((0, "#a5d8ff"), (1, "#ffc9c9")):
            e = cv.ellipse(cx + d * 34, cy, 56, 40, step, col if (j == "FULL" or (j == "LEFT" and d == 0) or
                                                               (j == "RIGHT" and d == 1)) else "#ffffff", INK, 1.5)
            e["opacity"] = 70
        if j == "INNER":
            cv.ellipse(cx + 34, cy + 6, 22, 28, step, "#b2f2bb", "transparent", 1)
        cv.text(cx + 100, cy + 20, f"{j} JOIN", step, 18, "#e8590c", "left", "middle", math=False)
        _table(cv, cx, cy + 74, "", ["id", "name", "course"], rows, step, W * 0.33, size=14)
        step += 1
    return step


def build_normalization(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    flat = [["1", "Ana", "CS 224", "Kim", "A"], ["1", "Ana", "CS 226", "Wong", "B"], ["2", "Ben", "CS 224", "Kim", "A-"]]
    _table(cv, x0, y0 + 30, "before: one wide table (repeats!)", ["sid", "name", "course", "teacher", "grade"], flat,
           step, W * 0.55, size=14)
    cv.text(x0 + W * 0.58, y0 + 40, "problems:\n• Ana's name stored twice\n• change Kim's course → many rows\n"
                                   "• delete Ben's only row → lose Ben", step, 17, "#e03131", math=False)
    step += 1
    tabs = [("students", ["sid PK", "name"], [["1", "Ana"], ["2", "Ben"]]),
            ("courses", ["course PK", "teacher"], [["CS 224", "Kim"], ["CS 226", "Wong"]]),
            ("enrollments", ["sid FK", "course FK", "grade"], [["1", "CS 224", "A"], ["1", "CS 226", "B"], ["2", "CS 224", "A-"]])]
    for i, (n, cols, rows) in enumerate(tabs):
        _table(cv, x0 + i * W * 0.33, y0 + 230, n, cols, rows, step, W * 0.3, size=14)
        step += 1
    cv.text(x0, y1 - 26, "after (3NF): every fact stored once; non-key columns depend on the key, the whole key, "
                         "and nothing but the key", step, 17, "#2f9e44", math=False)
    return step + 1


def build_star_schema(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cx, cy = x0 + W * 0.38, (y0 + y1) / 2
    fact = ["sale_id", "date_key FK", "product_key FK", "store_key FK", "customer_key FK", "quantity", "revenue"]
    dims = [("dim_date", ["date_key PK", "day, month, year"]), ("dim_product", ["product_key PK", "name, category"]),
            ("dim_store", ["store_key PK", "city, region"]), ("dim_customer", ["customer_key PK", "name, segment"])]
    fw, fh = 200, 40 + 26 * len(fact)
    cv.rect(cx - fw / 2, cy - fh / 2, fw, fh, step, "#fff3bf", INK, 3, rounded=False)
    cv.text(cx, cy - fh / 2 + 18, "fact_sales", step, 20, "#e8590c", "center", "middle", math=False)
    for i, f in enumerate(fact):
        cv.text(cx - fw / 2 + 12, cy - fh / 2 + 40 + i * 26, f, step, 15, "#1971c2" if "FK" in f else INK, math=False)
    step += 1
    spots = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    for (sx, sy), (n, cols) in zip(spots, dims):
        dx, dy = cx + sx * W * 0.27, cy + sy * H * 0.3
        cv.rect(dx - 95, dy - 40, 190, 80, step, "#d0ebff", INK, 2, rounded=False)
        cv.text(dx, dy - 24, n, step, 18, INK, "center", "middle", math=False)
        cv.text(dx, dy + 10, "\n".join(cols), step, 14, INK, "center", "middle", math=False)
        cv.line([(cx + sx * fw / 2, cy + sy * 20), (dx - sx * 95, dy)], step, MUTED, 2)
        step += 1
    cv.text(x0 + W * 0.74, y0 + 30, "data warehouse design:\none big fact table of events\n(numbers to add up), joined\n"
                                   "to small dimension tables\n(who, what, where, when)", step, 18, INK, math=False)
    return step + 1


def build_moving_average(cv, v, box, step):
    rnd = random.Random(8)
    data = v.get("data") or [20 + 0.6 * i + 6 * math.sin(i / 2.2) + rnd.gauss(0, 3) for i in range(40)]
    k = int(v.get("window") or 7)
    ma = [sum(data[i - k + 1:i + 1]) / k if i >= k - 1 else None for i in range(len(data))]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    lo, hi = min(data) - 2, max(data) + 2
    px = lambda i: x0 + 50 + i * W * 0.62 / (len(data) - 1)
    py = lambda y: y1 - 30 - (y - lo) / (hi - lo) * (H - 60)
    _axes(cv, x0 + 50, y0, x0 + 60 + W * 0.62, y1 - 30, step, "time", "value")
    cv.line([(px(i), py(y)) for i, y in enumerate(data)], step, "#adb5bd", 2)
    step += 1
    pts = [(px(i), py(m)) for i, m in enumerate(ma) if m is not None]
    for j in range(0, len(pts) - 1, 6):
        cv.line(pts[j:j + 7], step, "#e03131", 3.5)
        step += 1
    cv.text(x0 + W * 0.68, y0 + 40, f"grey: raw data (noisy)\nred: {k}-point moving average\n\n"
                                   f"each red point = mean of the\nlast {k} values: smooths noise,\nshows the trend, lags a bit",
            step, 19, INK, math=False)
    return step + 1


def build_correlation(cv, v, box, step):
    rnd = random.Random(4)
    sets = v.get("sets") or [("hours studied vs score", [(h, 50 + 6 * h + rnd.gauss(0, 6)) for h in [rnd.uniform(0, 8) for _ in range(30)]]),
                             ("shoe size vs score", [(s, 70 + rnd.gauss(0, 10)) for s in [rnd.uniform(5, 12) for _ in range(30)]])]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    pw = W / len(sets) - 30
    for k, (name, pts) in enumerate(sets):
        n = len(pts)
        mx, my = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
        r = sum((p[0] - mx) * (p[1] - my) for p in pts) / math.sqrt(sum((p[0] - mx) ** 2 for p in pts) *
                                                                    sum((p[1] - my) ** 2 for p in pts))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        bx = x0 + k * (pw + 30)
        sx = lambda x: bx + 30 + (x - min(xs)) / (max(xs) - min(xs)) * (pw - 50)
        sy = lambda y: y1 - 60 - (y - min(ys)) / (max(ys) - min(ys)) * (H - 140)
        _axes(cv, bx + 20, y0 + 40, bx + pw, y1 - 50, step)
        for x, y in pts:
            cv.ellipse(sx(x) - 5, sy(y) - 5, 10, 10, step, PAL[k], STRONG[k], 1)
        cv.text(bx + pw / 2, y0, name, step, 18, INK, "center", "top", math=False)
        cv.text(bx + pw / 2, y1 - 40, f"Pearson r = {r:.2f}  " + ("strong positive" if r > 0.7 else "weak / none" if abs(r) < 0.3
                                                                 else "moderate"), step, 18, STRONG[k], "center", "top",
                math=False)
        step += 1
    return step


# ======================================================================== maths

def build_sieve(cv, v, box, step):
    n = int(v.get("n") or 60)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cols = 10
    c = min(W * 0.62 / cols, (H - 20) / math.ceil(n / cols))
    cell = lambda k: (x0 + ((k - 1) % cols) * c, y0 + ((k - 1) // cols) * c)
    for k in range(1, n + 1):
        x, y = cell(k)
        cv.rect(x, y, c - 3, c - 3, step, "#ffffff", INK, 1, rounded=False)
        cv.text(x + c / 2, y + c / 2, str(k), step, 16, INK, "center", "middle", math=False)
    step += 1
    alive = set(range(2, n + 1))
    for p in range(2, int(n ** 0.5) + 1):
        if p not in alive:
            continue
        for m in range(p * p, n + 1, p):
            if m in alive:
                alive.discard(m)
                x, y = cell(m)
                cv.rect(x, y, c - 3, c - 3, step, PAL[p % len(PAL)], "transparent", 1, rounded=False)
                cv.text(x + c / 2, y + c / 2, str(m), step, 16, MUTED, "center", "middle", math=False)
        x, y = cell(p)
        cv.ellipse(x + 2, y + 2, c - 7, c - 7, step, "transparent", "#2f9e44", 3)
        cv.text(x0 + W * 0.66, y0 + 30 + (p - 2) * 30, f"{p} is prime → cross out {p * p}, {p * p + p}, …", step, 18,
                "#2f9e44", math=False)
        step += 1
    primes = sorted(alive)
    for p in primes:
        x, y = cell(p)
        cv.ellipse(x + 2, y + 2, c - 7, c - 7, step, "transparent", "#2f9e44", 3)
    cv.text(x0 + W * 0.66, y1 - 120, f"{len(primes)} primes up to {n}:\n" + ", ".join(map(str, primes)), step, 17, INK,
            math=False)
    return step + 1


def build_euclid_gcd(cv, v, box, step):
    a, b = int(v.get("a") or 1071), int(v.get("b") or 462)
    rows = []
    x, y = a, b
    while y:
        rows.append((x, y, x // y, x % y))
        x, y = y, x % y
    x0, y0, x1, y1 = box
    W = x1 - x0
    _title(cv, x0, y0, f"gcd({a}, {b}) by Euclid's algorithm", step, 22)
    step += 1
    for i, (p, q, k, r) in enumerate(rows):
        cv.text(x0, y0 + 50 + i * 40, f"{p} = {k} × {q} + {r}", step, 24, INK, math=False)
        step += 1
    cv.text(x0, y0 + 70 + len(rows) * 40, f"the last non-zero remainder is the gcd: {rows[-1][1]}", step, 24, "#2f9e44",
            math=False)
    # rectangle picture of the first step
    p, q = rows[0][0], rows[0][1]
    s = min(W * 0.35 / p, 220 / q)
    rx, ry = x0 + W * 0.55, y0 + 60
    cv.rect(rx, ry, p * s, q * s, step, "#ffffff", INK, 2, rounded=False)
    for j in range(rows[0][2]):
        cv.rect(rx + j * q * s, ry, q * s, q * s, step, PAL[j], INK, 1.5, rounded=False)
    cv.text(rx, ry + q * s + 8, f"{p} × {q} rectangle: cut {rows[0][2]} squares of {q},\nthe {rows[0][3]}-wide strip is "
                               "left; repeat on it", step, 16, MUTED, math=False)
    return step + 1


def build_pascal(cv, v, box, step):
    n = int(v.get("rows") or 8)
    tri = [[1]]
    for i in range(1, n):
        prev = tri[-1]
        tri.append([1] + [prev[j] + prev[j + 1] for j in range(len(prev) - 1)] + [1])
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    c = min(W * 0.66 / n, (H - 20) / n)
    for i, row in enumerate(tri):
        for j, val in enumerate(row):
            cx = x0 + W * 0.33 + (j - i / 2) * c
            col = "#ffec99" if val % 2 else "#ffffff"
            cv.ellipse(cx - c * 0.45, y0 + i * c, c * 0.9, c * 0.9, step, col, INK, 1.5)
            cv.text(cx, y0 + i * c + c * 0.45, str(val), step, max(12, int(c * 0.32)), INK, "center", "middle", math=False)
        step += 1
    cv.text(x0 + W * 0.7, y0 + 30, "each number = the two above it\n\nrow n gives the binomial\ncoefficients C(n, k):\n"
                                   f"(a + b)^{n - 1} = " + " + ".join(str(x) for x in tri[-1][:4]) + " …\n\nodd numbers (yellow)\n"
                                   "make Sierpinski's triangle", step, 19, INK, math=False)
    return step + 1


def build_binomial(cv, v, box, step):
    n, p = int(v.get("n") or 10), float(v.get("p") or 0.5)
    probs = [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    bw = W * 0.62 / (n + 1)
    mx = max(probs)
    for k, pr in enumerate(probs):
        h = (H - 70) * pr / mx
        cv.rect(x0 + 30 + k * bw, y1 - 30 - h, bw - 6, h, step, "#a5d8ff", "#1971c2", 1.5, rounded=False)
        cv.text(x0 + 30 + k * bw + bw / 2, y1 - 26, str(k), step, 15, INK, "center", "top", math=False)
        cv.text(x0 + 30 + k * bw + bw / 2, y1 - 34 - h, f"{pr:.3f}", step, 12, MUTED, "center", "bottom", math=False)
        step += 1
    cv.text(x0 + W * 0.68, y0 + 30, f"Binomial(n = {n}, p = {p:g})\nP(k) = C(n,k) pᵏ (1−p)ⁿ⁻ᵏ\n\nmean = np = {n * p:g}\n"
                                   f"sd = √(np(1−p)) = {math.sqrt(n * p * (1 - p)):.2f}\n\ne.g. {n} coin flips: "
                                   f"P({n // 2} heads) = {probs[n // 2]:.3f}", step, 19, INK, math=False)
    return step + 1


def build_bayes_test(cv, v, box, step):
    prev, sens, spec = float(v.get("prevalence") or 0.01), float(v.get("sensitivity") or 0.9), float(v.get("specificity") or 0.91)
    N = 1000
    sick = round(N * prev)
    tp = round(sick * sens)
    fp = round((N - sick) * (1 - spec))
    ppv = tp / (tp + fp)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cols = 40
    c = min(W * 0.6 / cols, (H - 40) / (N / cols))
    for i in range(N):
        r, k = divmod(i, cols)
        col = "#e03131" if i < tp else "#ffa8a8" if i < sick else "#fab005" if i < sick + fp else "#dee2e6"
        cv.rect(x0 + k * c, y0 + r * c, c - 2, c - 2, step, col, "transparent", 1, rounded=False)
        if i in (sick - 1, sick + fp - 1, N - 1):
            step += 1
    cv.text(x0 + W * 0.64, y0 + 20, f"{N} people, {prev:.0%} have the disease = {sick}\n\n"
                                   f"test catches {sens:.0%} of them: {tp} (red)\nmissed: {sick - tp} (pink)\n\n"
                                   f"healthy but test positive ({1 - spec:.0%}):\n{fp} (yellow)\n\n"
                                   f"P(sick | positive) = {tp} / ({tp} + {fp})\n= {ppv:.0%}  (not {sens:.0%}!)", step, 19, INK,
            math=False)
    return step + 1


def build_monte_carlo_pi(cv, v, box, step):
    rnd = random.Random(1)
    N = int(v.get("points") or 600)
    pts = [(rnd.random(), rnd.random()) for _ in range(N)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    S = min(W * 0.55, H - 20)
    cv.rect(x0, y0, S, S, step, "#ffffff", INK, 2, rounded=False)
    arc = [(x0 + S * math.cos(t), y0 + S - S * math.sin(t)) for t in [i * math.pi / 2 / 40 for i in range(41)]]
    cv.line(arc, step, INK, 2)
    step += 1
    inside = 0
    for b, cut in enumerate((50, 150, 300, N)):
        lo = (0, 50, 150, 300)[b]
        for x, y in pts[lo:cut]:
            ins = x * x + y * y <= 1
            inside += ins
            cv.ellipse(x0 + x * S - 3, y0 + S - y * S - 3, 6, 6, step, "#e03131" if ins else "#1971c2", "transparent", 1)
        tr = _Track(cv)
        cv.text(x0 + S + 30, y0 + 40, f"{cut} points, {inside} inside\nπ ≈ 4 × {inside}/{cut} = {4 * inside / cut:.3f}", step,
                24, "#e8590c", math=False)
        tr.tag(until=step + 1 if cut != N else None, fade=b > 0)
        step += 1
    cv.text(x0 + S + 30, y0 + 160, "quarter circle area / square area = π/4\n\nmore random points → closer to\n"
                                   "π = 3.14159…", step, 19, INK, math=False)
    return step + 1


# ======================================================================== physics

def build_collision(cv, v, box, step):
    m1, u1, m2, u2 = (float(v.get(k) or d) for k, d in (("m1", 2), ("u1", 3), ("m2", 1), ("u2", -1)))
    elastic = str(v.get("kind") or "elastic").startswith("el")
    if elastic:
        v1 = ((m1 - m2) * u1 + 2 * m2 * u2) / (m1 + m2)
        v2 = ((m2 - m1) * u2 + 2 * m1 * u1) / (m1 + m2)
    else:
        v1 = v2 = (m1 * u1 + m2 * u2) / (m1 + m2)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    gy = y0 + H * 0.4
    cv.line([(x0, gy + 40), (x0 + W * 0.95, gy + 40)], step, INK, 2)
    step += 1
    frames = 10
    for k in range(frames + 1):
        t = k / frames
        if t < 0.5:
            a, b = x0 + W * 0.15 + u1 * t * 80, x0 + W * 0.6 + u2 * t * 80
        else:
            ca, cb = x0 + W * 0.15 + u1 * 40, x0 + W * 0.6 + u2 * 40
            a, b = ca + v1 * (t - 0.5) * 80, cb + v2 * (t - 0.5) * 80
        tr = _Track(cv)
        cv.rect(a - 30 * m1 ** 0.5, gy + 40 - 50 * m1 ** 0.4, 60 * m1 ** 0.5, 50 * m1 ** 0.4, step, "#a5d8ff", INK, 2)
        cv.text(a, gy + 15, f"{m1:g} kg", step, 15, INK, "center", "middle", math=False)
        cv.rect(b - 30 * m2 ** 0.5, gy + 40 - 50 * m2 ** 0.4, 60 * m2 ** 0.5, 50 * m2 ** 0.4, step, "#ffc9c9", INK, 2)
        cv.text(b, gy + 15, f"{m2:g} kg", step, 15, INK, "center", "middle", math=False)
        tr.tag(until=step + 1 if k < frames else None, fade=k > 0)
        step += 1
    p0, p1 = m1 * u1 + m2 * u2, m1 * v1 + m2 * v2
    ke0, ke1 = 0.5 * m1 * u1 ** 2 + 0.5 * m2 * u2 ** 2, 0.5 * m1 * v1 ** 2 + 0.5 * m2 * v2 ** 2
    cv.text(x0, gy + 80, f"{'elastic' if elastic else 'perfectly inelastic (they stick)'} collision\n"
                         f"before: v = {u1:g} m/s and {u2:g} m/s    after: v = {v1:.2f} and {v2:.2f} m/s\n"
                         f"momentum: {p0:.2f} → {p1:.2f} kg·m/s (conserved)\n"
                         f"kinetic energy: {ke0:.2f} → {ke1:.2f} J" + (" (conserved)" if elastic else " (some lost to heat)"),
            step, 20, INK, math=False)
    return step + 1


def build_incline(cv, v, box, step):
    m, ang, mu = float(v.get("mass") or 5), float(v.get("angle") or 30), float(v.get("mu") or 0.2)
    g = 9.8
    th = math.radians(ang)
    W_, par, perp = m * g, m * g * math.sin(th), m * g * math.cos(th)
    fric = mu * perp
    a = max(0.0, (par - fric) / m)
    x0, y0, x1, y1 = box
    Wd, H = x1 - x0, y1 - y0
    bx, by = x0 + 40, y1 - 40
    L = min(Wd * 0.6, (H - 80) / math.sin(th) * 0.95)
    top = (bx + L * math.cos(th), by - L * math.sin(th))
    cv.polygon([(bx, by), (bx + L * math.cos(th), by), top], step, "#e9ecef", INK, 2)
    cv.text(bx + 70, by - 14, f"{ang:g}°", step, 18, INK, math=False)
    step += 1
    cx, cy = bx + L * 0.55 * math.cos(th), by - L * 0.55 * math.sin(th)
    ux, uy = -math.cos(th), math.sin(th)                    # down the slope
    nx, ny = -math.sin(th), -math.cos(th)                   # normal (out of surface)
    s = 4.0
    cv.rect(cx - 30 + nx * 30, cy - 30 + ny * 30, 60, 60, step, "#ffd8a8", INK, 2)
    cbx, cby = cx + nx * 30, cy + ny * 30
    step += 1
    for (dx, dy, mag, col, lab) in ((0, 1, W_, "#1971c2", f"mg = {W_:.1f} N"),
                                    (ux, uy, par, "#e03131", f"mg sin θ = {par:.1f} N"),
                                    (-nx, -ny, perp, "#7048e8", f"mg cos θ = {perp:.1f} N"),
                                    (nx, ny, perp, "#2f9e44", f"N = {perp:.1f} N"),
                                    (-ux, -uy, fric, "#e8590c", f"friction μN = {fric:.1f} N")):
        cv.arrow(cbx, cby, cbx + dx * mag * s, cby + dy * mag * s, step, color=col)
        cv.text(cbx + dx * (mag * s + 14), cby + dy * (mag * s + 14), lab, step, 15, col, "center", "middle", math=False)
        step += 1
    cv.text(x0 + Wd * 0.68, y0 + 30, f"m = {m:g} kg, μ = {mu:g}\n\nnet force down the slope:\n{par:.1f} − {fric:.1f} = "
                                    f"{par - fric:.1f} N\nacceleration a = {a:.2f} m/s²", step, 20, INK, math=False)
    return step + 1


def build_gas_law(cv, v, box, step):
    n, T0 = float(v.get("moles") or 1), float(v.get("temperature") or 300)
    Rg = 8.314
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    vols = [0.025, 0.0185, 0.0125]
    rnd = random.Random(3)
    for k, V in enumerate(vols):
        P = n * Rg * T0 / V
        cx = x0 + 40 + k * W * 0.22
        ch = (H - 80) * V / vols[0]
        tr = _Track(cv)
        cv.line([(cx, y0 + 20), (cx, y1 - 40), (cx + W * 0.18, y1 - 40), (cx + W * 0.18, y0 + 20)], step, INK, 3)
        cv.rect(cx + 2, y1 - 40 - ch - 14, W * 0.18 - 4, 14, step, "#868e96", INK, 1.5, rounded=False)
        cv.line([(cx + W * 0.09, y1 - 40 - ch - 14), (cx + W * 0.09, y0)], step, INK, 3)
        for _ in range(int(14 * vols[0] / V)):
            cv.ellipse(cx + rnd.uniform(8, W * 0.18 - 16), y1 - 46 - rnd.uniform(0, ch - 20), 9, 9, step, "#4dabf7",
                       "#1971c2", 1)
        cv.text(cx + W * 0.09, y1 - 30, f"V = {V * 1000:.1f} L\nP = {P / 1000:.0f} kPa", step, 16, INK, "center", "top",
                math=False)
        step += 1
    cv.text(x0 + W * 0.7, y0 + 30, f"PV = nRT\nn = {n:g} mol, T = {T0:g} K\nR = 8.314 J/(mol·K)\n\nsame temperature:\n"
                                   "squeeze the volume → the\nmolecules hit the walls more\noften → pressure rises\n"
                                   "(Boyle's law: P × V stays\nconstant)", step, 19, INK, math=False)
    return step + 1


def build_half_life(cv, v, box, step):
    N0, t_half = float(v.get("start") or 1000), float(v.get("half_life") or 5730)
    name = str(v.get("isotope") or "carbon-14")
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    px = lambda t: x0 + 50 + t / (5 * t_half) * W * 0.62
    py = lambda nn: y1 - 30 - nn / N0 * (H - 60)
    _axes(cv, x0 + 50, y0, x0 + 60 + W * 0.62, y1 - 30, step, "years", "atoms left")
    cv.line([(px(t), py(N0 * 0.5 ** (t / t_half))) for t in [i * 5 * t_half / 60 for i in range(61)]], step, "#1971c2", 3)
    step += 1
    for k in range(1, 5):
        t = k * t_half
        nn = N0 * 0.5 ** k
        cv.line([(px(t), py(0)), (px(t), py(nn)), (px(0), py(nn))], step, MUTED, 1, dashed=True)
        cv.ellipse(px(t) - 6, py(nn) - 6, 12, 12, step, "#e03131", "#e03131", 1)
        cv.text(px(t) + 8, py(nn) - 6, f"{nn:g}", step, 16, "#e03131", "left", "bottom", math=False)
        cv.text(px(t), py(0) + 6, f"{t:,.0f}", step, 13, MUTED, "center", "top", math=False)
        step += 1
    cv.text(x0 + W * 0.68, y0 + 30, f"{name}: half-life {t_half:,.0f} years\n\nN(t) = N₀ (½)^(t / t½)\n\n"
                                   "after each half-life, half of\nwhat is left decays", step, 20, INK, math=False)
    return step + 1


def build_doppler(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cy = (y0 + y1) / 2
    speed = 0.45                                   # source speed / wave speed
    n = 6
    for k in range(n):
        tr = _Track(cv)
        sx = x0 + W * 0.36 + k * 25
        for j in range(k + 1):
            emit_x = x0 + W * 0.36 + j * 25
            r = (k - j) * 50 + 6
            cv.ellipse(emit_x - r, cy - r, 2 * r, 2 * r, step, "transparent", "#1971c2", 1.5)
        draw_device(cv, "phone", sx, cy, 50, step)
        tr.tag(until=step + 1 if k < n - 1 else None, fade=k > 0)
        step += 1
    draw_device(cv, "user", x0 + W * 0.9, cy, 70, step)
    draw_device(cv, "user", x0 + 30, cy, 70, step)
    cv.text(x0 + W * 0.9, cy + 50, "ahead:\nwaves bunched\n→ higher pitch", step, 16, "#e03131", "center", "top", math=False)
    cv.text(x0 + 30, cy + 50, "behind:\nwaves spread\n→ lower pitch", step, 16, "#1971c2", "center", "top", math=False)
    cv.text(x0, y1 - 24, "a moving source (siren) squeezes the waves in front and stretches them behind", step, 18,
            "#e8590c", math=False)
    return step + 1


# ======================================================================== chemistry / biology

ORDER = ["1s", "2s", "2p", "3s", "3p", "4s", "3d", "4p", "5s", "4d", "5p"]
CAP = {"s": 2, "p": 6, "d": 10}


def build_electron_config(cv, v, box, step):
    from drawlib2 import ELEMENTS, NAMES
    el = str(v.get("element") or "Fe")
    extra = {"Fe": (26, "Iron"), "Cu": (29, "Copper"), "Zn": (30, "Zinc"), "Br": (35, "Bromine"), "Kr": (36, "Krypton")}
    if el in extra:
        z, name = extra[el]
    elif el in ELEMENTS:
        z, name = ELEMENTS.index(el) + 1, NAMES[ELEMENTS.index(el)]
    else:
        z, name = 26, "Iron"
    left, fill = z, []
    for o in ORDER:
        if left <= 0:
            break
        k = min(CAP[o[-1]], left)
        fill.append((o, k))
        left -= k
    x0, y0, x1, y1 = box
    W = x1 - x0
    _title(cv, x0, y0, f"{name} (Z = {z}): fill orbitals in energy order", step, 21)
    step += 1
    for i, (o, k) in enumerate(fill):
        y = y0 + 50 + i * 44
        cv.text(x0, y + 16, o, step, 22, "#1971c2", "left", "middle", math=False)
        boxes = CAP[o[-1]] // 2
        for b in range(boxes):                         # Hund's rule: one up in each, then pair
            ups = 1 if b < k else 0
            downs = 1 if b + boxes < k else 0
            _box(cv, x0 + 60 + b * 52, y, 48, 34, step, ("↑" if ups else "") + ("↓" if downs else ""), "#ffffff", 20)
        step += 1
    conf = " ".join(f"{o}{''.join('⁰¹²³⁴⁵⁶⁷⁸⁹'[int(d)] for d in str(k))}" for o, k in fill)
    cv.text(x0 + W * 0.55, y0 + 60, f"{conf}\n\nAufbau: lowest energy first\n(note 4s fills before 3d)\nPauli: 2 per box, "
                                   "opposite spins\nHund: one in each box first", step, 20, INK, math=False)
    return step + 1


def build_food_web(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    levels = [["grass", "seeds"], ["rabbit", "mouse", "grasshopper"], ["snake", "frog", "bird"], ["hawk"]]
    names = ["producers", "primary consumers", "secondary consumers", "top predator"]
    pos = {}
    for li, lvl in enumerate(levels):
        y = y1 - 40 - li * (H - 80) / 3
        cv.text(x0, y, names[li], step, 15, MUTED, "left", "middle", math=False)
        for j, n in enumerate(lvl):
            x = x0 + 200 + (j + 0.5) * (W * 0.62) / len(lvl)
            pos[n] = (x, y)
            _box(cv, x - 60, y - 22, 120, 44, step, n, ["#d3f9d8", "#fff3bf", "#ffd8a8", "#ffc9c9"][li], 17)
        step += 1
    eats = [("grass", "rabbit"), ("grass", "grasshopper"), ("seeds", "mouse"), ("seeds", "bird"), ("rabbit", "snake"),
            ("mouse", "snake"), ("grasshopper", "frog"), ("grasshopper", "bird"), ("frog", "snake"), ("snake", "hawk"),
            ("bird", "hawk"), ("rabbit", "hawk"), ("mouse", "hawk")]
    for a, b in eats:
        (ax, ay), (bx, by) = pos[a], pos[b]
        cv.arrow(ax, ay - 22, bx, by + 22, step, color="#2f9e44")
    step += 1
    cv.text(x0 + 200, y0 - 4, "arrows show where energy goes (from food to eater)", step, 17, "#e8590c", math=False)
    return step + 1


def build_energy_pyramid(cv, v, box, step):
    E0 = float(v.get("energy") or 10000)
    levels = ["producers (plants)", "primary consumers", "secondary consumers", "tertiary consumers"]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    lh = (H - 20) / len(levels)
    cx = x0 + W * 0.33
    for i, name in enumerate(levels):
        e = E0 * 0.1 ** i
        bw = W * 0.6 * (len(levels) - i) / len(levels)
        y = y1 - (i + 1) * lh
        cv.rect(cx - bw / 2, y, bw, lh - 6, step, ["#b2f2bb", "#fff3bf", "#ffd8a8", "#ffc9c9"][i], INK, 2, rounded=False)
        cv.text(cx, y + lh / 2 - 3, f"{name}\n{e:,.0f} kcal", step, 17 if i < 3 else 15, INK, "center", "middle", math=False)
        step += 1
    cv.text(x0 + W * 0.68, y0 + 40, "only about 10% of the energy\npasses to the next level;\nthe rest is used for living\n"
                                   "and lost as heat\n\nso food chains are short and\ntop predators are few", step, 19,
            INK, math=False)
    return step + 1


def build_natural_selection(cv, v, box, step):
    p, s = float(v.get("start") or 0.05), float(v.get("advantage") or 0.15)
    gens = int(v.get("generations") or 40)
    freq = [p]
    for _ in range(gens):
        q = freq[-1]
        freq.append(q * (1 + s) / (q * (1 + s) + (1 - q)))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    px = lambda g: x0 + 50 + g / gens * W * 0.62
    py = lambda f: y1 - 30 - f * (H - 60)
    _axes(cv, x0 + 50, y0, x0 + 60 + W * 0.62, y1 - 30, step, "generation", "share with the trait")
    step += 1
    for j in range(0, gens, 8):
        cv.line([(px(g), py(freq[g])) for g in range(j, min(gens, j + 8) + 1)], step, "#2f9e44", 3.5)
        step += 1
    rnd = random.Random(2)
    for gi, g in enumerate((0, gens // 2, gens)):
        bx = x0 + W * 0.68 + gi * 90
        for i in range(20):
            col = "#495057" if rnd.random() < freq[g] else "#e9ecef"
            cv.ellipse(bx + (i % 4) * 18, y0 + 80 + (i // 4) * 18, 14, 14, step, col, INK, 1)
        cv.text(bx + 30, y0 + 180, f"gen {g}\n{freq[g]:.0%}", step, 15, INK, "center", "top", math=False)
    step += 1
    cv.text(x0 + W * 0.68, y0 + 250, f"dark moths survive {s:.0%} better\non sooty trees: the trait spreads\nfrom {p:.0%} to "
                                    f"{freq[-1]:.0%} in {gens} generations", step, 18, INK, math=False)
    return step + 1


# ======================================================================== earth / GIS

def build_seasons(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cx, cy = x0 + W * 0.36, (y0 + y1) / 2
    rx, ry = W * 0.3, H * 0.33
    cv.ellipse(cx - rx, cy - ry, 2 * rx, 2 * ry, step, "transparent", MUTED, 1.5)
    cv.ellipse(cx - 34, cy - 34, 68, 68, step, "#ffd43b", "#f08c00", 2)
    step += 1
    spots = [(0, "June: north tilted toward the Sun\n→ northern summer"), (90, "September equinox"),
             (180, "December: north tilted away\n→ northern winter"), (270, "March equinox")]
    for ang, lab in spots:
        a = math.radians(ang)
        ex, ey = cx - rx * math.cos(a), cy + ry * math.sin(a)
        cv.ellipse(ex - 26, ey - 26, 52, 52, step, "#74c0fc", "#1971c2", 2)
        t = math.radians(23.5)
        cv.line([(ex - 40 * math.sin(t), ey - 40 * math.cos(t)), (ex + 40 * math.sin(t), ey + 40 * math.cos(t))], step,
                "#e03131", 2)
        cv.text(ex, ey + (36 if ang != 90 else -36), lab, step, 14, INK, "center", "top" if ang != 90 else "bottom",
                math=False)
        step += 1
    cv.text(x0 + W * 0.7, y0 + 40, "the axis stays tilted 23.5°\nand points the same way all\nyear; seasons come from the\n"
                                   "tilt, not from distance to\nthe Sun", step, 19, INK, math=False)
    return step + 1


def build_greenhouse(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    gy = y1 - 50
    cv.rect(x0, gy, W * 0.7, 50, step, "#b2f2bb", "#2b8a3e", 2, rounded=False)
    cv.text(x0 + W * 0.35, gy + 25, "Earth's surface", step, 18, INK, "center", "middle", math=False)
    ay = y0 + H * 0.3
    band = cv.rect(x0, ay, W * 0.7, 60, step, "#e9ecef", "transparent", 1, rounded=False)
    band["opacity"] = 70
    cv.text(x0 + 10, ay + 30, "atmosphere: CO₂, CH₄, H₂O", step, 16, MUTED, "left", "middle", math=False)
    cv.ellipse(x1 - 120, y0, 80, 80, step, "#ffd43b", "#f08c00", 2)
    step += 1
    cv.arrow(x1 - 120, y0 + 60, x0 + W * 0.4, gy - 4, step, "sunlight passes", color="#f08c00")
    step += 1
    for k in range(3):
        sx = x0 + W * (0.15 + k * 0.15)
        cv.line([(sx, gy - 2), (sx + 10, ay + 70)], step, "#e03131", 2.5, arrow_end=True)
        cv.line([(sx + 10, ay + 62), (sx + 30, gy - 6)], step, "#e03131", 2.5, arrow_end=True)
    cv.line([(x0 + W * 0.6, gy - 2), (x0 + W * 0.62, y0 + 10)], step, "#ffa8a8", 2, arrow_end=True)
    cv.text(x0 + W * 0.72, y0 + 220, "heat (infrared) radiated up,\nabsorbed and sent back down", step, 16, "#e03131",
            math=False)
    step += 1
    cv.text(x0 + W * 0.72, y0 + 120, "more greenhouse gas →\nmore heat held near the\nsurface → warmer climate", step,
            19, INK, math=False)
    return step + 1


def build_rock_cycle(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    pos = {"magma": (0.35, 0.85), "igneous rock": (0.1, 0.45), "sediment": (0.35, 0.1), "sedimentary rock": (0.62, 0.45),
           "metamorphic rock": (0.35, 0.5)}
    cols = {"magma": "#ff8787", "igneous rock": "#868e96", "sediment": "#ffe066", "sedimentary rock": "#e9a76b",
            "metamorphic rock": "#b197fc"}
    P = lambda k: (x0 + W * pos[k][0] + 40, y0 + (H - 40) * pos[k][1] + 10)
    for k in pos:
        x, y = P(k)
        _box(cv, x - 80, y - 26, 160, 52, step, k, cols[k], 18)
    step += 1
    for a, b, lab in (("magma", "igneous rock", "cooling"), ("igneous rock", "sediment", "weathering, erosion"),
                      ("sediment", "sedimentary rock", "compaction"), ("sedimentary rock", "metamorphic rock", "heat, pressure"),
                      ("metamorphic rock", "magma", "melting"), ("igneous rock", "metamorphic rock", "heat, pressure"),
                      ("sedimentary rock", "sediment", "weathering")):
        (ax, ay), (bx, by) = P(a), P(b)
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy)
        cv.arrow(ax + dx / L * 85, ay + dy / L * 30, bx - dx / L * 85, by - dy / L * 30, step, lab, color="#495057")
        step += 1
    return step


def build_map_projection(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    w, hmax = W * 0.6, H - 40
    lat_max = 80
    ymerc = lambda lat: math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    scale = hmax / 2 / ymerc(lat_max)
    cy = y0 + 20 + hmax / 2
    for lon in range(-180, 181, 30):
        x = x0 + (lon + 180) / 360 * w
        cv.line([(x, cy - scale * ymerc(lat_max)), (x, cy + scale * ymerc(lat_max))], step, "#dee2e6", 1)
    for lat in range(-75, 76, 15):
        y = cy - scale * ymerc(lat)
        cv.line([(x0, y), (x0 + w, y)], step, "#74c0fc" if lat else "#e03131", 1.5)
        cv.text(x0 + w + 6, y, f"{lat}°", step, 13, MUTED, "left", "middle", math=False)
    step += 1
    for lat in (0, 30, 60, 75):                       # equal-size circles on the globe grow on the map
        k = 1 / math.cos(math.radians(lat))
        r = 8 * k
        for lon in (-120, 0, 120):
            x = x0 + (lon + 180) / 360 * w
            cv.ellipse(x - r, cy - scale * ymerc(lat) - r, 2 * r, 2 * r, step, "#ffc9c9", "#e03131", 1.5)
        cv.text(x0 + w + 60, y0 + 40 + (lat // 15) * 34, f"{lat}°: scale × {k:.2f}", step, 17, INK, math=False)
        step += 1
    cv.text(x0 + w + 60, y0 + 230, "Mercator keeps angles (good for\nnavigation) but stretches areas:\nscale = 1 / cos(latitude)."
                                  "\nGreenland looks as big as Africa\n(Africa is 14× bigger).", step, 17, INK, math=False)
    return step + 1


def build_voronoi(cv, v, box, step):
    rnd = random.Random(6)
    sites = v.get("sites") or [(rnd.uniform(0.05, 0.95), rnd.uniform(0.05, 0.95)) for _ in range(7)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    S = min(W * 0.6, H - 20)
    g = 26
    c = S / g
    for i in range(g):
        for j in range(g):
            px, py = (j + 0.5) / g, (i + 0.5) / g
            k = min(range(len(sites)), key=lambda s: (sites[s][0] - px) ** 2 + (sites[s][1] - py) ** 2)
            cv.rect(x0 + j * c, y0 + i * c, c + 0.5, c + 0.5, step, PAL[k % len(PAL)], "transparent", 1, rounded=False)
        if i % 6 == 5:
            step += 1
    step += 1
    for k, (sx, sy) in enumerate(sites):
        cv.ellipse(x0 + sx * S - 8, y0 + sy * S - 8, 16, 16, step, STRONG[k % len(STRONG)], INK, 1.5)
        draw_device(cv, "building", x0 + sx * S, y0 + sy * S - 22, 26, step)
    cv.text(x0 + S + 30, y0 + 30, "Voronoi / Thiessen polygons:\neach place belongs to its\nnearest site (fire station,\n"
                                  "store, rain gauge)\n\nused for service areas and\nfor interpolating point data", step, 19,
            INK, math=False)
    return step + 1


def build_choropleth(cv, v, box, step):
    rnd = random.Random(11)
    vals = v.get("values") or [round(rnd.uniform(5, 95)) for _ in range(20)]
    classes = int(v.get("classes") or 4)
    lo, hi = min(vals), max(vals)
    w = (hi - lo) / classes
    ramp = ["#fff3bf", "#ffd8a8", "#ff922b", "#e8590c", "#c92a2a"]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cols, rows = 5, 4
    cw, ch = W * 0.6 / cols, (H - 20) / rows
    for i, val in enumerate(vals[:cols * rows]):
        r, c = divmod(i, cols)
        k = min(classes - 1, int((val - lo) / w)) if w else 0
        jit = [(x0 + c * cw + rnd.uniform(0, 8), y0 + r * ch + rnd.uniform(0, 8)),
               (x0 + (c + 1) * cw - rnd.uniform(0, 8), y0 + r * ch + rnd.uniform(0, 8)),
               (x0 + (c + 1) * cw - rnd.uniform(0, 8), y0 + (r + 1) * ch - rnd.uniform(0, 8)),
               (x0 + c * cw + rnd.uniform(0, 8), y0 + (r + 1) * ch - rnd.uniform(0, 8))]
        cv.polygon(jit, step, ramp[k], INK, 1.5)
        cv.text(x0 + c * cw + cw / 2, y0 + r * ch + ch / 2, str(val), step, 16, INK, "center", "middle", math=False)
        if i % cols == cols - 1:
            step += 1
    for k in range(classes):
        a, b = lo + k * w, lo + (k + 1) * w
        cv.rect(x0 + W * 0.66, y0 + 40 + k * 40, 40, 30, step, ramp[k], INK, 1, rounded=False)
        cv.text(x0 + W * 0.66 + 50, y0 + 55 + k * 40, f"{a:.0f} – {b:.0f}", step, 18, INK, "left", "middle", math=False)
    cv.text(x0 + W * 0.66, y0 + 60 + classes * 40, f"equal interval: ({hi} − {lo}) / {classes}\n= {w:.1f} per class\n\n"
                                                    "map rates or densities,\nnot raw counts", step, 18, INK, math=False)
    return step + 1


# ======================================================================== economics

def build_amortization(cv, v, box, step):
    P, rate, years = float(v.get("principal") or 300000), float(v.get("rate") or 0.065), int(v.get("years") or 30)
    r, n = rate / 12, years * 12
    pay = P * r / (1 - (1 + r) ** -n)
    bal, rows, yearly = P, [], []
    for m in range(1, n + 1):
        it = bal * r
        pr = pay - it
        bal -= pr
        rows.append((m, it, pr, bal))
        if m % 12 == 0:
            yearly.append((m // 12, sum(x[1] for x in rows[-12:]), sum(x[2] for x in rows[-12:]), bal))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    bw = W * 0.6 / len(yearly)
    mx = pay * 12
    _title(cv, x0, y0, f"${P:,.0f} at {rate:.2%} for {years} years: ${pay:,.2f} a month", step, 20)
    step += 1
    for i, (yr, it, pr, b) in enumerate(yearly):
        x = x0 + 30 + i * bw
        hi_, hp = (H - 90) * it / mx, (H - 90) * pr / mx
        cv.rect(x, y1 - 30 - hi_ - hp, bw - 3, hp, step, "#b2f2bb", "transparent", 1, rounded=False)
        cv.rect(x, y1 - 30 - hi_, bw - 3, hi_, step, "#ffc9c9", "transparent", 1, rounded=False)
        if yr % 5 == 0:
            cv.text(x + bw / 2, y1 - 26, str(yr), step, 13, MUTED, "center", "top", math=False)
        if i % 3 == 2:
            step += 1
    tot = pay * n
    cv.text(x0 + W * 0.66, y0 + 50, f"red = interest, green = principal\n(per year)\n\nyear 1 interest: ${yearly[0][1]:,.0f}\n"
                                   f"total paid: ${tot:,.0f}\ntotal interest: ${tot - P:,.0f}\n\nearly payments are mostly\n"
                                   "interest", step + 1, 18, INK, math=False)
    return step + 2


def build_ppf(cv, v, box, step):
    a, b = v.get("goods") or ("robots", "food")
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    S = min(W * 0.55, H - 60)
    px = lambda q: x0 + 60 + q / 100 * S
    py = lambda q: y1 - 30 - q / 100 * S
    _axes(cv, x0 + 60, y1 - 30 - S - 20, x0 + 80 + S, y1 - 30, step, a, b)
    curve = [(px(100 * math.sin(t)), py(100 * math.cos(t))) for t in [i * math.pi / 2 / 30 for i in range(31)]]
    cv.line(curve, step, "#1971c2", 3)
    step += 1
    pts = [("A", 30, 95, "#2f9e44", "efficient"), ("B", 70, 71, "#2f9e44", "efficient"), ("C", 40, 40, "#f08c00", "inefficient"),
           ("D", 80, 85, "#e03131", "impossible (for now)")]
    for name, qa, qb, col, lab in pts:
        cv.ellipse(px(qa) - 8, py(qb) - 8, 16, 16, step, col, col, 1)
        cv.text(px(qa) + 12, py(qb), f"{name}: {lab}", step, 17, col, "left", "middle", math=False)
        step += 1
    cv.text(x0 + W * 0.68, y0 + 30, f"opportunity cost: moving from A\nto B gives 40 more {a} but\ncosts 24 {b}\n\n"
                                   "bowed out: resources aren't\nequally good at both\n\ngrowth shifts the curve out",
            step, 18, INK, math=False)
    return step + 1


# ======================================================================== register

visuals.BUILDERS.update({
    "quick_sort": build_quick_sort, "maze_bfs": build_maze_bfs, "bloom_filter": build_bloom_filter,
    "consistent_hash": build_consistent_hash, "list_reverse": build_list_reverse, "sliding_window": build_sliding_window,
    "memory_layout": build_memory_layout, "fork_tree": build_fork_tree, "disk_sched": build_disk_sched, "raid": build_raid,
    "cpu_cycle": build_cpu_cycle, "ieee754": build_ieee754, "utf8": build_utf8, "nat_table": build_nat_table,
    "routing_lpm": build_routing_lpm, "traceroute": build_traceroute, "buffer_overflow": build_buffer_overflow,
    "brute_force": build_brute_force, "backprop": build_backprop, "roc_curve": build_roc_curve,
    "grid_world": build_grid_world, "pca": build_pca, "word_vectors": build_word_vectors,
    "transformer_block": build_transformer_block, "sql_joins": build_sql_joins, "normalization": build_normalization,
    "star_schema": build_star_schema, "moving_average": build_moving_average, "correlation": build_correlation,
    "sieve": build_sieve, "euclid_gcd": build_euclid_gcd, "pascal": build_pascal, "binomial": build_binomial,
    "bayes_test": build_bayes_test, "monte_carlo_pi": build_monte_carlo_pi, "collision": build_collision,
    "incline": build_incline, "gas_law": build_gas_law, "half_life": build_half_life, "doppler": build_doppler,
    "electron_config": build_electron_config, "food_web": build_food_web, "energy_pyramid": build_energy_pyramid,
    "natural_selection": build_natural_selection, "seasons": build_seasons, "greenhouse": build_greenhouse,
    "rock_cycle": build_rock_cycle, "map_projection": build_map_projection, "voronoi": build_voronoi,
    "choropleth": build_choropleth, "amortization": build_amortization, "ppf": build_ppf,
})
for _k in [k for k in list(visuals.ALIASES) if k in visuals.BUILDERS]:
    visuals.ALIASES.pop(_k)
visuals.ALIASES.update({
    "quicksort": "quick_sort", "partition": "quick_sort", "pathfinding": "maze_bfs", "maze": "maze_bfs",
    "bloom": "bloom_filter", "hash_ring": "consistent_hash", "reverse_linked_list": "list_reverse",
    "two_pointers": "sliding_window", "process_memory": "memory_layout", "fork": "fork_tree",
    "disk_scheduling": "disk_sched", "fetch_decode_execute": "cpu_cycle", "floating_point": "ieee754",
    "unicode": "utf8", "encoding": "utf8", "nat": "nat_table", "routing_table": "routing_lpm", "ttl": "traceroute",
    "stack_smashing": "buffer_overflow", "password_strength": "brute_force", "chain_rule": "backprop",
    "computational_graph": "backprop", "roc": "roc_curve", "auc": "roc_curve", "reinforcement_learning": "grid_world",
    "value_iteration": "grid_world", "principal_components": "pca", "embeddings": "word_vectors",
    "word2vec": "word_vectors", "encoder": "transformer_block", "joins": "sql_joins", "3nf": "normalization",
    "data_warehouse": "star_schema", "time_series": "moving_average", "scatter_correlation": "correlation",
    "primes": "sieve", "gcd": "euclid_gcd", "pascals_triangle": "pascal", "binomial_distribution": "binomial",
    "bayes": "bayes_test", "base_rate": "bayes_test", "monte_carlo": "monte_carlo_pi", "momentum": "collision",
    "inclined_plane": "incline", "ideal_gas": "gas_law", "boyle": "gas_law", "radioactive_decay": "half_life",
    "doppler_effect": "doppler", "orbitals": "electron_config", "trophic_levels": "energy_pyramid",
    "evolution": "natural_selection", "axial_tilt": "seasons", "greenhouse_effect": "greenhouse",
    "rocks": "rock_cycle", "mercator": "map_projection", "thiessen": "voronoi", "thematic_map": "choropleth",
    "mortgage": "amortization", "loan": "amortization", "production_possibilities": "ppf",
})
