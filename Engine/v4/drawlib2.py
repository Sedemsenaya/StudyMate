"""
StudyMate v4 - more things the engine draws itself (no AI needed for the picture).

Computed for real where possible (so the picture is correct, not guessed):
  gantt       CPU scheduling (FCFS, SJF, RR) computed from the processes
  graph_algo  BFS / DFS / Dijkstra run on the graph, visit order animated
  sql         query really run in SQLite; tables, query and result drawn
  asm_trace   tiny x86-style interpreter (mov add sub mul inc dec cmp jmp je jne jl jg push pop)
  ast         Python expression parsed into its syntax tree (compilers)
  kmeans      k-means really run, points recoloured per iteration
  gradient    gradient descent really run on a 1-D loss, ball rolling down the curve
  circuit     series / parallel resistors: total R and current from Ohm's law, current flowing
  projectile  real trajectory, ball moving
  atom        Bohr model of elements 1-20 (protons, neutrons, shells filled 2-8-8-2)

Drawn:
  pipeline    5-stage CPU pipeline, cycle by cycle
  hierarchy   memory hierarchy pyramid           gpu_grid   CUDA grid -> blocks -> threads
  logic       logic gates with input values flowing to the output
  ds_ops      stack / queue / linked list operations, step by step
  neural_net  layers of neurons, forward pass lighting up layer by layer
  attention   tokens with attention lines from one token (NLP)
  molecule    H2O, CO2, CH4, NH3, O2, N2, NaCl, HCl, C2H6
  free_body   a box with labelled force arrows
  nested      boxes inside boxes: Docker host/containers, Kubernetes cluster/nodes/pods,
              VPC/subnets, Azure resource groups, Terraform modules (logos where known)
  gis_layers  stacked map layers (basemap, raster, polygons, lines, points)
  osi         encapsulation: data -> segment -> packet -> frame, headers added layer by layer
  topology    Cisco-style network: routers, switches, PCs, servers, IPs on links, packet path
"""
import ast as pyast
import math
import random
import re
import sqlite3

import visuals
from canvas import INK, MUTED, fit_text, text_size
from drawlib import INKS, _envelope, _poly, _Track, draw_device, device_for

PAL = ["#a5d8ff", "#ffc9c9", "#b2f2bb", "#ffec99", "#d0bfff", "#ffd8a8", "#99e9f2", "#fcc2d7"]
STRONG = ["#1971c2", "#e03131", "#2f9e44", "#f08c00", "#7048e8", "#e8590c", "#0c8599", "#c2255c"]


def _box(cv, x, y, w, h, step, text, fill="#ffffff", size=20, stroke=INK, sw=2, mono=False):
    cv.rect(x, y, w, h, step, fill, stroke, sw)
    el = cv.label_in(x, y, w, h, str(text), step, size, pad=4)
    if el is not None and mono:
        el["fontFamily"] = 3
    return (x, y, w, h)


# ======================================================================== OS: scheduling

def schedule(procs, algo="fcfs", quantum=2):
    """procs: [{"name","arrival","burst"}] -> [(name, start, end)] really computed."""
    ps = sorted([dict(p) for p in procs], key=lambda p: (p.get("arrival", 0), p["name"]))
    for p in ps:
        p.setdefault("arrival", 0)
        p["left"] = p["burst"]
    t, out = 0, []
    algo = algo.lower()
    if algo in ("rr", "round robin", "round-robin"):
        queue, i = [], 0
        while i < len(ps) or queue:
            while i < len(ps) and ps[i]["arrival"] <= t:
                queue.append(ps[i])
                i += 1
            if not queue:
                t = ps[i]["arrival"]
                continue
            p = queue.pop(0)
            run = min(quantum, p["left"])
            out.append((p["name"], t, t + run))
            t += run
            p["left"] -= run
            while i < len(ps) and ps[i]["arrival"] <= t:
                queue.append(ps[i])
                i += 1
            if p["left"] > 0:
                queue.append(p)
        return out
    done = set()
    while len(done) < len(ps):
        ready = [p for p in ps if p["name"] not in done and p["arrival"] <= t]
        if not ready:
            t = min(p["arrival"] for p in ps if p["name"] not in done)
            continue
        p = min(ready, key=lambda q: q["burst"]) if algo in ("sjf", "shortest job first") else ready[0]
        out.append((p["name"], t, t + p["burst"]))
        t += p["burst"]
        done.add(p["name"])
    return out


def build_gantt(cv, v, box, step):
    procs = v.get("processes") or [{"name": "P1", "arrival": 0, "burst": 5}, {"name": "P2", "arrival": 1, "burst": 3},
                                   {"name": "P3", "arrival": 2, "burst": 1}, {"name": "P4", "arrival": 3, "burst": 2}]
    algo = str(v.get("algorithm") or "fcfs")
    sched = schedule(procs, algo, int(v.get("quantum") or 2))
    x0, y0, x1, y1 = box
    W = x1 - x0
    end = max(e for _, _, e in sched)
    unit = (W - 40) / end
    names = sorted({p["name"] for p in procs})
    col = {n: PAL[i % len(PAL)] for i, n in enumerate(names)}
    # process table
    ty = y0
    cv.text(x0, ty, f"{algo.upper()}" + (f" (quantum {v.get('quantum') or 2})" if algo.lower() in ("rr", "round robin") else ""),
            step, 26, "#e8590c", math=False)
    for j, h in enumerate(["process", "arrival", "burst"]):
        cv.text(x0 + j * 120, ty + 44, h, step, 18, MUTED, math=False)
    for i, p in enumerate(procs):
        cv.rect(x0 - 6, ty + 70 + i * 32, 22, 22, step, col[p["name"]], sw=1)
        for j, k in enumerate(["name", "arrival", "burst"]):
            cv.text(x0 + 24 + j * 120 - (24 if j else 0), ty + 70 + i * 32, str(p.get(k, 0)), step, 19, INK, math=False)
    step += 1
    gy = ty + 90 + len(procs) * 32
    cv.text(x0, gy - 30, "CPU timeline", step, 19, MUTED, math=False)
    for name, s, e in sched:                       # one block per step
        _box(cv, x0 + s * unit, gy, (e - s) * unit, 56, step, name, col[name], 20)
        cv.text(x0 + s * unit, gy + 62, str(s), step, 16, MUTED, "center", "top", math=False)
        step += 1
    cv.text(x0 + end * unit, gy + 62, str(end), step - 1, 16, MUTED, "center", "top", math=False)
    # waiting times (really computed)
    fin = {}
    for name, s, e in sched:
        fin[name] = e
    waits = []
    for p in procs:
        w = fin[p["name"]] - p.get("arrival", 0) - p["burst"]
        waits.append(w)
    cv.text(x0, gy + 100, "waiting: " + ", ".join(f"{p['name']}={w}" for p, w in zip(procs, waits)) +
            f"   average = {sum(waits) / len(waits):.2f}", step, 20, "#2f9e44", math=False)
    return step + 1


# ======================================================================== graphs

def _layout(nodes, edges, box):
    """Circle layout (good for small graphs)."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = min(x1 - x0, y1 - y0) * 0.38
    pos = {}
    for i, n in enumerate(nodes):
        a = -math.pi / 2 + 2 * math.pi * i / len(nodes)
        pos[n] = (cx + r * 1.3 * math.cos(a), cy + r * math.sin(a))
    return pos


def run_graph(nodes, edges, algo, start, directed=False):
    adj = {n: [] for n in nodes}
    for e in edges:
        a, b, w = e[0], e[1], (e[2] if len(e) > 2 else 1)
        adj[a].append((b, w))
        if not directed:
            adj[b].append((a, w))
    for n in adj:
        adj[n].sort()
    order, dist = [], {}
    if algo == "dfs":
        seen = set()

        def go(u):
            seen.add(u)
            order.append(u)
            for v, _ in adj[u]:
                if v not in seen:
                    go(v)
        go(start)
    elif algo == "dijkstra":
        import heapq
        dist = {start: 0}
        pq, seen = [(0, start)], set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            order.append(u)
            for v, w in adj[u]:
                if d + w < dist.get(v, 1e18):
                    dist[v] = d + w
                    heapq.heappush(pq, (d + w, v))
    else:
        seen, q = {start}, [start]
        while q:
            u = q.pop(0)
            order.append(u)
            for v, _ in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
    return order, dist


def build_graph_algo(cv, v, box, step):
    nodes = [str(n) for n in (v.get("nodes") or list("ABCDEF"))][:9]
    edges = v.get("edges") or [["A", "B", 4], ["A", "C", 2], ["B", "D", 5], ["C", "D", 8], ["C", "E", 10], ["D", "F", 2], ["E", "F", 3]]
    edges = [[str(e[0]), str(e[1])] + ([e[2]] if len(e) > 2 else []) if isinstance(e, (list, tuple)) else
             [str(e.get("from")), str(e.get("to"))] + ([e["weight"]] if "weight" in e else []) for e in edges]
    edges = [e for e in edges if e[0] in nodes and e[1] in nodes]
    algo = str(v.get("algorithm") or "bfs").lower()
    start = str(v.get("start") or nodes[0])
    order, dist = run_graph(nodes, edges, algo, start)
    x0, y0, x1, y1 = box
    gbox = (x0, y0 + 30, x0 + (x1 - x0) * 0.66, y1 - 10)
    pos = _layout(nodes, edges, gbox)
    R = 30
    for e in edges:
        (ax, ay), (bx, by) = pos[e[0]], pos[e[1]]
        cv.line([(ax, ay), (bx, by)], step, MUTED, 2)
        if len(e) > 2:
            cv.text((ax + bx) / 2, (ay + by) / 2 - 4, str(e[2]), step, 18, "#e8590c", "center", "bottom", math=False)
    for n in nodes:
        x, y = pos[n]
        cv.ellipse(x - R, y - R, 2 * R, 2 * R, step, "#ffffff")
        cv.text(x, y, n, step, 22, INK, "center", "middle", math=False)
    cv.text(x0, y0, {"bfs": "Breadth-first search", "dfs": "Depth-first search", "dijkstra": "Dijkstra"}.get(algo, algo)
            + f" from {start}", step, 24, "#e8590c", math=False)
    step += 1
    lx = gbox[2] + 30
    for k, n in enumerate(order):
        x, y = pos[n]
        cv.ellipse(x - R, y - R, 2 * R, 2 * R, step, "#b2f2bb", "#2f9e44", 3)
        cv.text(x, y, n, step, 22, INK, "center", "middle", math=False)
        cv.text(x + R, y - R, str(k + 1), step, 16, "#2f9e44", "left", "bottom", math=False)
        txt = f"{k + 1}. visit {n}" + (f"  (distance {dist[n]})" if dist else "")
        cv.text(lx, y0 + 50 + k * 34, txt, step, 20, INK, math=False)
        step += 1
    return step


# ======================================================================== data structures

def build_ds_ops(cv, v, box, step):
    """{"type":"ds_ops","structure":"stack|queue|linked_list","ops":["push 3","push 5","pop", ...]}"""
    kind = str(v.get("structure") or "stack").lower()
    ops = [str(o) for o in (v.get("ops") or ["push 3", "push 7", "push 1", "pop", "push 9", "pop"])][:10]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    items = []
    log = []
    cv.text(x0, y0, kind.replace("_", " ").title(), step, 26, "#e8590c", math=False)
    step += 1
    for k, op in enumerate(ops):
        m = re.match(r"(\w+)\s*\(?\s*([^)]*)\)?", op.strip())
        verb, arg = (m.group(1).lower(), m.group(2).strip()) if m else (op, "")
        res = ""
        if verb in ("push", "enqueue", "append", "insert", "add", "offer"):
            items.append(arg)
        elif verb in ("pop", "dequeue", "remove", "poll"):
            if items:
                res = items.pop() if kind == "stack" else items.pop(0)
        elif verb in ("peek", "top", "front"):
            res = items[-1] if kind == "stack" and items else (items[0] if items else "")
        until = step + 1 if k < len(ops) - 1 else None
        t = _Track(cv)
        cx = x0 + W * 0.3
        if kind == "stack":
            cap = max(4, sum(1 for o in ops if re.match(r"\s*(push|add|append|insert)", o, re.I)) + 1)
            bw, bh = 150, min(56, (H - 80) / cap)
            base = y1 - 20
            el = cv.line([(cx - bw / 2 - 12, base - bh * cap), (cx - bw / 2 - 12, base + 6), (cx + bw / 2 + 12, base + 6),
                          (cx + bw / 2 + 12, base - bh * cap)], step, INK, 3)
            el["roundness"] = None
            for i, it in enumerate(items):
                _box(cv, cx - bw / 2, base - (i + 1) * bh, bw, bh - 4, step, it,
                     "#b2f2bb" if i == len(items) - 1 else PAL[i % len(PAL)], 20)
            if items:
                cv.text(cx + bw / 2 + 24, base - len(items) * bh + bh / 2, "◀ top", step, 18, "#2f9e44", "left", "middle",
                        math=False)
        else:
            bw, bh = min(110, W * 0.6 / 8), 64
            cy = y0 + H * 0.45
            sx = x0 + 20
            for i, it in enumerate(items):
                _box(cv, sx + i * (bw + (34 if kind == "linked_list" else 6)), cy, bw, bh, step, it, PAL[i % len(PAL)], 20)
                if kind == "linked_list" and i < len(items) - 1:
                    xx = sx + i * (bw + 34) + bw
                    cv.arrow(xx + 2, cy + bh / 2, xx + 32, cy + bh / 2, step)
            if items and kind == "queue":
                cv.text(sx + bw / 2, cy - 8, "front", step, 17, "#2f9e44", "center", "bottom", math=False)
                cv.text(sx + (len(items) - 0.5) * (bw + 6), cy + bh + 8, "rear", step, 17, "#1971c2", "center", "top",
                        math=False)
            if kind == "linked_list" and items:
                cv.text(sx, cy - 8, "head", step, 17, "#2f9e44", "left", "bottom", math=False)
        t.tag(until=until, fade=k > 0)
        log.append(op + (f"  → {res}" if res else ""))
        cv.text(x0 + W * 0.62, y0 + 50 + k * 34, f"{k + 1}. {log[-1]}", step, 21, "#c92a2a" if res else INK, math=False)
        step += 1
    return step


# ======================================================================== architecture

def build_pipeline(cv, v, box, step):
    stages = v.get("stages") or ["IF", "ID", "EX", "MEM", "WB"]
    instr = v.get("instructions") or ["add r1, r2, r3", "sub r4, r1, r5", "lw r6, 0(r7)", "or r8, r6, r2"]
    instr = instr[:6]
    stalls = {int(k): int(n) for k, n in (v.get("stalls") or {}).items()} if isinstance(v.get("stalls"), dict) else {}
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    lw = W * 0.26
    start = []
    t = 0
    for i in range(len(instr)):
        t += stalls.get(i, 0)
        start.append(t)
        t += 1
    cycles = start[-1] + len(stages)
    cwid = (W - lw) / cycles
    rh = min(58, (H - 50) / (len(instr) + 1))
    for c in range(cycles):
        cv.text(x0 + lw + (c + 0.5) * cwid, y0, f"{c + 1}", step, 17, MUTED, "center", "top", math=False)
    cv.text(x0, y0, "cycle →", step, 17, MUTED, math=False)
    for i, ins in enumerate(instr):
        el = cv.text(x0, y0 + 34 + i * rh + rh / 2, ins, step, 19, INK, "left", "middle", math=False)
        if el:
            el["fontFamily"] = 3
    step += 1
    for c in range(cycles):                         # cycle by cycle
        for i in range(len(instr)):
            s = c - start[i]
            if 0 <= s < len(stages):
                _box(cv, x0 + lw + c * cwid + 2, y0 + 34 + i * rh + 3, cwid - 4, rh - 6, step, stages[s],
                     PAL[s % len(PAL)], 17)
            elif s < 0 and i > 0 and c >= start[i - 1] + 1 and stalls.get(i) and c < start[i]:
                _box(cv, x0 + lw + c * cwid + 2, y0 + 34 + i * rh + 3, cwid - 4, rh - 6, step, "stall", "#e9ecef", 15)
        step += 1
    return step


def build_hierarchy(cv, v, box, step):
    levels = v.get("levels") or [["Registers", "<1 ns", "bytes"], ["L1 cache", "~1 ns", "KB"],
                                 ["L2 / L3 cache", "3-30 ns", "MB"], ["Main memory (RAM)", "~100 ns", "GB"],
                                 ["SSD", "~100 µs", "TB"], ["Hard disk", "~10 ms", "TB"]]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    n = len(levels)
    pw, lh = W * 0.6, (H - 20) / n
    cx = x0 + pw / 2 + 20
    for i, lv in enumerate(levels):
        lv = lv if isinstance(lv, (list, tuple)) else [lv]
        top_w = pw * (0.18 + 0.82 * i / n)
        bot_w = pw * (0.18 + 0.82 * (i + 1) / n)
        y = y0 + i * lh
        cv.polygon([(cx - top_w / 2, y), (cx + top_w / 2, y), (cx + bot_w / 2, y + lh - 3), (cx - bot_w / 2, y + lh - 3)],
                   step, PAL[i % len(PAL)])
        cv.text(cx, y + lh / 2, str(lv[0]), step, 19, INK, "center", "middle", math=False)
        if len(lv) > 1:
            cv.text(cx + pw / 2 + 30, y + lh / 2, "  ·  ".join(str(x) for x in lv[1:]), step, 19, "#495057", "left",
                    "middle", math=False)
        step += 1
    cv.arrow(x0, y0 + 10, x0, y1 - 10, step, color="#e03131")
    cv.text(x0 + 8, y1 - 10, "bigger, slower, cheaper", step, 16, "#e03131", "left", "bottom", math=False)
    return step + 1


def build_gpu_grid(cv, v, box, step):
    gx, gy = int(v.get("grid_x") or 3), int(v.get("grid_y") or 2)
    tx, ty = int(v.get("block_x") or 4), int(v.get("block_y") or 2)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cv.rect(x0, y0, W * 0.98, H - 40, step, "#f8f9fa", INK, 3)
    cv.text(x0 + 10, y0 + 6, f"Grid: {gx} x {gy} blocks", step, 22, "#e8590c", math=False)
    step += 1
    bw, bh = (W * 0.98 - 30) / gx, (H - 100) / gy
    for by in range(gy):
        for bx in range(gx):
            X, Y = x0 + 15 + bx * bw, y0 + 44 + by * bh
            cv.rect(X + 4, Y + 4, bw - 8, bh - 8, step, "#e7f5ff", "#1971c2", 2)
            cv.text(X + 12, Y + 8, f"block ({bx},{by})", step, 16, "#1971c2", math=False)
            tw, th = (bw - 30) / tx, (bh - 44) / ty
            for j in range(ty):
                for i in range(tx):
                    cv.rect(X + 14 + i * tw, Y + 34 + j * th, tw - 4, th - 4, step, "#ffd8a8", sw=1)
            step += 1
    gid = "global id = blockIdx.x * blockDim.x + threadIdx.x"
    cv.text(x0, y1 - 30, gid, step, 19, INK, math=False)
    return step + 1


def build_logic(cv, v, box, step):
    """{"type":"logic","gate":"AND|OR|NOT|XOR|NAND|NOR","inputs":[1,0]} or a truth table of the gate."""
    gate = str(v.get("gate") or "AND").upper()
    fns = {"AND": lambda a, b: a & b, "OR": lambda a, b: a | b, "XOR": lambda a, b: a ^ b,
           "NAND": lambda a, b: 1 - (a & b), "NOR": lambda a, b: 1 - (a | b), "NOT": lambda a, b=0: 1 - a}
    f = fns.get(gate, fns["AND"])
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    gx, gy, gw, gh = x0 + W * 0.22, y0 + H * 0.25, 150, 120
    # gate shape
    if gate in ("AND", "NAND"):
        pts = [(gx, gy), (gx + gw * 0.5, gy)] + [(gx + gw * 0.5 + gh / 2 * math.sin(a), gy + gh / 2 - gh / 2 * math.cos(a))
                                                for a in [i * math.pi / 10 for i in range(11)]] + [(gx, gy + gh), (gx, gy)]
        _poly(cv, pts, step, INK, "#d0ebff", 3, smooth=False)
        out_x = gx + gw * 0.5 + gh / 2
    elif gate == "NOT":
        cv.polygon([(gx, gy), (gx + gw * 0.8, gy + gh / 2), (gx, gy + gh)], step, "#d0ebff", INK, 3)
        out_x = gx + gw * 0.8
    else:
        pts = [(gx, gy), (gx + gw * 0.6, gy + gh * 0.1), (gx + gw, gy + gh / 2), (gx + gw * 0.6, gy + gh * 0.9),
               (gx, gy + gh), (gx + gw * 0.22, gy + gh / 2), (gx, gy)]
        _poly(cv, pts, step, INK, "#d0ebff", 3)
        out_x = gx + gw
        if gate == "XOR":
            cv.line([(gx - 14, gy), (gx + gw * 0.22 - 14, gy + gh / 2), (gx - 14, gy + gh)], step, INK, 3)
    if gate in ("NAND", "NOR"):
        cv.ellipse(out_x, gy + gh / 2 - 9, 18, 18, step, "#ffffff", INK, 3)
        out_x += 18
    cv.text(gx + gw * 0.35, gy + gh / 2, gate, step, 26, INK, "center", "middle", math=False)
    ins = 1 if gate == "NOT" else 2
    iy = [gy + gh / 2] if ins == 1 else [gy + gh * 0.28, gy + gh * 0.72]
    for k, y in enumerate(iy):
        cv.line([(gx - 120, y), (gx + (gw * 0.2 if gate in ("OR", "NOR", "XOR") else 0), y)], step, INK, 3)
        cv.text(gx - 130, y, "AB"[k], step, 24, INK, "right", "middle", math=False)
    cv.line([(out_x, gy + gh / 2), (out_x + 120, gy + gh / 2)], step, INK, 3)
    cv.text(out_x + 130, gy + gh / 2, "Q", step, 24, INK, "left", "middle", math=False)
    step += 1
    # truth table, row by row; each row lights the wires
    rows = [(a,) for a in (0, 1)] if ins == 1 else [(a, b) for a in (0, 1) for b in (0, 1)]
    tx, ty = x0 + W * 0.7, y0 + 20
    cv.text(tx, ty, ("A" if ins == 1 else "A   B") + "   Q", step, 22, MUTED, math=False)
    for r, vals in enumerate(rows):
        q = f(*vals)
        until = step + 1 if r < len(rows) - 1 else None
        t = _Track(cv)
        for k, y in enumerate(iy):
            cv.text(gx - 60, y - 6, str(vals[k]), step, 26, "#2f9e44" if vals[k] else "#e03131", "center", "bottom",
                    math=False)
        cv.text(out_x + 60, gy + gh / 2 - 6, str(q), step, 30, "#2f9e44" if q else "#e03131", "center", "bottom",
                math=False)
        t.tag(until=until, fade=True)
        cv.text(tx, ty + 40 + r * 36, "   ".join(str(x) for x in vals) + "   " + str(q), step, 22,
                "#2f9e44" if q else INK, math=False)
        step += 1
    return step


# ======================================================================== assembly (tiny interpreter)

def run_asm(code, max_steps=40):
    """x86-style: mov add sub mul imul inc dec cmp jmp je jne jl jle jg jge push pop nop. Labels 'name:'."""
    lines = [l.split(";")[0].rstrip() for l in code.split("\n")]
    labels = {}
    for i, l in enumerate(lines):
        m = re.match(r"\s*(\w+):", l)
        if m:
            labels[m.group(1)] = i
    regs = {"eax": 0, "ebx": 0, "ecx": 0, "edx": 0}
    stack, flags, steps, pc = [], 0, [], 0

    def val(x):
        x = x.strip().lower()
        if x in regs:
            return regs[x]
        return int(x, 0)
    while pc < len(lines) and len(steps) < max_steps:
        l = re.sub(r"^\s*\w+:\s*", "", lines[pc]).strip()
        if not l:
            pc += 1
            continue
        op, _, rest = l.partition(" ")
        op = op.lower()
        args = [a.strip().lower() for a in rest.split(",")] if rest else []
        for a in args:
            if re.fullmatch(r"[a-z]{3}", a) and a not in regs and a not in labels:
                regs[a] = 0
        nxt = pc + 1
        note = ""
        if op == "mov":
            regs[args[0]] = val(args[1])
        elif op in ("add", "sub", "mul", "imul"):
            if len(args) == 1:
                regs["eax"] *= val(args[0])
            else:
                a, b = regs[args[0]], val(args[1])
                regs[args[0]] = a + b if op == "add" else a - b if op == "sub" else a * b
        elif op in ("inc", "dec"):
            regs[args[0]] += 1 if op == "inc" else -1
        elif op == "cmp":
            flags = val(args[0]) - val(args[1])
            note = f"compare {val(args[0])} with {val(args[1])}"
        elif op in ("jmp", "je", "jz", "jne", "jnz", "jl", "jle", "jg", "jge"):
            take = {"jmp": True, "je": flags == 0, "jz": flags == 0, "jne": flags != 0, "jnz": flags != 0,
                    "jl": flags < 0, "jle": flags <= 0, "jg": flags > 0, "jge": flags >= 0}[op]
            if take:
                nxt = labels.get(args[0], pc + 1)
            note = f"jump to {args[0]}" if take else "no jump"
        elif op == "push":
            stack.append(val(args[0]))
        elif op == "pop":
            regs[args[0]] = stack.pop() if stack else 0
        elif op in ("ret", "hlt", "halt"):
            steps.append({"line": pc + 1, "vars": dict(regs), "array": list(stack) or None, "array_name": "stack",
                          "note": "stop"})
            break
        steps.append({"line": pc + 1, "vars": dict(regs), "array": list(stack) or None, "array_name": "stack",
                      "note": note})
        pc = nxt
    for s in steps:
        if not s.get("array"):
            s.pop("array", None)
    if len(steps) > 16:                              # keep the first and last steps, sample the middle
        idx = sorted({round(i * (len(steps) - 1) / 15) for i in range(16)})
        steps = [steps[i] for i in idx]
    return steps


# ======================================================================== compilers: syntax tree

def ast_tree(expr):
    """Python expression/statement -> {"label","children"} tree."""
    ops = {pyast.Add: "+", pyast.Sub: "-", pyast.Mult: "*", pyast.Div: "/", pyast.Pow: "**", pyast.Mod: "%",
           pyast.Lt: "<", pyast.Gt: ">", pyast.Eq: "==", pyast.LtE: "<=", pyast.GtE: ">=", pyast.NotEq: "!=",
           pyast.And: "and", pyast.Or: "or", pyast.USub: "-"}

    def conv(n):
        if isinstance(n, pyast.Module):
            return conv(n.body[0])
        if isinstance(n, pyast.Expr):
            return conv(n.value)
        if isinstance(n, pyast.Assign):
            return {"label": "=", "children": [conv(n.targets[0]), conv(n.value)]}
        if isinstance(n, pyast.BinOp):
            return {"label": ops.get(type(n.op), "op"), "children": [conv(n.left), conv(n.right)]}
        if isinstance(n, pyast.UnaryOp):
            return {"label": ops.get(type(n.op), "op"), "children": [conv(n.operand)]}
        if isinstance(n, pyast.BoolOp):
            return {"label": ops.get(type(n.op), "op"), "children": [conv(x) for x in n.values]}
        if isinstance(n, pyast.Compare):
            return {"label": ops.get(type(n.ops[0]), "cmp"), "children": [conv(n.left), conv(n.comparators[0])]}
        if isinstance(n, pyast.Call):
            return {"label": (getattr(n.func, "id", None) or "call") + "()", "children": [conv(a) for a in n.args]}
        if isinstance(n, pyast.Name):
            return {"label": n.id}
        if isinstance(n, pyast.Constant):
            return {"label": repr(n.value)}
        return {"label": type(n).__name__}
    return conv(pyast.parse(expr))


def tokens_of(expr):
    found = re.findall(r"(\d+\.?\d*)|([A-Za-z_]\w*)|(==|<=|>=|!=|\*\*|[-+*/%=<>()])", expr)
    return [("number" if a else "identifier" if b else "operator", a or b or c) for a, b, c in found]


def build_ast(cv, v, box, step):
    expr = str(v.get("expr") or "x = a + b * 2")
    try:
        tree = ast_tree(expr)
    except SyntaxError:
        return step
    x0, y0, x1, y1 = box
    W = x1 - x0
    el = cv.text(x0, y0, expr, step, 26, INK, math=False)
    if el:
        el["fontFamily"] = 3
    step += 1
    toks = tokens_of(expr)
    tw = min(110, (W - 20) / max(1, len(toks)))
    for i, (kind, t) in enumerate(toks):          # lexer: tokens
        _box(cv, x0 + i * tw, y0 + 46, tw - 8, 44, step, t, {"number": "#ffec99", "identifier": "#d0ebff",
                                                            "operator": "#ffc9c9"}[kind], 18, mono=True)
    cv.text(x0 + len(toks) * tw + 10, y0 + 68, "tokens (lexer)", step, 17, MUTED, "left", "middle", math=False)
    step += 1
    return visuals.build_tree(cv, {"root": tree}, (x0, y0 + 120, x1, y1), step)


# ======================================================================== data: SQL

def run_sql(tables, query):
    con = sqlite3.connect(":memory:")
    for name, t in tables.items():
        cols = t["columns"]
        con.execute(f"CREATE TABLE {name} ({', '.join(cols)})")
        con.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(cols))})", t["rows"])
    cur = con.execute(query)
    return [d[0] for d in cur.description], [list(r) for r in cur.fetchall()]


def _table(cv, x, y, name, cols, rows, step, w, hl=None, size=17):
    cw = w / len(cols)
    rh = 32
    cv.text(x, y - 6, name, step, 19, "#e8590c", "left", "bottom", math=False)
    for j, c in enumerate(cols):
        _box(cv, x + j * cw, y, cw, rh, step, c, "#e9ecef", size)
    for i, r in enumerate(rows[:8]):
        for j, c in enumerate(r):
            _box(cv, x + j * cw, y + (i + 1) * rh, cw, rh, step, c,
                 "#b2f2bb" if hl and i in hl else "#ffffff", size)
    return y + (min(len(rows), 8) + 1) * rh


def build_sql(cv, v, box, step):
    tables = v.get("tables") or {
        "students": {"columns": ["id", "name"], "rows": [[1, "Ana"], [2, "Ben"], [3, "Chen"]]},
        "grades": {"columns": ["student_id", "course", "grade"],
                   "rows": [[1, "CS 224", "A"], [2, "CS 224", "B"], [1, "CS 226", "A-"]]}}
    query = v.get("query") or ("SELECT s.name, g.course, g.grade FROM students s "
                               "JOIN grades g ON s.id = g.student_id ORDER BY s.name")
    try:
        cols, rows = run_sql(tables, query)
    except Exception as ex:
        cv.text(box[0], box[1], f"SQL error: {ex}", step, 20, "#e03131", math=False)
        return step + 1
    x0, y0, x1, y1 = box
    W = x1 - x0
    n = len(tables)
    tw = (W * 0.55 - 20 * (n - 1)) / n
    for i, (name, t) in enumerate(tables.items()):
        _table(cv, x0 + i * (tw + 20), y0 + 30, name, t["columns"], t["rows"], step, tw)
        step += 1
    q, qs = fit_text(query, W * 0.42, 120, 19, 13)
    el = cv.text(x0 + W * 0.58, y0, q, step, qs, "#1971c2", math=False)
    if el:
        el["fontFamily"] = 3
    step += 1
    cv.arrow(x0 + W * 0.75, y0 + 130, x0 + W * 0.75, y0 + 175, step, "runs", color="#1971c2")
    _table(cv, x0 + W * 0.58, y0 + 210, "result", cols, rows, step, W * 0.42, hl=set(range(len(rows))))
    return step + 1


# ======================================================================== ML / DL / NLP

def build_neural_net(cv, v, box, step):
    layers = [int(x) for x in (v.get("layers") or [3, 4, 4, 2])][:6]
    names = v.get("names") or (["input"] + ["hidden"] * (len(layers) - 2) + ["output"])
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    pos = []
    for li, n in enumerate(layers):
        n_draw = min(n, 7)
        xs = x0 + W * (li + 0.5) / len(layers)
        ys = [y0 + 30 + (H - 70) * (k + 0.5) / n_draw for k in range(n_draw)]
        pos.append([(xs, y) for y in ys])
    R = min(24, (H - 70) / 7 / 2.6)
    for li in range(len(layers) - 1):                # weights first, faint
        for a in pos[li]:
            for b in pos[li + 1]:
                cv.line([a, b], step, "#ced4da", 1)
    for li, layer in enumerate(pos):
        for (x, y) in layer:
            cv.ellipse(x - R, y - R, 2 * R, 2 * R, step, "#ffffff")
        cv.text(layer[0][0], y1 - 10, f"{names[li] if li < len(names) else ''} ({layers[li]})", step, 18, MUTED,
                "center", "bottom", math=False)
    step += 1
    rnd = random.Random(4)
    for li, layer in enumerate(pos):                 # forward pass: layer by layer
        if li:
            for a in pos[li - 1]:
                for b in layer:
                    if rnd.random() < 0.5:
                        cv.line([a, b], step, "#f08c00", 1.5)
        for (x, y) in layer:
            act = rnd.random()
            cv.ellipse(x - R, y - R, 2 * R, 2 * R, step, "#ffd43b" if act > 0.5 else "#fff3bf", "#f08c00", 2)
            cv.text(x, y, f"{act:.1f}", step, max(11, int(R * 0.6)), INK, "center", "middle", math=False)
        step += 1
    return step


def build_gradient(cv, v, box, step):
    """Gradient descent really run on loss(w) = (w - target)^2 (or a given expression)."""
    lr = float(v.get("learning_rate") or 0.3)
    w = float(v.get("start") or -4.0)
    target = float(v.get("target") or 1.0)
    f = lambda x: (x - target) ** 2
    df = lambda x: 2 * (x - target)
    pts, ws = [], [w]
    for _ in range(int(v.get("steps") or 7)):
        w = w - lr * df(w)
        ws.append(w)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    lo, hi = min(-5, min(ws) - 1), max(5, max(ws) + 1)
    ymax = max(f(lo), f(hi))
    px = lambda x: x0 + 40 + (x - lo) / (hi - lo) * (W * 0.62 - 40)
    py = lambda y: y1 - 40 - y / ymax * (H - 80)
    cv.line([(x0 + 40, y1 - 40), (x0 + W * 0.62, y1 - 40)], step, INK, 2, arrow_end=True)
    cv.line([(x0 + 40, y1 - 40), (x0 + 40, y0)], step, INK, 2, arrow_end=True)
    cv.text(x0 + W * 0.62, y1 - 34, "w", step, 20, INK, "right", "top", math=False)
    cv.text(x0 + 46, y0, "loss", step, 20, INK, math=False)
    curve = [(px(lo + (hi - lo) * i / 40), py(f(lo + (hi - lo) * i / 40))) for i in range(41)]
    cv.line(curve, step, "#1971c2", 3)
    step += 1
    for k, wk in enumerate(ws):
        until = step + 1 if k < len(ws) - 1 else None
        t = _Track(cv)
        cv.ellipse(px(wk) - 12, py(f(wk)) - 12, 24, 24, step, "#e03131", "#c92a2a", 2)
        t.tag(until=until, fade=True)
        if k:
            cv.line([(px(ws[k - 1]), py(f(ws[k - 1]))), (px(wk), py(f(wk)))], step, "#e03131", 1.5, dashed=True)
        cv.text(x0 + W * 0.66, y0 + 10 + k * 34, f"step {k}: w = {wk:.3f}, loss = {f(wk):.3f}", step, 19,
                "#c92a2a" if k == len(ws) - 1 else INK, math=False)
        step += 1
    cv.text(x0 + W * 0.66, y0 + 30 + len(ws) * 34, f"w ← w − {lr} · dL/dw", step, 22, "#1971c2", math=False)
    return step + 1


def build_kmeans(cv, v, box, step):
    k = int(v.get("k") or 3)
    rnd = random.Random(7)
    pts = v.get("points") or [(rnd.gauss(cx, 0.7), rnd.gauss(cy, 0.7)) for cx, cy in [(2, 2), (6, 3), (4, 7)] for _ in range(12)]
    pts = [tuple(map(float, p)) for p in pts][:60]
    cents = [pts[0]]                                 # spread-out start (farthest point first)
    while len(cents) < k:
        cents.append(max(pts, key=lambda p: min((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in cents)))
    iters = []
    for _ in range(int(v.get("iterations") or 4)):
        lab = [min(range(k), key=lambda c: (p[0] - cents[c][0]) ** 2 + (p[1] - cents[c][1]) ** 2) for p in pts]
        iters.append((list(cents), lab))
        cents = [(sum(p[0] for p, l in zip(pts, lab) if l == c) / max(1, lab.count(c)),
                  sum(p[1] for p, l in zip(pts, lab) if l == c) / max(1, lab.count(c))) for c in range(k)]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    sx = lambda x: x0 + 20 + (x - min(xs)) / (max(xs) - min(xs) + 1e-9) * (W * 0.62)
    sy = lambda y: y1 - 20 - (y - min(ys)) / (max(ys) - min(ys) + 1e-9) * (H - 60)
    for p in pts:
        cv.ellipse(sx(p[0]) - 6, sy(p[1]) - 6, 12, 12, step, "#ced4da", "#868e96", 1)
    cv.text(x0, y0, f"k-means, k = {k}", step, 24, "#e8590c", math=False)
    step += 1
    for it, (cs, lab) in enumerate(iters):
        until = step + 1 if it < len(iters) - 1 else None
        t = _Track(cv)
        for p, l in zip(pts, lab):
            cv.ellipse(sx(p[0]) - 7, sy(p[1]) - 7, 14, 14, step, PAL[l], STRONG[l], 1.5)
        for c, (cx, cy) in enumerate(cs):
            cv.text(sx(cx), sy(cy), "✚", step, 34, STRONG[c], "center", "middle", math=False)
        cv.text(x0 + W * 0.7, y0 + 50, f"iteration {it + 1}", step, 24, INK, math=False)
        cv.text(x0 + W * 0.7, y0 + 90, "assign each point to the\nnearest centroid, then move\neach centroid to the mean",
                step, 18, MUTED, math=False)
        t.tag(until=until, fade=True)
        step += 1
    return step


def build_attention(cv, v, box, step):
    toks = [str(t) for t in (v.get("tokens") or ["The", "cat", "sat", "on", "the", "mat"])][:10]
    q = int(v.get("query", 2)) if str(v.get("query", "")).isdigit() else (toks.index(v["query"]) if v.get("query") in toks else 2)
    weights = v.get("weights")
    if not weights:
        rnd = random.Random(q)
        raw = [math.exp(rnd.uniform(0, 2) + (2.2 if i in (1, q) else 0)) for i in range(len(toks))]
        weights = [r / sum(raw) for r in raw]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    tw = W / len(toks)
    ytop, ybot = y0 + 40, y1 - 90
    for i, t in enumerate(toks):
        _box(cv, x0 + i * tw + 8, ytop, tw - 16, 50, step, t, "#ffec99" if i == q else "#e7f5ff", 21)
        _box(cv, x0 + i * tw + 8, ybot, tw - 16, 50, step, t, "#e7f5ff", 21)
    cv.text(x0, y0, f'attention from "{toks[q]}"', step, 22, "#e8590c", math=False)
    step += 1
    order = sorted(range(len(toks)), key=lambda i: -weights[i])
    for i in order:
        a = (x0 + q * tw + tw / 2, ytop + 52)
        b = (x0 + i * tw + tw / 2, ybot - 2)
        cv.line([a, b], step, "#e8590c", max(1, weights[i] * 14), arrow_end=True)
        cv.text(b[0], ybot + 56, f"{weights[i]:.2f}", step, 17, "#e8590c", "center", "top", math=False)
        step += 1
    return step


# ======================================================================== science

ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca"]
NAMES = ["Hydrogen", "Helium", "Lithium", "Beryllium", "Boron", "Carbon", "Nitrogen", "Oxygen", "Fluorine", "Neon",
         "Sodium", "Magnesium", "Aluminium", "Silicon", "Phosphorus", "Sulfur", "Chlorine", "Argon", "Potassium", "Calcium"]
NEUTRONS = [0, 2, 4, 5, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 22, 20, 20]


def build_atom(cv, v, box, step):
    el = str(v.get("element") or "C")
    z = ELEMENTS.index(el) + 1 if el in ELEMENTS else next((i + 1 for i, n in enumerate(NAMES) if n.lower() == el.lower()), 6)
    shells, left = [], z
    for cap in (2, 8, 8, 2):
        shells.append(min(cap, left))
        left -= shells[-1]
        if left <= 0:
            break
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cx, cy = x0 + W * 0.35, (y0 + y1) / 2
    rmax = min(W * 0.32, H * 0.46)
    cv.ellipse(cx - 34, cy - 34, 68, 68, step, "#ffc9c9", "#c92a2a", 2)
    cv.text(cx, cy, f"{z}p\n{NEUTRONS[z - 1]}n", step, 17, INK, "center", "middle", math=False)
    step += 1
    for s, n in enumerate(shells):
        r = 34 + (rmax - 34) * (s + 1) / len(shells)
        cv.ellipse(cx - r, cy - r, 2 * r, 2 * r, step, "transparent", MUTED, 1.5)
        for e in range(n):
            a = 2 * math.pi * e / n + s * 0.4
            cv.ellipse(cx + r * math.cos(a) - 9, cy + r * math.sin(a) - 9, 18, 18, step, "#4dabf7", "#1971c2", 1.5)
        step += 1
    cv.text(x0 + W * 0.72, cy - 80, f"{NAMES[z - 1]} ({ELEMENTS[z - 1]})", step, 30, "#e8590c", math=False)
    cv.text(x0 + W * 0.72, cy - 30, f"atomic number {z}\n{z} protons, {NEUTRONS[z - 1]} neutrons\n"
                                    f"electrons: {'-'.join(map(str, shells))}\nvalence electrons: {shells[-1]}",
            step, 21, INK, math=False)
    return step + 1


MOLECULES = {
    "H2O": ([("O", 0, 0), ("H", -1, 0.8), ("H", 1, 0.8)], [(0, 1, 1), (0, 2, 1)], "water: bent, 104.5°"),
    "CO2": ([("C", 0, 0), ("O", -1.4, 0), ("O", 1.4, 0)], [(0, 1, 2), (0, 2, 2)], "carbon dioxide: linear, two double bonds"),
    "CH4": ([("C", 0, 0), ("H", 0, -1.2), ("H", 1.2, 0), ("H", 0, 1.2), ("H", -1.2, 0)], [(0, i, 1) for i in range(1, 5)],
            "methane: tetrahedral (drawn flat)"),
    "NH3": ([("N", 0, 0), ("H", -1.1, 0.8), ("H", 1.1, 0.8), ("H", 0, 1.3)], [(0, 1, 1), (0, 2, 1), (0, 3, 1)],
            "ammonia: trigonal pyramidal"),
    "O2": ([("O", -0.7, 0), ("O", 0.7, 0)], [(0, 1, 2)], "oxygen: double bond"),
    "N2": ([("N", -0.7, 0), ("N", 0.7, 0)], [(0, 1, 3)], "nitrogen: triple bond"),
    "HCl": ([("H", -0.7, 0), ("Cl", 0.7, 0)], [(0, 1, 1)], "hydrogen chloride: polar single bond"),
    "NaCl": ([("Na", -0.8, 0), ("Cl", 0.8, 0)], [(0, 1, 0)], "sodium chloride: ionic bond, Na+ and Cl-"),
    "C2H6": ([("C", -0.7, 0), ("C", 0.7, 0), ("H", -1.4, -1), ("H", -1.4, 1), ("H", -1.9, 0), ("H", 1.4, -1),
              ("H", 1.4, 1), ("H", 1.9, 0)], [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (1, 5, 1), (1, 6, 1), (1, 7, 1)],
             "ethane"),
}
ATOM_COL = {"H": "#f1f3f5", "O": "#ff8787", "C": "#868e96", "N": "#74c0fc", "Cl": "#8ce99a", "Na": "#d0bfff"}


def build_molecule(cv, v, box, step):
    name = str(v.get("formula") or "H2O").replace(" ", "")
    if name not in MOLECULES:
        return step
    atoms, bonds, desc = MOLECULES[name]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cx, cy, s = x0 + W * 0.35, (y0 + y1) / 2, min(W, H) * 0.22
    P = [(cx + a[1] * s, cy + a[2] * s) for a in atoms]
    for a, b, order in bonds:                          # bonds first, then atoms on top
        (ax, ay), (bx, by) = P[a], P[b]
        dx, dy = by - ay, ax - bx
        L = math.hypot(dx, dy) or 1
        dx, dy = dx / L * 7, dy / L * 7
        if order == 0:
            cv.line([(ax, ay), (bx, by)], step, MUTED, 2, dashed=True)
        for k in range(order):
            off = (k - (order - 1) / 2)
            cv.line([(ax + dx * off, ay + dy * off), (bx + dx * off, by + dy * off)], step, INK, 3)
        step += 1
    for (sym, *_), (x, y) in zip(atoms, P):
        r = 38 if sym != "H" else 28
        cv.ellipse(x - r, y - r, 2 * r, 2 * r, step, ATOM_COL.get(sym, "#ffec99"), INK, 2)
        cv.text(x, y, sym + ("+" if name == "NaCl" and sym == "Na" else "-" if name == "NaCl" and sym == "Cl" else ""),
                step, 24, INK, "center", "middle", math=False)
    step += 1
    pretty = re.sub(r"(\d)", lambda m: "₀₁₂₃₄₅₆₇₈₉"[int(m.group(1))], name)
    cv.text(x0 + W * 0.68, cy - 40, pretty, step, 40, "#e8590c", math=False)
    d, sz = fit_text(desc, W * 0.3, 120, 22, 15)
    cv.text(x0 + W * 0.68, cy + 20, d, step, sz, INK, math=False)
    return step + 1


def build_circuit(cv, v, box, step):
    volts = float(v.get("voltage") or 12)
    rs = [float(r) for r in (v.get("resistors") or [2, 4])][:4]
    par = str(v.get("arrangement") or "series").lower().startswith("par")
    rt = 1 / sum(1 / r for r in rs) if par else sum(rs)
    i_tot = volts / rt
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    L, T, R, B = x0 + 60, y0 + 40, x0 + W * 0.62, y1 - 40
    if par:                                          # the rails end at the last branch
        R = L + (R - L) * len(rs) / (len(rs) + 1)
    # battery on the left
    cv.line([(L, T), (L, (T + B) / 2 - 22)], step, INK, 3)
    cv.line([(L - 26, (T + B) / 2 - 22), (L + 26, (T + B) / 2 - 22)], step, INK, 4)
    cv.line([(L - 14, (T + B) / 2 + 2), (L + 14, (T + B) / 2 + 2)], step, INK, 4)
    cv.line([(L, (T + B) / 2 + 2), (L, B)], step, INK, 3)
    cv.text(L - 36, (T + B) / 2 - 10, f"{volts:g} V", step, 20, "#e03131", "right", "middle", math=False)
    cv.line([(L, T), (R, T)], step, INK, 3)
    cv.line([(L, B), (R, B)], step, INK, 3)
    step += 1

    def resistor(x, y, horiz, label):
        n, a = 6, 14
        if horiz:
            pts = [(x - 50, y)] + [(x - 36 + k * 12, y + (a if k % 2 else -a)) for k in range(n)] + [(x + 50, y)]
        else:
            pts = [(x, y - 50)] + [(x + (a if k % 2 else -a), y - 36 + k * 12) for k in range(n)] + [(x, y + 50)]
        _poly(cv, pts, step, "#e8590c", "transparent", 3, smooth=False)
        cv.text(x + (0 if horiz else 26), y + (-26 if horiz else 0), label, step, 19, "#e8590c",
                "center" if horiz else "left", "bottom" if horiz else "middle", math=False)
    if par:
        cv.line([(R, T), (R, B)], step, "transparent", 1)
        xs = [L + (R - L) * (k + 1) / len(rs) for k in range(len(rs))]
        for x, r in zip(xs, rs):
            cv.line([(x, T), (x, (T + B) / 2 - 50)], step, INK, 3)
            resistor(x, (T + B) / 2, False, f"{r:g} Ω  ({volts / r:.2f} A)")
            cv.line([(x, (T + B) / 2 + 50), (x, B)], step, INK, 3)
            step += 1
    else:
        cv.line([(R, T), (R, B)], step, INK, 3)
        xs = [L + (R - L) * (k + 1) / (len(rs) + 1) for k in range(len(rs))]
        for x, r in zip(xs, rs):
            cv.rect(x - 52, T - 4, 104, 8, step, "#ffffff", "#ffffff", 1, rounded=False)
            resistor(x, T, True, f"{r:g} Ω  ({i_tot * r:.2f} V)")
            step += 1
    # current flowing: dots moving round the loop
    path = [(L, B), (L, T), (R, T), (R, B), (L, B)]
    per = sum(math.dist(path[i], path[i + 1]) for i in range(4))
    for k in range(8):
        until = step + 1 if k < 7 else None
        t = _Track(cv)
        for d in range(6):
            pos = (k / 8 + d / 6) * per % per
            for i in range(4):
                seg = math.dist(path[i], path[i + 1])
                if pos <= seg:
                    f = pos / seg
                    x = path[i][0] + (path[i + 1][0] - path[i][0]) * f
                    y = path[i][1] + (path[i + 1][1] - path[i][1]) * f
                    cv.ellipse(x - 6, y - 6, 12, 12, step, "#ffd43b", "#f08c00", 1)
                    break
                pos -= seg
        t.tag(until=until, fade=True)
        step += 1
    rr = " + ".join(f"{r:g}" for r in rs) if not par else "1/(" + " + ".join(f"1/{r:g}" for r in rs) + ")"
    cv.text(x0 + W * 0.68, T, f"{'Parallel' if par else 'Series'} circuit", step, 26, "#e8590c", math=False)
    cv.text(x0 + W * 0.68, T + 50, f"R total = {rr}\n        = {rt:.2f} Ω\nI = V / R = {volts:g} / {rt:.2f}\n   = {i_tot:.2f} A",
            step, 22, INK, math=False)
    return step + 1


def build_projectile(cv, v, box, step):
    v0 = float(v.get("speed") or 20)
    ang = math.radians(float(v.get("angle") or 45))
    g = 9.8
    T = 2 * v0 * math.sin(ang) / g
    Rng = v0 ** 2 * math.sin(2 * ang) / g
    Hmax = (v0 * math.sin(ang)) ** 2 / (2 * g)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    sx = (W * 0.62) / Rng
    sy = (H - 80) / max(Hmax, 1e-6) * 0.9
    s = min(sx, sy * 1.0)
    gx, gy = x0 + 30, y1 - 30
    cv.line([(gx, gy), (gx + Rng * s + 30, gy)], step, INK, 2)
    cv.arrow(gx, gy, gx + 70 * math.cos(ang), gy - 70 * math.sin(ang), step, color="#1971c2")
    cv.text(gx + 80, gy - 20, f"v₀ = {v0:g} m/s", step, 19, "#1971c2", "left", "bottom", math=False)
    step += 1
    n = 12
    for k in range(n + 1):
        t = T * k / n
        x, y = v0 * math.cos(ang) * t, v0 * math.sin(ang) * t - g * t * t / 2
        until = step + 1 if k < n else None
        tr = _Track(cv)
        cv.ellipse(gx + x * s - 10, gy - y * s - 10, 20, 20, step, "#e03131", "#c92a2a", 2)
        tr.tag(until=until, fade=True)
        cv.ellipse(gx + x * s - 5, gy - y * s - 5, 10, 10, step, "#ffc9c9", "#e03131", 1)
        step += 1
    cv.text(x0 + W * 0.7, y0 + 20, f"angle {math.degrees(ang):g}°\nflight time = {T:.2f} s\nrange = {Rng:.1f} m\n"
                                   f"max height = {Hmax:.1f} m\n(g = 9.8 m/s², no air)", step, 22, INK, math=False)
    return step + 1


def build_free_body(cv, v, box, step):
    forces = v.get("forces") or [{"label": "weight mg", "dir": "down"}, {"label": "normal N", "dir": "up"},
                                 {"label": "push F", "dir": "right"}, {"label": "friction f", "dir": "left"}]
    x0, y0, x1, y1 = box
    cx, cy = x0 + (x1 - x0) * 0.4, (y0 + y1) / 2
    cv.line([(cx - 260, cy + 60), (cx + 260, cy + 60)], step, INK, 3)
    cv.rect(cx - 60, cy - 60, 120, 120, step, "#ffd8a8")
    step += 1
    dirs = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
    for k, f in enumerate(forces[:6]):
        f = f if isinstance(f, dict) else {"label": str(f), "dir": "right"}
        dx, dy = dirs.get(str(f.get("dir", "right")).lower(), (1, 0))
        L = 140 * float(f.get("size", 1))
        cv.arrow(cx + dx * 60, cy + dy * 60, cx + dx * (60 + L), cy + dy * (60 + L), step, color=STRONG[k])
        cv.text(cx + dx * (70 + L), cy + dy * (70 + L), str(f.get("label")), step, 20, STRONG[k],
                "left" if dx >= 0 else "right", "middle", math=False)
        step += 1
    return step


# ======================================================================== cloud / devops / GIS / networks

def build_nested(cv, v, box, step):
    """{"type":"nested","title":"Kubernetes cluster","groups":[{"label":"Node 1","items":["Pod: web","Pod: api"]}, ...],
        "outer_icon":"kubernetes"}"""
    groups = [g for g in (v.get("groups") or []) if isinstance(g, dict)][:4]
    if not groups:
        return step
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cv.rect(x0, y0, W, H, step, "#f8f9fa", "#495057", 3)
    if v.get("outer_icon") and cv.icons:
        cv.icon(v["outer_icon"], x0 + 10, y0 + 6, 40, step)
    cv.text(x0 + 60, y0 + 14, str(v.get("title") or ""), step, 24, INK, math=False)
    step += 1
    gw = (W - 30 - 20 * (len(groups) - 1)) / len(groups)
    for gi, g in enumerate(groups):
        gx, gy = x0 + 15 + gi * (gw + 20), y0 + 60
        gh = H - 75
        cv.rect(gx, gy, gw, gh, step, "#ffffff", STRONG[gi % len(STRONG)], 2.5)
        if g.get("icon") and cv.icons:
            cv.icon(g["icon"], gx + 8, gy + 6, 34, step)
        cv.text(gx + 48, gy + 12, str(g.get("label", "")), step, 20, STRONG[gi % len(STRONG)], math=False)
        step += 1
        items = g.get("items") or []
        ih = min(90, (gh - 60) / max(1, len(items)))
        for ii, it in enumerate(items[:6]):
            iy = gy + 50 + ii * ih
            lab = it if isinstance(it, str) else it.get("label", "")
            icon = None if isinstance(it, str) else it.get("icon")
            cv.rect(gx + 12, iy, gw - 24, ih - 10, step, PAL[(gi + ii) % len(PAL)], INK, 1.5)
            placed = False
            if cv.icons:
                placed = cv.icon(icon or lab, gx + 18, iy + (ih - 10) / 2 - 18, 36, step)
            cv.label_in(gx + (60 if placed else 12), iy, gw - (72 if placed else 24), ih - 10, lab, step, 18)
            step += 1
    return step


def build_gis_layers(cv, v, box, step):
    layers = v.get("layers") or ["Basemap (imagery)", "Elevation raster", "Parcels (polygons)", "Roads (lines)",
                                 "Wells (points)"]
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    n = len(layers)
    lw, lh = W * 0.42, (H - 40) / (n + 1.2)
    rnd = random.Random(5)
    for i, name in enumerate(reversed(layers)):                  # bottom layer first
        k = n - 1 - i
        x, y = x0 + 40, y0 + 20 + k * lh * 0.9
        skew = lw * 0.25
        quad = [(x + skew, y), (x + skew + lw, y), (x + lw, y + lh), (x, y + lh)]
        low = name.lower()
        fill = "#d3f9d8" if "basemap" in low or "imagery" in low else "#fff3bf" if "raster" in low else "#ffffff"
        cv.polygon(quad, step, fill, INK, 2)
        cx, cy = x + skew / 2 + lw / 2, y + lh / 2
        if "point" in low or "well" in low:
            for _ in range(7):
                px, py = cx + rnd.uniform(-lw * 0.35, lw * 0.35), cy + rnd.uniform(-lh * 0.3, lh * 0.3)
                cv.ellipse(px - 5, py - 5, 10, 10, step, "#e03131", "#c92a2a", 1)
        elif "line" in low or "road" in low or "river" in low:
            for j in range(2):
                pts = [(x + skew * (1 - t) + lw * t, y + lh * (0.25 + 0.45 * j) + rnd.uniform(-6, 6)) for t in
                       [0.05, 0.3, 0.55, 0.8, 0.95]]
                cv.line(pts, step, "#495057", 3)
        elif "polygon" in low or "parcel" in low or "zone" in low:
            for j in range(3):
                px, py = x + skew / 2 + lw * (0.2 + j * 0.27), y + lh * 0.25
                cv.polygon([(px, py), (px + lw * 0.2, py + 4), (px + lw * 0.17, py + lh * 0.5), (px - 6, py + lh * 0.45)],
                           step, PAL[j], "#7048e8", 1.5)
        elif "raster" in low or "elevation" in low:
            for j in range(6):
                for kk in range(3):
                    cv.rect(x + skew * (1 - (kk + 0.5) / 3) + lw * (j / 6), y + lh * kk / 3, lw / 6 - 2, lh / 3 - 2, step,
                            ["#fff3bf", "#ffd8a8", "#ffa94d", "#e8590c"][(j + kk) % 4], "transparent", 1, rounded=False)
        cv.text(x + lw + skew + 30, y + lh / 2, name, step, 21, INK, "left", "middle", math=False)
        step += 1
    return step


OSI = [("Application", "data", "#e7f5ff"), ("Transport", "TCP header", "#fff3bf"), ("Network", "IP header", "#d3f9d8"),
       ("Data link", "frame header", "#ffe8cc"), ("Physical", "bits 0101", "#f3f0ff")]


def build_osi(cv, v, box, step):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    lh = (H - 20) / len(OSI)
    for i, (layer, hdr, col) in enumerate(OSI):
        y = y0 + i * lh
        cv.rect(x0, y, W * 0.22, lh - 8, step, col)
        cv.label_in(x0, y, W * 0.22, lh - 8, layer, step, 20)
        # the unit at this layer: headers in front of the data
        parts = [h for _, h, _ in OSI[1:i + 1]][::-1] + ["data"] if 0 < i < 4 else (["data"] if i == 0 else ["0 1 1 0 1 0 0 1 ..."])
        if i == 3:
            parts = parts + ["trailer"]
        px = x0 + W * 0.26
        for p in parts:
            pw = max(80, text_size(p, 18)[0] + 24)
            fill = next((c for _, h, c in OSI if h == p), "#e7f5ff")
            cv.rect(px, y + 6, pw, lh - 20, step, fill)
            cv.label_in(px, y + 6, pw, lh - 20, p, step, 17)
            px += pw + 4
        name = ["message", "segment", "packet", "frame", "bits"][i]
        cv.text(px + 16, y + (lh - 8) / 2, f"→ {name}", step, 19, "#e8590c", "left", "middle", math=False)
        step += 1
    return step


def draw_cisco(cv, kind, cx, cy, s, step):
    """Cisco-style symbols: router (round, cross arrows), switch (box, 4 arrows), pc, server, firewall, cloud."""
    if kind == "router":
        cv.ellipse(cx - s * 0.45, cy - s * 0.12, s * 0.9, s * 0.32, step, "#1c7ed6", "#1864ab", 2)
        cv.rect(cx - s * 0.45, cy - s * 0.27, s * 0.9, s * 0.3, step, "#1c7ed6", "transparent", 1, rounded=False)
        cv.ellipse(cx - s * 0.45, cy - s * 0.42, s * 0.9, s * 0.32, step, "#4dabf7", "#1864ab", 2)
        for dx, dy in ((-1, 0), (1, 0), (0, -0.4), (0, 0.4)):
            cv.arrow(cx + dx * s * 0.08, cy - s * 0.26 + dy * s * 0.08, cx + dx * s * 0.3, cy - s * 0.26 + dy * s * 0.3,
                     step, color="#ffffff")
        return (cx - s * 0.45, cy - s * 0.42, s * 0.9, s * 0.62)
    if kind == "switch":
        cv.rect(cx - s * 0.5, cy - s * 0.3, s, s * 0.45, step, "#1c7ed6", "#1864ab", 2, rounded=False)
        for k, (y, d) in enumerate(((cy - s * 0.18, 1), (cy - s * 0.04, -1))):
            for side in (-1, 1):
                x = cx + side * s * 0.12
                cv.arrow(x - d * s * 0.12, y, x + d * s * 0.12, y, step, color="#ffffff")
        return (cx - s * 0.5, cy - s * 0.3, s, s * 0.45)
    if kind == "pc":
        return draw_device(cv, "laptop", cx, cy, s, step)
    return draw_device(cv, kind, cx, cy, s, step)


def _cisco_kind(label):
    low = label.lower()
    if re.search(r"router|\br\d\b|gateway", low):
        return "router"
    if re.search(r"switch|\bsw\d?\b", low):
        return "switch"
    if re.search(r"firewall|asa", low):
        return "firewall"
    if re.search(r"server|dns|dhcp|web|srv", low):
        return "server"
    if re.search(r"cloud|internet|isp|wan", low):
        return "cloud"
    if re.search(r"phone|mobile", low):
        return "phone"
    return "pc"


def build_topology(cv, v, box, step):
    """{"type":"topology","nodes":[{"id":"R1","label":"R1 router"},...],
        "links":[{"from":"PC1","to":"SW1","label":"Fa0/1"}], "ips":{"PC1":"192.168.1.10"},
        "path":["PC1","SW1","R1","Server"]}"""
    nodes = [n if isinstance(n, dict) else {"id": str(n), "label": str(n)} for n in (v.get("nodes") or [])][:10]
    if len(nodes) < 2:
        return step
    links = [l for l in (v.get("links") or v.get("edges") or []) if isinstance(l, dict)]
    ids = [str(n.get("id") or n.get("label")) for n in nodes]
    # layers: by BFS from first node
    adj = {i: [] for i in ids}
    for l in links:
        a, b = str(l.get("from")), str(l.get("to"))
        if a in adj and b in adj:
            adj[a].append(b)
            adj[b].append(a)
    level, q = {ids[0]: 0}, [ids[0]]
    while q:
        u = q.pop(0)
        for w in adj[u]:
            if w not in level:
                level[w] = level[u] + 1
                q.append(w)
    for i in ids:
        level.setdefault(i, max(level.values()) + 1)
    cols = {}
    for i in ids:
        cols.setdefault(level[i], []).append(i)
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    nc = len(cols)
    s = min(110, W / nc * 0.5, H / max(len(c) for c in cols.values()) * 0.45)
    pos = {}
    for c, members in cols.items():
        for r, i in enumerate(members):
            pos[i] = (x0 + W * (c + 0.5) / nc, y0 + 20 + (H - 70) * (r + 0.5) / len(members))
    for l in links:                                   # cables first
        a, b = str(l.get("from")), str(l.get("to"))
        if a in pos and b in pos:
            cv.line([pos[a], pos[b]], step, "#212529", 2)
            if l.get("label"):
                mx, my = (pos[a][0] * 0.7 + pos[b][0] * 0.3), (pos[a][1] * 0.7 + pos[b][1] * 0.3)
                cv.text(mx, my - 4, str(l["label"]), step, 15, "#1864ab", "center", "bottom", math=False)
    step += 1
    ips = v.get("ips") or {}
    for n, i in zip(nodes, ids):
        x, y = pos[i]
        b = draw_cisco(cv, n.get("kind") or _cisco_kind(str(n.get("label") or i)), x, y, s, step)
        cv.text(x, b[1] + b[3] + 8, str(n.get("label") or i), step, 18, INK, "center", "top", math=False)
        if ips.get(i):
            cv.text(x, b[1] + b[3] + 30, str(ips[i]), step, 15, "#2f9e44", "center", "top", math=False)
    step += 1
    path = [str(p) for p in (v.get("path") or []) if str(p) in pos]
    for h in range(len(path) - 1):
        (ax, ay), (bx, by) = pos[path[h]], pos[path[h + 1]]
        for k in range(3):
            f = k / 3
            last = h == len(path) - 2 and k == 2
            _envelope(cv, ax + (bx - ax) * (f + 0.17), ay + (by - ay) * (f + 0.17) - s * 0.55, s * 0.36, step,
                      until=None if last else step + 1)
            step += 1
    return step


def build_asm_trace(cv, v, box, step):
    import drawlib
    code = str(v.get("code") or "")
    steps = run_asm(code)
    if not steps:
        return step
    return drawlib.build_code_trace(cv, {"code": code, "steps": steps}, box, step)


# ======================================================================== register

visuals.BUILDERS.update({
    "gantt": build_gantt, "graph_algo": build_graph_algo, "ds_ops": build_ds_ops, "pipeline": build_pipeline,
    "hierarchy": build_hierarchy, "gpu_grid": build_gpu_grid, "logic": build_logic, "ast": build_ast,
    "sql": build_sql, "neural_net": build_neural_net, "gradient": build_gradient, "kmeans": build_kmeans,
    "attention": build_attention, "atom": build_atom, "molecule": build_molecule, "circuit": build_circuit,
    "projectile": build_projectile, "free_body": build_free_body, "nested": build_nested,
    "gis_layers": build_gis_layers, "osi": build_osi, "topology": build_topology,
    "asm_trace": build_asm_trace,
})
for _k in [k for k in list(visuals.ALIASES) if k in visuals.BUILDERS]:   # a real drawing beats an old alias
    visuals.ALIASES.pop(_k)
visuals.ALIASES.update({
    "scheduling": "gantt", "cpu_scheduling": "gantt", "bfs": "graph_algo", "dfs": "graph_algo", "dijkstra": "graph_algo",
    "stack_ops": "ds_ops", "queue_ops": "ds_ops", "linked_list": "ds_ops", "cpu_pipeline": "pipeline",
    "memory_hierarchy": "hierarchy", "cuda": "gpu_grid", "logic_gates": "logic", "gate": "logic",
    "syntax_tree": "ast", "parse_tree": "ast", "query": "sql", "neural_network": "neural_net", "mlp": "neural_net",
    "gradient_descent": "gradient", "clustering": "kmeans", "transformer": "attention", "bohr": "atom",
    "chemistry": "molecule", "ohms_law": "circuit", "forces": "free_body", "containers": "nested",
    "kubernetes": "nested", "docker": "nested", "groups": "nested", "layers_gis": "gis_layers", "gis": "gis_layers",
    "encapsulation": "osi", "assembly": "asm_trace", "asm": "asm_trace", "network_topology": "topology", "cisco": "topology",
})
