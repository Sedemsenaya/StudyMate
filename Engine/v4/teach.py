"""
StudyMate v4 - AI teaching pass for ANY lesson (slides, notes, textbooks).

The practice-test idea, applied to every PDF:
  1. rules build the lesson first (instant, the source's words, nothing lost)
  2. pictures are kept: slide diagrams are cropped (extract.py), and pictures in
     documents are added as "figure" scenes at their page
  3. a vision model describes each picture in 1-2 sentences (cached) and the
     sentence becomes that scene's narration
  4. every few scenes, ONE short AI call (thinking off, JSON schema, <=450 tokens,
     time limit, cached) draws a NEW teaching scene: the idea of those scenes as a
     picture (architecture with icons, flow, cycle, comparison table, timeline,
     worked equation or tree) plus a plain explanation and a takeaway.
     A slow or bad answer just means no extra scene there; the run never stalls.
"""
import json
import re
import time

import drawlib     # noqa: F401  (registers cell, mitosis, packet_flow, sequence, code_trace)
import drawlib2    # noqa: F401  (gantt, graph_algo, sql, neural_net, circuit, topology, nested ...)
import drawlib3    # noqa: F401  (sorting, hashing, trees, security, data, biology, physics, maths ...)
import drawlib4    # noqa: F401  (DP, Huffman, caches, JWT, CI/CD, decision trees, genetics, optics, finance ...)
import drawlib7    # noqa: F401  (A*, K-maps, MIPS, VPC, IAM, Raft, RAG, PCR, gels, Gini ...)
import drawlib6    # noqa: F401  (MST, KMP, B-trees, pipelines, Kafka, Terraform, RSA, regression, titration ...)
import drawlib5    # noqa: F401  (quick sort, RAID, IEEE 754, NAT, backprop, ROC, joins, sieve, gas law, GIS ...)
import illustrate  # noqa: F401  (registers the arch/choices visuals)
from tracer import default_call, find_code, trace_python

TEACH_SYSTEM_FULL = """You are StudyMate's tutor-illustrator. You get a few slides or paragraphs of a
lesson. Draw ONE new picture that makes their main idea easier to understand than
the original, and explain it simply. Choose the visual type that SHOWS the thing itself (the engine draws it):
  cell        a cell and its organelles: kind "animal"|"plant", parts ["nucleus","mitochondria",...]
  mitosis     cell division, drawn phase by phase: phases ["prophase","metaphase",...]
  packet_flow data/packets travelling between devices: nodes ["Laptop","Router","Server"],
              hops [{"from","to","label"}] in order (requests and replies)
  sequence    a protocol exchange over time: actors [...], messages [{"from","to","label"}]
  code_trace  running code: code (short Python), call (e.g. "f([3,1,2])") - it is executed and traced
  arch        things/services/actors connected by arrows (servers, databases... are drawn)
  flow        ordered steps            cycle    steps that repeat
  table       comparison (columns + rows)   timeline dated events
  equation    formula + a worked example with real numbers      tree   a hierarchy
  More (fields in brackets; the engine computes the rest):
  gantt [algorithm fcfs|sjf|rr, quantum, processes:[{name,arrival,burst}]]   graph_algo [algorithm bfs|dfs|dijkstra,
  nodes, edges:[[a,b,weight]], start]   ds_ops [structure stack|queue|linked_list, ops:["push 3","pop"]]
  pipeline [instructions]   hierarchy   gpu_grid   logic [gate AND|OR|NOT|XOR|NAND|NOR]   asm_trace [code]
  ast [expr]   sql [tables:{name:{columns,rows}}, query]   neural_net [layers:[3,4,2]]   gradient   kmeans [k]
  attention [tokens, query]   atom [element]   molecule [formula H2O|CO2|CH4|NH3|O2|N2|HCl|NaCl|C2H6]
  circuit [voltage, resistors, arrangement series|parallel]   projectile [speed, angle]   free_body [forces:[{label,dir}]]
  nested [title, groups:[{label, items:[...]}]] for Docker/Kubernetes/VPC/Azure   gis_layers [layers]   osi
  topology [nodes, links:[{from,to,label}], ips:{node:ip}, path] for Cisco-style networks
  sort_bars [algorithm bubble|insertion|selection, array]   hash_table [keys, buckets]   bst [keys]   heap [keys]
  recursion [function factorial|fibonacci, n]   fsm [states, start, accept, transitions:{"q0,a":"q1"}, input]
  page_table [address, page_size, table]   mapreduce [documents]   git_graph   subnet [cidr]   binary [number]
  blockchain [blocks]   cipher [text, shift]   firewall [rules, packets]   load_balancer [servers, requests]   ddos
  layers_ring [layers] (shape "triangle" = CIA triad)   convolution [image, kernel]   confusion [tp, fn, fp, tn]
  stats [data]   bpe [words:{word:count}]   photosynthesis   dna [sequence]   central_dogma [dna]   heart
  water_cycle   neuron   states_matter   wave [wavelength, amplitude]   refraction [n1, n2, angle]   ph_scale
  solar_system   venn [sets:{A:[...],B:[...]}]   prob_tree [p, given:{"A":..,"not A":..}, names]
  matrix_mult [A, B]   riemann [expr, a, b]   vectors [vectors]   supply_demand [demand_a, demand_b, supply_c, supply_d]
  dp_table [a, b] (LCS)   merge_sort [array]   huffman [text]   trie [words]   topo_sort [nodes, edges]
  lru [refs, frames, policy lru|fifo]   cache_sim [addresses, lines, block]   tcp_cwnd [losses]   deadlock   race
  event_loop   twos_complement [number, bits]   jwt [payload]   sql_injection [input]   password_hash [password]
  cicd [stages]   autoscaling [load]   decision_tree [rows, label]   knn [k]   normal_dist [mean, sd]
  softmax [labels, logits]   overfitting   er_diagram [entities, relationships]   groupby [columns, rows]
  punnett [parent1, parent2, dominant, recessive]   meiosis   respiration   osmosis   plate_tectonics   moon_phases
  earth_layers   lens [focal, object, height]   pendulum [length]   energy_ramp [mass, height]
  balance [equation "CH4 + 2O2 -> CO2 + 2H2O"]   reaction_energy [kind exothermic|endothermic]   tangent [expr]
  compound [principal, rate, years]   linear_system [eq1:[a,b,c], eq2:[a,b,c]]   break_even [fixed, variable, price]
  gis_buffer [distance]
  quick_sort [array]   maze_bfs [grid]   bloom_filter [add, check]   consistent_hash [servers, keys]   list_reverse
  sliding_window [array, k]   memory_layout   fork_tree [forks]   disk_sched [requests, head, algorithm]
  raid [level 0|1|5]   cpu_cycle   ieee754 [number]   utf8 [text]   nat_table   routing_lpm [routes, destinations]
  traceroute [hops]   buffer_overflow   brute_force   backprop [x, w, b, target]   roc_curve   grid_world   pca
  word_vectors   transformer_block   sql_joins [left, right]   normalization   star_schema   moving_average [window]
  correlation   sieve [n]   euclid_gcd [a, b]   pascal [rows]   binomial [n, p]
  bayes_test [prevalence, sensitivity, specificity]   monte_carlo_pi   collision [m1, u1, m2, u2, kind]
  incline [mass, angle, mu]   gas_law   half_life [half_life, isotope]   doppler   electron_config [element]
  food_web   energy_pyramid   natural_selection   seasons   greenhouse   rock_cycle   map_projection   voronoi
  choropleth [values]   amortization [principal, rate, years]   ppf
  kruskal [edges]   kmp [pattern, text]   n_queens [n]   avl [insert]   big_o   b_tree [insert, search]   cpu_pipeline
  producer_consumer [size]   go_back_n [packets, window, lost]   ip_header [src, dst]   vpn_tunnel   cdn [requests]
  acid_transaction   docker_layers   rolling_update [replicas]   terraform_plan   kafka_partitions [messages, partitions]
  virtual_dom   token_bucket [capacity, refill]   diffie_hellman [p, g, a, b]   rsa [p, q, e, message]
  linear_regression [points]   logistic_regression   rnn_unrolled [words]   cross_validation [folds]   tfidf [docs]
  perceptron   boxplot [data]   histogram [data]   central_limit [n]   hypothesis_test [mu0, mean, sd, n, alpha]
  taylor_series   fourier_series   newtons_method [start]   eigen_transform [matrix]   number_bases [number]
  modular_clock [modulus, a, b]   spring_shm [k, mass, amplitude]   lever_torque [m1, d1, m2]   buoyancy   electric_field
  em_spectrum   titration   limiting_reagent [h2, o2]   vsepr   enzyme [vmax, km]   action_potential   hardy_weinberg [p]
  population_growth [r, K, N0]   idw_interpolation   topographic   raster_vector   spatial_join   npv [cash_flows, rate]
  elasticity [p1, p2]
  a_star   floyd_warshall   radix_sort   interval_scheduling   coin_change   lexer   gc_mark_sweep   branch_predictor   amdahl   ripple_adder   karnaugh   endianness   mips_encoding   memory_hierarchy   bankers   memory_alloc   distance_vector   hamming_code   crc   manchester   acl_rules   vlan   aws_vpc   iam_policy
  circuit_breaker   raft_election   sharding   merkle_tree   port_scan   cert_chain   activations   naive_bayes   dbscan   dendrogram   cosine_similarity   temperature_sampling   positional_encoding   learning_rate   rag_pipeline   window_functions   standardization   dot_product   complex_plane   quadratic   sequences   markov_chain   truth_table   permutations
  euler_path   rc_circuit   standing_waves   double_slit   circular_motion   kepler_orbit   heating_curve   photoelectric   maxwell_boltzmann   galvanic_cell   pcr   gel_electrophoresis   phylogenetic_tree   feedback_loop   ndvi   epicenter   slope_hillshade   lorenz_gini   prisoners_dilemma   comparative_advantage
  (optional fields for these: array, coins+amount, code, outcomes, a+b bits, minterms, value, instructions, allocation,
   blocks+requests, data bits, message+generator, bits, acl+packets, statements+requests, calls, expression, items+k,
   R+C+V, wavelength, eccentricity, mass, anode+cathode, cycles, samples, distances, income_shares, payoffs, hours)
Rules: labels 1-4 words, at most 7 nodes/steps/rows. Only facts from the text;
examples must be correct. explain: 2-3 short sentences (max 60 words).
takeaway: max 12 words. title: max 6 words. Answer with JSON only."""

TEACH_SYSTEM = """You are StudyMate's tutor-illustrator. You get a few slides or paragraphs of a
lesson. Explain the main idea simply and pick ONE picture for it.
topic: 2-6 words naming the exact concept (e.g. "Dijkstra shortest path", "TCP three-way handshake",
"mitosis phases", "Punnett square", "RSA encryption"). The engine has 300 exact drawings and picks one from it.
visual: only if no exact drawing fits, one of:
  sequence  actors [...], messages [{"from","to","label"}] (a protocol / conversation over time)
  arch      nodes [{"id","label"}], edges [{"from","to","label"}] (things connected by arrows)
  flow      steps [...] (ordered steps)      cycle  steps [...] (steps that repeat)
  table     columns [...], rows [[...]] (a comparison)      timeline  events [{"date","label"}]
  tree      root {"label","children":[...]} (a hierarchy)
  equation  lines [{"math","note"}] (formula + worked example with real numbers)
  code_trace code (short Python), call (e.g. "f([3,1,2])")
Rules: labels 1-4 words, at most 7 steps/nodes/rows. Only facts from the text.
explain: 2-3 short sentences (max 60 words). takeaway: max 12 words. title: max 6 words. JSON only."""

_NODE = {"type": "object", "properties": {"id": {"type": "string"}, "label": {"type": "string"},
                                          "highlight": {"type": "boolean"}}, "required": ["id", "label"]}
_EDGE = {"type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"},
                                          "label": {"type": "string"}}, "required": ["from", "to"]}
MORE_TYPES = ['a_star', 'floyd_warshall', 'radix_sort', 'interval_scheduling', 'coin_change', 'lexer', 'gc_mark_sweep', 'branch_predictor', 'amdahl', 'ripple_adder', 'karnaugh', 'endianness', 'mips_encoding', 'memory_hierarchy', 'bankers', 'memory_alloc', 'distance_vector', 'hamming_code', 'crc', 'manchester', 'acl_rules', 'vlan', 'aws_vpc', 'iam_policy', 'circuit_breaker', 'raft_election', 'sharding', 'merkle_tree', 'port_scan', 'cert_chain', 'activations', 'naive_bayes', 'dbscan', 'dendrogram', 'cosine_similarity', 'temperature_sampling', 'positional_encoding', 'learning_rate', 'rag_pipeline', 'window_functions', 'standardization', 'dot_product', 'complex_plane', 'quadratic', 'sequences', 'markov_chain', 'truth_table', 'permutations', 'euler_path', 'rc_circuit', 'standing_waves', 'double_slit', 'circular_motion', 'kepler_orbit', 'heating_curve', 'photoelectric', 'maxwell_boltzmann', 'galvanic_cell', 'pcr', 'gel_electrophoresis', 'phylogenetic_tree', 'feedback_loop', 'ndvi', 'epicenter', 'slope_hillshade', 'lorenz_gini', 'prisoners_dilemma', 'comparative_advantage',
              "kruskal", "kmp", "n_queens", "avl", "big_o", "b_tree", "cpu_pipeline", "producer_consumer", "go_back_n",
              "ip_header", "vpn_tunnel", "cdn", "acid_transaction", "docker_layers", "rolling_update", "terraform_plan",
              "kafka_partitions", "virtual_dom", "token_bucket", "diffie_hellman", "rsa", "linear_regression",
              "logistic_regression", "rnn_unrolled", "cross_validation", "tfidf", "perceptron", "boxplot", "histogram",
              "central_limit", "hypothesis_test", "taylor_series", "fourier_series", "newtons_method", "eigen_transform",
              "number_bases", "modular_clock", "spring_shm", "lever_torque", "buoyancy", "electric_field", "em_spectrum",
              "titration", "limiting_reagent", "vsepr", "enzyme", "action_potential", "hardy_weinberg",
              "population_growth", "idw_interpolation", "topographic", "raster_vector", "spatial_join", "npv", "elasticity",
              "quick_sort", "maze_bfs", "bloom_filter", "consistent_hash", "list_reverse", "sliding_window",
              "memory_layout", "fork_tree", "disk_sched", "raid", "cpu_cycle", "ieee754", "utf8", "nat_table",
              "routing_lpm", "traceroute", "buffer_overflow", "brute_force", "backprop", "roc_curve", "grid_world", "pca",
              "word_vectors", "transformer_block", "sql_joins", "normalization", "star_schema", "moving_average",
              "correlation", "sieve", "euclid_gcd", "pascal", "binomial", "bayes_test", "monte_carlo_pi", "collision",
              "incline", "gas_law", "half_life", "doppler", "electron_config", "food_web", "energy_pyramid",
              "natural_selection", "seasons", "greenhouse", "rock_cycle", "map_projection", "voronoi", "choropleth",
              "amortization", "ppf",
              "dp_table", "merge_sort", "huffman", "trie", "topo_sort", "lru", "cache_sim", "tcp_cwnd", "deadlock",
              "race", "event_loop", "twos_complement", "jwt", "sql_injection", "password_hash", "cicd", "autoscaling",
              "decision_tree", "knn", "normal_dist", "softmax", "overfitting", "er_diagram", "groupby", "punnett",
              "meiosis", "respiration", "osmosis", "plate_tectonics", "moon_phases", "earth_layers", "lens", "pendulum",
              "energy_ramp", "balance", "reaction_energy", "tangent", "compound", "linear_system", "break_even",
              "gis_buffer",
              "sort_bars", "hash_table", "bst", "heap", "recursion", "fsm", "page_table", "mapreduce", "git_graph",
              "subnet", "binary", "blockchain", "cipher", "firewall", "load_balancer", "ddos", "layers_ring",
              "convolution", "confusion", "stats", "bpe", "photosynthesis", "dna", "central_dogma", "heart",
              "water_cycle", "neuron", "states_matter", "wave", "refraction", "ph_scale", "solar_system", "venn",
              "prob_tree", "matrix_mult", "riemann", "vectors", "supply_demand",
              "gantt", "graph_algo", "ds_ops", "pipeline", "hierarchy", "gpu_grid", "logic", "asm_trace", "ast", "sql",
              "neural_net", "gradient", "kmeans", "attention", "atom", "molecule", "circuit", "projectile",
              "free_body", "nested", "gis_layers", "osi", "topology"]
_CORE = ["sequence", "arch", "flow", "cycle", "table", "timeline", "tree", "equation", "code_trace"]
TEACH_SCHEMA_SMALL = {"type": "object", "properties": {
    "title": {"type": "string"}, "topic": {"type": "string"},
    "visual": {"type": "object", "properties": {"type": {"type": "string", "enum": _CORE}},
               "required": ["type"], "additionalProperties": True},
    "explain": {"type": "string"}, "takeaway": {"type": "string"}},
    "required": ["title", "topic", "explain", "takeaway"]}
TEACH_SCHEMA = {"type": "object", "properties": {
    "title": {"type": "string"},
    "visual": {"type": "object", "properties": {
        "type": {"type": "string", "enum": ["cell", "mitosis", "packet_flow", "sequence", "code_trace", "arch",
                                            "flow", "cycle", "table", "timeline", "equation", "tree"] + MORE_TYPES},
        "kind": {"type": "string"}, "parts": {"type": "array", "items": {"type": "string"}},
        "phases": {"type": "array", "items": {"type": "string"}},
        "hops": {"type": "array", "items": _EDGE},
        "actors": {"type": "array", "items": {"type": "string"}},
        "messages": {"type": "array", "items": _EDGE},
        "code": {"type": "string"}, "call": {"type": "string"},
        "nodes": {"type": "array", "items": _NODE}, "edges": {"type": "array", "items": _EDGE},
        "steps": {"type": "array", "items": {"type": "string"}},
        "columns": {"type": "array", "items": {"type": "string"}},
        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
        "events": {"type": "array", "items": {"type": "object", "properties": {
            "date": {"type": "string"}, "label": {"type": "string"}}}},
        "lines": {"type": "array", "items": {"type": "object", "properties": {
            "math": {"type": "string"}, "note": {"type": "string"}}}},
        "root": {"type": "object"}},
        "required": ["type"], "additionalProperties": True},
    "explain": {"type": "string"}, "takeaway": {"type": "string"}},
    "required": ["title", "visual", "explain", "takeaway"]}


STUDENT_SYSTEM = ("You are StudyMate. Read the lesson text and answer with JSON: topic (2-6 words), "
                  "title (max 6 words), visual (one StudyMate drawing with its fields), explain (a tutor's "
                  "3 short sentences with a concrete example), takeaway (max 12 words).")
STUDENT_SCHEMA = {"type": "object", "properties": {
    "topic": {"type": "string"}, "title": {"type": "string"},
    "visual": {"type": "object", "properties": {"type": {"type": "string"}}, "required": ["type"],
               "additionalProperties": True},
    "explain": {"type": "string"}, "takeaway": {"type": "string"}},
    "required": ["topic", "title", "visual", "explain", "takeaway"]}


def _is_student(ai):
    """A StudyMate fine-tuned model (ollama create studymate-...) knows the formats: tiny prompt."""
    return "studymate" in str(getattr(ai, "model", "")).lower()


def _teach_call(ai, prompt, num_predict=320):
    student = _is_student(ai)

    def call():
        t = time.perf_counter()
        try:
            r = ai.client.chat(model=ai.model, messages=[
                {"role": "system", "content": STUDENT_SYSTEM if student else TEACH_SYSTEM},
                {"role": "user", "content": prompt}],
                format=STUDENT_SCHEMA if student else TEACH_SCHEMA_SMALL, think=False, keep_alive="30m",
                options={"temperature": 0.2, "num_predict": num_predict, "num_ctx": 3072})
            ai._count(r, t)
            txt = re.sub(r"<think>.*?</think>", "", r["message"]["content"], flags=re.S)
            return json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
        except Exception as ex:
            ai.log(f"[ai] teach call failed after {time.perf_counter() - t:.0f}s "
                   f"({type(ex).__name__}: {str(ex)[:80]}); no extra scene here")
            ai.stats["fallback"] += 1
            if "timeout" in type(ex).__name__.lower() or "timed out" in str(ex).lower():
                ai.timeouts = getattr(ai, "timeouts", 0) + 1
            return None
    if getattr(ai, "out_of_budget", lambda: False)():
        return None
    if getattr(ai, "timeouts", 0) >= 2:            # the machine is too slow for this model: stop wasting time
        if not getattr(ai, "_gave_up", False):
            ai._gave_up = True
            ai.log("[ai] 2 calls timed out: skipping the remaining AI scenes (try --model qwen3:4b or "
                   "--llm-parallel 1); the engine's drawings still run")
        return None
    out = ai._cached("teach2_" + ai.model.replace(":", "_"), (STUDENT_SYSTEM if student else TEACH_SYSTEM) + prompt,
                     call)
    if isinstance(out, dict) and not (student and _valid(out.get("visual"))):   # map the topic to an exact drawing
        rv = rules_visual(" ".join(str(out.get(k) or "") for k in ("topic", "title")))
        if rv:
            out = dict(out, visual=rv)
    return out


TUTOR_SYSTEM = """You are a patient tutor. The student is about to watch an animated drawing of: {what}.
Using only facts from the lesson text, explain the idea the drawing shows the way a good tutor would:
start from what the student already knows, give the intuition, then one concrete example with real
numbers or names from the text. explain: 3 short sentences (max 70 words). takeaway: max 12 words.
title: max 6 words. JSON only."""
TUTOR_SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}, "explain": {"type": "string"},
                                                 "takeaway": {"type": "string"}},
                "required": ["title", "explain", "takeaway"]}


def _tutor_call(ai, prompt, what, num_predict=170):
    """Short tutor narration for a drawing the engine already chose (no visual to invent: fast)."""
    if getattr(ai, "timeouts", 0) >= 2 or getattr(ai, "out_of_budget", lambda: False)():
        return None
    sysmsg = TUTOR_SYSTEM.format(what=what.replace("_", " "))

    def call():
        t = time.perf_counter()
        try:
            r = ai.client.chat(model=ai.model, messages=[
                {"role": "system", "content": sysmsg}, {"role": "user", "content": prompt[:1400]}],
                format=TUTOR_SCHEMA, think=False, keep_alive="30m",
                options={"temperature": 0.3, "num_predict": num_predict, "num_ctx": 2048})
            ai._count(r, t)
            txt = re.sub(r"<think>.*?</think>", "", r["message"]["content"], flags=re.S)
            return json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
        except Exception as ex:
            ai.log(f"[ai] tutor call failed after {time.perf_counter() - t:.0f}s ({type(ex).__name__})")
            ai.stats["fallback"] += 1
            if "timeout" in type(ex).__name__.lower() or "timed out" in str(ex).lower():
                ai.timeouts = getattr(ai, "timeouts", 0) + 1
            return None
    return ai._cached("tutor_" + ai.model.replace(":", "_"), sysmsg + prompt[:1400], call)


def _valid(v):
    """Keep only visuals the renderer can draw well."""
    if not isinstance(v, dict):
        return None
    t = v.get("type")
    if t == "cell":
        return {"type": "cell", "kind": v.get("kind") or "animal", "parts": v.get("parts") or []}
    if t == "mitosis":
        return {"type": "mitosis", "phases": v.get("phases") or []}
    if t == "packet_flow":
        nodes = [n.get("label") if isinstance(n, dict) else str(n) for n in v.get("nodes") or []]
        if len(nodes) >= 2:
            return {"type": "packet_flow", "nodes": nodes[:6], "hops": v.get("hops") or []}
    if t == "sequence":
        actors = [str(a) for a in v.get("actors") or []]
        if len(actors) >= 2 and v.get("messages"):
            return {"type": "sequence", "actors": actors[:5], "messages": v["messages"][:9]}
    if t in MORE_TYPES:                       # the engine computes and checks these itself
        return v
    if t == "code_trace" and v.get("code"):
        steps = trace_python(v["code"], v.get("call") or default_call(v["code"]))
        if steps:
            return {"type": "code_trace", "code": v["code"], "steps": steps}
        return None
    if t in ("arch",):
        nodes = [n for n in v.get("nodes") or [] if isinstance(n, dict) and n.get("label")]
        if 2 <= len(nodes) <= 9:
            return {"type": "arch", "nodes": nodes[:7], "edges": v.get("edges") or [],
                    "highlight_text": "★ key idea"}
    if t in ("flow", "cycle"):
        steps = [s for s in v.get("steps") or [] if isinstance(s, str) and s.strip()]
        if not steps and v.get("nodes"):
            steps = [n.get("label") for n in v["nodes"] if isinstance(n, dict) and n.get("label")]
        if 2 <= len(steps):
            return {"type": t, "steps": steps[:6]}
    if t == "table":
        cols = v.get("columns") or []
        rows = [r for r in v.get("rows") or [] if isinstance(r, list) and r]
        if cols and rows:
            return {"type": "table", "columns": cols[:4], "rows": [r[:len(cols)] for r in rows[:7]]}
    if t == "timeline" and v.get("events"):
        return {"type": "timeline", "events": v["events"][:7]}
    if t == "equation" and v.get("lines"):
        return {"type": "equation", "lines": v["lines"][:5]}
    if t == "tree" and isinstance(v.get("root"), dict):
        return {"type": "tree", "root": v["root"]}
    return None


# ---------------------------------------------------------------- rules: draw the real thing

ALGOS = {
    "binary search": """def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1""",
    "linear search": """def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1""",
    "bubble sort": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr""",
    "insertion sort": """def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr""",
    "selection sort": """def selection_sort(arr):
    for i in range(len(arr)):
        m = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[m]:
                m = j
        arr[i], arr[m] = arr[m], arr[i]
    return arr""",
}
ALGO_CALL = {"binary search": "binary_search([2, 5, 8, 12, 16, 23, 38], 23)",
             "linear search": "linear_search([7, 3, 9, 4, 6], 4)",
             "bubble sort": "bubble_sort([5, 1, 4, 2, 8])", "insertion sort": "insertion_sort([5, 2, 4, 6, 1])",
             "selection sort": "selection_sort([29, 10, 14, 37, 13])"}
ORGANELLE_WORDS = r"nucleus|mitochondri|ribosome|golgi|endoplasmic|lysosome|chloroplast|vacuole|cell wall|" \
                  r"plasma membrane|cell membrane|nucleolus|centriole"


def _seq(actors, msgs):
    return {"type": "sequence", "actors": actors, "messages": [{"from": a, "to": b, "label": l} for a, b, l in msgs]}


def rules_visual_more(text, low):
    """Topics added for the MS CEN / AWS / science list."""
    has = lambda rx: re.search(rx, low)
    m = re.search(r"\b(round robin|fcfs|first[- ]come|shortest job first|sjf)\b", low)
    if m or has(r"cpu scheduling|scheduling algorithm"):
        a = (m.group(1) if m else "fcfs")
        a = "rr" if "round" in a else "sjf" if "short" in a or a == "sjf" else "fcfs"
        return {"type": "gantt", "algorithm": a, "quantum": 2}
    for rx, algo in ((r"breadth[- ]first|\bbfs\b", "bfs"), (r"depth[- ]first|\bdfs\b", "dfs"),
                     (r"dijkstra|shortest path", "dijkstra")):
        if has(rx):
            return {"type": "graph_algo", "algorithm": algo}
    if has(r"\bmov\s+e?[abcd]x|\b(eax|ebx|ecx|edx)\b"):
        m = re.search(r"((?:^|\n)\s*(?:\w+:\s*)?(?:mov|add|sub|cmp|inc|dec|push|pop|jmp|jne|je)\b[\s\S]{10,600}?)(?:\n\s*\n|\Z)",
                      text, re.I)
        code = m.group(1).strip() if m else "mov ecx, 4\nmov eax, 0\nloop: add eax, ecx\ndec ecx\ncmp ecx, 0\njne loop\nhlt"
        return {"type": "asm_trace", "code": code}
    if has(r"assembly language|machine code|registers?\b.*instruction"):
        return {"type": "asm_trace", "code": "mov ecx, 4\nmov eax, 0\nloop: add eax, ecx\ndec ecx\ncmp ecx, 0\njne loop\nhlt"}
    if has(r"pipelin(e|ing)") and has(r"instruction|fetch|decode|cpu|stage|hazard"):
        return {"type": "pipeline", "stalls": {"1": 1} if has(r"hazard|stall|bubble") else {}}
    if has(r"memory hierarchy|l1 cache|cache memory|registers.*cache|cache.*main memory"):
        return {"type": "hierarchy"}
    if has(r"\bcuda\b|thread ?block|threadidx|blockidx|\bgpu\b.*thread|warp"):
        return {"type": "gpu_grid"}
    m = re.search(r"\b(nand|nor|xor|and|or|not)\s+gate", low)
    if m or has(r"logic gate|truth table|boolean algebra"):
        return {"type": "logic", "gate": (m.group(1) if m else "and").upper()}
    if has(r"lexer|lexical analy|tokeni[sz]er.*pars|parse tree|syntax tree|abstract syntax|compiler"):
        m = re.search(r"\b([a-z_]\w*\s*=\s*[\w\s+\-*/().]{3,40})", text)
        expr = m.group(1).strip().rstrip(".,;: ") if m else "x = a + b * 2"
        try:
            drawlib2.ast_tree(expr)
        except SyntaxError:
            expr = "x = a + b * 2"
        return {"type": "ast", "expr": expr}
    if has(r"\bselect\b.+\bfrom\b|\bsql\b|\bjoin\b.*\btable|relational database"):
        return {"type": "sql"} if has(r"join") else {"type": "sql", "query": "SELECT name FROM students WHERE id >= 2"}
    if has(r"\bredis\b|cache[- ]aside|cache hit|cache miss|in-memory (cache|key)"):
        return _seq(["App server", "Redis cache", "Database"],
                    [("App server", "Redis cache", "GET user:42"), ("Redis cache", "App server", "miss (nil)"),
                     ("App server", "Database", "SELECT ... id=42"), ("Database", "App server", "row"),
                     ("App server", "Redis cache", "SET user:42 EX 300"), ("App server", "Redis cache", "GET user:42 (later)"),
                     ("Redis cache", "App server", "hit: <1 ms")])
    if has(r"gradient descent|learning rate"):
        return {"type": "gradient"}
    if has(r"k-means|kmeans|clustering"):
        return {"type": "kmeans", "k": 3}
    if has(r"self-attention|attention (mechanism|weights|head)|transformer"):
        return {"type": "attention"}
    if has(r"neural network|perceptron|hidden layer|backpropagation|deep learning|activation function"):
        return {"type": "neural_net"}
    m = re.search(r"\b(hydrogen|helium|lithium|beryllium|boron|carbon|nitrogen|oxygen|fluorine|neon|sodium|magnesium|"
                  r"aluminium|aluminum|silicon|phosphorus|sulfur|chlorine|argon|potassium|calcium)\b", low)
    if has(r"atomic number|electron shell|bohr model|valence electron|protons?\b.*neutrons?"):
        el = (m.group(1).replace("aluminum", "aluminium") if m else "carbon").title()
        return {"type": "atom", "element": el}
    for f, rx in () if not has(r"molecul|\bbonds?\b|formula|lewis|covalent|ionic|atoms? of") else (("H2O", r"\bh2o\b|water molecule"), ("CO2", r"\bco2\b|carbon dioxide"), ("CH4", r"\bch4\b|methane"),
                  ("NH3", r"\bnh3\b|ammonia"), ("NaCl", r"\bnacl\b|sodium chloride"),
                  ("HCl", r"\bhcl\b|hydrogen chloride"), ("C2H6", r"\bc2h6\b|ethane")):
        if has(rx):
            return {"type": "molecule", "formula": f}
    if has(r"ohm'?s law|resistor|series circuit|parallel circuit|electric current"):
        return {"type": "circuit", "arrangement": "parallel" if has(r"parallel") else "series", "voltage": 12,
                "resistors": [4, 6] if has(r"parallel") else [2, 4]}
    if has(r"projectile"):
        return {"type": "projectile", "speed": 20, "angle": 45}
    if has(r"free[- ]body|net force|newton'?s (first|second|third) law|friction"):
        return {"type": "free_body"}
    if has(r"kubernetes|\bk8s\b|\bpods?\b|kubectl"):
        return {"type": "nested", "title": "Kubernetes cluster", "outer_icon": "kubernetes",
                "groups": [{"label": "Control plane", "items": ["API server", "Scheduler", "etcd"]},
                           {"label": "Node 1", "icon": "server", "items": ["Pod: web", "Pod: api", "kubelet"]},
                           {"label": "Node 2", "icon": "server", "items": ["Pod: web", "Pod: worker", "kubelet"]}]}
    if has(r"\bdocker\b|container image|dockerfile|\bcontainers?\b"):
        return {"type": "nested", "title": "Docker host", "outer_icon": "docker",
                "groups": [{"label": "Docker engine", "items": ["Image: python:3.12", "Image: nginx", "Image: redis"]},
                           {"label": "Running containers", "items": [{"label": "web (nginx)", "icon": "nginx"},
                                                                     {"label": "api (python)", "icon": "python"},
                                                                     {"label": "cache (redis)", "icon": "redis"}]}]}
    if has(r"terraform|infrastructure as code"):
        return _seq(["You", "Terraform", "Cloud API"],
                    [("You", "Terraform", "write main.tf"), ("You", "Terraform", "terraform init"),
                     ("You", "Terraform", "terraform plan"), ("Terraform", "You", "+ 3 to add, 0 to destroy"),
                     ("You", "Terraform", "terraform apply"), ("Terraform", "Cloud API", "create resources"),
                     ("Cloud API", "Terraform", "ids"), ("Terraform", "You", "state saved (terraform.tfstate)")])
    if has(r"\bazure\b"):
        return {"type": "nested", "title": "Azure subscription", "outer_icon": "azure",
                "groups": [{"label": "Resource group: web", "items": ["App Service", "Azure SQL Database", "Storage account"]},
                           {"label": "Resource group: network", "items": ["Virtual network", "Network security group",
                                                                         "Load balancer"]}]}
    if has(r"\bvpc\b|public subnet|private subnet"):
        return {"type": "nested", "title": "VPC 10.0.0.0/16",
                "groups": [{"label": "Public subnet 10.0.1.0/24", "items": ["Internet gateway", "NAT gateway",
                                                                             "Load balancer"]},
                           {"label": "Private subnet 10.0.2.0/24", "items": ["EC2 app servers", "RDS database"]}]}
    if has(r"\breact\b|\bcomponents?\b.*\b(props|state)\b|\bjsx\b"):
        return {"type": "tree", "root": {"label": "App (state: todos)", "children": [
            {"label": "Header"}, {"label": "TodoList (props: todos)", "children": [{"label": "TodoItem"}, {"label": "TodoItem"}]},
            {"label": "AddTodo (calls setTodos)"}]}}
    if has(r"\bgis\b|shapefile|raster|vector data|arcgis|qgis|geospatial|spatial data"):
        return {"type": "gis_layers"}
    if has(r"\bosi\b|encapsulation|tcp/ip model|protocol stack"):
        return {"type": "osi"}
    if has(r"packet tracer|cisco|\bvlan\b|router.*switch|switch.*router"):
        return {"type": "topology",
                "nodes": [{"id": "PC1", "label": "PC1"}, {"id": "PC2", "label": "PC2"}, {"id": "SW1", "label": "SW1 switch"},
                          {"id": "R1", "label": "R1 router"}, {"id": "R2", "label": "R2 router"},
                          {"id": "SRV", "label": "Web server"}],
                "links": [{"from": "PC1", "to": "SW1", "label": "Fa0/1"}, {"from": "PC2", "to": "SW1", "label": "Fa0/2"},
                          {"from": "SW1", "to": "R1", "label": "G0/0"}, {"from": "R1", "to": "R2", "label": "S0/0/0"},
                          {"from": "R2", "to": "SRV", "label": "G0/1"}],
                "ips": {"PC1": "192.168.1.10", "PC2": "192.168.1.11", "R1": "192.168.1.1", "SRV": "10.0.0.5"},
                "path": ["PC1", "SW1", "R1", "R2", "SRV"]}
    if has(r"man[- ]in[- ]the[- ]middle|\bmitm\b|arp spoof"):
        return _seq(["Alice laptop", "Attacker", "Bank server"],
                    [("Alice laptop", "Attacker", "login (thinks it's the bank)"), ("Attacker", "Bank server", "forwards login"),
                     ("Bank server", "Attacker", "session cookie"), ("Attacker", "Alice laptop", "page (cookie stolen)")])
    if has(r"phishing"):
        return _seq(["Attacker", "Employee", "Fake login site"],
                    [("Attacker", "Employee", "email: 'reset your password'"), ("Employee", "Fake login site", "clicks link"),
                     ("Employee", "Fake login site", "types password"), ("Fake login site", "Attacker", "stolen password")])
    if has(r"public key|asymmetric|\brsa\b|diffie"):
        return _seq(["Alice laptop", "Bob laptop"],
                    [("Bob laptop", "Alice laptop", "Bob's public key"),
                     ("Alice laptop", "Bob laptop", "message encrypted with it"),
                     ("Alice laptop", "Bob laptop", "only Bob's private key decrypts")])
    if has(r"pythagor"):
        return {"type": "right_triangle", "a": "a = 3", "b": "b = 4", "c": "c = 5", "squares": True,
                "formulas": ["a² + b² = c²", "9 + 16 = 25"]}
    if has(r"quadratic"):
        return {"type": "plot", "functions": [{"expr": "x^2 - 2*x - 3", "label": "y = x² - 2x - 3 (roots -1 and 3)"}],
                "x_range": [-3, 5], "y_range": [-5, 10]}
    if has(r"\bsine\b|\bcosine\b|unit circle"):
        return {"type": "unit_circle", "angle": 40}
    if has(r"linear regression|line of best fit|least squares"):
        xs = [1, 2, 3, 4, 5, 6, 7, 8]
        ys = [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = my - b * mx
        return {"type": "plot", "functions": [{"label": "data", "points": [[x, y] for x, y in zip(xs, ys)]},
                                              {"expr": f"{b:.3f}*x + {a:.3f}", "label": f"fit: y = {b:.2f}x + {a:.2f}"}],
                "x_range": [0, 9], "y_range": [0, 18]}
    return None


RULES_6 = [
    (r"kruskal|minimum spanning tree|prim'?s algorithm|union[- ]find|disjoint[- ]set", "kruskal"),
    (r"knuth[- ]morris|\bkmp\b|string matching|pattern matching algorithm", "kmp"),
    (r"n[- ]queens|eight queens|8 queens|backtracking", "n_queens"),
    (r"\bavl\b|self[- ]balancing|tree rotation|balance factor", "avl"),
    (r"big[- ]o\b|asymptotic|growth rates? of (functions|algorithms)|complexity classes", "big_o"),
    (r"\bb-tree|\bb\+ ?tree|\bbtree|database index", "b_tree"),
    (r"instruction pipeline|pipelin(e|ing) (hazard|stage)|data hazard|five[- ]stage pipeline|5[- ]stage pipeline|pipeline stall", "cpu_pipeline"),
    (r"producer[- ]consumer|bounded buffer|\bsemaphores?\b", "producer_consumer"),
    (r"go[- ]back[- ]n|selective repeat|stop[- ]and[- ]wait|automatic repeat request", "go_back_n"),
    (r"\bip header|ipv4 header|header checksum|packet header", "ip_header"),
    (r"\bvpn\b|ipsec|wireguard", "vpn_tunnel"),
    (r"\bcdn\b|content delivery network|cloudfront|edge location", "cdn"),
    (r"acid properties|\bacid\b.{0,40}(transaction|database)|atomicity|commit (and|or) rollback|\brollback\b", "acid_transaction"),
    (r"dockerfile|docker image|image layers?|container image|docker build", "docker_layers"),
    (r"rolling update|rolling deployment|kubectl rollout|zero[- ]downtime deploy|blue[- ]green", "rolling_update"),
    (r"terraform|infrastructure as code", "terraform_plan"),
    (r"\bkafka\b|consumer group|event streaming", "kafka_partitions"),
    (r"virtual dom|reconciliation|react (re-?render|state)|usestate", "virtual_dom"),
    (r"rate limit|token bucket|throttling", "token_bucket"),
    (r"diffie|key exchange", "diffie_hellman"),
    (r"\brsa\b|public[- ]key (cryptography|encryption)|asymmetric encryption", "rsa"),
    (r"linear regression|least squares|line of best fit|regression line", "linear_regression"),
    (r"logistic regression|sigmoid function|binary classification", "logistic_regression"),
    (r"recurrent neural|\brnn\b|\blstm\b|\bgru\b|vanishing gradient", "rnn_unrolled"),
    (r"cross[- ]validation|k-fold|train[/ -]test split", "cross_validation"),
    (r"tf-?idf|inverse document frequency", "tfidf"),
    (r"perceptron", "perceptron"),
    (r"box ?plot|box-and-whisker|interquartile|quartiles", "boxplot"),
    (r"histogram|frequency distribution", "histogram"),
    (r"central limit theorem|sampling distribution|distribution of (the )?sample means", "central_limit"),
    (r"hypothesis test|null hypothesis|p-value|significance level|z-test|t-test", "hypothesis_test"),
    (r"taylor series|maclaurin|power series", "taylor_series"),
    (r"fourier|frequency domain|\bfft\b", "fourier_series"),
    (r"newton'?s method|newton[- ]raphson|root[- ]finding", "newtons_method"),
    (r"eigenvalue|eigenvector|linear transformation", "eigen_transform"),
    (r"decimal to binary|binary to decimal|convert(ing)? .{0,25}(to|into) (binary|hexadecimal|hex)|number bases|base conversion", "number_bases"),
    (r"modular arithmetic|\bmodulo\b|congruen|clock arithmetic", "modular_clock"),
    (r"simple harmonic|hooke'?s law|spring constant|mass on a spring", "spring_shm"),
    (r"\btorque|\blevers?\b|fulcrum|moment of (a )?force", "lever_torque"),
    (r"buoyan|archimedes|float or sink|floats? (and|or) sinks?", "buoyancy"),
    (r"electric field|coulomb|point charges?|field lines", "electric_field"),
    (r"electromagnetic spectrum|electromagnetic (waves?|radiation)|em spectrum", "em_spectrum"),
    (r"titration|equivalence point|acid[- ]base (reaction|neutrali)", "titration"),
    (r"limiting (reagent|reactant)|excess (reagent|reactant)|stoichiometry|theoretical yield", "limiting_reagent"),
    (r"vsepr|molecular geometry|molecular shape|bond angles?|lone pairs?", "vsepr"),
    (r"active site|michaelis|lock[- ]and[- ]key|induced fit|enzyme activity|how enzymes work|enzymes? (are|act as) (biological )?catalysts", "enzyme"),
    (r"action potential|nerve impulse|resting membrane potential", "action_potential"),
    (r"hardy[- ]weinberg|genotype frequenc", "hardy_weinberg"),
    (r"population growth|carrying capacity|logistic growth", "population_growth"),
    (r"interpolat|kriging|inverse distance", "idw_interpolation"),
    (r"contour|topographic|elevation map|digital elevation|\bdem\b", "topographic"),
    (r"raster (and|vs\.?|versus) vector|vector (and|vs\.?|versus) raster|raster data|vector data model", "raster_vector"),
    (r"spatial join|point[- ]in[- ]polygon|overlay analysis", "spatial_join"),
    (r"net present value|\bnpv\b|discounted cash flow|present value|time value of money", "npv"),
    (r"elasticity|elastic demand|inelastic", "elasticity"),
]


RULES_7 = [
    (r"\ba\*|a-star|a star search|heuristic search", "a_star"),
    (r"floyd|all[- ]pairs shortest", "floyd_warshall"),
    (r"radix sort|counting sort|bucket sort", "radix_sort"),
    (r"activity selection|interval scheduling|greedy algorithm", "interval_scheduling"),
    (r"coin change|change[- ]making|making change", "coin_change"),
    (r"lexical analy|\blexer\b|lexical tokens|compiler front[- ]end", "lexer"),
    (r"garbage collect|mark[- ]and[- ]sweep|reference counting", "gc_mark_sweep"),
    (r"branch predict", "branch_predictor"),
    (r"amdahl|parallel speedup", "amdahl"),
    (r"ripple[- ]carry|full adder|half adder|binary adder", "ripple_adder"),
    (r"karnaugh|k-map|\bkmap\b|boolean (simplification|minimi[sz]ation)", "karnaugh"),
    (r"endian|byte order", "endianness"),
    (r"\bmips\b|instruction format|\br-type\b|\bi-type\b", "mips_encoding"),
    (r"memory hierarchy|storage hierarchy|latency numbers", "memory_hierarchy"),
    (r"banker'?s algorithm|safe state|deadlock avoidance", "bankers"),
    (r"first[- ]fit|best[- ]fit|worst[- ]fit|contiguous (memory )?allocation|external fragmentation", "memory_alloc"),
    (r"distance[- ]vector|bellman[- ]ford|routing information protocol", "distance_vector"),
    (r"hamming code|error[- ]correcting|ecc memory|parity bits", "hamming_code"),
    (r"\bcrc\b|cyclic redundancy", "crc"),
    (r"manchester (encoding|code)|line coding|\bnrz\b", "manchester"),
    (r"access control list|access[- ]list|\bacls?\b", "acl_rules"),
    (r"\bvlans?\b|802\.1q|trunk port", "vlan"),
    (r"\bvpc\b|virtual private cloud|nat gateway|internet gateway|public and private subnets", "aws_vpc"),
    (r"\biam\b|least privilege|identity and access management", "iam_policy"),
    (r"circuit breaker", "circuit_breaker"),
    (r"\braft\b|leader election|paxos|consensus algorithm", "raft_election"),
    (r"\bshard|horizontal partition", "sharding"),
    (r"merkle|hash tree", "merkle_tree"),
    (r"port scan|\bnmap\b|open ports", "port_scan"),
    (r"certificate (chain|authority)|chain of trust|\bpki\b|root ca\b", "cert_chain"),
    (r"activation functions?|\brelu\b|\bgelu\b", "activations"),
    (r"naive bayes|spam filter", "naive_bayes"),
    (r"dbscan|density[- ]based clustering", "dbscan"),
    (r"hierarchical clustering|dendrogram|agglomerative", "dendrogram"),
    (r"cosine similarity|semantic similarity|vector similarity", "cosine_similarity"),
    (r"(sampling|softmax|llm|decoding).{0,60}temperature|temperature.{0,60}(sampling|softmax|llm|tokens?)|top-p|nucleus sampling|top-k sampling", "temperature_sampling"),
    (r"positional (encoding|embedding)", "positional_encoding"),
    (r"learning rate", "learning_rate"),
    (r"retrieval[- ]augmented|\brag\b|vector database|vector store", "rag_pipeline"),
    (r"window functions?|over \(partition|running total|row_number|\blag\(", "window_functions"),
    (r"z-scores?|standardi[sz](e|ation)|feature scaling", "standardization"),
    (r"dot product|scalar product|vector projection", "dot_product"),
    (r"complex numbers?|imaginary numbers?|argand|complex plane", "complex_plane"),
    (r"quadratic (equation|function|formula)|discriminant|parabola", "quadratic"),
    (r"arithmetic (sequence|series|progression)|geometric (sequence|series|progression)|common ratio|common difference", "sequences"),
    (r"markov chain|transition matrix|steady[- ]state distribution|stationary distribution", "markov_chain"),
    (r"truth table|boolean expression|propositional logic", "truth_table"),
    (r"permutations?|n choose k|counting principle", "permutations"),
    (r"euler(ian)? (path|circuit)|k(ö|o)nigsberg|seven bridges", "euler_path"),
    (r"rc circuit|capacitor (charg|discharg)|time constant", "rc_circuit"),
    (r"standing waves?|harmonics on a string|fundamental frequency|overtones|vibrating string", "standing_waves"),
    (r"double[- ]slit|young'?s (double|experiment|slits)|interference pattern|interference fringes", "double_slit"),
    (r"circular motion|centripetal|centrifugal", "circular_motion"),
    (r"kepler|elliptical orbit|planetary orbit|orbital motion", "kepler_orbit"),
    (r"heating curve|latent heat|heat of (fusion|vaporization)|specific heat capacity", "heating_curve"),
    (r"photoelectric|work function|photon energy", "photoelectric"),
    (r"maxwell[- ]boltzmann|boltzmann distribution|collision theory|kinetic molecular theory", "maxwell_boltzmann"),
    (r"galvanic|voltaic|electrochemical cell|standard reduction potential|anode and cathode", "galvanic_cell"),
    (r"polymerase chain reaction|\bpcr\b", "pcr"),
    (r"gel electrophoresis|restriction enzymes?|dna fingerprint", "gel_electrophoresis"),
    (r"phylogen|evolutionary tree|cladogram", "phylogenetic_tree"),
    (r"homeostasis|negative feedback|blood glucose|blood sugar|\binsulin\b", "feedback_loop"),
    (r"\bndvi\b|vegetation index|multispectral", "ndvi"),
    (r"epicent|seismic waves?|p[- ]waves?|s[- ]waves?|seismograph", "epicenter"),
    (r"hillshade|slope (and|&) aspect|terrain analysis|slope analysis", "slope_hillshade"),
    (r"gini coefficient|lorenz curve|income inequality|wealth inequality", "lorenz_gini"),
    (r"prisoner'?s dilemma|nash equilibrium|game theory|payoff matrix", "prisoners_dilemma"),
    (r"comparative advantage|absolute advantage", "comparative_advantage"),
]


def rules_visual_7(text, low):
    """Library part 7 + more exchanges: checked first."""
    has = lambda rx: re.search(rx, low)
    for rx, typ in RULES_7:
        if has(rx):
            v = {"type": typ}
            if typ == "quadratic":
                m = re.search(r"([+-]?\s*\d*\.?\d*)\s*x\s*(?:\^|\*\*)?\s*2|([+-]?\s*\d*\.?\d*)\s*x²", low)
                q = re.search(r"(?:y\s*=|f\(x\)\s*=)?\s*([+-]?\s*\d*\.?\d*)\s*x\s*(?:\^|\*\*)?\s*2\s*([+-]\s*\d*\.?\d*)\s*x"
                              r"\s*([+-]\s*\d+\.?\d*)", low.replace("²", "^2"))
                if q:
                    num = lambda s, d: float(s.replace(" ", "")) if s.replace(" ", "") not in ("", "+", "-") else \
                        (-d if s.strip() == "-" else d)
                    a_, b_, c_ = num(q.group(1), 1), num(q.group(2), 1), num(q.group(3), 0)
                    if a_:
                        v["coefficients"] = [a_, b_, c_]
            return v
    if has(r"aws lambda|serverless|api gateway"):
        return _seq(["Mobile app", "API Gateway", "AWS Lambda", "DynamoDB"],
                    [("Mobile app", "API Gateway", "POST /orders"), ("API Gateway", "AWS Lambda", "invoke (cold start ~200 ms)"),
                     ("AWS Lambda", "DynamoDB", "PutItem order"), ("DynamoDB", "AWS Lambda", "ok"),
                     ("AWS Lambda", "API Gateway", "201 Created"), ("API Gateway", "Mobile app", "201 + order id")])
    if has(r"multi[- ]factor|\bmfa\b|two[- ]factor|\b2fa\b|one-time (password|code)|\btotp\b"):
        return _seq(["User", "Website", "Authenticator app"],
                    [("User", "Website", "username + password (something you know)"),
                     ("Website", "User", "enter the 6-digit code"), ("Authenticator app", "User", "TOTP code from shared secret + time"),
                     ("User", "Website", "code (something you have)"), ("Website", "User", "logged in")])
    if has(r"interrupt (handling|service routine|handler|vector)|\bisr\b|hardware interrupt"):
        return _seq(["Keyboard", "CPU", "Interrupt handler"],
                    [("Keyboard", "CPU", "IRQ: key pressed"), ("CPU", "CPU", "finish instruction, save registers + PC"),
                     ("CPU", "Interrupt handler", "jump via interrupt vector table"),
                     ("Interrupt handler", "CPU", "read key, acknowledge, return (iret)"),
                     ("CPU", "CPU", "restore registers, resume the program")])
    if has(r"\bssh\b|secure shell"):
        return _seq(["Laptop", "Server"],
                    [("Laptop", "Server", "connect: key exchange (Diffie-Hellman)"), ("Server", "Laptop", "host key (check fingerprint)"),
                     ("Laptop", "Server", "I am sed, with public key X"), ("Server", "Laptop", "challenge: sign this"),
                     ("Laptop", "Server", "signature made with the private key"), ("Server", "Laptop", "authenticated: shell")])
    return None


def rules_visual_6(text, low):
    """Library part 6 + more protocol exchanges: checked first."""
    has = lambda rx: re.search(rx, low)
    for rx, typ in RULES_6:
        if has(rx):
            return {"type": typ}
    if has(r"two[- ]phase commit|\b2pc\b|distributed transaction"):
        return _seq(["Coordinator", "Database A", "Database B"],
                    [("Coordinator", "Database A", "PREPARE"), ("Coordinator", "Database B", "PREPARE"),
                     ("Database A", "Coordinator", "YES (ready)"), ("Database B", "Coordinator", "YES (ready)"),
                     ("Coordinator", "Database A", "COMMIT"), ("Coordinator", "Database B", "COMMIT"),
                     ("Database A", "Coordinator", "done"), ("Database B", "Coordinator", "done")])
    if has(r"cache[- ]aside|\bredis\b|memcached|caching layer"):
        return _seq(["App server", "Redis cache", "Database"],
                    [("App server", "Redis cache", "GET user:42"), ("Redis cache", "App server", "miss (nil)"),
                     ("App server", "Database", "SELECT * FROM users WHERE id = 42"), ("Database", "App server", "row"),
                     ("App server", "Redis cache", "SET user:42 EX 300"), ("App server", "Redis cache", "next time: GET user:42"),
                     ("Redis cache", "App server", "hit in < 1 ms")])
    if has(r"cross[- ]site scripting|\bxss\b"):
        return _seq(["Attacker", "Web server", "Victim's browser"],
                    [("Attacker", "Web server", "post comment: <script>steal(cookie)</script>"),
                     ("Victim's browser", "Web server", "open the page"),
                     ("Web server", "Victim's browser", "page with the unescaped script"),
                     ("Victim's browser", "Attacker", "script runs: sends the session cookie"),
                     ("Web server", "Victim's browser", "fix: escape output + Content-Security-Policy")])
    if has(r"cross[- ]site request forgery|\bcsrf\b"):
        return _seq(["Victim's browser", "Evil site", "Bank"],
                    [("Victim's browser", "Bank", "log in (session cookie set)"), ("Victim's browser", "Evil site", "visit"),
                     ("Evil site", "Victim's browser", "hidden form: POST /transfer"),
                     ("Victim's browser", "Bank", "POST /transfer + cookie (sent automatically!)"),
                     ("Bank", "Victim's browser", "fix: CSRF token + SameSite cookies")])
    if has(r"publish[- ]subscribe|\bpub/?sub\b|message broker"):
        return _seq(["Order service", "Broker (topic: orders)", "Email service", "Inventory service"],
                    [("Order service", "Broker (topic: orders)", "publish OrderPlaced"),
                     ("Broker (topic: orders)", "Email service", "deliver"), ("Broker (topic: orders)", "Inventory service", "deliver"),
                     ("Email service", "Broker (topic: orders)", "ack"), ("Inventory service", "Broker (topic: orders)", "ack")])
    if has(r"websocket"):
        return _seq(["Browser", "Chat server"],
                    [("Browser", "Chat server", "GET /chat  Upgrade: websocket"), ("Chat server", "Browser", "101 Switching Protocols"),
                     ("Browser", "Chat server", "message: hi"), ("Chat server", "Browser", "push: new message from Ana"),
                     ("Chat server", "Browser", "push: Ana is typing…"), ("Browser", "Chat server", "close")])
    if has(r"kerberos|ticket[- ]granting"):
        return _seq(["User PC", "KDC (AS + TGS)", "File server"],
                    [("User PC", "KDC (AS + TGS)", "log in (password never sent)"),
                     ("KDC (AS + TGS)", "User PC", "ticket-granting ticket (TGT)"),
                     ("User PC", "KDC (AS + TGS)", "TGT + 'I want the file server'"),
                     ("KDC (AS + TGS)", "User PC", "service ticket"), ("User PC", "File server", "service ticket"),
                     ("File server", "User PC", "access granted")])
    return None


def rules_visual_5(text, low):
    """Library part 5 + protocol exchanges: checked first."""
    has = lambda rx: re.search(rx, low)
    if has(r"\bdhcp\b") and has(r"discover|offer|lease|assign|gives? out|hands? out|how dhcp"):
        return _seq(["New laptop", "DHCP server"], [("New laptop", "DHCP server", "DISCOVER (broadcast)"),
                    ("DHCP server", "New laptop", "OFFER 192.168.1.23"), ("New laptop", "DHCP server", "REQUEST that address"),
                    ("DHCP server", "New laptop", "ACK: lease 24 h, gateway, DNS")])
    if has(r"address resolution protocol|\barp\b.{0,80}\bmac\b|\bmac\b.{0,80}\barp\b|arp (request|reply|cache)"):
        return _seq(["PC 192.168.1.10", "Switch", "Router 192.168.1.1"],
                    [("PC 192.168.1.10", "Switch", "who has 192.168.1.1? (broadcast)"),
                     ("Switch", "Router 192.168.1.1", "flooded to all ports"),
                     ("Router 192.168.1.1", "PC 192.168.1.10", "192.168.1.1 is at aa:bb:cc:dd:ee:01"),
                     ("PC 192.168.1.10", "Router 192.168.1.1", "frame sent to that MAC (now cached)")])
    if has(r"(tls|ssl|https) handshake|how https works|tls 1\.[23]|certificate.{0,60}(tls|ssl|https)|(tls|ssl|https).{0,60}certificate"):
        return _seq(["Browser", "Web server"], [("Browser", "Web server", "ClientHello (ciphers, random)"),
                    ("Web server", "Browser", "ServerHello + certificate"),
                    ("Browser", "Web server", "key share (verify cert with CA)"),
                     ("Web server", "Browser", "Finished (keys agreed)"), ("Browser", "Web server", "encrypted HTTP request"),
                     ("Web server", "Browser", "encrypted response")])
    if has(r"\boauth|single sign-on|sign in with google|authorization code flow"):
        return _seq(["User", "App", "Auth server (Google)", "API"],
                    [("User", "App", "click 'Sign in with Google'"), ("App", "Auth server (Google)", "redirect: client_id, scope"),
                     ("User", "Auth server (Google)", "log in + consent"), ("Auth server (Google)", "App", "authorization code"),
                     ("App", "Auth server (Google)", "code + client secret"), ("Auth server (Google)", "App", "access token"),
                     ("App", "API", "request + Bearer token"), ("API", "App", "user's data")])
    if has(r"email (delivery|is (sent|delivered))|how email works|(smtp|imap|pop3).{0,120}mail server|mail server.{0,120}(smtp|imap|pop3)"):
        return _seq(["Alice's laptop", "Alice's mail server", "Bob's mail server", "Bob's phone"],
                    [("Alice's laptop", "Alice's mail server", "SMTP: send"), ("Alice's mail server", "Bob's mail server",
                      "SMTP (found via DNS MX)"), ("Bob's phone", "Bob's mail server", "IMAP: any new mail?"),
                     ("Bob's mail server", "Bob's phone", "the message")])
    if has(r"digital signature|signing a message|signed with (a|the|her|his) private key"):
        return _seq(["Alice laptop", "Certificate authority", "Bob laptop"],
                    [("Alice laptop", "Bob laptop", "message + signature (hash signed with Alice's PRIVATE key)"),
                     ("Bob laptop", "Certificate authority", "get Alice's certificate"),
                     ("Certificate authority", "Bob laptop", "Alice's PUBLIC key (vouched for by the CA)"),
                     ("Bob laptop", "Alice laptop", "verified: hashes match, so it's from Alice and unchanged")])
    if has(r"http request|rest api|\bget /api|request and response cycle") and not has(r"\bsyn\b|handshake"):
        return _seq(["Browser", "Web server", "Database"], [("Browser", "Web server", "GET /api/users/42"),
                    ("Web server", "Database", "SELECT * FROM users WHERE id = 42"), ("Database", "Web server", "row"),
                    ("Web server", "Browser", "200 OK + JSON {\"id\": 42, ...}")])
    for rx, typ in ((r"quick ?sort|partition(ing)? around a pivot", "quick_sort"), (r"bloom filter", "bloom_filter"),
                    (r"consistent hashing|hash ring", "consistent_hash"), (r"revers(e|ing) a linked list", "list_reverse"),
                    (r"sliding window|two pointers", "sliding_window"),
                    (r"memory layout|text segment|data segment|stack and (the )?heap|heap and (the )?stack", "memory_layout"),
                    (r"\bfork\(\)|fork system call|child process", "fork_tree"),
                    (r"disk scheduling|\bsstf\b|seek time|elevator algorithm", "disk_sched"), (r"\braid\b", "raid"),
                    (r"fetch[- ]decode[- ]execute|program counter|instruction register|instruction cycle", "cpu_cycle"),
                    (r"ieee[- ]?754|floating[- ]point", "ieee754"), (r"utf-?8|unicode|\bascii\b|character encoding", "utf8"),
                    (r"\bnat\b|network address translation|port address translation", "nat_table"),
                    (r"routing table|longest prefix|next hop", "routing_lpm"),
                    (r"traceroute|tracert", "traceroute"),
                    (r"buffer overflow|stack smashing|return address", "buffer_overflow"),
                    (r"brute[- ]force|password (strength|entropy|length)|key ?space", "brute_force"),
                    (r"backpropagation|backprop|chain rule|computational graph", "backprop"),
                    (r"\broc\b|\bauc\b|receiver operating", "roc_curve"),
                    (r"value iteration|markov decision|q-learning|bellman equation|grid ?world|reinforcement learning.{0,200}\b(agent|policy)\b.{0,200}\bstates?\b", "grid_world"),
                    (r"principal component|\bpca\b|dimensionality reduction", "pca"),
                    (r"word embedding|word2vec|word vectors?|king\s*[-−]\s*man", "word_vectors"),
                    (r"transformer (architecture|block|layer)|encoder layer|attention is all you need", "transformer_block"),
                    (r"inner join|left join|right join|outer join|types of joins?", "sql_joins"),
                    (r"normal form|\b[123]nf\b|database normali[sz]ation|normali[sz]e the (table|database)", "normalization"),
                    (r"star schema|fact table|dimension table|data warehouse", "star_schema"),
                    (r"moving average|rolling average|rolling mean", "moving_average"),
                    (r"correlation|pearson|correlation coefficient", "correlation"),
                    (r"prime numbers?|sieve of eratosthenes", "sieve"), (r"greatest common divisor|\bgcd\b|euclid", "euclid_gcd"),
                    (r"pascal'?s triangle|binomial coefficient", "pascal"), (r"binomial distribution", "binomial"),
                    (r"base rate|medical test|screening test|disease.*test.*positive|test.*positive.*disease", "bayes_test"),
                    (r"monte carlo", "monte_carlo_pi"), (r"momentum|elastic collision|inelastic collision", "collision"),
                    (r"inclined plane|incline|ramp with friction", "incline"),
                    (r"ideal gas|pv\s*=\s*nrt|boyle'?s law|charles'?s law|gas laws?", "gas_law"),
                    (r"half-life|half life|radioactive decay|carbon dating", "half_life"), (r"doppler", "doppler"),
                    (r"electron configuration|\baufbau\b|orbital diagram|hund'?s rule", "electron_config"),
                    (r"food web|food chain", "food_web"), (r"energy pyramid|trophic level|10% rule|ten percent rule", "energy_pyramid"),
                    (r"natural selection|\bevolution\b|allele frequenc|survival of the fittest|adaptation", "natural_selection"),
                    (r"seasons|axial tilt|solstice|equinox", "seasons"),
                    (r"greenhouse (effect|gas)|global warming|climate change", "greenhouse"),
                    (r"rock cycle|igneous|sedimentary rock|metamorphic", "rock_cycle"),
                    (r"map projection|mercator", "map_projection"), (r"voronoi|thiessen", "voronoi"),
                    (r"choropleth|thematic map", "choropleth"), (r"amortiz|mortgage|loan payment", "amortization"),
                    (r"production possibilit|opportunity cost", "ppf"),
                    (r"\bmaze\b|grid pathfinding|shortest path on a grid", "maze_bfs")):
        if has(rx):
            return {"type": typ}
    return None


def rules_visual_4(text, low):
    """Library part 4: checked before parts 1-3 (more specific topics)."""
    has = lambda rx: re.search(rx, low)
    nums = [float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", text)]
    if has(r"ci/cd|continuous (integration|delivery|deployment)|github actions|build pipeline|deployment pipeline"):
        return {"type": "cicd"}
    if has(r"sql injection|sqli\b"):
        return {"type": "sql_injection"}
    if has(r"\bjwt\b|json web token"):
        return {"type": "jwt"}
    if has(r"\bsalt(ed)?\b|password hash|bcrypt|argon2|rainbow table"):
        return {"type": "password_hash"}
    if has(r"two'?s complement|signed binary|negative numbers? in binary"):
        m = re.search(r"(-\d{1,3})\b", text)
        return {"type": "twos_complement", **({"number": int(m.group(1))} if m else {})}
    if has(r"race condition|\bmutex|lost update|critical section"):
        return {"type": "race"}
    if has(r"deadlock|circular wait"):
        return {"type": "deadlock"}
    if has(r"event loop|microtask|callback queue|settimeout"):
        return {"type": "event_loop"}
    if has(r"congestion (control|window)|slow start|\baimd\b|\bcwnd\b"):
        return {"type": "tcp_cwnd"}
    if has(r"page replacement|least recently used|\blru\b|page fault"):
        return {"type": "lru", "policy": "fifo" if has(r"\bfifo\b") and not has(r"\blru\b") else "lru"}
    if has(r"direct[- ]mapped|cache line|set[- ]associative|cache hit rate|tag,? index|index and offset"):
        return {"type": "cache_sim"}
    if has(r"longest common subsequence|\blcs\b|dynamic programming|memoi[sz]ation|edit distance"):
        return {"type": "dp_table"}
    if has(r"merge sort|mergesort|divide and conquer"):
        return {"type": "merge_sort"}
    if has(r"huffman"):
        return {"type": "huffman"}
    if has(r"\btries?\b|prefix tree|autocomplete"):
        return {"type": "trie"}
    if has(r"topological (sort|order)|\bdag\b|directed acyclic|prerequisites? graph|dependency graph"):
        return {"type": "topo_sort"}
    if has(r"auto ?scal|scale out|scale in|horizontal scaling"):
        return {"type": "autoscaling"}
    if has(r"decision tree|gini|information gain|entropy split"):
        return {"type": "decision_tree"}
    if has(r"k-nearest|nearest neighbou?rs?|\bknn\b"):
        return {"type": "knn"}
    if has(r"normal distribution|bell curve|standard normal|68[-–]95|z-score|gaussian distribution"):
        return {"type": "normal_dist"}
    if has(r"softmax"):
        return {"type": "softmax"}
    if has(r"overfit|underfit|early stopping|validation loss|bias[- ]variance"):
        return {"type": "overfitting"}
    if has(r"entity[- ]relationship|\ber diagram|\berd\b|foreign key|primary key|one-to-many|many-to-many"):
        return {"type": "er_diagram"}
    if has(r"groupby|group by|\bpandas\b|aggregate function"):
        return {"type": "groupby"}
    if has(r"punnett|genotype|phenotype|dominant allele|recessive allele|monohybrid"):
        m = re.search(r"\b([A-Z][a-z])\s*[x×]\s*([A-Z][a-z]|[A-Z]{2}|[a-z]{2})\b", text)
        return {"type": "punnett", **({"parent1": m.group(1), "parent2": m.group(2)} if m else {})}
    if has(r"meiosis|crossing over|gametes?\b|haploid"):
        return {"type": "meiosis"}
    if has(r"cellular respiration|krebs|glycolysis|electron transport chain|citric acid cycle"):
        return {"type": "respiration"}
    if has(r"osmosis|hypertonic|hypotonic|isotonic"):
        return {"type": "osmosis"}
    if has(r"plate tectonics|tectonic plates?|subduction|divergent boundar|convergent boundar"):
        return {"type": "plate_tectonics"}
    if has(r"phases? of the moon|moon phases|lunar phases?|waxing|waning|gibbous"):
        return {"type": "moon_phases"}
    if has(r"inner core|outer core|layers of the earth|earth'?s layers|the mantle"):
        return {"type": "earth_layers"}
    if has(r"focal (length|point)|converging lens|convex lens|thin lens|lens equation"):
        return {"type": "lens"}
    if has(r"pendulum"):
        return {"type": "pendulum"}
    if has(r"kinetic energy|potential energy|conservation of energy|mechanical energy"):
        return {"type": "energy_ramp"}
    m = re.search(r"((?:\d*[A-Z][A-Za-z0-9]*\s*\+\s*)+\d*[A-Z][A-Za-z0-9]*\s*(?:->|→)\s*(?:\d*[A-Z][A-Za-z0-9]*\s*\+?\s*)+)",
                  text)
    if m or has(r"balanc(e|ing) (the |a )?(chemical )?equation|law of conservation of mass|stoichiometr"):
        return {"type": "balance", **({"equation": m.group(1).strip()} if m else {})}
    if has(r"activation energy|catalyst|exothermic|endothermic|enzymes? lower"):
        return {"type": "reaction_energy", "kind": "endothermic" if has(r"endothermic") else "exothermic"}
    if has(r"derivative|tangent line|instantaneous rate|slope of (the|a) curve"):
        return {"type": "tangent"}
    if has(r"compound interest|compounded|future value"):
        return {"type": "compound"}
    if has(r"system of (linear )?equations|simultaneous equations|two equations"):
        return {"type": "linear_system"}
    if has(r"break[- ]even|fixed costs?.*variable costs?"):
        return {"type": "break_even"}
    if has(r"buffer analysis|proximity analysis|buffer zone|within \d+ ?(m|km|miles|meters|feet) of|gis buffer"):
        return {"type": "gis_buffer"}
    return None


def rules_visual_3(text, low):
    """Library part 3: specific topics, checked first."""
    has = lambda rx: re.search(rx, low)
    nums = [int(x) for x in re.findall(r"(?<![\w.])-?\d{1,3}(?![\w.])", text)]
    if has(r"photosynthesis|light-dependent|calvin cycle"):
        return {"type": "photosynthesis"}
    if has(r"transcription|translation\b.*ribosome|\bmrna\b|codon|protein synthesis|central dogma"):
        m = re.search(r"\b[ATGC]{9,}\b", text)
        return {"type": "central_dogma", **({"dna": m.group(0)} if m else {})}
    if has(r"double helix|base pair|dna replication|helicase|complementary bases?"):
        m = re.search(r"\b[ATGC]{6,}\b", text)
        return {"type": "dna", **({"sequence": m.group(0)} if m else {})}
    if has(r"\bventricle|\batri(um|a)\b|the heart pumps|circulatory system|pulmonary"):
        return {"type": "heart"}
    if has(r"\bneurons?\b|\baxons?\b|\bdendrites?\b|action potential") and \
            not has(r"neural network|perceptron|hidden layer|\bmodel|parameter|transformer|\bllm|activation function"):
        return {"type": "neuron"}
    if has(r"water cycle|hydrologic cycle|evaporation.*condensation|precipitation.*evaporation"):
        return {"type": "water_cycle"}
    if has(r"states of matter|solids?, liquids? and gas|solid, liquid"):
        return {"type": "states_matter"}
    if has(r"wavelength|amplitude|transverse wave|frequency of a wave"):
        return {"type": "wave"}
    if has(r"refraction|snell'?s law|index of refraction|refractive index"):
        return {"type": "refraction"}
    if has(r"\bph scale|\bph\b of|acidic|alkaline"):
        return {"type": "ph_scale"}
    if has(r"solar system|the planets|planetary orbits?"):
        return {"type": "solar_system"}
    if has(r"supply and demand|demand curve|supply curve|equilibrium price"):
        return {"type": "supply_demand"}
    if has(r"venn diagram|union and intersection|intersection of (two )?sets"):
        return {"type": "venn"}
    if has(r"probability tree|tree diagram|conditional probability|bayes"):
        return {"type": "prob_tree"}
    if has(r"matrix multiplication|multiply(ing)? (two )?matrices|product of (two )?matrices"):
        return {"type": "matrix_mult"}
    if has(r"riemann|area under (the )?curve|definite integral"):
        return {"type": "riemann"}
    if has(r"vector addition|resultant vector|head[- ]to[- ]tail|adding vectors"):
        return {"type": "vectors"}
    if has(r"blockchain|hash chain|bitcoin|distributed ledger"):
        return {"type": "blockchain"}
    if has(r"caesar|shift cipher|substitution cipher"):
        m = re.search(r"shift(?:ed)? (?:of |by )?(\d+)", low)
        return {"type": "cipher", **({"shift": int(m.group(1))} if m else {})}
    if has(r"\bddos\b|denial[- ]of[- ]service|botnet"):
        return {"type": "ddos"}
    if has(r"cia triad|confidentiality,? integrity,? (and )?availability"):
        return {"type": "layers_ring", "shape": "triangle"}
    if has(r"defen[cs]e in depth|layered security"):
        return {"type": "layers_ring"}
    if has(r"firewall rules?|access control list|packet filter"):
        return {"type": "firewall"}
    if has(r"load balanc"):
        return {"type": "load_balancer"}
    m = re.search(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2})\b", text)
    if m or has(r"subnet mask|subnetting|\bcidr\b"):
        return {"type": "subnet", **({"cidr": m.group(1)} if m else {})}
    if has(r"binary number|convert(ing)? .*to binary|base[- ]2|hexadecimal|decimal to binary"):
        m = re.search(r"\b(\d{2,4})\b", text)
        return {"type": "binary", **({"number": int(m.group(1))} if m else {})}
    if has(r"\bgit\b|feature branch|merge (the )?branch|pull request"):
        return {"type": "git_graph"}
    if has(r"mapreduce|map[- ]reduce|word count"):
        return {"type": "mapreduce"}
    if has(r"page table|virtual address|virtual memory|paging"):
        return {"type": "page_table"}
    if has(r"finite (state )?automat|\bdfa\b|state machine|regular language"):
        return {"type": "fsm"}
    if has(r"recursion|recursive|base case"):
        return {"type": "recursion", "function": "fibonacci" if has(r"fibonacci") else "factorial"}
    if has(r"binary search tree|\bbst\b"):
        ks = [n for n in nums if n >= 0][:10]
        return {"type": "bst", **({"keys": ks} if len(ks) >= 4 else {})}
    if has(r"min-heap|max-heap|binary heap|priority queue|heapify"):
        return {"type": "heap"}
    if has(r"hash table|hash map|hashing|hash function|collision"):
        return {"type": "hash_table"}
    m = re.search(r"\b(bubble|insertion|selection) sort", low)
    if (m or has(r"sorting algorithm")) and not find_code(text):
        arr = [n for n in nums if 0 <= n < 100][:8]
        return {"type": "sort_bars", "algorithm": m.group(1) if m else "bubble", **({"array": arr} if len(arr) >= 4 else {})}
    if has(r"convolution|\bcnn\b|feature map|convolutional"):
        return {"type": "convolution"}
    if has(r"confusion matrix|precision and recall|false positive|true positive"):
        return {"type": "confusion"}
    if has(r"\bmedian\b|quartile|box plot|boxplot|histogram|standard deviation|descriptive statistics"):
        return {"type": "stats"}
    if has(r"byte[- ]pair|\bbpe\b|subword"):
        return {"type": "bpe"}
    return None


def rules_visual(text):
    """Recognise ideas the engine can draw exactly (no AI needed)."""
    text = re.sub(r"(?is)(key terms?( review)?|glossary|vocabulary)\s*:.*?(\n\n|$)", " ", text)   # word lists mislead
    low = text.lower()
    third = rules_visual_7(text, low) or rules_visual_6(text, low) or rules_visual_5(text, low) or rules_visual_4(text, low) or rules_visual_3(text, low)
    if third:
        return third
    atomish = re.search(r"atomic number|electron shell|bohr model|valence electron", low)
    if not atomish and not re.search(r"mitosis|prophase|organelle|nucleus|mitochondri", low):
        more = rules_visual_more(text, low)
        if more:
            return more
    ph = [p for p in drawlib.PHASES if p in low]
    if len(re.findall(r"mitosis|mitotic|prophase|metaphase|anaphase|telophase|cytokinesis", low)) >= 2:
        order = [p for p in drawlib.PHASES if p in ph]
        return {"type": "mitosis", "phases": order if len(order) >= 3 else drawlib.PHASES}
    orgs = re.findall(ORGANELLE_WORDS, low)
    about_genes = re.search(r"replication|transcription|translation|\bmrna\b|codon|central dogma", low)
    if not about_genes and (len(set(orgs)) >= 3 or
                            (len(set(orgs)) >= 2 and re.search(r"organelle|eukaryot|prokaryot|cell structure", low))):
        parts = []
        for o in dict.fromkeys(orgs):
            parts.append({"mitochondri": "mitochondria", "endoplasmic": "rough er", "golgi": "golgi apparatus",
                          "ribosome": "ribosomes", "plasma membrane": "cell membrane", "centriole": "centrioles",
                          "chloroplast": "chloroplast"}.get(o, o))
        plant = bool(re.search(r"chloroplast|cell wall|plant cell", low))
        return {"type": "cell", "kind": "plant" if plant else "animal", "parts": parts}
    if re.search(r"three-way handshake|\bsyn\b|syn-ack", low):
        msgs = [{"from": "Client", "to": "Server", "label": "SYN"},
                {"from": "Server", "to": "Client", "label": "SYN-ACK"},
                {"from": "Client", "to": "Server", "label": "ACK"},
                {"from": "Client", "to": "Server", "label": "data"},
                {"from": "Server", "to": "Client", "label": "ACK"}]
        if "fin" in low:
            msgs += [{"from": "Client", "to": "Server", "label": "FIN"}, {"from": "Server", "to": "Client", "label": "ACK"}]
        return {"type": "sequence", "actors": ["Client", "Server"], "messages": msgs}
    if re.search(r"\bdns\b|domain name|resolver|name server", low):
        return {"type": "sequence", "actors": ["Laptop", "DNS resolver", "Root server", "TLD server", "Authoritative"],
                "messages": [{"from": "Laptop", "to": "DNS resolver", "label": "where is example.com?"},
                             {"from": "DNS resolver", "to": "Root server", "label": "query"},
                             {"from": "Root server", "to": "DNS resolver", "label": "ask .com TLD"},
                             {"from": "DNS resolver", "to": "TLD server", "label": "query"},
                             {"from": "TLD server", "to": "DNS resolver", "label": "ask authoritative"},
                             {"from": "DNS resolver", "to": "Authoritative", "label": "query"},
                             {"from": "Authoritative", "to": "DNS resolver", "label": "93.184.216.34"},
                             {"from": "DNS resolver", "to": "Laptop", "label": "93.184.216.34 (cached)"}]}
    if re.search(r"\budp\b", low) and not re.search(r"\btcp\b", low):
        return {"type": "packet_flow", "nodes": ["Laptop", "Router", "Game server"],
                "hops": [{"from": "Laptop", "to": "Router", "label": "datagram 1"},
                         {"from": "Router", "to": "Game server", "label": "datagram 1"},
                         {"from": "Laptop", "to": "Router", "label": "datagram 2"},
                         {"from": "Router", "to": "Game server", "label": "datagram 2 (no ACK)"}]}
    if re.search(r"\bpackets?\b.*\b(router|hop|routing)|\brout(er|ing)\b.*\bpackets?\b", low):
        return {"type": "packet_flow", "nodes": ["Laptop", "Home router", "ISP router", "Web server"],
                "hops": [{"from": "Laptop", "to": "Home router", "label": "IP packet"},
                         {"from": "Home router", "to": "ISP router", "label": "next hop"},
                         {"from": "ISP router", "to": "Web server", "label": "delivered"},
                         {"from": "Web server", "to": "Laptop", "label": "reply"}]}
    code = find_code(text)
    if code:
        steps = trace_python(code, default_call(code, text))
        if steps:
            return {"type": "code_trace", "code": code, "steps": steps}
    for name, src in ALGOS.items():
        if name in low:
            steps = trace_python(src, ALGO_CALL[name])
            if steps:
                return {"type": "code_trace", "code": src, "steps": steps}
    if re.search(r"\bstack\b", low) and re.search(r"\bpush|\bpop\b|lifo", low):
        return {"type": "ds_ops", "structure": "stack", "ops": ["push 3", "push 7", "push 1", "pop", "push 9", "pop"]}
    if re.search(r"enqueue|dequeue|\bfifo\b|queue data structure", low):
        return {"type": "ds_ops", "structure": "queue", "ops": ["enqueue A", "enqueue B", "enqueue C", "dequeue",
                                                                 "enqueue D", "dequeue"]}
    if re.search(r"linked list", low):
        return {"type": "ds_ops", "structure": "linked_list", "ops": ["insert 10", "insert 20", "insert 30", "remove",
                                                                       "insert 40"]}
    return rules_visual_more(text, low) if atomish else None


def _first_sentences(text, n=2):
    s = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    return " ".join(s[:n])[:420]


def _scene_text(sc):
    txt = sc.get("narration") or ""
    if sc.get("title"):
        txt = sc["title"] + ": " + txt
    return re.sub(r"\s+", " ", txt).strip()


def caption_figures(lessons, ai, log=print, max_pictures=0):
    """Vision model -> 1-2 sentences per picture scene (cached). Returns count."""
    figs = [(L, s, v) for L in lessons for s in L["scenes"] for v in s.get("visuals", [])
            if isinstance(v, dict) and v.get("type") == "figure" and v.get("png")]
    if max_pictures:
        figs = figs[:max_pictures]
    if not figs:
        return 0
    if not (ai and ai.vision):
        from illustrate import ocr_caption
        done = 0
        for _, s, v in figs:
            c = ocr_caption(v["png"])
            if c:
                s["narration"] = (s.get("narration", "") + " " + c).strip()
                done += 1
        return done
    log(f"[vision] {ai.vision}: {len(figs)} pictures")
    t0 = time.perf_counter()
    done = 0
    for j, (_, s, v) in enumerate(figs, 1):
        c = ai.see(v["png"])
        if c:
            s["narration"] = (s.get("narration", "") + " " + c).strip() if \
                not s.get("narration", "").endswith(c) else s["narration"]
            done += 1
        if j % 10 == 0 or j == len(figs):
            el = time.perf_counter() - t0
            log(f"[vision] {j}/{len(figs)} pictures, {el:.0f}s, ETA {el / j * (len(figs) - j):.0f}s")
    return done


def teach_lessons(lessons, ai, log=print, every=4, max_chars=1800, parallel=1):
    """Insert one AI teaching scene after every `every` scenes of each lesson."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = []                                 # (lesson, insert_after_index, prompt)
    for L in lessons:
        sc = L["scenes"]
        i = 0
        while i < len(sc):
            group = sc[i:i + every]
            text, n = [], 0
            for s in group:
                t = _scene_text(s)
                if n + len(t) > max_chars and text:
                    break
                text.append(t)
                n += len(t)
            k = len(text) or 1
            if n > 120:                        # something worth teaching
                prompt = f"Lesson: {L.get('title', '')}\n\n" + "\n\n".join(text)
                jobs.append((L, i + k - 1, prompt, group[:k]))
            i += k
    if not jobs:
        return 0
    drawn = sum(1 for j in jobs if rules_visual(_scene_text(j[3][0])) or rules_visual(j[2]))
    log(f"[teach] {len(jobs)} teaching scenes to make: {drawn} drawn by the engine's rules "
        f"(cell, mitosis, protocol, code trace...), " +
        (f"AI ({ai.model}, {parallel} at once) tutors all of them and draws the other {len(jobs) - drawn}"
         if ai else "no AI: rules only"))
    t0 = time.perf_counter()

    def one(job):
        a = time.perf_counter()
        rv = rules_visual(_scene_text(job[3][0])) or rules_visual(job[2])   # the group's first scene decides first
        if rv:                                   # the engine draws it; the AI only tutors (short, fast call)
            out = _tutor_call(ai, job[2], rv["type"]) if ai else None
            out = dict(out or {}, visual=rv, _rules=True)
        else:
            out = _teach_call(ai, job[2]) if ai else None
            out.setdefault("explain", _first_sentences(job[2].split("\n\n", 1)[-1]))
            out.setdefault("title", job[3][0].get("title", ""))
        return job, out, time.perf_counter() - a

    results = []
    with ThreadPoolExecutor(max(1, parallel)) as ex:
        for j, (job, out, secs) in enumerate(ex.map(one, jobs), 1):
            results.append((job, out))
            if j % 5 == 0 or j == len(jobs):
                el = time.perf_counter() - t0
                log(f"[teach] {j}/{len(jobs)}, last {secs:.1f}s, ETA {el / j * (len(jobs) - j):.0f}s")
    added = 0
    for L in lessons:                          # the same drawing twice in a row is dropped
        last = None
        for job, out in sorted([(j, o) for j, o in results if j[0] is L], key=lambda x: x[0][1]):
            if isinstance(out, dict) and out.get("_rules"):
                key = json.dumps(out["visual"], sort_keys=True, default=str)[:4000]
                if key == last:
                    out.clear()
                last = key
    for L in lessons:                          # insert from the back so indexes stay valid
        mine = sorted([(job, out) for job, out in results if job[0] is L], key=lambda x: -x[0][1])
        for (_, after, _, group), out in mine:
            if not isinstance(out, dict):
                continue
            vis = out["visual"] if out.get("_rules") else _valid(out.get("visual"))
            if not vis:
                continue
            src = group[0].get("source", {})
            L["scenes"].insert(after + 1, {
                "id": (group[-1].get("id") or "S") + "-T",
                "title": ("Drawn: " if out.get("_rules") else "Explained: ") +
                         str(out.get("title") or group[0].get("title", ""))[:60],
                "idea_kind": "teach", "visual_type": "ai:" + vis["type"], "visuals": [vis],
                "note": str(out.get("takeaway") or "")[:110],
                "narration": str(out.get("explain") or "")[:420],
                "units": [], "source": src, "concept": group[0].get("concept", "")})
            added += 1
    return added
