# StudyMate v4

PDFs (one or 100+) → **narrated**, hand-drawn, animated lesson videos, in parallel, with run-time logs.

```
PDF ──► extract ──► IR ──► visual selection ──► critic (evaluate → refine) ──► Excalidraw scenes
         (pdfium /         (concept, idea kind,        (coverage, overflow,          │
          Docling)          source pages, units)        long labels, dupes)          ▼
                                                     voice (Piper/Kokoro) ──► render tabs ──► mux ──► lesson MP4
```

## Run
```
.venv\Scripts\activate
pip install pypdfium2 piper-tts playwright imageio-ffmpeg sympy
playwright install chromium
mkdir Engine\voices   &  copy en_US-lessac-medium.onnx + .onnx.json into it  (v4 also finds them in Downloads)

python main_v4.py "C:\Users\Colli\Downloads\AWS Certified Solutions Architect Slides v49 (reduced).pdf"
python main_v4.py "...Slides v49 (reduced).pdf" --pages 194-267          # just Route 53
python main_v4.py C:\path\to\folder_with_100_pdfs                          # whole folder, recursive
python main_v4.py notes.pdf --planner llm --model qwen3:8b --llm-parallel 2
python main_v4.py uploads\math_02_pythagorean_theorem.pdf --planner llm --model qwen3:8b --llm-parallel 1
```

## Options
| option | default | what |
|---|---|---|
| `--planner auto/rules/llm` | auto | slides → rules engine (instant, word for word); documents → Ollama (rules if Ollama is off) |
| `--workers N` | cores-1 | processes reading PDFs (big decks split in 50-page chunks) |
| `--tts-workers N` | cores/4 | voice threads |
| `--render-workers N` | cores/3 | Chromium tabs drawing scenes at the same time |
| `--llm-parallel N` | 2 | Ollama requests at once. **Also set** `setx OLLAMA_NUM_PARALLEL 2` and restart Ollama |
| `--tts piper/kokoro/none` | piper | `none` = captions only (fastest) |
| `--max-slides` | 40 | slides per lesson video (long sections → Part 1, Part 2…) |
| `--fps`, `--size` | 24, 1280x720 | lower fps = faster render |
| `--max-lessons N` | all | test on the first N lessons |

## Output (nothing is overwritten; re-runs skip finished scenes)
```
Engine/Output/
  batch_<stamp>/batch.log  summary.csv  summary.json        ← run times for every stage + every lesson
  <pdf>__<lesson>/
     run.log         one line per scene: video length, tts s, render s, frames drawn, mux s
     timings.json    the same, machine-readable
     ir.json         the intermediate representation (concept, idea kind, visual, source pages)
     critic.json     every check that failed and how it was fixed, coverage score
     scenes/001_<title>.excalidraw / .mp4
     <lesson>_StudyMate.mp4
  .tts_cache/        narration wavs, reused across runs
```

## Files
| file | job |
|---|---|
| `extract.py` | slides: title, bullets (+indent), diagram labels and a crop of the slide's own diagram; docs: Docling/pdfium; splits decks at divider slides |
| `select_visual.py` | IR + visual selection engine: idea kind (definition, process, parts, comparison, figure, formula, list) → visual, subject independent |
| `critic.py` | evaluate → refine: coverage, overflow split, long labels, empty, duplicates |
| `figure.py` | "figure" visual (the slide's diagram, framed) |
| `voice.py` | Piper / Kokoro with a cache |
| `render4.py` | Chromium tab pool; scene length follows its audio; mux |
| `batch.py` | the parallel pipeline + logs |

v3 (`Engine/v3`) is reused for drawing (visuals.py, scene.py, web/bundle.js) and for the Ollama planner.

## Practice tests: AI illustration mode (new)
Results PDFs ("Question 12 Skipped ... Correct answer ... Overall explanation ... Domain") are
detected automatically and read with `quiz.py` (no Docling, ~6 s for 174 pages).
Each question becomes 4-6 scenes: the scenario drawn as an AWS architecture, the choices,
the answer drawn with the answer's service circled, why each other option fails, and the
pictures from the explanation (read by a vision model).

The AI only draws: one short call per question (thinking off, JSON schema, <=520 tokens,
time limit) instead of writing the whole lesson. Slow or bad answers fall back to the
rules sketch for that question; every answer is cached in `Engine/Output/.ai_cache`.

```
ollama pull qwen3:8b            # text (or qwen3:4b: about 2x faster on a CPU)
ollama pull qwen2.5vl:3b        # pictures (optional; without it: --vision ocr or off)
python main_v4.py "uploads\Practice_Test#3-....pdf" --questions 1-10      # first test
python main_v4.py "uploads\Practice_Test#3-....pdf"                       # all 65
```
| option | default | what |
|---|---|---|
| `--ai auto/on/off` | auto | AI sketches if Ollama is running |
| `--model` | qwen3:8b | text model for the sketches |
| `--vision` | auto | first vision model you have pulled; or a name, `ocr` (pip install rapidocr_onnxruntime), `off` |
| `--ai-timeout` | 150 | seconds per AI call before the rules sketch is used |
| `--llm-parallel` | 1 | 1 on a CPU; 2-4 on a GPU (also set OLLAMA_NUM_PARALLEL) |
| `--ollama-host` | local | another PC's Ollama, e.g. Louis's GPU over Tailscale |
| `--questions` | all | e.g. `1-10` |
| `--per-lesson` | 10 | questions per video |
| `--sketches file.json` | | hand-written sketches/captions per question (see samples/) |
`--pages` now works for documents and practice tests too.
Icons: `icons_aws.json` (AWS logos CC0, Fluent Emoji MIT; licenses inside the file).

## Every PDF: pictures + AI teaching scenes (new)
Slides and notes now get the same treatment as practice tests:
- pictures are kept (slide diagrams were already cropped; pictures in documents are now added
  as scenes at their page) and the vision model describes each one in the narration
- every 4 scenes, one short AI call adds an "Explained: ..." scene that draws the idea in a new
  way (architecture with icons, flow, cycle, comparison table, timeline, worked equation, tree)
  with a plain explanation and a takeaway. A failed/slow call just means no extra scene there.
- documents now use the fast pdfium reader by default (Docling took ~280 s); `--docling` brings it back
- the old "AI writes the whole lesson" planner is still there: `--planner llm`

| option | default | what |
|---|---|---|
| `--teach-every N` | 4 | one AI teaching scene per N scenes (0 = off) |
| `--vision-max N` | 0 (all) | describe at most N pictures (pictures are the slowest AI step on a CPU) |
| `--plan-only` | | stop after planning: check ir.json / scene files before rendering |
| `--ai off` | | no AI at all (rules only, fastest) |

## The engine draws the thing itself (new: drawlib.py, tracer.py)
When a lesson is about something the engine can draw, the teaching scene is drawn
exactly by rules (no AI needed); the AI only adds the explanation:
| idea in the text | drawn as |
|---|---|
| mitosis / prophase, metaphase... | one cell going through every phase: chromosomes condense, line up, split, two nuclei, the cell pinches in two |
| organelles (nucleus, mitochondria, ER, Golgi...) | an animal or plant cell with each organelle drawn and labelled one by one |
| TCP handshake / SYN | client and server with SYN, SYN-ACK, ACK, data, ACK (FIN if mentioned) |
| DNS | laptop -> resolver -> root -> TLD -> authoritative and back |
| UDP, packets and routers | devices in a row and a packet (envelope) travelling hop by hop |
| Python code in the PDF, or binary search / linear search / bubble, insertion, selection sort | the code really runs (tracer.py, 3 s limit, separate process): running line highlighted, variables, the array with lo/mid/hi arrows, step by step |
| servers, databases, laptops, phones, routers, firewalls, clouds, users, files... in any diagram | drawn by the engine (AWS services keep their logos) |
The AI can choose these too (`cell`, `mitosis`, `packet_flow`, `sequence`, `code_trace`).
Moving things use two new timeline keys in render4: `until` (disappear at step k) and `fade`.

## More drawings (drawlib2.py) - picked by the rules or chosen by the AI
| subject | type | what you see (computed for real where it says so) |
|---|---|---|
| OS (CS 202/153) | `gantt` | FCFS / SJF / Round Robin schedule computed, waiting times |
| algorithms (CS 141) | `graph_algo`, `ds_ops` | BFS / DFS / Dijkstra run on the graph; stack, queue, linked list operations |
| architecture (CS 203/162) | `pipeline`, `hierarchy`, `logic` | 5-stage pipeline with stalls; memory pyramid; gates with truth table |
| GPU (CS/EE 217) | `gpu_grid` | grid -> blocks -> threads, global index formula |
| assembly | `asm_trace` | tiny x86-style interpreter: registers and stack step by step |
| compilers (CS 201) | `ast` | tokens from the lexer + the real Python syntax tree |
| SQL / data (CS 226) | `sql` | the query runs in SQLite; tables and result drawn |
| Redis | `sequence` | cache-aside: miss, read DB, SET with TTL, hit |
| ML / DL / NLP (CS 224/228/222/229) | `gradient`, `kmeans`, `neural_net`, `attention` | gradient descent and k-means really run; forward pass; attention weights |
| chemistry | `atom`, `molecule` | Bohr model of elements 1-20; H2O, CO2, CH4, NH3, O2, N2, HCl, NaCl, C2H6 |
| physics | `circuit`, `projectile`, `free_body` | Ohm's law computed with current flowing; real trajectory; force arrows |
| Docker / Kubernetes / VPC / Azure | `nested` | boxes in boxes with logos |
| Terraform | `sequence` | write -> init -> plan -> apply -> state |
| React | `tree` | component tree with props/state |
| GIS | `gis_layers` | basemap, raster, polygons, lines, points stacked |
| networks (CS 204) | `osi`, `topology` | encapsulation layer by layer; Cisco-style routers/switches, ports, IPs, packet path |
| security | `sequence` | man-in-the-middle, phishing, public-key exchange (attacker drawn) |
| maths | `right_triangle`, `plot`, `unit_circle` | Pythagoras, quadratic roots, least-squares fit computed |
Logos bundled in icons_aws.json: 65 AWS services + 76 tools (Docker, Kubernetes, Helm, Terraform, Redis,
Azure, GCP, React, Postgres, MySQL, Mongo, Python, Kafka, Spark, PyTorch, TensorFlow, Nginx, Linux ...).

## Drawing library part 3 (drawlib3.py): 38 more
| subject | types |
|---|---|
| algorithms / data structures | `sort_bars` (bubble, insertion, selection, every swap computed), `hash_table` (buckets + chaining), `bst`, `heap` (sift-up), `recursion` (call stack), `fsm` (DFA run on an input) |
| systems | `page_table` (virtual -> physical computed), `mapreduce` (word count), `git_graph` (branch + merge) |
| networks / security | `subnet` (CIDR maths), `binary` (division method + hex), `blockchain` (real SHA-256 links, tamper breaks the chain), `cipher` (Caesar), `firewall` (ordered rules, allow/deny computed), `load_balancer`, `ddos`, `layers_ring` (defense in depth, CIA triad) |
| data / ML / NLP | `convolution` (each output computed), `confusion` (accuracy, precision, recall, F1), `stats` (histogram + box plot, quartiles, outliers), `bpe` (real merges) |
| biology | `photosynthesis`, `dna` (helix + replication), `central_dogma` (transcription + translation with the full genetic code), `heart` (blood path), `neuron` (signal travels) |
| earth / chemistry / physics | `water_cycle`, `states_matter`, `ph_scale`, `wave` (moving), `refraction` (Snell's law computed), `solar_system` |
| maths / economics | `venn` (regions computed), `prob_tree` (products + Bayes), `matrix_mult` (cell by cell), `riemann` (n = 2, 4, 8, 16 vs exact), `vectors` (resultant), `supply_demand` (equilibrium solved) |
The rules check the first scene of each group first, so a section's own topic wins over a neighbour's.

## Drawing library part 4 (drawlib4.py): 41 more
| subject | types |
|---|---|
| algorithms | `dp_table` (LCS filled + traceback), `merge_sort` (split/merge levels), `huffman` (tree, codes, bits saved), `trie`, `topo_sort` (Kahn) |
| systems | `lru` (LRU/FIFO faults), `cache_sim` (direct-mapped hits/misses), `tcp_cwnd` (slow start, AIMD), `deadlock`, `race` (lost update), `event_loop` (JS), `twos_complement` |
| web / security | `jwt` (real base64 + HMAC-SHA256), `sql_injection` (the built query + the fix), `password_hash` (salt + SHA-256) |
| cloud / DevOps | `cicd` (logos), `autoscaling` (instances follow load) |
| ML / data | `decision_tree` (Gini split computed), `knn` (vote), `normal_dist`, `softmax`, `overfitting`, `er_diagram`, `groupby` (pandas) |
| biology | `punnett` (ratios computed), `meiosis`, `respiration` (ATP), `osmosis` |
| earth / space | `plate_tectonics`, `moon_phases` (lit fraction computed), `earth_layers` |
| physics | `lens` (thin-lens equation), `pendulum` (period), `energy_ramp` (PE/KE bars) |
| chemistry | `balance` (atom counts + finds the coefficients), `reaction_energy` (activation energy, catalyst) |
| maths / business | `tangent` (derivative), `compound` (vs simple interest), `linear_system` (solved), `break_even` |
| GIS | `gis_buffer` (buffer a river, select houses inside) |

## Drawing library part 5 (drawlib5.py): 52 more + 6 protocol exchanges

Checked first (`teach.rules_visual_5`), then part 4, part 3 and the main rules. (computed) = the engine works it out.

- **Algorithms / data structures:** quick_sort (Lomuto, computed), maze_bfs (computed), bloom_filter (real hashes, shows a false positive), consistent_hash (ring; removes the busiest server and shows which keys move), list_reverse, sliding_window (computed)
- **Systems / architecture:** memory_layout, fork_tree (binary process tree), disk_sched (FCFS/SSTF/SCAN totals, computed), raid (0/1/5 with real XOR parity and a rebuild), cpu_cycle (tiny accumulator machine run step by step), ieee754 (real float bits), utf8 (real bytes)
- **Networks:** nat_table, routing_lpm (longest prefix match, computed), traceroute; sequences for DHCP, ARP, TLS handshake, OAuth, email (SMTP/IMAP), digital signatures, HTTP request
- **Security:** buffer_overflow (stack frame, return address overwritten), brute_force (key space and crack time, computed)
- **ML / DL / NLP:** backprop (forward values and chain-rule gradients, computed), roc_curve (AUC computed), grid_world (value iteration + policy arrows), pca (eigenvectors, computed), word_vectors (king − man + woman), transformer_block
- **Data / SQL:** sql_joins (inner/left/right/full, computed), normalization (to 3NF), star_schema, moving_average, correlation (Pearson r)
- **Maths / probability:** sieve, euclid_gcd, pascal, binomial, bayes_test (1000 people, natural frequencies), monte_carlo_pi
- **Physics / chemistry:** collision (momentum and energy checked), incline (force components), gas_law (PV = nRT), half_life, doppler, electron_config (Aufbau + Hund)
- **Biology / earth / GIS:** food_web, energy_pyramid, natural_selection (allele frequency), seasons, greenhouse, rock_cycle, map_projection (Mercator scale), voronoi, choropleth (equal-interval classes)
- **Economics:** amortization (real loan schedule), ppf

Rules are kept strict to avoid false matches: "https" in a link doesn't trigger TLS, DNS "TTL" doesn't trigger traceroute, "revolution" doesn't trigger evolution, RLHF slides don't get a grid world, and an HTTP rule never overrides a TCP handshake.
Gallery: `gallery5.py` renders all 58.

## Drawing library part 6 (drawlib6.py): 55 more + 7 protocol exchanges

Checked first (`teach.rules_visual_6`). (computed) = the engine works it out.

- **Algorithms:** kruskal (union-find MST), kmp (prefix table + search), n_queens (backtracking), avl (rotations), big_o, b_tree (splits + search path) — all computed
- **Systems / networks:** cpu_pipeline (load-use stall), producer_consumer (semaphores), go_back_n (lost packet), ip_header (real checksum), vpn_tunnel, cdn (hit rate), acid_transaction (rollback then commit)
- **Cloud / DevOps / web:** docker_layers, rolling_update (maxSurge 1), terraform_plan (desired vs state → + ~ -), kafka_partitions (key hashing, consumer group), virtual_dom (React diff), token_bucket (rate limiter)
- **Security:** diffie_hellman, rsa (small primes, real modular maths); sequences for XSS, CSRF, Kerberos
- **ML / NLP:** linear_regression (least squares, R²), logistic_regression (gradient descent), rnn_unrolled, cross_validation (k folds), tfidf, perceptron
- **Statistics:** boxplot (quartiles, fences, outliers), histogram (Sturges), central_limit (simulated), hypothesis_test (z, p-value)
- **Maths:** taylor_series, fourier_series, newtons_method, eigen_transform, number_bases, modular_clock
- **Physics / chemistry:** spring_shm, lever_torque, buoyancy, electric_field (Coulomb), em_spectrum, titration (pH curve), limiting_reagent, vsepr
- **Biology / GIS / economics:** enzyme (Michaelis-Menten), action_potential, hardy_weinberg, population_growth, idw_interpolation, topographic, raster_vector, spatial_join (point in polygon), npv, elasticity
- **More sequences:** two-phase commit, Redis cache-aside, pub/sub, WebSockets

Also fixed in `v3/mathtext.py`: identifiers like `student_id` or `aws_db_instance` were shown as subscripts (`student_(id)`); real subscripts like `x_1` still work.
Rules are strict on purpose: "enzyme" in a DNA chapter keeps its DNA drawing, "exponential function" in maths doesn't become population growth, and O(log n) in a binary search PDF keeps the binary search trace.
Gallery: `gallery6.py` renders all 62.

## Drawing library part 7 (drawlib7.py): 68 more + 4 exchanges (301 drawing types in total)

Checked first (`teach.rules_visual_7`). Nearly everything here is computed by the engine.

- **Algorithms:** a_star (vs BFS cell count), floyd_warshall, radix_sort, interval_scheduling, coin_change (greedy vs DP)
- **Compilers / runtimes:** lexer (real tokenizer), gc_mark_sweep (cycles freed)
- **Architecture / digital logic:** branch_predictor (2-bit vs 1-bit accuracy), amdahl, ripple_adder, karnaugh (Quine-McCluskey), endianness, mips_encoding (R/I-type bits + hex), memory_hierarchy
- **OS:** bankers (safe sequence), memory_alloc (first/best/worst fit)
- **Networks / Cisco:** distance_vector, hamming_code (fixes a flipped bit), crc, manchester, acl_rules (first match, implicit deny), vlan
- **AWS / distributed:** aws_vpc (real AWS icons), iam_policy (explicit deny > allow > implicit deny), circuit_breaker, raft_election, sharding (range vs hash hotspot)
- **Security:** merkle_tree (real SHA-256), port_scan, cert_chain; sequences for MFA and SSH keys
- **ML / LLMs:** activations, naive_bayes, dbscan, dendrogram, cosine_similarity, temperature_sampling, positional_encoding, learning_rate, rag_pipeline
- **Data:** window_functions (real SQLite), standardization (z-scores)
- **Maths:** dot_product, complex_plane, quadratic (reads y = ax² + bx + c from the text), sequences, markov_chain, truth_table, permutations, euler_path
- **Physics / chemistry:** rc_circuit, standing_waves, double_slit, circular_motion, kepler_orbit, heating_curve, photoelectric, maxwell_boltzmann, galvanic_cell
- **Biology:** pcr, gel_electrophoresis (real EcoRI digest), phylogenetic_tree (UPGMA), feedback_loop (glucose/insulin)
- **Earth / GIS:** ndvi, epicenter (S−P triangulation), slope_hillshade (from a DEM)
- **Economics:** lorenz_gini, prisoners_dilemma (Nash computed), comparative_advantage
- **More sequences:** serverless (API Gateway → Lambda → DynamoDB), interrupt handling

Guards against false matches: "Gini" in a decision-tree page stays a decision tree, "quadratic time" in Big-O stays Big-O, and X-ray "diffraction" in a DNA chapter doesn't become the double slit.
Gallery: `gallery7.py` renders all 72.
