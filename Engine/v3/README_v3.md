# StudyMate v3

PDF → Docling → Ollama (free JSON, any number of scenes) → hand-drawn Excalidraw scenes
→ local stroke-by-stroke animation → MP4. No Excalimate website, no MCP, no npm.

## Run
```
.venv\Scripts\activate
pip install ollama sympy playwright imageio-ffmpeg
playwright install chromium
python main_v3.py uploads\cs_B_algorithms_data_structures.pdf
python main_v3.py uploads\math_B_geometry_and_trigonometry.pdf --size 1920x1080
python main_v3.py --from-json Engine\Output\<folder>\lesson.json      # edit JSON, re-render
python Engine\v3\render.py Engine\Output\<folder>\scenes               # re-render edited .excalidraw
```

## Output (never overwrites)
```
Engine/Output/algorithms_and_data_structures_20261118-1006/
  lesson.json
  scenes/01_what_is_an_algorithm.excalidraw  .timeline.json  .mp4
  scenes/02_searching_linear_vs_binary_search.mp4 ...
  Algorithms_And_Data_Structures_StudyMate.mp4
```

## Files
| file | job |
|---|---|
| `prompt_v3.txt` | what Ollama is told: cover everything, explain, add worked examples, pick any visual |
| `planner.py` | one Ollama call per section, tolerant JSON parsing, coverage check (nothing missing) |
| `visuals.py` | flow, diagram, tree (+search path), table, plot, bar, array trace, stack, queue, equation, right triangle (+squares), unit circle, area shapes, cycle, timeline, list |
| `mathtext.py` | `sin^2(x)` → `sin²(x)`, `pi x r^2` → `π × r²`, LaTeX → Unicode (math stays real text) |
| `scene.py` | title, sticky note, paged captions, step timeline |
| `render.py` + `web/` | local Chromium draws Excalidraw SVG stroke by stroke, ffmpeg encodes |
| `pipeline.py` | names files, renders every scene, joins the lesson video |
