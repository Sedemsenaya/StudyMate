# StudyMate: Local AI Tutorial Generation with Excalimate

## Project direction

Build a locally hosted AI-powered tutorial production pipeline. The user
provides learning material (such as a PDF or Markdown file), and a local
language model plans an educational tutorial with narration, visuals,
and animation instructions. Excalimate is used as the diagram and
animation engine.

The main product focus should be the **tutorial-generation
intelligence**, rather than building a new animation editor from
scratch.

## Proposed architecture

``` text
                 USER
                  |
                  v
          PDF / MARKDOWN INPUT
                  |
                  v
            PYTHON BACKEND
                  |
                  v
              OLLAMA API
          (Qwen / fine-tuned model)
                  |
                  v
         TUTORIAL GENERATOR
                  |
             +----+----+
             |         |
             v         v
         VOICEOVER   SCENE PLAN
             |         |
             v         v
          LOCAL TTS  PYTHON MCP CLIENT
                         |
                         v
                  EXCALIMATE MCP
                   HTTP :3001
                         |
                         v
                   EXCALIMATE APP
                         |
                         v
                     ANIMATION
                         |
                         v
                   VIDEO EXPORT
```

### Component responsibilities

  -----------------------------------------------------------------------
  Component                           Responsibility
  ----------------------------------- -----------------------------------
  Ollama                              Runs the local language model and
                                      exposes its API

  Foundation / fine-tuned model       Understands material and generates
                                      tutorial plans

  Python backend                      Coordinates document processing,
                                      model calls, MCP calls, narration,
                                      and export

  MCP client                          Connects the Python application to
                                      Excalimate's MCP server

  Excalimate                          Creates diagrams, applies
                                      animations, and provides live
                                      preview

  Local TTS                           Converts narration scripts into
                                      voice audio

  FFmpeg (optional)                   Helps combine or process rendered
                                      video and narration
  -----------------------------------------------------------------------

**Important:** Ollama is the model runtime; it is not, by itself, the
MCP client in this architecture. The Python application acts as the
bridge between Ollama and Excalimate.

## Tutorial generation responsibilities

The model should act as an instructional designer and animation
director. For each topic, it should generate a structured tutorial
containing:

1.  **Instructional plan** --- break the topic into clear, progressive
    scenes.
2.  **Narration** --- write a natural voiceover script for each scene.
3.  **Visual plan** --- describe the objects, labels, relationships, and
    layout needed.
4.  **Animation plan** --- specify what appears, moves, changes, or gets
    highlighted, and in what order.
5.  **Timing and transitions** --- estimate scene timing and how scenes
    connect.

Example input:

``` text
Explain the TCP three-way handshake.
```

Example of the intended structured output (illustrative schema):

``` json
{
  "tutorial": "TCP Three-Way Handshake",
  "scenes": [
    {
      "scene": 1,
      "title": "Introduction",
      "narration": "Imagine two computers preparing to communicate reliably.",
      "visuals": [
        {"id": "client", "description": "Client computer"},
        {"id": "server", "description": "Server computer"}
      ],
      "actions": [
        {"type": "create", "target": "client"},
        {"type": "create", "target": "server"}
      ]
    },
    {
      "scene": 2,
      "title": "SYN",
      "narration": "The client begins by sending a synchronization request.",
      "visuals": [
        {"id": "syn", "description": "SYN message traveling from client to server"}
      ],
      "actions": [
        {
          "type": "animate",
          "target": "syn",
          "direction": "client_to_server"
        }
      ]
    }
  ]
}
```

This JSON is a conceptual example, not a confirmed Excalimate MCP
schema. The Python backend must translate the model's scene plan into
the actual arguments accepted by the tools exposed by the installed
Excalimate MCP server.

## Excalimate MCP setup

The Excalimate guide documents an HTTP MCP server.

Start it in a separate terminal:

``` powershell
npx @excalimate/mcp-server
```

By default, the server listens at:

``` text
http://localhost:3001/mcp
```

A custom port can be selected:

``` powershell
npx @excalimate/mcp-server --port 4000
```

Keep the server terminal open while using it.

### Connect the Excalimate application

1.  Start the MCP server.
2.  Open Excalimate.
3.  Click the broadcast icon in the toolbar to connect to the running
    server.
4.  If required, configure the server URL under **File → Preferences**.
5.  Confirm that the live preview is connected.

The Excalimate MCP setup guide says live preview is enabled by default
in HTTP mode. Follow the current in-app connection flow and the server's
terminal output for any pairing or connection details.

### Python MCP client

Install the Python MCP SDK:

``` powershell
pip install mcp
```

The exact client API should be checked against the installed SDK
version. The first integration milestone is simply to connect to the
HTTP MCP endpoint and list the available tools.

Do not assume tool names, argument schemas, or export capabilities.
Discover them from the running server using MCP's `list_tools`
operation, then inspect each tool's description and input schema.

## Ollama setup and model testing

Ollama is already installed and responding locally. Its API is normally
available at:

``` text
http://localhost:11434
```

Check installed models:

``` powershell
ollama list
```

Current locally available models:

  --------------------------------------------------------------------------
  Model                               Approximate size Potential role
  ----------------------- ---------------------------- ---------------------
  `qwen2.5-coder:7b`                            4.7 GB Code and structured
                                                       animation
                                                       instructions

  `qwen2.5:7b-instruct`                         4.7 GB Instructional
                                                       explanations and
                                                       narration

  `llama3.2:latest`                             2.0 GB Lightweight
                                                       experimentation

  `llama3.2:1b`                                 1.3 GB Simple lightweight
                                                       tasks

  `qwen2.5-coder:3b`                            1.9 GB Lightweight
                                                       code-generation
                                                       experiments
  --------------------------------------------------------------------------

Start with the existing `qwen2.5:7b-instruct` model before downloading
or training another model.

``` powershell
ollama run qwen2.5:7b-instruct
```

A first test prompt:

``` text
You are an educational animation director.

Create a tutorial explaining the TCP three-way handshake.

Requirements:
1. Break the explanation into scenes.
2. Give every scene a voiceover script.
3. Describe the visual elements.
4. Describe the animation sequence.
5. Explain concepts progressively.
6. Avoid unnecessary technical complexity.
7. Return valid JSON matching the requested schema.
```

Also test the local API:

``` python
import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "qwen2.5:7b-instruct",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain the TCP three-way handshake "
                    "as a five-scene animated tutorial."
                )
            }
        ],
        "stream": False
    },
    timeout=300
)

response.raise_for_status()
print(response.json()["message"]["content"])
```

## Fine-tuning strategy

Do not fine-tune the model before the end-to-end workflow is understood.

Recommended stages:

### Stage 1 --- Prompting and structured output

-   Use an existing Ollama model.
-   Define a stable tutorial output schema.
-   Require valid structured output.
-   Test it on several subjects.
-   Record failures and revise the prompt and schema.

### Stage 2 --- Build an approved example dataset

Manually create and approve approximately 5--10 high-quality tutorials.

Save each example with: - Source topic or learning material. - Scene
breakdown. - Narration. - Visual descriptions. - Animation
instructions. - Any corrections made after rendering.

These examples establish the target teaching style and output
conventions.

### Stage 3 --- Fine-tune

Once the schema and desired behavior are stable, evaluate fine-tuning an
open-weight foundation model with a method such as LoRA or QLoRA.
Unsloth is one possible training framework.

The available GPU model and VRAM must be checked before selecting a
model size or training configuration.

### Stage 4 --- Local deployment

A possible deployment workflow is:

1.  Fine-tune the selected foundation model.
2.  Evaluate the fine-tuned model on tutorials it has not seen.
3.  Save the adapter or merge it with the base model as appropriate.
4.  Export to a format supported by the intended runtime.
5.  Import the model or compatible adapter into Ollama.
6.  Test generation through the local Ollama API.

The exact export and import procedure depends on the base model, adapter
format, quantization, and Ollama compatibility. Fine-tuning does not
automatically guarantee that the resulting model will be directly
importable.

## Voiceover and video synchronization

Treat narration and animation as related but independently generated
assets.

Initial approach:

1.  Generate narration text for each scene.
2.  Generate audio with a local text-to-speech engine.
3.  Render the Excalimate animation.
4.  Use scene timing or audio duration to synchronize the narration and
    visuals.
5.  Combine or process the outputs with a video tool such as FFmpeg when
    needed.

Piper and Kokoro are possible local TTS options to evaluate. Compare
voice quality, installation requirements, licensing, and hardware
performance before choosing one.

## Development milestones

### Milestone 1 --- Verify local services

-   [X] Confirm Ollama responds at `localhost:11434`.
-   [X] Confirm the selected model generates a useful tutorial plan.
-   [X] Start Excalimate MCP with `npx @excalimate/mcp-server`.
-   [X] Connect Excalimate's live preview.
-   [X] Confirm the MCP endpoint is reachable.

### Milestone 2 --- Discover and test MCP tools

-   [X] Connect from a small Python MCP client.
-   [X] List all available tools.
-   [x] Inspect tool descriptions and input schemas.
-   [X] Call one simple creation tool.
-   [X] Confirm the result appears in Excalimate.
-   [X] Test an animation operation.

### Milestone 3 --- Generate a complete scene

-   [ ] Ask Ollama for one structured scene.
-   [ ] Validate the JSON before execution.
-   [ ] Translate the scene into actual Excalimate MCP calls.
-   [ ] Render and review the scene.
-   [ ] Correct the plan or mapping when necessary.

### Milestone 4 --- Produce a narrated tutorial

-   [ ] Generate multiple scenes.
-   [ ] Generate local voiceover audio.
-   [ ] Render the animation.
-   [ ] Synchronize narration and visuals.
-   [ ] Export and review a complete video.

### Milestone 5 --- Improve the model

-   [ ] Collect approved tutorial examples.
-   [ ] Build a training and evaluation dataset.
-   [ ] Select a foundation model based on GPU VRAM.
-   [ ] Fine-tune and evaluate.
-   [ ] Deploy the resulting model locally through Ollama.

## Key design principles

-   **Model decides; tools execute.** The model plans the teaching
    sequence and desired visuals. Excalimate performs the actual drawing
    and animation operations.
-   **Validate before execution.** Never send unvalidated model output
    directly to MCP tools.
-   **Use real tool schemas.** Discover the running server's tool
    definitions rather than inventing MCP tool names or arguments.
-   **Keep components replaceable.** The LLM, TTS engine, animation
    engine, and video processor should be independent modules.
-   **Build before fine-tuning.** A working baseline reveals what the
    model actually needs to learn.
-   **Keep the MVP local.** Use Ollama, a local Python backend,
    Excalimate's local MCP server, and local TTS wherever practical.

## MVP target

The first successful MVP should allow a user to provide a short topic or
Markdown document, then:

1.  Generate a clear scene-by-scene lesson locally.
2.  Generate narration for each scene.
3.  Create and animate the visuals through Excalimate MCP.
4.  Produce a narrated video export.

The initial goal is one reliable end-to-end tutorial---not a fully
autonomous production system.
