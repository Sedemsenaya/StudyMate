from docling_engine import DoclingEngine

from ollama import chat

# Use a model tag you actually have pulled (`ollama list`).
MODEL = "qwen2.5:7b-instruct"

SYSTEM_PROMPT = (
    "You are an expert teacher who turns documents into a visual Mermaid "
    "lesson with narration. Output ONLY valid Mermaid syntax.\n"
    "\n"
    "STRICT RULES:\n"
    "1. Node IDs must be a single token with NO spaces (use camelCase), and the\n"
    "   human-readable label goes in brackets. Example:\n"
    "       lightDependent[Light-dependent reactions]\n"
    "       calvinCycle[Calvin cycle]\n"
    "   Never write an ID with a space in it (NOT 'calvin cycle-->stroma').\n"
    "2. Reuse the same ID once defined; connect nodes with -->.\n"
    "3. Put narration as comment lines starting with %% .\n"
    "4. Define each node once. Do not repeat lines.\n"
    "5. Output ONLY the raw Mermaid diagram: no Markdown code fences, nothing else.\n"
    "\n"
    "Example of the required style:\n"
    "graph LR\n"
    "    co2[Carbon dioxide] --> calvinCycle[Calvin cycle]\n"
    "    calvinCycle --> glucose[Glucose]\n"
    "    %% The Calvin cycle fixes carbon dioxide into sugar."
)


class OllamaEngine:
    def generateLesson(self, markdown_file: str):
        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f":\n\n{markdown_file}"},
        ]

        stream = chat(
            model=MODEL,
            messages=message,
            options={
                "temperature": 0.2,
                "repeat_penalty": 1.3,  # discourage the endless-loop failure mode
                "num_predict": 900,     # hard cap so it can never loop forever
            },
            stream=True,
        )

        out = ""  # collect the text as it streams
        for chunk in stream:
            piece = chunk["message"]["content"]
            print(piece, end="", flush=True)  # watch the model think in real time
            out += piece
        return out


def strip_code_fences(text: str) -> str:
    """Remove a surrounding ```mermaid ... ``` fence if the model added one,
    so diagram.mmd holds raw Mermaid that a parser can render directly."""
    lines = text.strip().splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    print("=== STAGE 1: Docling — reading PDF into Markdown ===")
    docling_engine = DoclingEngine()
    markdown_file = docling_engine.open_file()
    print(markdown_file)

    print("\n\n=== STAGE 2: Ollama — generating Mermaid lesson (streaming) ===")
    ollama_engine = OllamaEngine()
    ollama_lesson = ollama_engine.generateLesson(markdown_file)

    with open("diagram.mmd", "w", encoding="utf-8") as file:
        file.write(strip_code_fences(ollama_lesson))
    print("\n\n=== Saved Mermaid output to diagram.mmd ===")
