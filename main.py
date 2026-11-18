import json
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from ollama import chat

from typing import Literal, Optional
from pydantic import BaseModel, Field

ENGINE = Path(r"C:\Users\Colli\PycharmProjects\StudyMate\Engine")
PROMPT_FILE = ENGINE / "scene_prompt.txt"
OUT_FILE = ENGINE / "visualization" / "diagram.json"


# ---------- the JSON shape Ollama must return (enforced, so it always parses) ----------

Layout = Literal["flowchart", "sequence", "cycle", "concept_map", "hierarchy", "anatomy",
                 "comparison", "table", "timeline", "network", "architecture", "equation",
                 "graph", "circuit", "geography"]
ElementType = Literal["rectangle", "rounded_rectangle", "circle", "ellipse", "diamond",
                      "text", "equation", "icon_placeholder"]
Relationship = Literal["causes", "produces", "uses", "contains", "part_of", "leads_to",
                       "converts_to", "transports", "compares_with", "depends_on"]


class Element(BaseModel):
    id: str
    type: ElementType
    text: str
    icon: Optional[str]
    color: str
    description: str
    expression: Optional[str] = None


class Connection(BaseModel):
    model_config = {"populate_by_name": True}
    from_: str = Field(alias="from")
    to: str
    label: str
    relationship: Relationship


class Scene(BaseModel):
    scene: int
    title: str
    learning_objective: str
    narration: str
    key_fact: str
    layout: Layout
    elements: list[Element]
    connections: list[Connection]
    x_range: Optional[list[float]] = None
    y_range: Optional[list[float]] = None


class SectionLesson(BaseModel):
    section_title: str
    subject: str
    scenes: list[Scene] = Field(min_length=1, max_length=3)


# ---------- engines ----------

class DoclingEngine:
    def __init__(self):
        # FIX 1: build the converter ONCE (it loads the AI models), and skip OCR /
        # table detection - your PDFs already contain real text.
        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = False
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )

    def open_file(self, file_path):
        result = self.converter.convert(file_path)
        return result.document.export_to_markdown()


class OllamaEngine:
    def __init__(self, model="qwen3:8b"):
        self.model = model

    # FIX 2: take the messages as an argument instead of reading a global
    def generateLesson(self, messages):
        stream = chat(
            model=self.model,
            messages=messages,
            format=SectionLesson.model_json_schema(by_alias=True),  # FIX 3: valid JSON every time
            options={
                "temperature": 0.2,
                "num_ctx": 8192,        # FIX 4: room for the prompt + document
            },
            think=False,                # FIX 5: no hidden "thinking" before the answer
            keep_alive="10m",           # FIX 6: keep the model loaded between runs
            stream=True,
        )
        out = ""
        for chunk in stream:
            piece = chunk["message"]["content"]
            print(piece, end="", flush=True)
            out += piece
        print()
        return out


if __name__ == "__main__":
    # file_path = r"C:\Users\Colli\PycharmProjects\StudyMate\uploads\Biology_Foundations.pdf"
    file_path = r"C:\Users\Colli\PycharmProjects\StudyMate\uploads\cs_B_algorithms_data_structures.pdf"
    # file_path = r"C:\Users\Colli\PycharmProjects\StudyMate\uploads\networking_transport_layer.pdf"

    SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8")   # FIX 7: prompt lives in one file

    t = time.time()
    docling_engine = DoclingEngine()
    print(f"[time] Docling models loaded: {time.time() - t:.1f}s")

    t = time.time()
    markdown_file = docling_engine.open_file(file_path)
    print(f"[time] PDF converted: {time.time() - t:.1f}s  ({len(markdown_file):,} characters)")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Create visual scenes from this section:\n\n{markdown_file}"},
    ]

    t = time.time()
    ollama_lesson = OllamaEngine().generateLesson(messages)
    print(f"[time] Ollama: {time.time() - t:.1f}s")

    lesson = SectionLesson.model_validate_json(ollama_lesson)      # fails loudly if broken
    OUT_FILE.write_text(json.dumps(lesson.model_dump(by_alias=True), indent=2), encoding="utf-8")
    print(f"Saved {len(lesson.scenes)} scene(s) to {OUT_FILE}")