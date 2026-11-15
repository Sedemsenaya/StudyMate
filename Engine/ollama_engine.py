from docling_engine import DoclingEngine
from add_icons import available_icons

from ollama import chat

class OllamaEngine:
    def generateLesson(self):
        stream = chat(
            model='llama3.2',
            messages=message,
            options={'temperature': 0.2},
            stream=True,
        )

        out = ""  # FIX: collect the text as it streams
        for chunk in stream:
            piece = chunk['message']['content']
            print(piece, end='', flush=True)
            out += piece
        return out


if __name__ == "__main__":
    docling_engine =  DoclingEngine()
    markdown_file = docling_engine.open_file()
    ollama_engine = OllamaEngine()

    SYSTEM_PROMPT = """You are an expert teacher who turns documents into visual Mermaid lessons with narration (put narration in %% ) and output mermaid syntax."""


    message = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f':\n\n{markdown_file}'},
    ]
    ollama_lesson = ollama_engine.generateLesson()
    with open("diagram.mmd", "w", encoding="utf-8") as file:
        file.write(ollama_lesson)
