from docling.document_converter import DocumentConverter
from ollama import chat


class DoclingEngine:

    def open_file(self):
        # file_path =  r"C:\Users\Colli\PycharmProjects\StudyMate\uploads\transport_layer_test.pdf"
        # file_path =  r"C:\Users\Colli\PycharmProjects\StudyMate\uploads\biology_photosynthesis.pdf"
        file_path =  r"C:\Users\Colli\PycharmProjects\StudyMate\uploads\networking_transport_layer.pdf"
        converter = DocumentConverter()
        result = converter.convert(file_path)
        markdown = result.document.export_to_markdown()
        return markdown



class OllamaEngine:
    def generateLesson(self):
        stream = chat(
            model='qwen3:8b',
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
    docling_engine = DoclingEngine()
    markdown_file = docling_engine.open_file()
    ollama_engine = OllamaEngine()

    SYSTEM_PROMPT = """
    You are an expert educational content generator.

    Your task is to transform documents into structured visual lessons using Mermaid.js flowchart syntax.

    STRICT OUTPUT RULES:
    1. Output ONLY valid Mermaid syntax.
    2. Do NOT include Markdown code fences.
    3. Do NOT include introductory text, explanations, summaries, or conclusions outside Mermaid syntax.
    4. Do NOT output Python, HTML, JSON, XML, or any other programming language.
    5. Do NOT include unsupported Mermaid syntax.
    6. All narration, explanations, scene titles, and teaching notes MUST be written inside Mermaid comments using %%.
    7. Use graph TD as the diagram declaration.
    8. Use Mermaid nodes and connections to represent educational concepts and relationships.
    9. Organize complex lessons into clearly labeled subgraphs.
    10. Ensure all node IDs are unique.
    11. Keep node labels concise and readable.
    12. Preserve the important educational information from the source document.
    13. Organize the lesson into sequential scenes.

    SCENE STRUCTURE:
    For every scene, include:
    %% scene: [scene number]
    %% title: [scene title]
    %% explain: [educational explanation]
    %% voice: [natural narration for the scene]

    DIAGRAM STRUCTURE:
    - Use subgraph to organize each scene.
    - Use nodes to represent concepts, processes, and examples.
    - Use arrows to demonstrate relationships, dependencies, and sequences.
    - Use appropriate Mermaid flowchart syntax.
    - Ensure connections are logically correct.
    - Make the diagram educational and easy to follow.

    FINAL VALIDATION:
    Before responding, verify that:
    - The entire output is valid Mermaid syntax.
    - Every narration and explanation is inside %% comments.
    - No content exists outside the Mermaid diagram.
    - No Markdown fences are present.
    - The output can be passed directly to a Mermaid renderer.

    Return ONLY the Mermaid source.
    """


    message = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f':\n\n{markdown_file}'},
    ]
    ollama_lesson = ollama_engine.generateLesson()
    with open("Engine/excalidraw-app/public/diagram.mmd", "w", encoding="utf-8") as file:
        file.write(ollama_lesson)