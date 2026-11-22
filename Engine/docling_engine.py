import os

from docling.document_converter import DocumentConverter

# Default sample PDF (relative to the project root, works on any OS)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PDF = os.path.join(PROJECT_ROOT, "uploads", "biology_photosynthesis.pdf")


class DoclingEngine:

    def open_file(self, file_path: str = DEFAULT_PDF):
        converter = DocumentConverter()
        result = converter.convert(file_path)
        markdown = result.document.export_to_markdown()
        return markdown


if __name__ == "__main__":
    docling_engine = DoclingEngine()
    content = docling_engine.open_file()

    print(content)
    print()
