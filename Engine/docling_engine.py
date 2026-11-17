from docling.document_converter import DocumentConverter


class DoclingEngine:

    def open_file(self):
        # file_path =  r"C:\Users\Colli\PycharmProjects\StudyMate\uploads\transport_layer_test.pdf"
        # file_path =  r"C:\Users\Colli\PycharmProjects\StudyMate\uploads\biology_photosynthesis.pdf"
        file_path =  r"C:\Users\Colli\PycharmProjects\StudyMate\uploads\networking_transport_layer.pdf"
        converter = DocumentConverter()
        result = converter.convert(file_path)
        markdown = result.document.export_to_markdown()
        return markdown

if __name__ == "__main__":
    docling_engine = DoclingEngine()
    content = docling_engine.open_file(
        # r"C:\Users\Colli\PycharmProjects\StudyMate\uploads\transport_layer_test.pdf"
    )

    print(content)
    print()