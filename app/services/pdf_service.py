import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

def procesar_pdf(file_path: str) -> list[str]:
    # 1. Cargar el PDF
    doc = fitz.open(file_path)
    texto_completo = "".join([page.get_text() for page in doc])

    # 2. Dividir en fragmentos (1000 caracteres, 200 de solapamiento)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # 3. Retornar la lista de textos
    return text_splitter.split_text(texto_completo)
