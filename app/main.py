import os
import shutil
import traceback
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF

from app.config import DOCUMENTS_DIR
from app.services.rag_service import (
    guardar_chunks,
    listar_documentos,
    reset_vectorstore,
    responder_pregunta,
    hay_documentos
)

app = FastAPI(
    title="Chatbot RAG API",
    version="1.0.0"
)

# Configuración de CORS para permitir conexiones desde Vercel o local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear la carpeta temporal de documentos si no existe al arrancar
os.makedirs(DOCUMENTS_DIR, exist_ok=True)


def extraer_chunks_de_pdf(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Extrae el texto de un archivo PDF y lo divide en fragmentos (chunks)."""
    doc = fitz.open(file_path)
    texto_completo = ""
    for page in doc:
        texto_completo += page.get_text() + "\n"
    doc.close()

    if not texto_completo.strip():
        raise ValueError("El archivo PDF no contiene texto legible (puede ser una imagen o escaneo sin OCR).")

    # Fragmentar el texto en bloques con superposición
    chunks = []
    inicio = 0
    while inicio < len(texto_completo):
        fin = inicio + chunk_size
        chunk = texto_completo[inicio:fin]
        chunks.append(chunk)
        inicio += (chunk_size - overlap)

    return chunks


@app.get("/")
def read_root():
    return {"status": "ok", "message": "API Chatbot RAG activa"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "has_documents": hay_documentos()}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    print(f"📥 [Upload] Recibiendo archivo: {file.filename}")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos en formato PDF.")

    file_path = os.path.join(DOCUMENTS_DIR, file.filename)

    try:
        # 1. Asegurar la existencia del directorio
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)

        # 2. Guardar el archivo recibido en el disco del contenedor
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"✅ Archivo guardado temporalmente en: {file_path}")

        # 3. Leer y fragmentar el PDF
        print("📄 Extrayendo texto con PyMuPDF...")
        chunks = extraer_chunks_de_pdf(file_path)
        print(f"🧩 Se generaron {len(chunks)} chunks de texto.")

        # 4. Enviar los embeddings a la nube de Pinecone
        print("🚀 Guardando vectores en Pinecone Cloud...")
        guardar_chunks(chunks, filename=file.filename)
        print("🎉 Proceso de indexación finalizado con éxito.")

        return {
            "message": "Documento subido e indexado correctamente.",
            "filename": file.filename,
            "chunks_processed": len(chunks)
        }

    except Exception as e:
        print(f"❌ ERROR CRÍTICO en /documents/upload al procesar {file.filename}:")
        traceback.print_exc()  # Muestra el error exacto y la línea en los Logs de Render

        # Limpieza del archivo temporal en caso de fallo
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

        raise HTTPException(
            status_code=500,
            detail=f"Error interno procesando '{file.filename}': {str(e)}"
        )


@app.get("/documents/list")
def list_documents():
    try:
        documentos = listar_documentos()
        return {"documents": documentos}
    except Exception as e:
        print("❌ Error en /documents/list:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/reset")
def reset_documents():
    try:
        reset_vectorstore()
        return {"message": "Base de datos vectorial y documentos borrados correctamente."}
    except Exception as e:
        print("❌ Error en /documents/reset:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat(payload: dict = Body(...)):
    try:
        pregunta = payload.get("question") or payload.get("pregunta")
        chat_history = payload.get("chat_history", [])

        if not pregunta:
            raise HTTPException(status_code=400, detail="Debe proporcionar una pregunta.")

        resultado = responder_pregunta(pregunta, chat_history)
        return resultado

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print("❌ Error en /chat:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al procesar la respuesta: {str(e)}")
