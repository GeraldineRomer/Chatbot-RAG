import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.config import DOCUMENTS_DIR
from app.schemas import UploadResponse, DocumentListResponse, MessageResponse
from app.services.pdf_service import procesar_pdf
from app.services.rag_service import guardar_chunks, listar_documentos, reset_vectorstore

router = APIRouter(prefix="/documents", tags=["Documentos"])

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Solo se permiten archivos con formato PDF."
        )

    file_path = os.path.join(DOCUMENTS_DIR, file.filename)
    
    try:
        # Guardar el archivo en el directorio documents/
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Procesar PDF en chunks
        chunks = procesar_pdf(file_path)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El archivo PDF no contiene texto extraíble."
            )
            
        # Vectorizar y guardar en Chroma
        guardar_chunks(chunks, filename=file.filename)

        return UploadResponse(
            filename=file.filename,
            message="Documento procesado y vectorizado correctamente",
            chunks_count=len(chunks)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar el archivo: {str(e)}"
        )

@router.get("/list", response_model=DocumentListResponse)
def list_documents():
    docs = listar_documentos()
    return DocumentListResponse(documents=docs, total=len(docs))

@router.delete("/reset", response_model=MessageResponse)
def reset_documents():
    reset_vectorstore()
    return MessageResponse(message="Vector store y documentos eliminados exitosamente.")
