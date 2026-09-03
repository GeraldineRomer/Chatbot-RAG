import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Header, status
from app.config import DOCUMENTS_DIR
from app.schemas import UploadResponse, DocumentListResponse, MessageResponse
from app.services.pdf_service import procesar_pdf
from app.services.rag_service import guardar_chunks, listar_documentos, reset_vectorstore

router = APIRouter(prefix="/documents", tags=["Documentos"])

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    x_session_id: str = Header(None, alias="X-Session-ID")
):
    if not x_session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El header 'X-Session-ID' es obligatorio para operar."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Solo se permiten archivos con formato PDF."
        )

    session_dir = os.path.join(DOCUMENTS_DIR, x_session_id)
    os.makedirs(session_dir, exist_ok=True)

    file_path = os.path.join(session_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        chunks = procesar_pdf(file_path)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El archivo PDF no contiene texto extraíble."
            )
            
        guardar_chunks(chunks, filename=file.filename, session_id=x_session_id)

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
def list_documents(
    x_session_id: str = Header(None, alias="X-Session-ID")
):
    if not x_session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El header 'X-Session-ID' es obligatorio para operar."
        )
    docs = listar_documentos(session_id=x_session_id)
    return DocumentListResponse(documents=docs, total=len(docs))

@router.delete("/reset", response_model=MessageResponse)
def reset_documents(
    x_session_id: str = Header(None, alias="X-Session-ID")
):
    if not x_session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El header 'X-Session-ID' es obligatorio para operar."
        )
    reset_vectorstore(session_id=x_session_id)
    return MessageResponse(message="Vector store y documentos eliminados exitosamente.")
