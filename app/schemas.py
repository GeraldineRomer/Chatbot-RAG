from pydantic import BaseModel, Field
from typing import List, Optional

class ChatMessage(BaseModel):
    user: str = Field(..., description="Pregunta enviada por el usuario")
    ai: str = Field(..., description="Respuesta generada por la IA")

class ChatRequest(BaseModel):
    question: str = Field(..., description="Pregunta actual del usuario")
    chat_history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Historial de conversación")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Respuesta generada por el RAG")
    sources: List[str] = Field(default_factory=list, description="Fragmentos fuente de contexto utilizados")

class UploadResponse(BaseModel):
    filename: str = Field(..., description="Nombre del archivo cargado")
    message: str = Field(..., description="Mensaje de confirmación")
    chunks_count: int = Field(..., description="Cantidad de fragmentos generados y vectorizados")

class DocumentListResponse(BaseModel):
    documents: List[str] = Field(..., description="Lista de nombres de documentos vectorizados")
    total: int = Field(..., description="Total de documentos cargados")

class MessageResponse(BaseModel):
    message: str = Field(..., description="Mensaje informativo o de estado")
