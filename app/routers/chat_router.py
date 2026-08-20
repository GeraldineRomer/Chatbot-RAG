from fastapi import APIRouter, HTTPException, status
from app.schemas import ChatRequest, ChatResponse
from app.services.rag_service import responder_pregunta

router = APIRouter(prefix="", tags=["Chat"])

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        resultado = responder_pregunta(
            pregunta=request.question,
            chat_history=request.chat_history
        )
        return ChatResponse(
            response=resultado["response"],
            sources=resultado["sources"]
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=400, 
            detail="No hay ningún documento procesado. Sube un PDF primero. \nError: " + str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Por favor verifica que si hayas subido un archivo PDF. Error en el servicio de chat: {str(e)}"
        )
