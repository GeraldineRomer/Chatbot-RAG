from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import docs_router, chat_router
import os
from app.config import DOCUMENTS_DIR

app = FastAPI(
    title="RAG PDF Chatbot API",
    description="API para Chatbot RAG que analiza documentos PDF y responde únicamente con su información.",
    version="1.0.0"
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #react + vite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear el directorio automáticamente si no existe en el contenedor
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Inclusión de Routers
app.include_router(docs_router.router)
app.include_router(chat_router.router)

@app.get("/health", tags=["Salud"])
def health_check():
    return {"status": "ok"}
