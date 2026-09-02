import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from app.config import DOCUMENTS_DIR, GOOGLE_API_KEY

INDEX_NAME = "chatbot-rag-index"

def get_pinecone_api_key() -> str:
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        raise ValueError("PINECONE_API_KEY no está configurada en las variables de entorno.")
    return key

def get_api_key():
    return GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

class GeminiEmbeddings(Embeddings):
    """Embeddings optimizados procesando elementos de forma individual para evitar batch 404/400 en gRPC."""
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API Key de Google no configurada.")
        genai.configure(api_key=api_key)
        self.model_name = "models/text-embedding-004"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            exito = False
            intentos = 0
            while not exito and intentos < 3:
                try:
                    res = genai.embed_content(
                        model=self.model_name,
                        content=text,  # Pasar string individual fuerza embed_content en lugar de batch
                        task_type="retrieval_document"
                    )
                    embeddings.append(res["embedding"])
                    exito = True
                except Exception as e:
                    intentos += 1
                    if "429" in str(e) or "ResourceExhausted" in str(e):
                        time.sleep(2 * intentos)
                    else:
                        raise e
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        res = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return res["embedding"]

def get_embeddings():
    return GeminiEmbeddings(api_key=get_api_key())

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=get_api_key(),
        temperature=0
    )

def get_vectorstore():
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=get_embeddings(),
        pinecone_api_key=get_pinecone_api_key()
    )

def hay_documentos() -> bool:
    try:
        pc = Pinecone(api_key=get_pinecone_api_key())
        indexes = [index.name for index in pc.list_indexes()]
        if INDEX_NAME in indexes:
            index = pc.Index(INDEX_NAME)
            stats = index.describe_index_stats()
            return stats.total_vector_count > 0
        return False
    except Exception as e:
        print(f"Error verificando documentos en Pinecone: {e}")
        return False

def guardar_chunks(chunks: list[str], filename: str = "documento.pdf"):
    if chunks:
        metadatas = [{"source": filename} for _ in chunks]
        PineconeVectorStore.from_texts(
            texts=chunks,
            embedding=get_embeddings(),
            metadatas=metadatas,
            index_name=INDEX_NAME,
            pinecone_api_key=get_pinecone_api_key()
        )

def listar_documentos() -> list[str]:
    documentos = set()
    if os.path.exists(DOCUMENTS_DIR):
        for f in os.listdir(DOCUMENTS_DIR):
            if f.lower().endswith('.pdf') and not f.startswith('.'):
                documentos.add(f)
    return sorted(list(documentos))

def reset_vectorstore():
    try:
        pc = Pinecone(api_key=get_pinecone_api_key())
        indexes = [index.name for index in pc.list_indexes()]
        if INDEX_NAME in indexes:
            index = pc.Index(INDEX_NAME)
            stats = index.describe_index_stats()
            if stats.total_vector_count > 0:
                index.delete(delete_all=True)
                print("Vectores eliminados de Pinecone con éxito.")
    except Exception as e:
        print(f"Aviso al resetear vectorstore en Pinecone: {e}")

    if os.path.exists(DOCUMENTS_DIR):
        for f in os.listdir(DOCUMENTS_DIR):
            path = os.path.join(DOCUMENTS_DIR, f)
            if os.path.isfile(path) and not f.startswith('.'):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error borrando {path}: {e}")

def responder_pregunta(pregunta: str, chat_history: list = None) -> dict:
    if not hay_documentos():
        raise ValueError("No hay documentos cargados en el sistema. Por favor sube un archivo PDF primero.")

    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(pregunta, k=4)
    
    contexto = "\n\n".join([doc.page_content for doc in docs])
    fuentes = [doc.page_content for doc in docs]

    historial_texto = ""
    if chat_history:
        for msg in chat_history:
            if isinstance(msg, dict):
                user_msg = msg.get("user", "")
                ai_msg = msg.get("ai", "")
            else:
                user_msg = getattr(msg, "user", "")
                ai_msg = getattr(msg, "ai", "")
            historial_texto += f"Usuario: {user_msg}\nAsistente: {ai_msg}\n"

    prompt = f"""Eres un asistente corporativo. Responde a la pregunta basándote ÚNICAMENTE en este contexto y en el historial de la conversación. Si no sabes la respuesta, di que no hay información en el documento.
    
    Contexto: {contexto}
    
    Historial de conversación:
    {historial_texto if historial_texto else 'Sin mensajes previos.'}
    
    Pregunta: {pregunta}
    Respuesta:"""

    llm = get_llm()
    respuesta = llm.invoke(prompt)
    
    return {
        "response": respuesta.content,
        "sources": fuentes
    }
