import os
import time
import shutil
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from app.config import DOCUMENTS_DIR, GOOGLE_API_KEY

# Nombre del índice en la nube de Pinecone
INDEX_NAME = "chatbot-rag-index"

def get_pinecone_api_key() -> str:
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        raise ValueError("PINECONE_API_KEY no está configurada en las variables de entorno.")
    return key

class GeminiEmbeddings(Embeddings):
    """Embeddings optimizados con lotes grandes y reintento automático ante límite 429."""
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY / GEMINI_API_KEY no está configurada en el entorno.")
        genai.configure(api_key=api_key)
        self.model_name = self._resolver_modelo()

    def _resolver_modelo(self) -> str:
        try:
            for m in genai.list_models():
                if "embedContent" in m.supported_generation_methods:
                    if "text-embedding-004" in m.name:
                        return m.name
                    return m.name
        except Exception:
            pass
        return "models/text-embedding-004"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        batch_size = 30  # Lotes más grandes = menos llamadas API
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            exito = False
            intentos = 0
            
            while not exito and intentos < 3:
                try:
                    res = genai.embed_content(
                        model=self.model_name,
                        content=batch,
                        task_type="retrieval_document"
                    )
                    all_embeddings.extend(res["embedding"])
                    exito = True
                    time.sleep(1)
                except ResourceExhausted:
                    intentos += 1
                    print(f"⚠️ Límites de cuota alcanzado. Esperando 15 segundos para reintentar ({intentos}/3)...")
                    time.sleep(15)
                except Exception as e:
                    raise e
                    
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        try:
            res = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            return res["embedding"]
        except ResourceExhausted:
            time.sleep(10)
            res = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            return res["embedding"]

def get_api_key():
    return GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

def get_embeddings():
    return GeminiEmbeddings(api_key=get_api_key())

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=get_api_key(),
        temperature=0
    )

def get_vectorstore():
    """Obtiene la referencia al vectorstore alojado en Pinecone Cloud."""
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=get_embeddings(),
        pinecone_api_key=get_pinecone_api_key()
    )

def hay_documentos() -> bool:
    """Verifica si existen vectores almacenados en el índice de Pinecone."""
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
    """Indexa y almacena los textos directamente en la nube de Pinecone."""
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
    """Obtiene la lista de nombres de PDFs subidos al sistema."""
    documentos = set()
    if os.path.exists(DOCUMENTS_DIR):
        for f in os.listdir(DOCUMENTS_DIR):
            if f.lower().endswith('.pdf') and not f.startswith('.'):
                documentos.add(f)
    return sorted(list(documentos))

def reset_vectorstore():
    """Elimina todos los vectores del índice en Pinecone y los archivos locales."""
    try:
        pc = Pinecone(api_key=get_pinecone_api_key())
        indexes = [index.name for index in pc.list_indexes()]
        if INDEX_NAME in indexes:
            index = pc.Index(INDEX_NAME)
            index.delete(delete_all=True)
            print("Vectores eliminados de Pinecone con éxito.")
    except Exception as e:
        print(f"Aviso al resetear vectorstore en Pinecone: {e}")

    # Eliminar archivos locales en la carpeta documents/
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

    print("   [Debug] Obteniendo vectorstore de Pinecone...")
    vectorstore = get_vectorstore()
    print("   [Debug] Ejecutando similarity_search en la nube...")
    docs = vectorstore.similarity_search(pregunta, k=4)
    print(f"   [Debug] Búsqueda finalizada. Documentos encontrados: {len(docs)}")
    
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

    print("   [Debug] Llamando al LLM...")
    llm = get_llm()
    respuesta = llm.invoke(prompt)
    print("   [Debug] Respuesta del LLM recibida.")
    
    return {
        "response": respuesta.content,
        "sources": fuentes
    }
