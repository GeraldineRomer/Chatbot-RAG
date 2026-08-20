import os
import time
import shutil
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from app.config import VECTORSTORE_DIR, DOCUMENTS_DIR, GOOGLE_API_KEY

os.environ["ANONYMIZED_TELEMETRY"] = "False"

VECTORSTORE_PATH = str(VECTORSTORE_DIR)

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
        model="gemini-3.6-flash",
        google_api_key=get_api_key(),
        temperature=0
    )

def get_vectorstore():
    return Chroma(
        collection_name="pdf_rag",
        embedding_function=get_embeddings(),
        persist_directory=VECTORSTORE_PATH
    )

def hay_documentos() -> bool:
    try:
        vectorstore = get_vectorstore()
        count = vectorstore._collection.count()
        return count > 0
    except Exception:
        return False

def guardar_chunks(chunks: list[str], filename: str = "documento.pdf"):
    if chunks:
        metadatas = [{"source": filename} for _ in chunks]
        Chroma.from_texts(
            texts=chunks,
            embedding=get_embeddings(),
            metadatas=metadatas,
            collection_name="pdf_rag",
            persist_directory=VECTORSTORE_PATH
        )

def listar_documentos() -> list[str]:
    documentos = set()
    # 1. Archivos en carpeta documents
    if os.path.exists(DOCUMENTS_DIR):
        for f in os.listdir(DOCUMENTS_DIR):
            if f.lower().endswith('.pdf') and not f.startswith('.'):
                documentos.add(f)
    
    # 2. Metadatos de Chroma
    try:
        vectorstore = get_vectorstore()
        data = vectorstore._collection.get()
        metadatas = data.get("metadatas") or []
        for meta in metadatas:
            if meta and "source" in meta:
                documentos.add(meta["source"])
    except Exception:
        pass

    return sorted(list(documentos))

def reset_vectorstore():
    try:
        vectorstore = get_vectorstore()
        vectorstore.delete_collection()
    except Exception as e:
        print(f"Aviso al resetear vectorstore: {e}")

    # Eliminar archivos en carpeta documents/
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

    print("   [Debug] Obteniendo vectorstore...")
    vectorstore = get_vectorstore()
    print("   [Debug] Ejecutando similarity_search...")
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

    print("   [Debug] Llamando al LLM (gemini-3.5-flash)...")
    llm = get_llm()
    respuesta = llm.invoke(prompt)
    print("   [Debug] Respuesta del LLM recibida.")
    
    return {
        "response": respuesta.content,
        "sources": fuentes
    }
