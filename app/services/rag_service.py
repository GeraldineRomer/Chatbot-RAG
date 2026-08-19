import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

os.environ["ANONYMIZED_TELEMETRY"] = "False"
load_dotenv()

VECTORSTORE_DIR = "../../vectorstore"

class GeminiEmbeddings(Embeddings):
    """Embeddings optimizados con lotes grandes y reintento automático ante límite 429."""
    def __init__(self, api_key: str):
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
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

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
        persist_directory=VECTORSTORE_DIR
    )

def guardar_chunks(chunks: list[str]):
    if chunks:
        Chroma.from_texts(
            texts=chunks,
            embedding=get_embeddings(),
            collection_name="pdf_rag",
            persist_directory=VECTORSTORE_DIR
        )

def responder_pregunta(pregunta: str, chat_history: list[dict] = None) -> str:
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(pregunta, k=4)
    contexto = "\n\n".join([doc.page_content for doc in docs])
    # Formatear el historial conversacional
    historial_texto = ""
    if chat_history:
        for msg in chat_history:
            historial_texto += f"Usuario: {msg['user']}\nAsistente: {msg['ai']}\n"
    prompt = f"""Eres un asistente corporativo. Responde a la pregunta basándote ÚNICAMENTE en este contexto y en el historial de la conversación. Si no sabes la respuesta, di que no hay información en el documento.
    
    Contexto: {contexto}
    
    Historial de conversación:
    {historial_texto if historial_texto else 'Sin mensajes previos.'}
    
    Pregunta: {pregunta}
    Respuesta:"""

    llm = get_llm()
    respuesta = llm.invoke(prompt)
    return respuesta.content
