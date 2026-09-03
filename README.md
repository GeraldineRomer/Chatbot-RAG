# 🤖 Chatbot RAG API - Multi-Tenant Backend

Servicio de backend RESTful para un sistema de **Generación Aumentada por Recuperación (RAG)** de alto rendimiento y arquitectura *Multi-Tenant*. Diseñado con **FastAPI**, **LangChain**, **Pinecone Cloud** y los modelos generativos de **Google Gemini**.

Permite la ingesta, fragmentación (*chunking*), vectorización y consulta de documentos PDF, garantizando el aislamiento absoluto de datos entre usuarios mediante espacios de nombres (*namespaces*) en la base de datos vectorial.

---

## 🛠️ Tecnologías y Herramientas

* **Framework API:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3.12)
* **Orquestador RAG:** [LangChain](https://www.langchain.com/) (`langchain-google-genai`, `langchain-pinecone`)
* **Base de Datos Vectorial:** [Pinecone](https://www.pinecone.io/) (Serverless)
* **Modelos de IA (Google Gemini):**
  * **Embeddings:** `gemini-embedding-001` (Ajustado a 768 dimensiones)
  * **LLM:** `gemini-3.5-flash`
* **Procesamiento de Documentos:** PyMuPDF (`fitz`)
* **Servidor ASGI:** Uvicorn
* **Despliegue:** Render Web Service

---

## 🚀 Arquitectura y Características Clave

1. **Aislamiento Multi-Tenant (Seguridad):**
   * Implementación de *Pinecone Namespaces* vinculados a un encabezado HTTP `X-Session-ID`.
   * Cada usuario posee su propio entorno aislado de búsqueda vectorial, evitando fugas de información (*data leakage*) entre clientes.
2. **Pipelines de Ingesta Eficientes:**
   * Extracción limpia de texto en PDF mediante PyMuPDF.
   * Fragmentación en bloques (*chunks*) con superposición (*overlap*) para preservar contexto sintáctico.
   * Mapeo de embeddings a 768 dimensiones para optimización de almacenamiento y precisión de similitud por coseno.
3. **Estructura Modular por Routers:**
   * Separación clara de responsabilidades (`docs_router`, `chat_router`) y manejo centralizado de configuración.

---

## 📂 Estructura del Proyecto

```text
├── app/
│   ├── config.py           # Variables de entorno y rutas del sistema
│   ├── main.py             # Punto de entrada de FastAPI, CORS y Middlewares
│   ├── routers/            # Definición de endpoints REST
│   │   ├── docs_router.py  # Endpoints para /upload, /list y /reset
│   │   └── chat_router.py  # Endpoint para /chat RAG
│   └── services/           # Lógica de negocio RAG y conexión a Pinecone/Gemini
│       └── rag_service.py
├── documents/              # Directorio temporal local para archivos PDF
├── requirements.txt        # Dependencias del proyecto
└── README.md
```
---

## 🔌 Endpoints de la API

Todos los endpoints requieren el header **`X-Session-ID`** para operar sobre el namespace correcto.

| Método | Ruta | Descripción | Headers Requeridos |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Verificación de estado del servidor e índice vectorial. | - |
| `POST` | `/documents/upload` | Carga un PDF, extrae texto y lo indexa en Pinecone. | `X-Session-ID` |
| `GET` | `/documents/list` | Lista los documentos activos de la sesión actual. | `X-Session-ID` |
| `DELETE`| `/documents/reset` | Elimina todos los vectores y archivos del namespace. | `X-Session-ID` |
| `POST` | `/chat` | Consulta RAG. Recibe la pregunta y el historial. | `X-Session-ID` |

---

## ⚙️ Configuración e Instalación Local

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/GeraldineRomer/chatbot-rag-backend.git
   cd chatbot-rag-backend
   ```

2. **Crear y activar un entorno virtual:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Variables de Entorno (`.env`):**
   Crea un archivo `.env` en la raíz con las siguientes claves:
   ```env
   GOOGLE_API_KEY=tu_api_key_de_google_ai_studio
   PINECONE_API_KEY=tu_api_key_de_pinecone
   DOCUMENTS_DIR=documents
   ```

5. **Iniciar el servidor localmente:**
   ```bash
   uvicorn app.main:app --reload
   ```

## Ver Swagger del proyecto
Esto puede demorar entre 30 a 60 segundos en activar el servidor debido a que se usa en la capa gratuita de Render
(https://chatbot-rag-8ws5.onrender.com/docs)
