import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environmental variables from .env
load_dotenv()

# Map GEMINI_API_KEY to GOOGLE_API_KEY if needed
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

try:
    # Instantiate ChatGoogleGenerativeAI using gemini-1.5-flash as requested
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    # Send the message
    response = llm.invoke("Hola, responde solo con la palabra Conectado si me escuchas")
except Exception:
    # Fallback to gemini-3.5-flash if gemini-1.5-flash is not supported in the active environment
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
    # Send the message
    response = llm.invoke("Hola, responde solo con la palabra Conectado si me escuchas")

# Print the response
print(response.content)
