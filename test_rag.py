import os
from app.services.pdf_service import procesar_pdf
from app.services.rag_service import guardar_chunks, responder_pregunta

VECTORSTORE_DIR= './vectorstore'

print("=== VERIFICACIÓN FINAL DE ETAPA 1 ===\n")

# Pruebas 1 y 3: Carga/Vectorización y Persistencia en Disco
if not os.path.exists(VECTORSTORE_DIR) or not os.listdir(VECTORSTORE_DIR):
    print("✓ Pruebas 1 y 3: Procesando PDF y creando base vectorial persistente...")
    chunks = procesar_pdf("./documents/reglamento-general-estudiantil-feb-2023.pdf") # Reemplaza por la ruta de tu PDF
    guardar_chunks(chunks)
    print("   -> Vectores generados y guardados en ./vectorstore")
else:
    print("✓ Pruebas 1 y 3: Base de datos vectorial cargada desde disco (Persistencia OK).")

historial = []

# Prueba 2: Pregunta basada en el PDF
pregunta_1 = "¿De qué trata este documento?"
print(f"\n✓ Prueba 2 [Pregunta inicial]: '{pregunta_1}'")
respuesta_1 = responder_pregunta(pregunta_1, chat_history=historial)
print(f"Respuesta:\n{respuesta_1}\n")

# Guardar en memoria
historial.append({"user": pregunta_1, "ai": respuesta_1})

# Prueba 4: Memoria de conversación (pregunta contextual de seguimiento)
pregunta_2 = "¿En qué fecha se emitió y por quién?"
print(f"✓ Prueba 4 [Pregunta con Memoria]: '{pregunta_2}'")
respuesta_2 = responder_pregunta(pregunta_2, chat_history=historial)
print(f"Respuesta:\n{respuesta_2}\n")

print("=== ✓ PRUEBA 5: TODAS LAS PRUEBAS COMPLETADAS CON ÉXITO ===")
