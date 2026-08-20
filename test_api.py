import os
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_flow():
    print("--- 1. Probar /health ---")
    res = client.get("/health")
    print(f"Status: {res.status_code}, Body: {res.json()}")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

    print("\n--- 2. Resetear vector store ---")
    res = client.delete("/documents/reset")
    print(f"Status: {res.status_code}, Body: {res.json()}")
    assert res.status_code == 200

    print("\n--- 3. Probar /chat sin documentos cargados (Debe dar error 400) ---")
    res = client.post("/chat", json={"question": "¿De qué trata el documento?", "chat_history": []})
    print(f"Status: {res.status_code}, Body: {res.json()}")
    assert res.status_code == 400
    assert "No hay documentos" in res.json()["detail"]

    print("\n--- 4. Listar documentos (debe estar vacío) ---")
    res = client.get("/documents/list")
    print(f"Status: {res.status_code}, Body: {res.json()}")
    assert res.status_code == 200
    assert res.json()["total"] == 0

    pdf_path = "./documents/reglamento-general-estudiantil-feb-2023.pdf"
    if not os.path.exists(pdf_path):
        print("Creando PDF de prueba dinámicamente con fitz (PyMuPDF)...")
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        texto_reglamento = (
            "REGLAMENTO GENERAL ESTUDIANTIL - FEBRERO 2023\n\n"
            "Este documento regula los derechos, deberes y la vida académica de los estudiantes.\n"
            "Se aplica a todos los estudiantes de pregrado, posgrado y educación continua de la universidad.\n\n"
            "Derechos principales de los estudiantes:\n"
            "1. Recibir educación de alta calidad académica.\n"
            "2. Respeto a su integridad física, moral y libre expresión.\n"
            "3. Acceso a los recursos y servicios de la institución.\n\n"
            "Deberes principales de los estudiantes:\n"
            "1. Cumplir con las normas de conducta académicas.\n"
            "2. Mantener la honestidad académica en exámenes y trabajos.\n\n"
            "Sanciones disciplinarias:\n"
            "Las faltas serán sancionadas según su gravedad con amonestación verbal, suspensión temporal, o expulsión definitiva.\n\n"
            "Este reglamento fue aprobado y emitido en la fecha de Febrero de 2023 por el Consejo Universitario."
        )
        page.insert_textbox(fitz.Rect(50, 50, 550, 750), texto_reglamento, fontsize=11)
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        doc.save(pdf_path)
        doc.close()

    print("\n--- 5. Subir PDF (/documents/upload) ---")
    with open(pdf_path, "rb") as f:
        res = client.post("/documents/upload", files={"file": ("reglamento.pdf", f, "application/pdf")})
    print(f"Status: {res.status_code}, Body: {res.json()}")
    assert res.status_code == 201
    assert res.json()["chunks_count"] > 0

    print("\n--- 6. Listar documentos (debe mostrar el PDF subido) ---")
    res = client.get("/documents/list")
    print(f"Status: {res.status_code}, Body: {res.json()}")
    assert res.status_code == 200
    assert res.json()["total"] >= 1

    print("\n--- 7. Hacer 5 preguntas en /chat ---")
    preguntas = [
        "¿De qué trata este reglamento?",
        "¿A quiénes se aplica este reglamento?",
        "¿Cuáles son los derechos principales de los estudiantes?",
        "¿Cuáles son las sanciones disciplinarias señaladas?",
        "¿En qué fecha o año fue publicado o aprobado este reglamento?"
    ]

    historial = []
    for i, q in enumerate(preguntas, 1):
        print(f"\nPregunta {i}: {q}")
        payload = {
            "question": q,
            "chat_history": historial
        }
        res = client.post("/chat", json=payload)
        print(f"Status: {res.status_code}")
        assert res.status_code == 200
        data = res.json()
        print(f"Respuesta: {data['response'][:200]}...")
        print(f"Cantidad de fuentes usadas: {len(data['sources'])}")
        assert len(data['sources']) > 0
        historial.append({"user": q, "ai": data["response"]})

    print("\n=== TODAS LAS PRUEBAS RESULTARON EXITOSAS ===")

if __name__ == "__main__":
    test_api_flow()
