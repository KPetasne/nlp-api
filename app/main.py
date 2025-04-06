from fastapi import FastAPI, Request, HTTPException
import os
from dotenv import load_dotenv
from app.train import entrenar_modelo
from app.evaluate import evaluar_texto
from app.auto_train import autoentrenar_modelo

load_dotenv()  # Cargar variables de entorno
API_KEY_SECRETA = os.getenv("API_KEY_SECRETA")

app = FastAPI()

@app.middleware("http")
async def validar_api_key(request: Request, call_next):
    """Intercepta cada request para validar la API Key."""
    api_key = request.headers.get("X-API-KEY")

    if not api_key or api_key != API_KEY_SECRETA:
        raise HTTPException(status_code=403, detail="Acceso denegado: API Key inválida")

    response = await call_next(request)
    return response

@app.post("/entrenar/")
def entrenar():
    """Ejecuta el entrenamiento inicial del modelo."""
    entrenar_modelo()
    return {"mensaje": "✅ Entrenamiento completado"}

@app.post("/autoentrenamiento/")
def autoentrenar(frase: str, entidades: list):
    """Entrena el modelo con una nueva frase y entidades definidas."""
    autoentrenar_modelo(frase, entidades)
    return {"mensaje": "✅ Autoentrenamiento ejecutado"}

@app.get("/evaluar/")
def evaluar(texto: str):
    """Evalúa un texto con el modelo entrenado y devuelve las entidades reconocidas."""
    resultado = evaluar_texto(texto)
    return {"texto": texto, "entidades": resultado}
