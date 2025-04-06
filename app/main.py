from fastapi import FastAPI
from app.train import entrenar_modelo
from app.evaluate import evaluar_texto
from app.auto_train import autoentrenar_modelo

app = FastAPI()

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
