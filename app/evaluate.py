import spacy

# Cargar el modelo entrenado
nlp = spacy.load("mi_modelo_personalizado")

def evaluar_texto(texto):
    """Procesa el texto con el modelo y devuelve las entidades encontradas."""
    doc = nlp(texto)
    entidades = [{"texto": ent.text, "etiqueta": ent.label_} for ent in doc.ents]
    return entidades
