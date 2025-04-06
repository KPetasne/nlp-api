import spacy
from spacy.training.example import Example

# Cargar modelo entrenado
nlp = spacy.load("mi_modelo_personalizado")

def autoentrenar_modelo(frase, entidades):
    """Actualiza el modelo con una nueva frase y entidades anotadas."""
    doc = nlp.make_doc(frase)
    ents = []
    for start, end, label in entidades:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span:
            ents.append(span)

    doc.ents = ents
    example = Example.from_dict(doc, {"entities": entidades})

    # Entrenar con la nueva frase
    losses = {}
    nlp.update([example], drop=0.5, losses=losses)

    print(f"✅ Modelo actualizado con la frase: '{frase}'")
    print(f"🔹 Pérdida en entrenamiento: {losses}")

    # Guardar el modelo actualizado
    nlp.to_disk("mi_modelo_personalizado")
    print("💾 Modelo guardado exitosamente.")
