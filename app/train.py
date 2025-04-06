import spacy
import random
from spacy.training.example import Example
from spacy.tokens import DocBin

# Cargar el modelo base en español
nlp = spacy.load("es_core_news_sm")

# Definir etiquetas y añadir al pipeline
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

labels = ["EMPRESA", "INDUSTRIA", "MONTO", "FECHA"]
for label in labels:
    ner.add_label(label)

# Datos de entrenamiento
TRAIN_DATA = [
    ("La factura de Metrogas vence el 10 de marzo por 3200 pesos", 
     {"entities": [(14, 22, "EMPRESA"), (32, 43, "FECHA"), (48, 58, "MONTO")]}),
]

# Convertir datos a formato `DocBin`
doc_bin = DocBin()
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annotations["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="expand")
        if span:
            ents.append(span)
    doc.ents = ents
    doc_bin.add(doc)

doc_bin.to_disk("training_data.spacy")

# Entrenar el modelo
nlp.initialize()
for epoch in range(30):
    random.shuffle(TRAIN_DATA)
    losses = {}
    for text, annotations in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], drop=0.5, losses=losses)

    print(f"Epoch {epoch + 1}, Loss: {losses}")

# Guardar el modelo entrenado
nlp.to_disk("mi_modelo_personalizado")
print("✅ Modelo entrenado y guardado")
