# pip install spacytextblob#
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# idioma
nlp = spacy.load("en_core_web_sm")
# añado extras
nlp.add_pipe('spacytextblob')
# Cargar el conjunto de datos de texto
doc = nlp("Fuck Apple es una empresa de tecnología con sede en Cupertino, California. Fundada en 1976, ha sido pionera en la industria de la informática y ha lanzado muchos productos populares, como el iPhone y el MacBook.")

## xxxxxxx

print(f"sentimientos ", doc._.blob.polarity)
# -0.125

print(f"subjetivo ",doc._.blob.subjectivity)
# 0.9

print(doc._.blob.sentiment_assessments.assessments)

# Eliminar signos de puntuación y convertir todas las palabras a minúsculas
tokens = [token.text.lower() for token in doc if not token.is_punct]
print(f"\n{tokens}")

# Tokenizar el texto en palabras individuales
tokens = [token for token in doc]
print(f"\n{tokens}")

# Identificar entidades en el texto
entities = [(entity.text, entity.label_) for entity in doc.ents]
print("\nEntidades")
print(entities)

# Identificar temas en el texto
topics = [(token.text, token.lemma_) for token in doc if token.pos_ == "NOUN"]
print("\nTEMAS:")
print(topics)
