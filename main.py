# Importamos las librerías necesarias
import pandas as pd  # Para manejar datos en formato tabular (DataFrames)
import numpy as np  # Para cálculos numéricos (aunque aquí no se usa directamente)
import re  # Para usar expresiones regulares y limpiar texto
import nltk  # Librería para procesamiento de lenguaje natural
from nltk.corpus import stopwords  # Para eliminar palabras vacías (como "the", "is", "in", etc.)
from nltk.stem import PorterStemmer  # Para aplicar stemming (reducir palabras a su raíz)

# Descargamos las stopwords de NLTK (solo la primera vez que ejecutes el script)
nltk.download("stopwords")

# Creamos el objeto "stemmer" (para aplicar el algoritmo de stemming de Porter)
stemmer = PorterStemmer()

# Cargamos las stopwords del idioma inglés y las convertimos en un conjunto (más eficiente para búsqueda)
stop_words = set(stopwords.words("english"))

# ----------------------------------------------------------------------
# 1️⃣ Cargar el dataset original
# ----------------------------------------------------------------------
# Leemos el archivo CSV que contiene los mensajes de texto y sus etiquetas ("ham" o "spam")
# La codificación "latin-1" se usa porque el archivo puede contener caracteres especiales
# Solo seleccionamos las columnas 'v1' (etiqueta) y 'v2' (mensaje)
df = pd.read_csv("files/spam.csv", encoding="latin-1")[["v1", "v2"]]

# ----------------------------------------------------------------------
# 2️⃣ Renombrar columnas para mayor claridad
# ----------------------------------------------------------------------
# Cambiamos los nombres originales ("v1", "v2") por nombres más descriptivos:
# "label" → etiqueta (ham/spam)
# "message" → texto del mensaje
df.columns = ["label", "message"]

# ----------------------------------------------------------------------
# 3️⃣ Convertir etiquetas a valores binarios
# ----------------------------------------------------------------------
# Transformamos las etiquetas de texto a números:
# Si la etiqueta es "ham" → 0 (mensaje normal)
# Si la etiqueta es "spam" → 1 (mensaje de spam)
df["label"] = df["label"].apply(lambda x: 0 if x == "ham" else 1)

# Mostramos las primeras filas para comprobar
print(df.head())


# ----------------------------------------------------------------------
# 4️⃣ Definir una función de preprocesamiento del texto
# ----------------------------------------------------------------------
def proprocess_text(text):
    # 🔹 Eliminamos todos los caracteres que no sean letras o números (dejamos espacios)
    text = re.sub(r"\W", " ", text)

    # 🔹 Convertimos todo el texto a minúsculas
    text = text.lower()

    # 🔹 Dividimos el texto en palabras individuales
    words = text.split()

    # 🔹 Aplicamos stemming a cada palabra (reduce "running", "runs", "ran" → "run")
    #     y eliminamos las stopwords (palabras sin valor semántico relevante)
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    # 🔹 Volvemos a unir las palabras procesadas en una sola cadena de texto
    return " ".join(words)


# ----------------------------------------------------------------------
# 5️⃣ Aplicar la limpieza a la columna 'message'
# ----------------------------------------------------------------------
# Creamos una nueva columna llamada "cleaned_message"
# que contiene la versión procesada de cada mensaje original
df["cleaned_message"] = df["message"].apply(proprocess_text)

# Mostramos las primeras filas para verificar el resultado
print(df.head())
