# 📧 Detector de Spam con Python y NLTK

Este proyecto implementa un **sistema básico de detección de mensajes spam** utilizando técnicas de **procesamiento de lenguaje natural (NLP)** con la librería **NLTK** y **pandas**.

---

## 🚀 Descripción

El objetivo es limpiar y transformar un conjunto de mensajes SMS para que puedan ser usados en un modelo de clasificación (por ejemplo, Naive Bayes, Logistic Regression, etc.).

Durante este proceso se:
- Carga el dataset original (`spam.csv`)
- Renombra las columnas para mayor claridad
- Convierte las etiquetas (`ham` → 0, `spam` → 1)
- Preprocesa los mensajes:
  - Elimina caracteres especiales
  - Convierte a minúsculas
  - Elimina *stopwords* (palabras vacías)
  - Aplica *stemming* (reducción a raíz de palabra)

---

## 🧩 Estructura del proyecto

📁 Detector_Spam/
│
├── files/
│ └── spam.csv # Dataset original
│
├── main.py # Script principal de preprocesamiento
├── README.md # Documentación del proyecto
└── requirements.txt # Dependencias del entorno



---

## 🧠 Tecnologías utilizadas

- **Python 3.12+**
- **pandas** → Manejo de datos tabulares
- **NumPy** → Operaciones numéricas
- **re (Regex)** → Limpieza de texto
- **NLTK** → Procesamiento de lenguaje natural
  - `stopwords`
  - `PorterStemmer`

---

## ⚙️ Instalación y ejecución

### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/tu_usuario/Detector_Spam.git
cd Detector_Spam
