# ----------------------------------------------------------------------
# 📘 Detector de Spam - Python + NLTK + Scikit-learn
# ----------------------------------------------------------------------

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------------------------------------------
# 0️⃣ Configuración inicial
# ----------------------------------------------------------------------
nltk.download("stopwords")
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ----------------------------------------------------------------------
# 1️⃣ Cargar y preparar el dataset
# ----------------------------------------------------------------------
df = pd.read_csv("files/spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "message"]
df["label"] = df["label"].apply(lambda x: 0 if x == "ham" else 1)
print(df.head())

# ----------------------------------------------------------------------
# 2️⃣ Preprocesamiento de texto
# ----------------------------------------------------------------------
def preprocess_text(text):
    text = re.sub(r"\W", " ", text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df["cleaned_message"] = df["message"].apply(preprocess_text)
print(df.head())

# ----------------------------------------------------------------------
# 3️⃣ Vectorización y entrenamiento del modelo
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["cleaned_message"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 4️⃣ Evaluación del modelo
# ----------------------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------------------
# 5️⃣ Función de predicción
# ----------------------------------------------------------------------
def predict_email(email_text):
    processed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# ----------------------------------------------------------------------
# 6️⃣ Ejemplos de prueba
# ----------------------------------------------------------------------
email = "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize."
result = predict_email(email)
print(f"\n🔎 Email 1 → {result}")

email_text = """
🎉 Congratulations! You have been selected to win a FREE iPhone 15 Pro 🎁

Dear user,

Your email has been randomly chosen to receive a brand new iPhone 15 Pro completely FREE.
All you need to do is confirm your details by clicking the link below:

👉 https://freeiphone-now.example.com

Hurry up! This offer is valid only for the next 24 hours.

Don’t miss this unique opportunity to get the most desired smartphone of the year without paying anything.
Click the link now and complete the confirmation form.

Thank you for participating,
The Online Rewards™ Team

© 2025 Online Rewards Inc. All rights reserved.
If you no longer wish to receive these emails, click here to unsubscribe.
"""

resultado = predict_email(email_text)
print(f"🔎 Email 2 → {resultado}")
