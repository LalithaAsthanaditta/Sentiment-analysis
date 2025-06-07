# web_app.py

import streamlit as st
import joblib
import re
import nltk
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# File paths
model_path = 'models/logistic_regression_model.pkl'
vectorizer_path = 'models/tfidf_vectorizer.pkl'
history_path = 'models/prediction_history.csv'

# Load model & vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(cleaned)

# Save prediction history
def save_history(review, prediction, confidence):
    new_entry = pd.DataFrame([[datetime.now(), review, prediction, round(confidence * 100, 2)]],
                             columns=["Time", "Review", "Prediction", "Confidence (%)"])
    if os.path.exists(history_path):
        old = pd.read_csv(history_path)
        df = pd.concat([new_entry, old], ignore_index=True)
    else:
        df = new_entry
    df.to_csv(history_path, index=False)

# UI
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# ✅ Updated Background Color: Purple to Pink Gradient
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to right, #fbc2eb, #a6c1ee);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎭 Sentiment Analysis Web App")
st.write("Enter your movie review or any text below to analyze the sentiment.")

review = st.text_area("**Your Review**", height=150)

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0].max()

        label = "Positive" if prediction == 1 else "Negative"
        st.success(f"**Predicted Sentiment:** {label}")
        st.info(f"**Confidence:** {round(confidence * 100, 2)}%")

        save_history(review, label, confidence)

# History Section
if os.path.exists(history_path):
    st.markdown("---")
    st.subheader("📊 Prediction History")
    history = pd.read_csv(history_path)
    st.dataframe(history.head(10), use_container_width=True)

    # Chart
    sentiment_counts = history['Prediction'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red'])
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Number of Predictions")
    st.pyplot(fig)
