import re
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load saved model and encoder
lr_model = joblib.load("models/logistic_model_glove.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
glove_embeddings = joblib.load("models/glove_embeddings.pkl")

print("Label classes:", label_encoder.classes_)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+|[^A-Za-z\s]", "", text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def text_to_glove_vector(text, embeddings, dim=100):
    words = text.split()
    word_vecs = [embeddings[word] for word in words if word in embeddings]
    if not word_vecs:
        return np.zeros(dim)
    return np.mean(word_vecs, axis=0)

review = input("Enter your review: ")
cleaned = preprocess_text(review)
vector = text_to_glove_vector(cleaned, glove_embeddings).reshape(1, -1)

pred_num = lr_model.predict(vector)[0]
pred_label = label_encoder.inverse_transform([pred_num])[0]

print(f"🔮 Predicted Sentiment: {pred_label} (encoded as {pred_num})")
