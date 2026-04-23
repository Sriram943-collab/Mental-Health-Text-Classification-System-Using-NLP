import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim.models import FastText
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# ✅ NLTK FIX (for Hugging Face)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# -----------------------------
# CACHE MODEL (RUN ONCE)
# -----------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("Mental_health.csv")

    # 🔥 BALANCE DATASET (VERY IMPORTANT FIX)
    df = df.dropna(subset=['statement'])
    df = df[df['statement'].astype(str).str.strip() != ""]

    # Equal samples per class
    df = df.groupby('status').apply(lambda x: x.sample(min(len(x), 1000))).reset_index(drop=True)

    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        tokens = word_tokenize(text)
        return [w for w in tokens if w not in stop_words]

    df['tokens'] = df['statement'].apply(clean_text)

    # FastText model
    ft_model = FastText(
        df['tokens'],
        vector_size=50,
        window=3,
        min_count=3,
        workers=4
    )

    # Improved sentence vector
    def sentence_vector(tokens):
        vec = np.zeros(ft_model.vector_size)
        count = 0

        for word in tokens:
            vec += ft_model.wv[word]
            count += 1

        return vec / (count + 1e-6)

    X = np.array([sentence_vector(t) for t in df['tokens']])

    le = LabelEncoder()
    y = le.fit_transform(df['status'])

    # 🔥 BEST MODEL FOR THIS CASE
    model = LogisticRegression(max_iter=300)
    model.fit(X, y)

    return model, le, ft_model, stop_words, sentence_vector


model, le, ft_model, stop_words, sentence_vector = load_model()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Mental Health Predictor")

st.title("🧠 Mental Health Predictor")
st.write("Analyze text and predict mental health condition")

user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        text = user_input.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in stop_words]

        vec = sentence_vector(tokens).reshape(1, -1)

        prediction = model.predict(vec)
        result = le.inverse_transform(prediction)[0]

        st.success(f"Prediction: {result}")