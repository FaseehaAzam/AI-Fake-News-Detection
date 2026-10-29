
import streamlit as st
import joblib
import re
from pathlib import Path

# App settings
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Load saved AI model and vectorizer
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "fake_news_model.pkl")
    vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Use the same cleaning function as training
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Header
st.title("📰 AI Fake News Detection & Analysis")
st.write("Analyze news article text using machine learning.")

st.info(
    "This tool identifies patterns learned from labeled news data. "
    "It does not independently verify whether a story is true."
)

# User input
news_text = st.text_area(
    "Paste your news article here:",
    height=220,
    placeholder="Enter the news article text..."
)

if st.button("🔍 Analyze News", type="primary"):
    cleaned = clean_text(news_text)

    if len(cleaned.split()) < 10:
        st.warning("Please enter at least 10 words of news text.")
    else:
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        label = "Real-like" if prediction == 1 else "Fake-like"
        confidence = probabilities[list(model.classes_).index(prediction)]

        st.subheader("Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Category", label)

        with col2:
            st.metric("Model Confidence", f"{confidence * 100:.2f}%")

        st.progress(float(confidence))

        st.caption(
            "Confidence is the model's estimated class probability, "
            "not the probability that the article is factually true."
        )

        st.subheader("Article Statistics")
        st.write(f"Word Count: {len(news_text.split())}")
        st.write(f"Character Count: {len(news_text)}")

        # Show words that influenced the prediction
        st.subheader("Influential Words")

        feature_names = vectorizer.get_feature_names_out()
        contributions = (
            features.multiply(model.coef_[0]).toarray().ravel()
        )

        nonzero = features.nonzero()[1]
        ranked = sorted(
            nonzero,
            key=lambda i: abs(contributions[i]),
            reverse=True
        )[:10]

        for i in ranked:
            direction = "Real-like" if contributions[i] > 0 else "Fake-like"
            st.write(
                f"**{feature_names[i]}** — {direction} "
                f"(contribution: {contributions[i]:+.3f})"
            )

st.divider()

# Model information
st.subheader("About the AI Model")
st.write("Algorithm: Logistic Regression")
st.write("Text Features: TF-IDF")
st.write("Dataset: Fake and Real News Dataset (Kaggle)")
st.write("Test Accuracy: 98.58% on a deduplicated random split")

st.caption(
    "Test accuracy may overestimate performance on new sources "
    "because the dataset may contain source-specific patterns."
)

# Project charts
st.subheader("Model Visualizations")

chart1 = BASE_DIR / "confusion_matrix.png"
chart2 = BASE_DIR / "news_distribution.png"

if chart1.exists():
    st.image(str(chart1), caption="Confusion Matrix")

if chart2.exists():
    st.image(str(chart2), caption="News Dataset Distribution")
