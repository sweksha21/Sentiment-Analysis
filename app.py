import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="✨",
    layout="centered"
)

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Models
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))



# Input
review = st.text_area(
    "",
    height=180,
    placeholder="Type your review here..."
)

analyze = st.button("Analyze Sentiment")

if analyze:

    if review.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Analyzing..."):
        time.sleep(1)

        X = tfidf.transform([review])

        pred = model.predict(X)

        probs = model.predict_proba(X)[0]

        sentiment = label_encoder.inverse_transform(pred)[0]

        confidence = float(np.max(probs) * 100)

    emoji = "😐"
    color = "#FFD166"

    if sentiment.lower() == "positive":
        emoji = "😊"
        color = "#4ADE80"

    elif sentiment.lower() == "negative":
        emoji = "😞"
        color = "#FB7185"

    st.markdown(
        f"""
        <div class="glass-card">
            <h2 style="color:{color};">
                {emoji} {sentiment}
            </h2>
            <p>Confidence: {confidence:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Confidence")

    st.progress(confidence / 100)

    st.subheader("Probabilities")

    classes = label_encoder.classes_

    df = pd.DataFrame({
        "Sentiment": classes,
        "Probability (%)": [round(float(x*100), 2) for x in probs]
    })

    st.dataframe(df, use_container_width=True)