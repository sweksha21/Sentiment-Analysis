
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="✨",
    layout="centered"
)

# ======================================
# LOAD MODELS
# ======================================

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ======================================
# CUSTOM STYLING
# ======================================

st.markdown("""
<style>

.stApp{
    background:
    radial-gradient(circle at top left,#dbeafe 0%,transparent 35%),
    radial-gradient(circle at bottom right,#e0e7ff 0%,transparent 35%),
    linear-gradient(135deg,#f8fafc,#eef2ff);
}

.block-container{
    max-width:900px;
    padding-top:2rem;
}

/* Text Area */

[data-testid="stTextArea"] textarea{
    background:rgba(255,255,255,0.85)!important;
    border-radius:20px!important;
    border:1px solid #dbeafe!important;
    color:#111827!important;
    font-size:16px!important;
}

/* Button */

.stButton > button{
    width:100%;
    height:55px;
    border:none;
    border-radius:18px;
    background:#111827;
    color:white;
    font-size:17px;
    font-weight:700;
}

.stButton > button:hover{
    background:#1f2937;
}

/* Progress */

.stProgress > div > div{
    background:
    linear-gradient(
    90deg,
    #60a5fa,
    #818cf8
    );
}

</style>
""", unsafe_allow_html=True)

# ======================================
# HEADER
# ======================================

st.markdown(
"""
<h1 style='text-align:center;color:#111827;'>
✨ AI Sentiment Analyzer
</h1>

<p style='text-align:center;color:#6b7280;font-size:18px;'>
Machine Learning Powered Review Intelligence
</p>
""",
unsafe_allow_html=True
)

st.write("")

# ======================================
# INPUT
# ======================================

review = st.text_area(
    "Enter Review",
    height=180,
    placeholder="Type customer review here..."
)

predict = st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
)

# ======================================
# PREDICTION
# ======================================

if predict:

    if review.strip() == "":
        st.warning("Please enter review text.")
        st.stop()

    with st.spinner("Analyzing sentiment..."):
        time.sleep(1)

        vector = tfidf.transform([review])

        prediction = model.predict(vector)

        probabilities = model.predict_proba(vector)[0]

        sentiment = label_encoder.inverse_transform(prediction)[0]

        confidence = float(np.max(probabilities) * 100)

    st.divider()

    # RESULT METRICS

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Sentiment",
            sentiment
        )

    with col2:
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    # CONFIDENCE BAR

    st.subheader("Confidence Level")

    st.progress(confidence / 100)

    if confidence >= 80:
        st.success("High confidence prediction")

    elif confidence >= 60:
        st.info("Moderate confidence prediction")

    else:
        st.warning(
            "Low confidence prediction. Consider providing more context."
        )

    # PROBABILITIES

    st.subheader("Sentiment Probabilities")

    classes = label_encoder.classes_

    sorted_probs = sorted(
        zip(classes, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    for label, prob in sorted_probs:

        st.write(
            f"**{label}** — {prob*100:.2f}%"
        )

        st.progress(float(prob))

    # TABLE

    st.subheader("Detailed Analysis")

    df = pd.DataFrame({
        "Sentiment": classes,
        "Probability (%)":
        [round(float(x*100), 2) for x in probabilities]
    })

    st.dataframe(
        df,
        use_container_width=True
    )

# ======================================
# FOOTER
# ======================================

st.markdown(
"""
<br><br>

<center style="color:#6b7280;">
Built with Streamlit • TF-IDF • Logistic Regression
</center>
""",
unsafe_allow_html=True
)

