import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="✨",
    layout="centered"
)

# =====================================
# LOAD MODELS
# =====================================

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# =====================================
# APPLE GLASSMORPHISM CSS
# =====================================

st.markdown("""
<style>

/* Background */

.stApp{
background:
linear-gradient(
135deg,
#dbeafe 0%,
#e0e7ff 50%,
#f8fafc 100%
);
}

/* Main container */

.block-container{
max-width:900px;
padding-top:2rem;
}

/* Header */

.title{
text-align:center;
font-size:3rem;
font-weight:800;
color:#111827;
margin-bottom:0;
}

.subtitle{
text-align:center;
font-size:1rem;
color:#6b7280;
margin-bottom:2rem;
}

/* Glass Card */

.glass{
background:rgba(255,255,255,0.45);

backdrop-filter:blur(25px);
-webkit-backdrop-filter:blur(25px);

border:1px solid rgba(255,255,255,0.6);

border-radius:28px;

padding:30px;

box-shadow:
0 8px 32px rgba(31,38,135,0.10);

margin-top:20px;
}

/* Text Area */

textarea{
background:rgba(255,255,255,0.55)!important;

color:#111827!important;

border-radius:20px!important;

border:1px solid rgba(255,255,255,0.7)!important;

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

transition:0.3s;
}

.stButton > button:hover{

background:#1f2937;

transform:translateY(-2px);
}

/* Result */

.result{
text-align:center;
}

.result h2{
font-size:40px;
margin-bottom:10px;
}

.result p{
font-size:20px;
font-weight:600;
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

/* DataFrame */

[data-testid="stDataFrame"]{
border-radius:16px;
overflow:hidden;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.markdown(
"""
<div class="title">
✨ AI Sentiment Analyzer
</div>

<div class="subtitle">
Machine Learning Powered Review Intelligence
</div>
""",
unsafe_allow_html=True
)

# =====================================
# INPUT
# =====================================

review = st.text_area(
"",
height=180,
placeholder="Type your review here..."
)

predict = st.button("Analyze Sentiment")

# =====================================
# PREDICTION
# =====================================

if predict:

    if review.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Analyzing sentiment..."):
        time.sleep(1)

        vector = tfidf.transform([review])

        prediction = model.predict(vector)

        probabilities = model.predict_proba(vector)[0]

        sentiment = label_encoder.inverse_transform(prediction)[0]

        confidence = float(np.max(probabilities) * 100)

    # Sentiment Style

    emoji = "😐"
    color = "#f59e0b"

    if sentiment.lower() == "positive":
        emoji = "😊"
        color = "#22c55e"

    elif sentiment.lower() == "negative":
        emoji = "😞"
        color = "#ef4444"

    # =================================
    # RESULT CARD
    # =================================

    st.markdown(
        f"""
        <div class="glass result">

            <h2 style="color:{color};">
            {emoji} {sentiment}
            </h2>

            <p>
            Confidence: {confidence:.2f}%
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # =================================
    # CONFIDENCE
    # =================================

    st.subheader("Confidence Level")

    st.progress(confidence / 100)

    if confidence >= 80:
        st.success("High Confidence Prediction")

    elif confidence >= 60:
        st.info("Moderate Confidence Prediction")

    else:
        st.warning(
            "Low confidence prediction. Consider providing more context."
        )

    # =================================
    # PROBABILITIES
    # =================================

    st.subheader("Sentiment Probabilities")

    classes = label_encoder.classes_

    df = pd.DataFrame({
        "Sentiment": classes,
        "Probability (%)":
        [round(float(x * 100), 2) for x in probabilities]
    })

    st.dataframe(
        df,
        use_container_width=True
    )

# =====================================
# FOOTER
# =====================================

st.markdown(
"""
<br><br>

<center style="color:#6b7280;">
Built with Streamlit • TF-IDF • Logistic Regression
</center>
""",
unsafe_allow_html=True
)
