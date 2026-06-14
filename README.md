# ✨ AI Sentiment Analyzer

A Machine Learning-powered Sentiment Analysis web application built using **Python**, **Scikit-Learn**, and **Streamlit**.

The application analyzes customer reviews and predicts whether the sentiment is **Positive**, **Negative**, or **Neutral** using a trained **Logistic Regression** model and **TF-IDF Vectorization**.

---

## 🚀 Features

* Real-time sentiment prediction
* Positive, Negative, and Neutral classification
* Confidence score display
* Sentiment probability analysis
* Interactive Streamlit dashboard
* Clean and responsive user interface
* Machine Learning model deployment using Streamlit

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-Learn
* Pandas
* NumPy
* TF-IDF Vectorizer
* Logistic Regression
* Pickle

---

## 📂 Project Structure

```text
Sentiment-Analysis/
│
├── app.py
├── model.pkl
├── tfidf.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md
```

---

## 🧠 Machine Learning Workflow

### Data Preprocessing

* Text Cleaning
* Lowercasing
* Stopword Removal
* Lemmatization

### Feature Engineering

TF-IDF Vectorization converts text reviews into numerical feature vectors.

### Model Training

Algorithm Used:

```text
Logistic Regression
```

### Sentiment Classes

```text
Positive
Negative
Neutral
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/sentiment-analysis.git

cd sentiment-analysis
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Application will open automatically in your browser.

---

## 📊 Example Predictions

### Input

```text
This product is amazing and exceeded my expectations.
```

### Output

```text
Positive
Confidence: 92.4%
```

---

### Input

```text
Worst purchase I have ever made.
```

### Output

```text
Negative
Confidence: 95.8%
```

---

## 📈 Model Components

### model.pkl

Stores the trained Logistic Regression model.

### tfidf.pkl

Stores the TF-IDF Vectorizer used during training.

### label_encoder.pkl

Converts encoded predictions back into human-readable sentiment labels.

---

## 🎯 Future Improvements

* Deep Learning Models (LSTM/BERT)
* Sentiment Visualization Charts
* Batch Review Analysis
* API Integration
* Model Explainability Dashboard

---

## 👨‍💻 Author

**Sweksha Sharma**


Skills:

* Python
* Machine Learning
* Data Analysis
* Power BI
* SQL
* Streamlit

---

## 📄 License

This project is developed for educational and portfolio purposes.
