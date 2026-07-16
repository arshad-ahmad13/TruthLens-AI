import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

from pathlib import Path
from dotenv import load_dotenv
import os

from google import genai


st.set_page_config(
    page_title="TruthLens AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


st.markdown("""
<style>


.stApp{

    background:#f4f7fc;

}


#MainMenu{

    visibility:hidden;

}
header {
    background: transparent;
}

footer{

    visibility:hidden;

}


.hero{

    background:linear-gradient(135deg,#2563EB,#0F172A);

    padding:35px;

    border-radius:20px;

    color:white;

    margin-bottom:25px;

    box-shadow:0px 12px 30px rgba(0,0,0,.18);

}

.hero h1{

    font-size:42px;

    font-weight:800;

    margin-bottom:8px;

}

.hero p{

    font-size:18px;

    opacity:.95;

}


.card{

    background:white;

    padding:22px;

    border-radius:18px;

    box-shadow:0px 5px 18px rgba(0,0,0,.08);

    margin-bottom:20px;

}


section[data-testid="stSidebar"]{

    background:#0F172A;

}

/* Sidebar Text */

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span{

    color:white !important;

}

/* Sidebar Expanders */

section[data-testid="stSidebar"] details{

    background:transparent !important;

    border:none !important;

    box-shadow:none !important;

}

section[data-testid="stSidebar"] summary{

    color:white !important;

    background:transparent !important;

    font-weight:600;

}

section[data-testid="stSidebar"] summary:hover{

    background:transparent !important;

}

section[data-testid="stSidebar"] details[open] summary{

    background:transparent !important;

}


.stButton>button{

    background:#2563EB;

    color:white;

    border:none;

    border-radius:12px;

    height:52px;

    font-size:18px;

    font-weight:600;

    width:100%;

}

.stButton>button:hover{

    background:#1D4ED8;

}


textarea{

    font-size:16px !important;

}


[data-testid="metric-container"]{

    background:white;

    border-radius:12px;

    padding:12px;

    box-shadow:0px 3px 10px rgba(0,0,0,.05);

}


.footer{

    text-align:center;

    color:#64748B;

    padding:30px;

    font-size:15px;

}

</style>
""", unsafe_allow_html=True)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "svm.pkl"

try:

    model = joblib.load(MODEL_PATH)
  
 # GEMINI CLIENT

    client = genai.Client(api_key=GEMINI_API_KEY)

except Exception as e:

    st.error(f"Unable to load model.\n\n{e}")

    st.stop()


st.markdown("""
<div class="hero">

<h1>📰 TruthLens AI</h1>

<p>

Machine Learning Powered Fake News Detection Platform

<br><br>

Analyze • Verify • Detect

</p>

</div>
""", unsafe_allow_html=True)


# AI EXPLANATION FUNCTION

def explain_prediction(news_text, prediction):

    label = "REAL" if prediction == 1 else "FAKE"

    prompt = f"""
You are an AI assistant integrated into TruthLens AI.

A Machine Learning model (Support Vector Machine with TF-IDF Vectorization)
has already classified the following news article as **{label}**.

Your task is NOT to verify whether the article is factually correct.

Instead:

1. Explain why a machine learning model may have produced this prediction.
2. Focus only on writing style, language, vocabulary, tone, sentence structure and textual patterns.
3. Give 5 concise bullet points.
4. End with a short disclaimer that this is an AI-generated explanation and not factual verification.

News Article:

{news_text}
"""

    models = [
    "gemini-3.5-flash",
    "gemini-flash-latest"
    ]

    last_error = None

    for model_name in models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text

        except Exception as e:
            last_error = e
            continue

    return f"""
⚠️ AI Explanation is temporarily unavailable.

Google Gemini servers are currently busy or unavailable.

Last error:
{last_error}
"""


with st.sidebar:

    st.markdown("# 📰 TruthLens AI")

    st.success("🟢 System Ready")

    st.metric(
        label="Final Model",
        value="SVM"
    )

    st.metric(
        label="Accuracy",
        value="94.67%"
    )

    st.markdown("---")

    # MACHINE LEARNING

    with st.expander("🤖 Machine Learning", expanded=False):

        st.success("Support Vector Machine (SVM)")
        st.info("Logistic Regression")
        st.info("Decision Tree")
        st.info("Random Forest")
        st.info("Naive Bayes")
        st.warning("Hyperparameter Tuning (GridSearchCV)")


    with st.expander("🧠 Natural Language Processing", expanded=False):

        st.success("Text Cleaning")
        st.success("Lowercase Conversion")
        st.info("Regular Expressions (Regex)")
        st.info("Tokenization")
        st.info("Stopword Removal")
        st.info("Stemming")
        st.info("Lemmatization")
        st.warning("TF-IDF Vectorization")

    with st.expander("✨ Generative AI", expanded=False):
        st.success("Google Gemini 3.5 Flash")
        st.success("AI-Powered Explanation")
        st.info("Google GenAI SDK")
        st.warning("Natural Language Explanation")



    with st.expander("📚 Libraries Used", expanded=False):

        st.success("Pandas")
        st.success("NumPy")
        st.info("NLTK")
        st.info("Scikit-learn")
        st.info("Matplotlib")
        st.info("Seaborn")
        st.info("Joblib")
        st.warning("Streamlit")
        st.warning("Google GenAI SDK")

   
    # DEVELOPMENT TOOLS

    with st.expander("⚙ Development Tools", expanded=False):

        st.success("Python")
        st.info("Jupyter Notebook")
        st.info("Visual Studio Code")
        st.warning("GitHub")

    # PROJECT FEATURES
    with st.expander("🚀 Project Features", expanded=False):

        st.success("Data Cleaning")
        st.success("Exploratory Data Analysis")
        st.success("Text Preprocessing")
        st.success("Feature Engineering (TF-IDF)")
        st.success("Machine Learning Model Training")
        st.success("Hyperparameter Tuning")
        st.success("Model Evaluation")
        st.success("Model Comparison")
        st.success("Model Serialization (Joblib)")
        st.success("Interactive Web Application")
        st.success("Real-Time Fake News Detection")
        st.success("AI-Powered Prediction Explanation")

    # DEVELOPER
    
    with st.expander("👨‍💻 About the Developer", expanded=False):

        st.success("Arshad Ahmad")

        st.markdown("""
🎓 **B.Tech CSE (Artificial Intelligence & Machine Learning)**

💻 Python Developer

🤖 Machine Learning & NLP

✨ Generative AI

🌐 Streamlit Application Development

📊 Data Science Enthusiast

🔧 Git & GitHub

🚀 Passionate about building intelligent AI-powered applications.
""")
    

badge1, badge2, badge3, badge4 = st.columns(4)

with badge1:
    st.success("🤖 Machine Learning")

with badge2:
    st.success("🧠 NLP")

with badge3:
    st.success("⚡ Real-Time Analysis")

with badge4:
    st.success("📊 Accuracy 94.67%")

st.markdown("<br>", unsafe_allow_html=True)



left, right = st.columns([2.2, 1], gap="large")



with left:

    st.markdown("""
<div class="card">

<h2>📝 Verify News Article</h2>

<p style="font-size:16px;color:#555;">
Paste a complete news article below.
TruthLens AI will analyze the content using
Natural Language Processing and Machine Learning
to classify it as <b>REAL</b> or <b>FAKE</b>.
</p>

</div>
""", unsafe_allow_html=True)

    news_text = st.text_area(
        label="News Article",
        placeholder="Paste the complete news article here...",
        height=320,
        label_visibility="collapsed"
    )

    btn1, btn2 = st.columns([4,1])

    with btn1:

        verify = st.button(
            "🔍 Verify News",
            use_container_width=True
        )

    with btn2:

        clear = st.button(
            "🗑 Clear",
            use_container_width=True
        )

    if clear:

        st.rerun()


with right:

    if not verify:

        st.markdown("""
<div class="card">

<h2 style="text-align:center;">
📊 Verification Result
</h2>

<hr>

<h4 style="text-align:center;color:#64748B;">
Waiting for Verification...
</h4>

<br>

<b>Status</b>

<br>

🟢 Ready

<hr>

<b>Model</b>

<br>

Support Vector Machine (SVM)

<hr>

<b>Vectorizer</b>

<br>

TF-IDF

</div>
""", unsafe_allow_html=True)

    else:

        if news_text.strip() == "":

            st.warning("⚠ Please enter a news article.")

        else:

            start_time = time.time()

            with st.spinner("🔍 Initializing TruthLens AI..."):

                progress = st.progress(0)

                progress.progress(20)

                time.sleep(0.3)

                progress.progress(45)

                time.sleep(0.3)

                progress.progress(70)

                time.sleep(0.3)

                prediction = model.predict([news_text])[0]

                st.session_state["prediction"] = prediction
                st.session_state["news_text"] = news_text

                progress.progress(100)

                time.sleep(0.2)

            processing_time = time.time() - start_time

            confidence = None

            try:

                if hasattr(model, "decision_function"):

                    score = abs(model.decision_function([news_text])[0])

                    confidence = min(score * 20, 99.9)

            except:

                confidence = None

            if prediction == 1:

                st.success("✅ REAL NEWS")

                summary = """
The submitted article has been classified as **REAL NEWS**.

The prediction is based on textual patterns extracted
using TF-IDF Vectorization and classified using a
Support Vector Machine model.
"""

            else:

                st.error("❌ FAKE NEWS")

                summary = """
The submitted article has been classified as **FAKE NEWS**.

The prediction is based on textual patterns extracted
using TF-IDF Vectorization and classified using a
Support Vector Machine model.
"""

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Model",
                    "SVM"
                )

            with col2:

                st.metric(
                    "Time",
                    f"{processing_time:.3f} sec"
                )

            if confidence:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            st.markdown("---")

            st.subheader("📄 Verification Summary")

            st.info(summary)

# AI EXPLANATION
st.markdown("---")

st.subheader("🤖 AI Explanation")

if st.button("✨ Explain with AI", use_container_width=True):

    with st.spinner("Generating AI explanation..."):

        explanation = explain_prediction(
            st.session_state["news_text"],
            st.session_state["prediction"]
        )

    st.success("AI Explanation Generated")

    st.markdown(explanation)

# ABOUT PROJECT
st.markdown("<br>", unsafe_allow_html=True)

about_tab, model_tab, limitation_tab = st.tabs(
    [
        "📖 About Project",
        "🤖 Model Information",
        "⚠ Model Limitations"
    ]
)


with about_tab:

    st.markdown("""
### 📰 TruthLens AI

TruthLens AI is a Machine Learning-based Fake News Detection
application that classifies news articles as **REAL** or **FAKE**
using Natural Language Processing (NLP).

The application preprocesses news articles, converts them
into numerical features using **TF-IDF Vectorization**, and
classifies them using a trained **Support Vector Machine (SVM)** model.

The project follows a complete Machine Learning pipeline
including:

- Data Collection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Text Preprocessing
- Feature Engineering
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- Model Comparison
- Model Deployment using Streamlit
""")


# MODEL

with model_tab:

    st.markdown("## 🤖 Final Model")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Algorithm",
            "Support Vector Machine"
        )

        st.metric(
            "Accuracy",
            "94.67%"
        )

        st.metric(
            "Vectorizer",
            "TF-IDF"
        )

    with col2:

        st.metric(
            "Programming",
            "Python"
        )

        st.metric(
            "Framework",
            "Streamlit"
        )

        st.metric(
            "Model File",
            "svm.pkl"
        )

    st.markdown("---")

    st.subheader("Algorithms Evaluated")

    comparison = pd.DataFrame({

        "Algorithm":[
            "Support Vector Machine",
            "Logistic Regression",
            "Random Forest",
            "Naive Bayes",
            "Decision Tree"
        ],

        "Accuracy":[
            "94.67%",
            "92.81%",
            "90.53%",
            "88.35%",
            "84.54%"
        ]

    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


with limitation_tab:

    st.warning("""
This model was trained using a specific fake news dataset.

Although it performs well on similar data,
its predictions on recent or unseen news topics
may be less accurate.

The prediction should be considered as an
AI-assisted classification rather than
a definitive fact verification.
""")


st.markdown("---")

st.markdown("""
<div class="footer">

<h3>📰 TruthLens AI</h3>

Machine Learning Powered Fake News Detection Platform

<br><br>

Developed by <b>Arshad Ahmad</b>

<br><br>

Powered by Python • Scikit-learn • Streamlit

</div>
""", unsafe_allow_html=True)