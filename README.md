# 📰 TruthLens AI

> **Machine Learning Powered Fake News Detection Platform**

TruthLens AI is an end-to-end **Machine Learning** and **Natural Language Processing (NLP)** application that detects whether a news article is **REAL** or **FAKE**. The project follows the complete machine learning workflow, from data preprocessing and feature engineering to model training, evaluation, and deployment through an interactive **Streamlit** web application.

---

## 📌 Project Overview

The rapid spread of misinformation on digital platforms has made automated fake news detection an important research area. TruthLens AI addresses this challenge by leveraging Natural Language Processing (NLP) techniques and supervised Machine Learning algorithms to classify news articles.

The application preprocesses textual news data, transforms it into numerical feature vectors using **TF-IDF Vectorization**, and classifies articles using a **Support Vector Machine (SVM)** model, which achieved the highest accuracy among all evaluated models.

---

# ✨ Features

- 📰 Real-time Fake News Detection
- 🤖 Support Vector Machine (SVM) Classifier
- 🧠 Natural Language Processing Pipeline
- 📊 TF-IDF Feature Engineering
- 📈 Model Comparison
- ⚙ Hyperparameter Tuning using GridSearchCV
- 💾 Model Serialization using Joblib
- 🌐 Interactive Streamlit Web Application
- 📋 Clean and Organized Project Structure
- 🎯 Professional Dashboard UI
- 🤖 AI-Powered Explanation using Google Gemini

---

# 🧠 Machine Learning Workflow

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Text Preprocessing
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Best Model Selection
      │
      ▼
Model Deployment
```

---

# 🧹 Natural Language Processing Pipeline

The preprocessing pipeline consists of:

- Text Cleaning
- Lowercase Conversion
- Regular Expressions (Regex)
- Tokenization
- Stopword Removal
- Stemming
- Lemmatization
- TF-IDF Vectorization

---

# 🤖 Machine Learning Models Evaluated

The following algorithms were trained and evaluated:

| Model | Accuracy |
|-------|---------:|
| Support Vector Machine (SVM) | **94.67%** |
| Logistic Regression | 92.82% |
| Random Forest | 90.53% |
| Naive Bayes | 88.35% |
| Decision Tree | 84.54% |

**Best Performing Model:** Support Vector Machine (SVM)

---

# 🛠 Technologies Used

## Programming Language

- Python

### Generative AI

- Google Gemini 3.5 Flash API
- Prompt Engineering

---

## Machine Learning

- Support Vector Machine (SVM)
- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes
- GridSearchCV

---

## Natural Language Processing

- Text Cleaning
- Tokenization
- Stopword Removal
- Stemming
- Lemmatization
- TF-IDF Vectorization

---

## Libraries

- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## Development Tools

- Jupyter Notebook
- Visual Studio Code
- GitHub

---

# 📁 Project Structure

```text
TruthLens-AI/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── svm.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── naive_bayes.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── notebooks/
│   ├── 1_Data_Loading.ipynb
│   ├── 2_EDA.ipynb
│   ├── 3_Text_Preprocessing.ipynb
│   ├── 4_Feature_Engineering.ipynb
│   ├── 5_Model_Training.ipynb
│   ├── 6_Model_Evaluation.ipynb
│   ├── 7_Hyperparameter_Tuning.ipynb
│   └── 8_Model_Comparison.ipynb
│
├── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/arshad-ahmad13/TruthLens-AI.git
```

Move into the project folder:

```bash
cd TruthLens-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
cd app

streamlit run app.py
```

or

```bash
python -m streamlit run app.py
```

---

# 📊 Model Performance

The Support Vector Machine achieved the highest performance among all evaluated models after hyperparameter tuning.

**Final Accuracy:** **94.67%**

Evaluation metrics included:

- Accuracy
- Precision
- Recall
- F1-Score

---

# 📸 Application Preview

> Add screenshots of the Streamlit application here after deployment.

Example:

```
images/
│
├── home_page.png
├── prediction_real.png
├── prediction_fake.png
```

---

# ⚠ Limitations

- Predictions depend on the quality and diversity of the training dataset.
- Performance may decrease on topics significantly different from the training data.
- The application performs AI-assisted classification and should not be considered a definitive fact-checking system.

---

# 🔮 Future Improvements

- Deep Learning Models (LSTM / GRU)
- Transformer Models (BERT)
- Explainable AI (XAI)
- Live News API Integration
- Multilingual Fake News Detection
- Cloud Deployment
- User Authentication
- Prediction History

---

# 👨‍💻 Author

## Arshad Ahmad

**B.Tech Student**  
Computer Science & Engineering (Artificial Intelligence & Machine Learning)

**Institution**  
United Institute of Technology (UIT), Prayagraj  
Affiliated with Dr. A.P.J. Abdul Kalam Technical University (AKTU), Lucknow

---

### About Me

I am an undergraduate student specializing in Artificial Intelligence and Machine Learning with a strong interest in Machine Learning, Natural Language Processing, Data Science, and Generative AI. I enjoy developing intelligent software solutions that combine theoretical concepts with practical implementation.

TruthLens AI was developed as an end-to-end machine learning project to strengthen my understanding of NLP workflows, feature engineering, supervised learning algorithms, model evaluation, hyperparameter tuning, and deployment using Streamlit.

---

### Areas of Interest

- Artificial Intelligence
- Machine Learning
- Natural Language Processing
- Data Science
- Generative AI
- Python Development

---

### Technical Skills

- Python
- Machine Learning
- Natural Language Processing
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Matplotlib
- Seaborn
- Streamlit
- Git & GitHub

---

### Contact

- GitHub: https://github.com/arshad-ahmad13
- LinkedIn: https://linkedin.com/in/arshad-ahmad13
- Email: arshadahmad8737@gmail.com

# 📄 License

This project is intended for educational and research purposes.

---

## ⭐ If you found this project useful, consider giving it a star on GitHub.