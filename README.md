# Mental Health Text Classification

A multi-class NLP pipeline that classifies social media statements into seven mental health categories using TF-IDF vectorisation and Logistic Regression.

![Python](https://img.shields.io/badge/Python-3.x-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow)

---

## Problem Statement

Mental health discussions on social media are growing in volume and urgency. Automated classification of such content could support triage tools, content moderation systems, and research pipelines. This project builds a baseline classifier capable of distinguishing between seven categories: Bipolar, Suicidal, Personality Disorder, Anxiety, Depression, Stress, and Normal.

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle - Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) |
| Rows | 26,350 |
| Columns | 2 (`statement`, `status`) |
| Task | Multi-class text classification (7 classes) |

The dataset contains raw text statements paired with a mental health label. Class distribution is imbalanced, which directly informs the modelling approach.

---

## Pipeline

```
Raw Text
   ↓
Text Preprocessing
(lowercase, remove URLs, punctuation, emojis, stop words, lemmatization)
   ↓
TF-IDF Vectorisation
(10,000 features, unigrams + bigrams)
   ↓
Train/Test Split (80/20)
   ↓
Logistic Regression (class_weight='balanced')
   ↓
Evaluation (Precision, Recall, F1-Score, Confusion Matrix)
```

---

## Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Anxiety | 0.84 | 0.79 | 0.81 |
| Bipolar | 0.77 | 0.71 | 0.74 |
| Depression | 0.59 | 0.59 | 0.59 |
| Normal | 0.96 | 0.98 | **0.97** |
| Personality Disorder | 0.80 | 0.77 | 0.78 |
| Stress | 0.72 | 0.75 | 0.74 |
| Suicidal | 0.72 | 0.77 | 0.74 |
| **Macro Average** | **0.77** | **0.77** | **0.77** |

**Overall accuracy: 77%** across 7 imbalanced classes using a baseline model.

---

## Key Findings

- **Normal** achieves the highest F1 (0.97). The language is sufficiently distinct from clinical categories that the model separates it cleanly.
- **Depression** has the lowest F1 (0.59) due to heavy lexical overlap with Suicidal, Anxiety, and Stress. The confusion matrix shows 204 depression posts misclassified as Suicidal and 173 suicidal posts misclassified as Depression -- a bidirectional overlap driven by shared language around hopelessness and exhaustion.
- **class_weight='balanced'** was critical. Without it, the model would have collapsed toward high-frequency classes and failed minority classes entirely.
- **Accuracy is a misleading metric here.** A model that always predicted Normal would still achieve reasonable accuracy given class imbalance. F1-score is the correct measure for this task.

---

## Tech Stack

- Python 3.x
- Pandas, NumPy
- NLTK (stopwords, WordNetLemmatizer)
- Scikit-learn (TfidfVectorizer, LogisticRegression, classification_report, ConfusionMatrixDisplay)
- Matplotlib, Seaborn

---

## Project Structure

```
mental-health-nlp-analysis/
├── notebooks/
│   └── mental_health_nlp_analysis.ipynb
├── data/
│   └── Sentiment_Mental_health_dataset.csv   ← not tracked (see .gitignore)
├── .gitignore
└── README.md
```

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/i-waseem/mental-health-nlp-analysis.git
cd mental-health-nlp-analysis
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) and place it in the `data/` folder as `Sentiment_Mental_health_dataset.csv`

4. Open the notebook
```bash
jupyter notebook notebooks/mental_health_nlp_analysis.ipynb
```

5. Run all cells in order

---

## Limitations

- TF-IDF treats words independently and has no understanding of tone, severity, or context. This is the primary reason Depression and Suicidal blur together.
- The dataset originates from a single Kaggle source. Real-world deployment would require more diverse data collection.
- This is a baseline model. It is not suitable for clinical or production use without significant further validation.

---

## Future Improvements

- Replace TF-IDF with sentence embeddings (SBERT) or a fine-tuned transformer such as Mental-BERT for semantic understanding
- Tune the classification threshold for the Suicidal class to maximise recall -- in any real application, missing a high-risk post is more costly than a false positive
- Explore ensemble methods (Random Forest, XGBoost) as alternative baselines
- Deploy the classifier as a REST API using FastAPI

---

## Author

**Iqra Waseem**
BSc Computer Science, First Class Honours -- University of Bradford
[GitHub](https://github.com/i-waseem) | [LinkedIn](https://www.linkedin.com/in/i-waseem)
