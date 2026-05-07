 🛍️ Sentiment Analysis on Kosmos Amazon Reviews

## Overview

This project applies **Sentiment Analysis** on the Kosmos dataset from Kaggle — a collection of Amazon reviews for a company selling home textiles and daily wear products. The goal is to automatically classify customer reviews as **positive** or **negative** to help the business understand customer feedback, maintain strengths, and address areas for improvement.

---
## 📁 Project Structure

```sentiment-analysis/
├── README.md
├── NLP_Project_Sentiment_analysis_.ipynb    ← Main notebook
├── NLP_Project_Sentiment_analysis_.html     ← Static HTML export
└── data/
    └── kozmos.csv                           ← Source: Kaggle (vedatgul/kozmos)
```
---

## 📊 Dataset

- **Source:** [Kaggle – Kozmos Amazon Reviews](https://www.kaggle.com/datasets/vedatgul/kozmos)
- **Size:** 5,611 entries
- **Columns:**
  - `Star` — Numerical rating (1–5)
  - `HelpFul` — Helpfulness count
  - `Title` — Review title
  - `Review` — Customer review text (main feature)

---

## 🔧 Methodology

### 1. Exploratory Data Analysis
- **Word Cloud** — Most frequently used terms across all reviews
- **Bar Chart** — Top words appearing more than 500 times

### 2. Sentiment Labeling (VADER)
Reviews were automatically labeled using **VADER** (Valence Aware Dictionary and Sentiment Reasoner), a rule-based sentiment tool suited for informal/social text like product reviews.
- Compound score **> 0** → `pos`
- Compound score **≤ 0** → `neg`
- **Validation:** Average star rating by label confirmed the approach — `neg` averaged 3.40 stars, `pos` averaged 4.58 stars ✅

### 3. Text Preprocessing
| Step | Description |
|---|---|
| Lowercasing | Normalize all text to lowercase |
| Punctuation removal | Strip all non-word characters |
| Stopword removal | Remove common English words (NLTK) |
| Rare word filtering | Remove bottom 1,000 least frequent words |
| Lemmatization | Reduce words to their root/base form |
| TF-IDF Vectorization | Convert text to numerical features |

### 4. Modeling
Two classifiers were trained on TF-IDF features with a **75/25 train-test split** and evaluated using **5-fold cross-validation**.

---
## 🤖 Model Results

| Model | Cross-Validation Score |
|---|---|
| Logistic Regression | **85.39%** |
| Random Forest | **89.02%** |

### Logistic Regression — Classification Report

```
              precision    recall  f1-score   support

         neg       0.32      0.91      0.48        79
         pos       0.99      0.89      0.94      1324

    accuracy                           0.89      1403
   macro avg       0.66      0.90      0.71      1403
weighted avg       0.96      0.89      0.91      1403
```
---

## 🧪 Sample Prediction
```
Review:   love super cool think visibility door black curtain 
          behind make easier see nice money
Prediction: ['pos'] ✅
```
---
## 💡 Conclusion

Logistic Regression demonstrated competitive performance on TF-IDF features, confirming that linear models remain effective for sentiment classification when preprocessing is sufficiently rigorous. Random Forest served as a non-linear comparative baseline and achieved a slightly higher cross-validation score (89.02%).

The VADER-based labeling strategy showed construct validity through its correlation with ground-truth star ratings. Future work could explore contextualized embeddings such as **BERT** or use raw star ratings as direct supervision for more fine-grained sentiment classification.

---
