# 🚨 Real or Not? Disaster Tweets Classification

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-yellow.svg)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red.svg)](https://streamlit.io/)

This repository contains a Natural Language Processing (NLP) solution for the Kaggle **Real or Not? Disaster Tweets** challenge. The goal is to predict whether a given tweet is about a real disaster or just using disaster-related words in a metaphorical or everyday context.

---

## 📊 Project Overview
* **Objective:** Build a robust text classification pipeline to accurately distinguish real emergency tweets from non-emergency ones (F1-score evaluation metric).
* **Key Steps:**
  * Text cleaning and preprocessing (removing URLs, handles, punctuation, numbers, and stop words).
  * Feature extraction using **TF-IDF Vectorization** (Term Frequency-Inverse Document Frequency).
  * Model Training using classification algorithms (e.g., Logistic Regression, Naive Bayes, or Support Vector Machines).
  * Interactive web dashboard deployment via **Streamlit** for real-time tweet classification.

---

## 🛠️ Tech Stack & Libraries
* **Python**
* **Pandas & NumPy** (Data Management)
* **Scikit-Learn** (Text Vectorization & Classification)
* **Matplotlib & Seaborn** (Data Visualization)
* **Streamlit** (Interactive Dashboard)

---

## 🚀 Getting Started & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/tomris17/disaster-tweets-nlp-classification.git](https://github.com/tomris17/disaster-tweets-nlp-classification.git)
   cd disaster-tweets-nlp-classification
