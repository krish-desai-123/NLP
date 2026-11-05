# Summary Report

## Business Problem & Datasets
An OTT streaming platform receives thousands of viewer text reviews daily. Manual inspection is infeasible, so we built an automated sentiment analysis pipeline that processes raw reviews and classifies them as Positive or Negative. We used the SMS Spam Collection dataset (5,572 messages) for RegEx and preprocessing fundamentals, and a balanced 5,000-review sample from the IMDb 50K Movie Reviews dataset for word embeddings, sentiment classification, and POS analysis.

## Text Cleaning & Feature Engineering
Raw text was cleaned by removing HTML tags (`<br />`), URLs, and non-alphabetic characters using regular expressions, converting characters to lowercase, tokenising with NLTK, removing English stop words, and lemmatizing with WordNetLemmatizer. We extracted numerical features using Bag-of-Words (`CountVectorizer`), TF-IDF (`TfidfVectorizer`), N-Grams (`ngram_range=(1,2)`), and 100-dimensional Word2Vec embeddings (CBOW and Skip-gram).

## Model Performance
Four Logistic Regression classifiers were trained and evaluated on an 80/20 stratified train-test split:
- **TF-IDF with (1,2) N-Grams (Model C)** achieved the highest F1-score (~86%), capturing both word importance and local phrase context such as negations ("not good").
- **Standard TF-IDF (Model B)** and **Bag-of-Words (Model A)** followed closely (~84-85%).
- **Averaged Word2Vec (Model D)** achieved ~76% accuracy because averaging dense vectors across an entire review dilutes individual sentiment-bearing words on a small 5,000-review corpus.

## Heuristic, Pre-built Tool & ML Comparison
On a 1,000-review test subset, the trained Machine Learning pipeline outperformed both VADER (~69% accuracy) and the 30-word Heuristic rule-based scorer (~64% accuracy). While heuristics and VADER require zero training data and work well as rapid baselines, they struggle with domain-specific expressions and nuanced phrasing.

## Streamlit App & Future Improvements
The `app.py` Streamlit interface loads `sentiment_pipeline.pkl`, takes user reviews, displays the predicted sentiment with confidence percentage, and provides an expandable POS tag breakdown. Future improvements include training on all 50,000 IMDb reviews, supporting Hinglish code-mixed text, and fine-tuning a pre-trained transformer like DistilBERT.
