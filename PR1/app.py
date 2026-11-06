from pathlib import Path

import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.tokenize import word_tokenize

from nlp_utils import clean_text, heuristic_score, heuristic_sentiment, vader_compound, vader_sentiment

st.set_page_config(page_title='Airline Tweet Sentiment Analyzer', page_icon='✈️')

MAX_CHARS = 500
EXAMPLES = {
    'Positive': 'Thanks @united, the crew was friendly and the flight was smooth! #greatservice',
    'Negative': "@AmericanAir my flight was cancelled, I'm stuck at the airport and nobody helps. Never again!",
    'Tricky': "Not happy with the delay, but the staff were not rude at all.",
}


@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / 'model/sentiment_model.joblib')


model = load_model()

if 'tweet' not in st.session_state:
    st.session_state['tweet'] = ''

with st.sidebar:
    st.header('About this project')
    st.write('Airline Tweet Sentiment Analyzer: classifies a tweet as negative, neutral or positive with three NLP approaches '
             '(heuristic lexicon, VADER and a TF-IDF + Logistic Regression model).')
    st.write('Dataset: [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (Kaggle).')

st.title('✈️ Airline Tweet Sentiment Analyzer')
st.caption('Type or paste a tweet, choose an approach and press Predict.')

for col, (name, example) in zip(st.columns(3), EXAMPLES.items()):
    if col.button(f'Example: {name}'):
        st.session_state['tweet'] = example

tweet = st.text_area('Tweet', key='tweet', height=120)
approach = st.radio('Approach', ['Heuristic', 'VADER', 'ML model'], horizontal=True)

if st.button('Predict', type='primary'):
    if not tweet.strip():
        st.warning('Please enter a tweet.')
    else:
        if len(tweet) > MAX_CHARS:
            tweet = tweet[:MAX_CHARS]
            st.info(f'Long input: only the first {MAX_CHARS} characters are analysed.')
        cleaned = clean_text(tweet)
        if not cleaned:
            st.warning('Nothing is left after cleaning (only emojis, symbols, numbers or a link?). Please write some words.')
        else:
            st.write('**Cleaned text:**', cleaned)
            if approach == 'Heuristic':
                label, detail = heuristic_sentiment(tweet), f'lexicon score = {heuristic_score(tweet)}'
            elif approach == 'VADER':
                label, detail = vader_sentiment(tweet), f'compound = {vader_compound(tweet):.3f}'
            else:
                proba = model.predict_proba([cleaned])[0]
                label, detail = model.classes_[proba.argmax()], f'confidence = {proba.max():.1%}'
            message = f'**{label.upper()}** ({detail})'
            {'positive': st.success, 'neutral': st.warning, 'negative': st.error}[label](message)
            if approach == 'ML model':
                st.bar_chart(pd.Series(proba, index=model.classes_, name='probability'))
            with st.expander('POS tags of the tweet'):
                st.dataframe(pd.DataFrame(nltk.pos_tag(word_tokenize(cleaned)), columns=['Word', 'POS tag']))
