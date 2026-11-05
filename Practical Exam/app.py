import streamlit as st
import joblib
import nltk
import pandas as pd

for pkg in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    nltk.download(pkg, quiet=True)

@st.cache_resource
def load_model():
    return joblib.load('model/sentiment_pipeline.pkl')

pipe = load_model()

st.title("OTT Review Sentiment Analyser")
st.write("Enter a viewer review below to predict whether the sentiment is Positive or Negative.")

st.sidebar.header("Example Reviews")
st.sidebar.code("Amazing movie with brilliant performances and a gripping storyline.")
st.sidebar.code("Complete waste of time, terrible direction and boring plot.")
st.sidebar.code("Not a good movie at all, the acting was very dull.")
st.sidebar.subheader("About this project")
st.sidebar.write("Built with TF-IDF and Logistic Regression trained on 5,000 IMDb reviews.")

review_text = st.text_area("Viewer Review")

if st.button("Analyse"):
    if not review_text.strip():
        st.warning("Please enter a review")
    else:
        pred = pipe.predict([review_text])[0]
        prob = pipe.predict_proba([review_text])[0].max() * 100
        label = "Positive" if pred == 1 else "Negative"
        st.subheader(f"Prediction: {label} ({prob:.2f}% confidence)")

        with st.expander("POS Tags"):
            tokens = nltk.word_tokenize(review_text)
            tags = nltk.pos_tag(tokens)
            st.dataframe(pd.DataFrame(tags, columns=['Word', 'POS Tag']))
