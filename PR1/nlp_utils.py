"""Shared NLP code: the notebook and app.py both import from here, so they use exactly the same cleaning and rules."""
import re
import nltk

for resource in ['stopwords', 'punkt', 'punkt_tab', 'vader_lexicon',
                 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

NEGATIONS = {'no', 'not', 'nor', 'never'}
STOP_WORDS = set(stopwords.words('english')) - NEGATIONS      # negation words are kept on purpose

POSITIVE_WORDS = {'thanks', 'thank', 'great', 'awesome', 'love', 'loved', 'friendly', 'helpful', 'excellent', 'amazing',
                  'best', 'good', 'nice', 'smooth', 'comfortable', 'perfect', 'happy', 'appreciate', 'appreciated',
                  'wonderful', 'fantastic', 'kind', 'quick', 'fast', 'easy', 'efficient', 'professional', 'polite',
                  'pleasant', 'impressed', 'enjoyed', 'fun', 'clean', 'welcome', 'ontime'}
NEGATIVE_WORDS = {'delay', 'delayed', 'delays', 'cancelled', 'canceled', 'cancellation', 'lost', 'late', 'rude', 'worst',
                  'terrible', 'awful', 'horrible', 'bad', 'stuck', 'waiting', 'waited', 'hold', 'disappointed',
                  'disappointing', 'angry', 'unacceptable', 'ridiculous', 'broken', 'refund', 'mess', 'failed', 'stranded',
                  'overbooked', 'missed', 'useless', 'nightmare', 'poor', 'slow', 'sucks', 'hate', 'hated', 'complaint'}


def clean_text(text):
    """lowercase -> remove URLs and @mentions -> keep hashtag word -> expand n't -> keep letters only -> collapse spaces"""
    text = text.lower().replace('’', "'")
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r"won't", 'will not', text)
    text = re.sub(r"can't", 'can not', text)
    text = re.sub(r"n't\b", ' not', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(clean):
    """tokenise a cleaned tweet and drop stop words (negations are kept)"""
    return [w for w in word_tokenize(clean) if w not in STOP_WORDS]


def heuristic_score(text):
    """positive words - negative words; a word flips polarity if one of the previous two words is a negation"""
    words = clean_text(text).split()
    score = 0
    for i, w in enumerate(words):
        polarity = (w in POSITIVE_WORDS) - (w in NEGATIVE_WORDS)
        if polarity and NEGATIONS & set(words[max(0, i - 2):i]):
            polarity = -polarity
        score += polarity
    return score


def heuristic_sentiment(text):
    score = heuristic_score(text)
    return 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'


_vader = SentimentIntensityAnalyzer()


def vader_compound(text):
    return _vader.polarity_scores(text)['compound']


def vader_sentiment(text):
    compound = vader_compound(text)
    return 'positive' if compound >= 0.05 else 'negative' if compound <= -0.05 else 'neutral'
