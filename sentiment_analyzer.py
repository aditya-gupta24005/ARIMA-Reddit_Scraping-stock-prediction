import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


def _load_stopwords():
    """
    Load NLTK English stopwords, downloading them on first use if missing.
    """
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


class RedditSentimentAnalyzer:
    """
    Enhanced Reddit sentiment analyzer combining VADER scores,
    text-quality heuristics, and TF-IDF + RandomForest for classification.
    """

    def __init__(self):
        # Initialize VADER sentiment scorer
        self.sia = SentimentIntensityAnalyzer()

        # TF-IDF vectorizer for text classification features
        self.vectorizer = TfidfVectorizer(max_features=1000)

        # Random forest classifier for optional custom classification pipeline
        self.classifier = RandomForestClassifier(n_estimators=100)

        # Load stopwords with auto-download fallback
        self.stop_words = _load_stopwords()

        # Threshold for filtering low-quality posts
        self.quality_threshold = 0.5

    def preprocess_text(self, text: str) -> str:
        """
        Lowercase, remove non-alpha chars, tokenize, and remove stopwords.
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [tok for tok in tokens if tok and tok not in self.stop_words]
        return ' '.join(tokens)

    def calculate_quality_score(self, post: dict) -> float:
        """
        Heuristic quality score: combines text length, karma, and VADER magnitude.
        """
        score = 0.0

        # Reward longer posts
        if len(post.get('text', '')) > 100:
            score += 0.3

        # Reward posts with higher upvotes
        if post.get('score', 0) > 10:
            score += 0.3

        # Add magnitude of sentiment
        vader_scores = self.sia.polarity_scores(post.get('text', ''))
        score += abs(vader_scores.get('compound', 0.0)) * 0.4
        return score

    def filter_low_quality_posts(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Annotate and filter out posts below the quality threshold.
        """
        posts = posts_df.copy()
        posts['quality_score'] = posts.apply(self.calculate_quality_score, axis=1)
        return posts[posts['quality_score'] > self.quality_threshold]

    def analyze_sentiment(self, text: str) -> float:
        """
        Compute a composite sentiment score combining VADER and custom signals.
        Returns a single float for "compound" sentiment.
        """
        # VADER provides polarity_scores dict with keys: neg, neu, pos, compound
        vader = self.sia.polarity_scores(text)

        # Features: length, punctuation signals
        text_length = len(text)
        has_question = 1 if '?' in text else 0
        has_exclamation = 1 if '!' in text else 0

        # Composite formula (weights may be tuned)
        composite = (
            vader['compound'] * 0.7 +      # primary VADER signal
            (text_length / 1000) * 0.1 +    # longer text adds mild positivity
            (has_exclamation * 0.1) +       # excitement boosts
            (has_question * -0.1)           # questions dampen positivity slightly
        )
        return composite

    def analyze_trend(self,
                      posts_df: pd.DataFrame,
                      window_size: int = 7) -> dict:
        """
        Analyze daily sentiment trend and moving average for a set of posts.

        Returns a dict with:
         - 'trend': 'Bullish'|'Bearish'|'Neutral'
         - 'daily_sentiment': pd.Series (D-indexed)
         - 'moving_avg': pd.Series (rolling mean)
         - 'current_sentiment': float
        """
        if posts_df.empty:
            return {
                'trend': 'Neutral',
                'daily_sentiment': pd.Series(dtype=float),
                'moving_avg': pd.Series(dtype=float),
                'current_sentiment': 0.0
            }

        df = posts_df.copy()
        df['date'] = pd.to_datetime(df['created_utc'], unit='s')
        df.set_index('date', inplace=True)

        # Ensure posts_df has a 'sentiment' column (float per post)
        daily = df['sentiment'].resample('D').mean().fillna(0.0)
        rolling = daily.rolling(window=window_size, min_periods=1).mean()

        latest = rolling.iloc[-1]
        if latest > 0.2:
            trend_label = 'Bullish'
        elif latest < -0.2:
            trend_label = 'Bearish'
        else:
            trend_label = 'Neutral'

        return {
            'trend': trend_label,
            'daily_sentiment': daily,
            'moving_avg': rolling,
            'current_sentiment': float(latest)
        }
