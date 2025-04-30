# sentiment_analyzer.py
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download as nltk_download

# Ensure we have the VADER lexicon; download only if needed
try:
    _ = SentimentIntensityAnalyzer()
except LookupError:
    nltk_download("vader_lexicon")

import praw
import pandas as pd


class RedditSentimentAnalyzer:
    def __init__(self):
        # Read Reddit API credentials from environment
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "stock-sentiment/1.0")
        )
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text: str) -> float:
        """
        Return the compound sentiment score for a piece of text.
        """
        return self.sia.polarity_scores(text)["compound"]

    def analyze_trend(self, symbol: str, start_date, end_date) -> pd.Series:
        """
        Search the r/stocks subreddit for posts mentioning `symbol`,
        compute each post's sentiment, and return a Series of daily averages.
        """
        records = []
        for post in self.reddit.subreddit("stocks").search(symbol, limit=200):
            post_date = pd.to_datetime(post.created_utc, unit="s").date()
            text = post.title + " " + (post.selftext or "")
            score = self.analyze_sentiment(text)
            records.append({"date": post_date, "sentiment": score})

        df = pd.DataFrame(records)
        if df.empty:
            return pd.Series(dtype=float)

        # average sentiment per day
        daily_sent = df.groupby("date")["sentiment"].mean()
        # restrict to the requested window
        return daily_sent.loc[start_date:end_date]
