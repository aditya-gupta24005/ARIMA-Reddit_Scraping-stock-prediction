# stock_sentiment.py
import sys
import os
import socket
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.offsets import BDay
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from nltk.sentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import praw
from plotter import SentimentPlotter

load_dotenv()

# Set timeout for yfinance
socket.setdefaulttimeout(10)

# Global sentiment analyzer
sia = SentimentIntensityAnalyzer()

#Uses reddit post to analyse sentiments
class StockSentimentAnalyzer:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        self.reddit.read_only = True
#Uses arima forecast to predict 5 days
    def arima_forecast(self, series: pd.Series, order=(5, 1, 0), steps=5) -> pd.Series:
        history = series.copy()
        preds = []

        for _ in range(steps):
            model = ARIMA(history, order=order).fit()
            yhat = model.forecast(steps=1).iloc[0]
            preds.append(yhat)
            next_day = history.index[-1] + BDay(1)
            history.loc[next_day] = yhat

        start = series.index[-1] + BDay(1)
        fc_index = pd.bdate_range(start=start, periods=steps)
        return pd.Series(preds, index=fc_index)

    def get_reddit_sentiment(self, symbol: str, limit=5) -> list:
        subreddits = ['stocks', 'investing', 'wallstreetbets']
        posts = []
        queries = [
            f'title:"{symbol}"',
            f'title:"${symbol}"',
            f'title:"{symbol} stock"'
        ]

        for subreddit in subreddits:
            try:
                for query in queries:
                    submissions = self.reddit.subreddit(subreddit).search(
                        query, limit=limit, sort='top', time_filter='month', syntax='lucene'
                    )

                    for submission in submissions:
                        url = f"https://reddit.com{submission.permalink}"
                        if any(p['url'] == url for p in posts):
                            continue

                        text = f"{submission.title} {submission.selftext}"
                        sentiment = sia.polarity_scores(text)
                        posts.append({
                            'title': submission.title,
                            'score': submission.score,
                            'subreddit': subreddit,
                            'sentiment': sentiment['compound'],
                            'url': url,
                            'created_utc': datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                        })
            except Exception as e:
                print(f"Error fetching from r/{subreddit}: {e}")

        posts.sort(key=lambda x: x['score'], reverse=True)
        return posts[:limit]

def safe_get_stock_data(symbol: str, months_back=6):
    today = pd.Timestamp.today().floor('D')
    if today.weekday() >= 5:  # Adjust for weekends
        today -= BDay(1)

    start = today - pd.DateOffset(months=months_back)
    print(f"\nFetching {symbol} price data from {start.date()} to {today.date()}...")

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.strftime('%Y-%m-%d'), end=(today + BDay(1)).strftime('%Y-%m-%d'))

        if df.empty or 'Close' not in df.columns:
            raise ValueError("No valid 'Close' price data found.")
        return df['Close']
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")
        return pd.Series(dtype=float)



def main():
    analyzer = StockSentimentAnalyzer()

    while True:
        symbol = input("\nEnter stock symbol (e.g., AAPL, TSLA, MSFT) or 'quit' to exit: ").strip().upper()
        if symbol == 'QUIT':
            print("Exiting program...")
            sys.exit(0)

        prices = safe_get_stock_data(symbol)

        if prices.empty:
            print("No price data available. Please try another symbol.")
            continue

        print("\nFetching and analyzing Reddit posts...")
        reddit_posts = analyzer.get_reddit_sentiment(symbol)

        if reddit_posts:
            print(f"\nTop Reddit Posts about {symbol}:")
            print("-" * 80)
            for post in reddit_posts:
                sentiment_label = "Positive" if post['sentiment'] > 0 else "Negative" if post['sentiment'] < 0 else "Neutral"
                print(f"Title: {post['title']}")
                print(f"Subreddit: r/{post['subreddit']}")
                print(f"Score: {post['score']}")
                print(f"Posted: {post['created_utc']}")
                print(f"Sentiment: {sentiment_label} ({post['sentiment']:.2f})")
                print(f"URL: {post['url']}")
                print("-" * 80)
        else:
            print("No relevant Reddit posts found.")

        print("\nRunning 5-day ARIMA forecast...")
        forecast = analyzer.arima_forecast(prices)
        
        # Get last known price for comparison
        last_price = prices.iloc[-1]
        
        print(f"\nForecast for {symbol} - Next 5 Business Days:")
        print("-" * 70)
        print(f"{'Date':<12} | {'Price':>10} | {'Change':>8} | {'Change %':>9}")
        print("-" * 70)
        
        for date, price in forecast.items():
            price_change = price - last_price
            pct_change = (price_change / last_price) * 100
            direction = "↑" if price_change > 0 else "↓" if price_change < 0 else "→"
            
            print(f"{date.strftime('%Y-%m-%d'):<12} | "
                  f"${price:>9.2f} | "
                  f"{direction} {abs(price_change):>6.2f} | "
                  f"{direction} {abs(pct_change):>7.2f}%")
            last_price = price
            
        print("-" * 70)
        print(f"Starting price: ${prices.iloc[-1]:.2f}")
        total_change = ((forecast.iloc[-1] - prices.iloc[-1]) / prices.iloc[-1]) * 100
        print(f"Total predicted change: {total_change:+.2f}%")
        
        print("\nGenerating plot...")
        plotter = SentimentPlotter()
        fig = plotter.plot_price_forecast(
            historical=prices,
            forecast=forecast,
            symbol=symbol
        )

        print(f"\nForecast for {symbol}:")
        for date, price in forecast.items():
            print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")

        print("\nPlot should now be visible. Enter another symbol or 'quit' to exit.")


if __name__ == "__main__":
    print("Stock Price Forecasting Tool")
    print("----------------------------")
    main()
