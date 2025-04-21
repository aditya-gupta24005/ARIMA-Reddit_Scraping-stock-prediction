import praw
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import nltk
import requests
from time import sleep

# Add a 2-second delay between requests
sleep(2)

# Download required NLTK data
nltk.download('vader_lexicon')

# Load environment variables
load_dotenv()

class StockSentimentAnalyzer:
    def __init__(self):
        # Initialize Reddit API
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        self.sia = SentimentIntensityAnalyzer()

    def get_reddit_posts(self, stock_symbol, limit=100):
        """Fetch Reddit posts about a specific stock."""
        posts = []
        for post in self.reddit.subreddit('stocks+investing+wallstreetbets+IndiaInvestments+StockMarket').search(
            f'{stock_symbol} stock', limit=limit
        ):
            posts.append({
                'title': post.title,
                'text': post.selftext,
                'score': post.score,
                'created_utc': datetime.fromtimestamp(post.created_utc)
            })
        return pd.DataFrame(posts)

    def analyze_sentiment(self, text):
        """Analyze sentiment of text using VADER."""
        return self.sia.polarity_scores(text)['compound']

    def get_stock_data(self, stock_symbol, days=30):
        """Fetch historical stock data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.Ticker(stock_symbol)
        return stock.history(start=start_date, end=end_date)

    def predict_trend(self, stock_symbol):
        """Predict stock trend based on Reddit sentiment."""
        # Get Reddit posts
        posts_df = self.get_reddit_posts(stock_symbol)
        
        # Calculate average sentiment
        posts_df['sentiment'] = posts_df['text'].apply(self.analyze_sentiment)
        avg_sentiment = posts_df['sentiment'].mean()
        
        # Get stock data
        stock_data = self.get_stock_data(stock_symbol)
        
        # Simple trend prediction
        if avg_sentiment > 0.2:
            return "Bullish (Positive sentiment detected)"
        elif avg_sentiment < -0.2:
            return "Bearish (Negative sentiment detected)"
        else:
            return "Neutral (Mixed sentiment)"



def get_stock_history(stock_symbol, period="1y"):
    """
    Fetch 1 year of historical data (daily) by default.
    """
    stock = yf.Ticker(stock_symbol)
    df = stock.history(period=period)
    df.sort_index(inplace=True)
    return df

def prepare_data_for_arima(df):
   
    close_series = df['Close'].asfreq('B')  # asfreq('B') sets business-day frequency
    # Optionally forward fill missing days
    close_series.fillna(method='ffill', inplace=True)
    return close_series

def train_arima_model(series, order=(5,1,0)):
    """
    Fits an ARIMA model to the provided time series.
    :param series: Pa  ndas Series of stock prices.
    :param order: (p, d, q) ARIMA order.
    :return: Fitted ARIMA model.
    """
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit
       
def predict_next_close(model_fit, steps=1):

    
    forecast = model_fit.forecast(steps=steps)
    return forecast

def predict_tomorrows_closing_price(stock_symbol):
    # 1. Fetch data
    df = get_stock_history(stock_symbol, period="1y")  # last year
    close_series = prepare_data_for_arima(df)
    
    # 2. Train ARIMA model
    model_fit = train_arima_model(close_series, order=(1,1,1))
    
    # 3. Forecast tomorrow
    forecast = predict_next_close(model_fit, steps=1)
    
    # 4. Return forecasted price
    return float(forecast.iloc[-1])


def main():
    # Initialize analyzer
    analyzer = StockSentimentAnalyzer()
    
    # Get user input
    stock_symbol = input("Enter stock symbol (e.g., TSLA): ").upper()
    
    for _ in range(3):  # Retry up to 3 times
        try:
            stock = yf.Ticker(stock_symbol)
            data = stock.info
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("Rate limit hit. Retrying...")
                sleep(5)  # Wait 5 seconds before retrying
            else:
                raise

    if not stock.info:
        print("Invalid stock symbol. Please try again.")
        return
    company_name = stock.info["longName"]
    print(f"Company Name: {company_name}")
    
    try:
        # Get prediction
        prediction = analyzer.predict_trend(stock_symbol)
        print(f"\nAnalysis for {stock_symbol}:")
        print(prediction)
        
        # Show some sample posts
        posts = analyzer.get_reddit_posts(stock_symbol, limit=5)
        print("\nRecent Reddit posts about this stock:")
        for _, post in posts.iterrows():
            print(f"\nTitle: {post['title']}")
            print(f"Sentiment: {analyzer.analyze_sentiment(post['text']):.2f}")

        prediction = predict_tomorrows_closing_price(stock_symbol)
        print(f"Predicted closing price for {stock_symbol} tomorrow: ${prediction:.2f}")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
