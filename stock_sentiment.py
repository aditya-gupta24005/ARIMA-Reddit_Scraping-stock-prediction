# stock_sentiment.py

import sys
import os
import time
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

sia = SentimentIntensityAnalyzer()


class StockSentimentAnalyzer:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id = os.getenv("REDDIT_CLIENT_ID"),
            client_secret = os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent = os.getenv("REDDIT_USER_AGENT", "stock-sentiment/1.0")
        )

    def arima_forecast(self, series: pd.Series, order=(5, 1, 0), steps=5) -> pd.Series:
        """
        Iterative, 1-step-ahead ARIMA forecast.
        Each predicted point is appended to the history so
        the next forecast is anchored on the last prediction.
        """
        history = series.copy()
        preds = []

        for _ in range(steps):
            model = ARIMA(history, order=order).fit()
            # use .iloc[0] to grab first forecast element regardless of index labels
            yhat = model.forecast(steps=1).iloc[0]
            preds.append(yhat)

            # append to history on next business day
            next_day = history.index[-1] + BDay(1)
            history.loc[next_day] = yhat

        # build business-day index for forecast series
        start = series.index[-1] + BDay(1)
        fc_index = pd.bdate_range(start=start, periods=steps)
        return pd.Series(preds, index=fc_index)


def main():
    # Get stock symbol from user input
    while True:
        symbol = input("\nEnter stock symbol (e.g., AAPL, TSLA, MSFT) or 'quit' to exit: ").upper()
        if symbol == 'QUIT':
            print("Exiting program...")
            sys.exit(0)

        # define a 6-month window
        end = pd.to_datetime("today")
        start = end - pd.DateOffset(months=6)

        print(f"\nDownloading {symbol} prices from {start.date()} to {end.date()}...")
        try:
            df = yf.download(symbol, start=start, end=end)
            prices = df.get("Close", pd.Series(dtype=float))

            if prices.empty:
                print("No price data found. Check if the stock symbol is correct.")
                continue

            print("\nRunning 5-day iterative ARIMA forecast...")
            analyzer = StockSentimentAnalyzer()
            forecast = analyzer.arima_forecast(prices, order=(5, 1, 0), steps=5)

            print("\nGenerating plot...")
            plotter = SentimentPlotter()
            fig = plotter.plot_price_forecast(
                historical=prices,
                forecast=forecast,
                symbol=symbol
            )

            print(f"\nForecast for {symbol} - next 5 business days:")
            for date, price in forecast.items():
                print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
            
            print("\nA plot window should open automatically. Close it to continue.")
            print("Enter a new symbol or 'quit' to exit.")

        except Exception as e:
            print(f"\nError processing {symbol}: {str(e)}")
            print("Please try another symbol.")


if __name__ == "__main__":
    print("Stock Price Forecasting Tool")
    print("----------------------------")
    main()
