# plotter.py

import matplotlib
matplotlib.use("TkAgg")   # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt
from datetime import datetime


class SentimentPlotter:
    def __init__(self):
        
        self.figsize = (12, 6)

    def plot_sentiment_trend(self, sentiment_series, stock_df=None, title="Sentiment Trend"):
        """
        Plot daily sentiment scores over time.
        If you pass a stock_df with a 'Close' column, it will overlay the closing price.
        """
        fig, ax1 = plt.subplots(figsize=self.figsize)
        ax1.plot(sentiment_series.index, sentiment_series.values,
                 label="Sentiment", linewidth=2)
        ax1.set_ylabel("Sentiment Score")
        ax1.set_title(title)
        ax1.grid(True)

        if stock_df is not None and "Close" in stock_df:
            ax2 = ax1.twinx()
            ax2.plot(stock_df.index, stock_df["Close"],
                     label="Close Price", alpha=0.7)
            ax2.set_ylabel("Stock Close Price")
            # Combine legends
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper left")

        return fig

    def plot_sentiment_distribution(self, sentiment_values, title="Sentiment Distribution"):
        """
        Show a histogram of sentiment scores to see their spread.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.hist(sentiment_values, bins=20, edgecolor="black")
        ax.set_xlabel("Sentiment Score")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.grid(True)
        return fig

    def plot_price_forecast(self, historical, forecast, symbol):
        """Plot historical prices and forecast."""
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(historical.index, historical, 
                label='Historical Prices', 
                color='blue',
                linewidth=2)
        
        # Plot forecast
        plt.plot(forecast.index, forecast, 
                label='Forecast', 
                color='red', 
                linewidth=2)
        
        # Add vertical line at the last historical date
        plt.axvline(x=historical.index[-1], 
                   color='gray', 
                   linestyle=':', 
                   alpha=0.5, 
                   label='Forecast Start')
        
        # Add today's date and format title
        today = datetime.now().strftime('%Y-%m-%d')
        plt.title(f'{symbol} Price Forecast\nAs of {today}')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        
        # Show the plot in a new window
        plt.show()
        
        return plt.gcf()
