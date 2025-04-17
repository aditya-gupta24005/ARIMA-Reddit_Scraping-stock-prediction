import matplotlib
matplotlib.use("Agg")  # Non‑interactive backend for Streamlit/CI

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Optional

# -----------------------------------------------------------------------------
# Visualization helper class
# -----------------------------------------------------------------------------
class SentimentPlotter:
    """
    Centralized matplotlib/seaborn helper for sentiment & price visualizations.
    """

    def __init__(self, figsize: tuple[int, int] = (12, 8)) -> None:
        # Use seaborn default theme instead of deprecated plt.style.use("seaborn")
        sns.set_theme(style="darkgrid")
        self.figsize = figsize

    # 1. Sentiment time series ------------------------------------------------------------------
    def plot_sentiment_trend(
        self,
        sentiment_data: pd.Series,
        stock_data: Optional[pd.DataFrame] = None,
        title: str = "Sentiment Analysis",
    ) -> plt.Figure:
        """
        Time‑series plot of sentiment scores with optional stock‑price overlay.
        """
        if sentiment_data.empty:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(
                0.5,
                0.5,
                "No sentiment data available",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_axis_off()
            fig.suptitle(title)
            return fig

        fig, ax1 = plt.subplots(figsize=self.figsize)

        # Plot sentiment
        color_sent = "tab:blue"
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Sentiment Score", color=color_sent)
        ax1.plot(sentiment_data.index, sentiment_data.values, color=color_sent, label="Sentiment")
        ax1.tick_params(axis="y", labelcolor=color_sent)

        # Optional price overlay
        if stock_data is not None and not stock_data.empty:
            ax2 = ax1.twinx()
            color_price = "tab:red"
            ax2.set_ylabel("Stock Price", color=color_price)
            ax2.plot(stock_data.index, stock_data["Close"], color=color_price, label="Close Price")
            ax2.tick_params(axis="y", labelcolor=color_price)

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    # 2. Sentiment distribution -----------------------------------------------------------------
    def plot_sentiment_distribution(
        self, sentiment_scores: np.ndarray | list, title: str = "Sentiment Distribution"
    ) -> plt.Figure:
        """
        Histogram + KDE for sentiment score distribution.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        if len(sentiment_scores) > 0:
            sns.histplot(sentiment_scores, kde=True, ax=ax)
            ax.set_xlabel("Sentiment Score")
            ax.set_ylabel("Frequency")
        else:
            ax.text(
                0.5,
                0.5,
                "No sentiment data available",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_axis_off()

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    # 3. Post quality distribution --------------------------------------------------------------
    def plot_quality_scores(
        self, posts_df: pd.DataFrame, title: str = "Post Quality Distribution"
    ) -> plt.Figure:
        """
        Histogram of quality scores stored in `posts_df['quality_score']`.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        if "quality_score" in posts_df.columns and not posts_df.empty:
            sns.histplot(posts_df["quality_score"], kde=True, ax=ax)
            ax.set_xlabel("Quality Score")
            ax.set_ylabel("Frequency")
        else:
            ax.text(
                0.5,
                0.5,
                "No quality score data available",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_axis_off()

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    # 4. Sentiment‑vs‑price scatter -------------------------------------------------------------
    def plot_sentiment_vs_price(
        self,
        sentiment_data: pd.Series | np.ndarray | list,
        stock_data: pd.DataFrame,
        title: str = "Sentiment vs Stock Price",
    ) -> plt.Figure:
        """
        Scatterplot of sentiment scores against closing price with optional trend line.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if len(sentiment_data) > 0 and not stock_data.empty:
            ax.scatter(sentiment_data, stock_data["Close"], alpha=0.6, s=40)

            # Trend line
            try:
                z = np.polyfit(sentiment_data, stock_data["Close"], 1)
                p = np.poly1d(z)
                ax.plot(sentiment_data, p(sentiment_data), "r--")
            except Exception as exc:
                # Fail silently but log message
                print(f"[plot_sentiment_vs_price] trend‑line failed: {exc}")

            ax.set_xlabel("Sentiment Score")
            ax.set_ylabel("Stock Price")
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for correlation plot",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_axis_off()

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    # 5. Save utility ---------------------------------------------------------------------------
    def save_plot(self, fig: plt.Figure, filename: str) -> None:
        """
        Persist a matplotlib Figure to disk, creating parent dirs as needed.
        """
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fig.savefig(filename, bbox_inches="tight")
        finally:
            plt.close(fig)


# -----------------------------------------------------------------------------
# Backwards‑compatibility façade
# -----------------------------------------------------------------------------
# Many callers (e.g. streamlit_app.py) still import the legacy
# `plot_sentiment_vs_price` function directly.  Provide a thin wrapper so we
# don't break them.
_plotter_singleton = SentimentPlotter()

def plot_sentiment_vs_price(
    sentiment_data: pd.Series | np.ndarray | list,
    stock_data: pd.DataFrame,
    title: str = "Sentiment vs Stock Price",
) -> plt.Figure:
    """
    Wrapper delegating to SentimentPlotter.plot_sentiment_vs_price.
    """
    return _plotter_singleton.plot_sentiment_vs_price(sentiment_data, stock_data, title)
