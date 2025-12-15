# pipeline.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from sentiment import daily_sentiment


@st.cache_data(ttl=60 * 60, show_spinner="Downloading stock data...")
def fetch_stock(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV from yfinance and return a normalized dataframe with Date, Close.
    Robust to multi-index columns and empty responses.
    """
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)

    # Handle empty result
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Close"])

    # Flatten multi-index columns if present (e.g., ('Close','AAPL')).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c != ""]) for col in df.columns]

    df = df.reset_index()

    # Normalize date column name
    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        else:
            # Fall back to first column as Date
            df = df.rename(columns={df.columns[0]: "Date"})

    # Try to locate a close column robustly
    close_col = None
    for cand in ["Close", "Adj Close", "Close_"+ticker, "Adj Close_"+ticker]:
        if cand in df.columns:
            close_col = cand
            break

    if close_col is None:
        # Last resort: pick the first column containing "Close"
        close_like = [c for c in df.columns if "Close" in str(c)]
        if close_like:
            close_col = close_like[0]
        else:
            # No close prices found
            return pd.DataFrame(columns=["Date", "Close"])

    out = df[["Date", close_col]].copy()
    out = out.rename(columns={close_col: "Close"})
    out["Date"] = pd.to_datetime(out["Date"]).dt.date.astype(str)
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Close"]).reset_index(drop=True)
    return out


def merge_stock_sentiment(stock_df: pd.DataFrame, sentiment_daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a dataframe with: Date, Close, sentiment_score
    Missing sentiment is forward-filled then filled with 0.
    """
    s = stock_df.copy()
    s["Date"] = pd.to_datetime(s["Date"]).dt.date.astype(str)

    if sentiment_daily_df is None or sentiment_daily_df.empty:
        s["sentiment_score"] = 0.0
        return s

    d = sentiment_daily_df.copy()
    d = d.rename(columns={"date": "Date"})
    d["Date"] = pd.to_datetime(d["Date"]).dt.date.astype(str)
    d["sentiment_score"] = pd.to_numeric(d["sentiment_score"], errors="coerce")

    merged = s.merge(d[["Date", "sentiment_score"]], on="Date", how="left").sort_values("Date")
    merged["sentiment_score"] = merged["sentiment_score"].ffill().fillna(0.0)
    return merged.reset_index(drop=True)


def run_sentiment_step(ticker: str, start: str, end: str, max_records: int = 250) -> Dict[str, pd.DataFrame]:
    """
    Returns {"articles": df, "daily": df}
    """
    return daily_sentiment(ticker=ticker, start_date=start, end_date=end, max_records=max_records)


def run_merge_step(ticker: str, start: str, end: str, sentiment_daily_df: pd.DataFrame) -> pd.DataFrame:
    stock = fetch_stock(ticker, start, end)
    return merge_stock_sentiment(stock, sentiment_daily_df)
