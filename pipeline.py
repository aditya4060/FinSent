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
    """
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Close"])

    # yfinance returns index as DatetimeIndex
    df = df.reset_index()
    # normalize column names
    if "Date" not in df.columns:
        # sometimes it's "Datetime"
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
    out = df[["Date", "Close"]].copy()
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
