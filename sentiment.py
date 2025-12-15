# sentiment.py
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict

import pandas as pd
import requests
import streamlit as st
from transformers import pipeline


GDELT_DOC_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "streamlit-sentiment-lstm-app/1.0"

# Map tickers to company-name search queries for GDELT
TICKER_TO_QUERY: Dict[str, str] = {
    "TSLA": '"Tesla Inc"',
    "AAPL": '"Apple Inc"',
    "MSFT": '"Microsoft Corp"',
    "GOOGL": '"Alphabet Inc"',
    "AMZN": '"Amazon.com Inc"',
    "META": '"Meta Platforms Inc"',
    "NVDA": '"NVIDIA Corp"',
    "JPM": '"JPMorgan Chase"',
    "BAC": '"Bank of America"',
    "^GSPC": '"S&P 500" OR "Standard & Poor\'s 500"',
}


def _gdelt_query_for_ticker(ticker: str) -> str:
    # Prefer mapped company-name query; fall back to ticker phrase.
    q = TICKER_TO_QUERY.get(ticker)
    if q is not None:
        return q
    return f'"{ticker}" OR "{ticker} stock"'


def _to_utc_date_str(x) -> str:
    # returns YYYY-MM-DD
    if isinstance(x, str):
        return pd.to_datetime(x).date().isoformat()
    if isinstance(x, (datetime, pd.Timestamp)):
        return pd.to_datetime(x).date().isoformat()
    return pd.to_datetime(x).date().isoformat()


@st.cache_resource(show_spinner="Loading FinBERT model...")
def get_finbert_pipeline():
    # Force PyTorch backend so transformers doesn't try to import TF. [web:66][web:71]
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        truncation=True,
        framework="pt",
    )


@st.cache_data(ttl=60 * 60, show_spinner="Fetching news from GDELT...")
def fetch_gdelt_articles(
    ticker: str,
    start_date: str,
    end_date: str,
    max_records: int = 250,
) -> pd.DataFrame:
    """
    Fetch news articles from GDELT DOC API for a ticker in date range.
    Returns DataFrame with: date, title, url, sourceCountry (if present).
    Handles HTTP / JSON errors gracefully and returns empty dataframe on failure.
    """
    start_date = _to_utc_date_str(start_date)
    end_date = _to_utc_date_str(end_date)

    # Inclusive range by extending end to next day 00:00.
    start_dt = f"{start_date} 00:00:00"
    end_dt = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")

    query = _gdelt_query_for_ticker(ticker)

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": int(max_records),
        "startdatetime": pd.to_datetime(start_dt).strftime("%Y%m%d%H%M%S"),
        "enddatetime": pd.to_datetime(end_dt).strftime("%Y%m%d%H%M%S"),
        "sort": "HybridRel",
    }

    headers = {"User-Agent": USER_AGENT}

    try:
        r = requests.get(GDELT_DOC_BASE, params=params, headers=headers, timeout=30)
    except requests.RequestException as e:
        print("GDELT request error:", e)
        return pd.DataFrame(columns=["date", "title", "url", "sourceCountry"])

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        print("GDELT HTTP error:", e, "body snippet:", r.text[:500])
        return pd.DataFrame(columns=["date", "title", "url", "sourceCountry"])

    try:
        data = r.json()
    except (requests.exceptions.JSONDecodeError, json.JSONDecodeError, ValueError) as e:
        print("GDELT JSON decode error:", e, "body snippet:", r.text[:500])
        return pd.DataFrame(columns=["date", "title", "url", "sourceCountry"])

    articles = data.get("articles", [])
    rows = []
    for a in articles:
        url = a.get("url")
        title = a.get("title")
        seendate = a.get("seendate") or a.get("datetime") or a.get("date")
        if not title or not seendate:
            continue
        d = pd.to_datetime(seendate, errors="coerce")
        if pd.isna(d):
            continue
        rows.append(
            {
                "date": d.date().isoformat(),
                "title": str(title),
                "url": str(url) if url else None,
                "sourceCountry": a.get("sourceCountry"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "title", "url", "sourceCountry"])
    return df.sort_values(["date"]).reset_index(drop=True)


def _score_to_signed(label: str, score: float) -> float:
    # Convert FinBERT label+confidence to signed score.
    # positive -> +score, negative -> -score, neutral -> 0
    label = (label or "").lower()
    if "pos" in label:
        return float(score)
    if "neg" in label:
        return -float(score)
    return 0.0


@st.cache_data(ttl=60 * 60, show_spinner="Scoring sentiment with FinBERT...")
def score_articles_finbert(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add FinBERT sentiment label/score columns to the articles dataframe.
    """
    if articles_df is None or articles_df.empty:
        return pd.DataFrame(
            columns=["date", "title", "url", "sourceCountry", "label", "confidence", "sentiment_score"]
        )

    sa = get_finbert_pipeline()

    titles = articles_df["title"].astype(str).tolist()
    preds = sa(titles)

    out = articles_df.copy()
    out["label"] = [p.get("label") for p in preds]
    out["confidence"] = [float(p.get("score", 0.0)) for p in preds]
    out["sentiment_score"] = [
        _score_to_signed(p.get("label", ""), float(p.get("score", 0.0))) for p in preds
    ]
    return out


def daily_sentiment(
    ticker: str,
    start_date: str,
    end_date: str,
    max_records: int = 250,
) -> Dict[str, pd.DataFrame]:
    """
    Orchestrates: fetch -> finbert -> aggregate daily.
    Returns:
      {
        "articles": article-level df,
        "daily": daily-aggregated df (date, sentiment_score, n_articles)
      }
    """
    articles = fetch_gdelt_articles(ticker, start_date, end_date, max_records=max_records)
    scored = score_articles_finbert(articles)

    if scored.empty:
        daily = pd.DataFrame(columns=["date", "sentiment_score", "n_articles"])
        return {"articles": scored, "daily": daily}

    daily = (
        scored.groupby("date", as_index=False)
        .agg(sentiment_score=("sentiment_score", "mean"), n_articles=("sentiment_score", "size"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    return {"articles": scored, "daily": daily}
