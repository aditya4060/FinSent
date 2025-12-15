# app.py
from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from pipeline import run_sentiment_step, run_merge_step
from lstm_functions import lstm_predict_with_sentiment


ART_DIR = Path("artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)


def _artifact_paths(ticker: str, start: str, end: str):
    safe = f"{ticker}_{start}_{end}".replace("^", "IDX_").replace("/", "_")
    return {
        "sent_articles": ART_DIR / f"{safe}_articles.parquet",
        "sent_daily": ART_DIR / f"{safe}_sent_daily.parquet",
        "merged": ART_DIR / f"{safe}_merged.parquet",
        "lstm": ART_DIR / f"{safe}_lstm_results.json",
    }


def _plot_actual_vs_pred(test_dates, actual, pred, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(test_dates), y=list(actual), mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=list(test_dates), y=list(pred), mode="lines", name="Predicted"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Close")
    return fig


def _metrics_table(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame([metrics]).T.rename(columns={0: "value"})


st.set_page_config(page_title="Sentiment → LSTM → Results", layout="wide")
st.title("Sentiment → LSTM → Results")

with st.sidebar:
    st.header("Controls")
    ticker = st.selectbox(
        "Select company (ticker)",
        ["TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "BAC", "^GSPC"],
        index=0,
    )
    start = st.date_input("Start date", value=date(2021, 1, 1))
    end = st.date_input("End date", value=date.today())
    max_records = st.slider("Max news articles (GDELT)", 50, 500, 250, 50)

    lookback = st.slider("LSTM lookback (days)", 10, 120, 30, 5)
    test_ratio = st.slider("Test ratio", 0.1, 0.4, 0.2, 0.05)
    epochs = st.slider("Epochs", 5, 100, 20, 5)
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)

start_s = pd.to_datetime(start).date().isoformat()
end_s = pd.to_datetime(end).date().isoformat()
paths = _artifact_paths(ticker, start_s, end_s)

# Session state keys
st.session_state.setdefault("sentiment_daily", None)
st.session_state.setdefault("sentiment_articles", None)
st.session_state.setdefault("merged_df", None)
st.session_state.setdefault("lstm_results", None)

colA, colB, colC = st.columns([1, 1, 1])

with colA:
    st.subheader("Step 1: Sentiment")
    if st.button("Run Sentiment Analysis", use_container_width=True):
        out = run_sentiment_step(ticker, start_s, end_s, max_records=max_records)
        st.session_state["sentiment_articles"] = out["articles"]
        st.session_state["sentiment_daily"] = out["daily"]

        # persist
        out["articles"].to_parquet(paths["sent_articles"], index=False)
        out["daily"].to_parquet(paths["sent_daily"], index=False)

    if st.button("Load Sentiment Artifact", use_container_width=True):
        if paths["sent_daily"].exists():
            st.session_state["sentiment_daily"] = pd.read_parquet(paths["sent_daily"])
        if paths["sent_articles"].exists():
            st.session_state["sentiment_articles"] = pd.read_parquet(paths["sent_articles"])

with colB:
    st.subheader("Step 2: Merge + LSTM")
    disabled_lstm = st.session_state["sentiment_daily"] is None

    if st.button("Run LSTM With Sentiment", disabled=disabled_lstm, use_container_width=True):
        merged = run_merge_step(ticker, start_s, end_s, st.session_state["sentiment_daily"])
        st.session_state["merged_df"] = merged
        merged.to_parquet(paths["merged"], index=False)

        results = lstm_predict_with_sentiment(
            merged_df=merged,
            lookback=lookback,
            test_ratio=float(test_ratio),
            epochs=int(epochs),
            batch_size=int(batch_size),
        )
        st.session_state["lstm_results"] = results

        # persist results as JSON
        serializable = {
            "metrics": results["metrics"],
            "plot": {
                "train_dates": list(results["plot"]["train_dates"]),
                "test_dates": list(results["plot"]["test_dates"]),
                "test_actual": [float(x) for x in results["plot"]["test_actual"]],
                "test_pred": [float(x) for x in results["plot"]["test_pred"]],
            },
            "sample_last10": results["sample_last10"].to_dict(orient="records"),
        }
        paths["lstm"].write_text(json.dumps(serializable, indent=2))

    if st.button("Load Merge/LSTM Artifacts", use_container_width=True):
        if paths["merged"].exists():
            st.session_state["merged_df"] = pd.read_parquet(paths["merged"])
        if paths["lstm"].exists():
            obj = json.loads(paths["lstm"].read_text())
            st.session_state["lstm_results"] = {
                "metrics": obj["metrics"],
                "plot": {
                    "train_dates": obj["plot"]["train_dates"],
                    "test_dates": obj["plot"]["test_dates"],
                    "test_actual": obj["plot"]["test_actual"],
                    "test_pred": obj["plot"]["test_pred"],
                },
                "sample_last10": pd.DataFrame(obj["sample_last10"]),
            }

with colC:
    st.subheader("Artifacts")
    st.write("Saved under:", str(ART_DIR.resolve()))
    st.write({k: str(v) for k, v in paths.items()})

st.divider()

# --- Display Step 1 outputs ---
st.header("Sentiment outputs")
if st.session_state["sentiment_daily"] is not None:
    daily = st.session_state["sentiment_daily"]
    st.write("Daily sentiment (mean score) and article count:")
    tail_daily = daily.tail(20).reset_index(drop=True)
    tail_daily.index = tail_daily.index + 1
    st.dataframe(tail_daily, use_container_width=True)
    # st.dataframe(daily.tail(20), use_container_width=True)

if st.session_state["sentiment_articles"] is not None:
    art = st.session_state["sentiment_articles"]
    st.write("Sample scored articles:")
    st.dataframe(art[["date", "title", "label", "confidence", "sentiment_score"]].head(15), use_container_width=True)

st.divider()

# --- Display Step 3 results ---
st.header("LSTM results")
if st.session_state["lstm_results"] is None:
    st.info("Run Step 2 to see charts + metrics + last 10 days sample.")
else:
    res = st.session_state["lstm_results"]
    metrics_df = _metrics_table(res["metrics"])

    left, right = st.columns([2, 1])
    with left:
        fig = _plot_actual_vs_pred(
            res["plot"]["test_dates"],
            res["plot"]["test_actual"],
            res["plot"]["test_pred"],
            title=f"{ticker}: Actual vs Predicted (test set)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.write("Metrics:")
        st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Last 10 test days")
    st.dataframe(res["sample_last10"], use_container_width=True)
