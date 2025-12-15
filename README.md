# Sentiment Analysis of Finantial News and its effect on Stocks using LSTM

This repository contains a **Streamlit application** that predicts stock prices using a three-step pipeline combining **news sentiment analysis (FinBERT)** with **historical stock prices** and an **LSTM neural network**.

The project demonstrates how financial news sentiment can be integrated with time-series market data to improve predictive modeling and visualization.

---

## Overview

The application allows you to:

* Select a stock ticker (TSLA, AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, BAC, ^GSPC).
* Fetch and analyze financial news sentiment over a selected date range using **GDELT + FinBERT**.
* Aggregate sentiment into **daily sentiment scores**.
* Merge daily sentiment with **historical stock price data** from Yahoo Finance.
* Train an **LSTM model** to predict closing prices using both price and sentiment features.
* Visualize predictions and evaluate model performance.

---

## Features

### 🔁 Three-Step Workflow (UI Driven)

1. **Run Sentiment Analysis**

   * Fetches news from GDELT.
   * Scores article titles using FinBERT.
   * Aggregates sentiment into daily averages.

2. **Run LSTM With Sentiment**

   * Downloads historical stock prices.
   * Merges price data with daily sentiment.
   * Trains an LSTM model and generates predictions.

3. **View Results**

   * Interactive Plotly chart of actual vs predicted prices.
   * Error metrics and recent prediction samples.

---

### 💾 Caching & Artifacts

* **In-memory caching** (Streamlit):

  * FinBERT model
  * GDELT API responses
  * yfinance stock data

* **On-disk artifacts** (`artifacts/` directory):

  * Raw scored news articles
  * Daily sentiment aggregates
  * Merged stock + sentiment datasets
  * LSTM metrics and predictions (JSON)

---

### 🛡 Robust Data Handling

* Defensive parsing of GDELT responses (HTTP / JSON failures).
* Ticker-to-company-name mapping for improved news coverage.
* Handles empty or multi-index responses from yfinance.
* Normalizes stock data to `Date` and `Close` columns.
* Forces **PyTorch backend** for FinBERT to avoid TensorFlow/Keras dependency issues.

---

## Project Structure

```
.
├── app.py              # Streamlit UI & step orchestration
├── pipeline.py         # Sentiment + stock merge pipeline
├── sentiment.py        # GDELT fetching, FinBERT scoring, aggregation
├── lstm_functions.py   # LSTM training, prediction, metrics
├── requirements.txt    # Python dependencies
├── .env                # Optional environment variables
└── artifacts/          # Cached outputs (auto-created)
```

---

## File Descriptions

### `app.py`

* Defines the Streamlit UI layout and sidebar controls.
* Provides buttons for:

  * **Step 1:** Run Sentiment Analysis
  * **Step 2:** Run LSTM With Sentiment
  * Load saved artifacts from disk
* Uses Streamlit session state to store intermediate results.
* Displays:

  * Daily sentiment tables
  * Sample scored articles
  * LSTM metrics (MAE, MSE, RMSE, R², MAPE)
  * Plotly chart of actual vs predicted prices
  * Last 10 test-day predictions

---

### `pipeline.py`

* `fetch_stock(ticker, start, end)`

  * Downloads stock prices using yfinance.
  * Handles empty or malformed responses.
  * Normalizes output to `Date` and `Close`.

* `merge_stock_sentiment(stock_df, sentiment_daily_df)`

  * Left-joins sentiment to stock data by date.
  * Forward-fills missing sentiment values.

* `run_sentiment_step(...)`

  * Wraps sentiment pipeline and returns article-level and daily data.

* `run_merge_step(...)`

  * Fetches stock data and merges it with sentiment.

---

### `sentiment.py`

* Maps stock tickers to full company-name queries for GDELT.
* Loads **ProsusAI/finbert** using PyTorch backend.
* Fetches news articles from the GDELT DOC API.
* Scores article titles with FinBERT sentiment labels.
* Aggregates daily sentiment scores and article counts.

---

### `lstm_functions.py`

* Expects merged data with:

  * `Date`
  * `Close`
  * `sentiment_score`
* Workflow:

  1. Sorts and cleans data
  2. Scales features using MinMaxScaler
  3. Creates sliding-window sequences
  4. Splits train/test sets
  5. Builds and trains a two-layer LSTM with dropout
  6. Applies early stopping
  7. Inverse-scales predictions
  8. Computes performance metrics
* Returns metrics, plot data, and recent prediction samples.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aditya4060/FinSent.git
cd FinSent
```

### 2. Create and Activate a Virtual Environment

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Environment Variables

Create a `.env` file at the project root if needed. GDELT and FinBERT do **not** require API keys.

---

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

In the browser:

1. Select a **ticker** and **date range**.
2. Click **Run Sentiment Analysis**.
3. Review daily sentiment and scored articles.
4. Click **Run LSTM With Sentiment**.
5. Inspect predictions, plots, and metrics.
6. Optionally reload saved artifacts using the provided buttons.

---

## Configuration

* **Tickers**: Update the ticker list in `app.py`.
* **GDELT Queries**: Extend `TICKER_TO_QUERY` in `sentiment.py`.
* **Model Hyperparameters**:

  * `lookback`
  * `test_ratio`
  * `epochs`
  * `batch_size`
* **Artifacts Directory**: Modify `ART_DIR` in `app.py`.

---

## Troubleshooting

* **No sentiment data**:

  * Verify ticker-to-company mapping.
  * Expand the date range.

* **App appears stuck**:

  * First-time FinBERT download may take several minutes.
  * Check terminal logs for GDELT or network errors.

* **Empty stock data**:

  * Ensure the ticker is valid on Yahoo Finance.
  * Avoid future-only or extremely short date ranges.

---
