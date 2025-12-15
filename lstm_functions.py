# lstm_functions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# TensorFlow/Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def _mape(y_true, y_pred, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    xs, ys = [], []
    for i in range(lookback, len(X)):
        xs.append(X[i - lookback : i, :])
        ys.append(y[i])
    return np.array(xs), np.array(ys)


def lstm_predict_with_sentiment(
    merged_df: pd.DataFrame,
    lookback: int = 30,
    test_ratio: float = 0.2,
    epochs: int = 20,
    batch_size: int = 32,
    seed: int = 7,
) -> Dict[str, Any]:
    """
    Input merged_df columns: Date, Close, sentiment_score
    Returns:
      {
        metrics: {MAE, MSE, RMSE, R2, MAPE},
        plot: {train_dates, train_actual, test_dates, test_actual, test_pred},
        sample_last10: DataFrame
      }
    """
    np.random.seed(seed)

    df = merged_df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # required columns
    for c in ["Date", "Close", "sentiment_score"]:
        if c not in df.columns:
            raise ValueError(f"merged_df missing required column: {c}")

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["Close"]).reset_index(drop=True)

    if len(df) < lookback + 10:
        raise ValueError(f"Not enough rows ({len(df)}) for lookback={lookback}. Increase date range or reduce lookback.")

    features = df[["Close", "sentiment_score"]].values.astype(np.float32)

    # scale features jointly
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Predict Close (scaled) using past window of [Close, sentiment]
    X_all = features_scaled
    y_all = features_scaled[:, 0]  # scaled Close

    X_seq, y_seq = _make_sequences(X_all, y_all, lookback=lookback)

    # dates aligned to y_seq indices
    seq_dates = df["Date"].iloc[lookback:].to_numpy()

    n = len(X_seq)
    test_n = max(1, int(n * test_ratio))
    train_n = n - test_n

    X_train, y_train = X_seq[:train_n], y_seq[:train_n]
    X_test, y_test = X_seq[train_n:], y_seq[train_n:]
    test_dates = seq_dates[train_n:]
    train_dates = seq_dates[:train_n]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    cb = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cb],
        verbose=0,
        shuffle=False,
    )

    # predict (scaled close)
    y_pred_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    y_test_scaled = y_test.reshape(-1, 1)

    # inverse transform: need 2 features for inverse, so pad sentiment with zeros
    pad0 = np.zeros_like(y_pred_scaled)
    inv_pred = scaler.inverse_transform(np.hstack([y_pred_scaled, pad0]))[:, 0]
    inv_true = scaler.inverse_transform(np.hstack([y_test_scaled, pad0]))[:, 0]

    mae = float(mean_absolute_error(inv_true, inv_pred))
    mse = float(mean_squared_error(inv_true, inv_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(inv_true, inv_pred))
    mape = _mape(inv_true, inv_pred)

    sample = pd.DataFrame({
        "Date": test_dates,
        "Actual": inv_true,
        "Predicted": inv_pred,
        "AbsError": np.abs(inv_true - inv_pred),
    }).tail(10).reset_index(drop=True)

    return {
        "metrics": {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape},
        "plot": {
            "train_dates": train_dates,
            "test_dates": test_dates,
            "test_actual": inv_true,
            "test_pred": inv_pred,
        },
        "sample_last10": sample,
    }
