"""Walk-forward one-step forecasting with train-only scaling."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def backtest(frame, lags=5, min_train=30):
    if not {"date", "close"}.issubset(frame.columns):
        raise ValueError("CSV requires date and close columns")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    frame = frame.sort_values("date").reset_index(drop=True)
    values = frame.close.to_numpy(dtype=float)
    if lags < 1 or min_train < 2 or len(frame) <= lags + min_train:
        raise ValueError("Not enough history for the requested training window")
    if (
        frame.date.duplicated().any()
        or not np.isfinite(values).all()
        or (values <= 0).any()
    ):
        raise ValueError("Dates must be unique and prices finite and positive")
    x = np.array([values[t - lags : t][::-1] for t in range(lags, len(values))])
    y = values[lags:]
    rows = []
    for i in range(min_train, len(y)):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x[:i], y[:i])
        rows.append(
            {
                "date": str(frame.date.iloc[lags + i].date()),
                "actual": float(y[i]),
                "prediction": float(model.predict(x[i : i + 1])[0]),
                "persistence": float(x[i, 0]),
            }
        )
    predictions = pd.DataFrame(rows)
    report = {
        "observations": len(rows),
        "model_mae": float((predictions.actual - predictions.prediction).abs().mean()),
        "persistence_mae": float(
            (predictions.actual - predictions.persistence).abs().mean()
        ),
        "protocol": "expanding-window one-step; no future observations in fit",
    }
    return predictions, report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="examples/prices.csv")
    p.add_argument("--output", default="artifacts")
    a = p.parse_args()
    pred, report = backtest(pd.read_csv(a.data))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
