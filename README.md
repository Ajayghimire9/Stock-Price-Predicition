# MarketBench

Walk-forward forecasting research.

MarketBench evaluates a simple one-step price forecast against persistence. It makes the time boundary explicit so that future observations cannot influence earlier fits.

## Run locally

Use Python 3.11 or newer in a virtual environment.

```bash
pip install -r requirements-portfolio.txt
python -m marketbench.backtest --data examples/prices.csv
```

## Design decisions

Every prediction fits a scaler and Ridge model using only prior rows. The training window expands one observation at a time.

The report compares model MAE with the last-observation baseline and writes row-level predictions for inspection.

Input dates are sorted and must be unique; prices must be finite and positive. A regression test changes the final price and checks that earlier predictions remain unchanged.

## Technology

Python, pandas, NumPy, scikit-learn, pytest.

## Validation

Run `python -m pytest tests -q` from the repository root. CI runs the maintained test suite and lint checks. Tests use local fixtures or mocks and do not deploy cloud resources.

## Scope and limitations

The bundled series is synthetic and exists only for reproducibility checks. This is a forecasting experiment, not a trading strategy; it does not model fees, fills, corporate actions or returns. LSTM claims from the original README were removed because no runnable LSTM implementation was present.
