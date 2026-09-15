import numpy as np
import pandas as pd

from marketbench.backtest import backtest


def test_future_change_does_not_change_earlier_predictions():
    frame = pd.read_csv("examples/prices.csv")
    first, report = backtest(frame)
    frame.loc[len(frame) - 1, "close"] *= 5
    changed, _ = backtest(frame)
    np.testing.assert_allclose(first.prediction, changed.prediction)
    assert report["observations"] == 35
