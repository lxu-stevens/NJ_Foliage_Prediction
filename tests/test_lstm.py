# tests/test_model_lstm.py

import numpy as np
import pandas as pd

from maple.lstm import FoliageLSTMModel


def test_lstm_training_and_prediction():
    """
    Test that the LSTM model can be trained on synthetic data
    and produce peak probabilities.
    """
    n = 40
    df = pd.DataFrame({
        "tavg_mean": np.random.uniform(5, 20, size=n),
        "diurnal_range_mean": np.random.uniform(3, 10, size=n),
        "temp_change_rate": np.random.uniform(-2, 2, size=n),
        "cold_day_count": np.random.randint(0, 5, size=n),
        "prcp_total": np.random.uniform(0, 80, size=n),
        "dry_fraction": np.random.uniform(0, 1, size=n),
    })
    df["week_start"] = pd.date_range("2023-09-01", periods=n, freq="W-MON")

    df["is_peak"] = (
        (df["tavg_mean"].between(8, 18)) &
        (df["cold_day_count"] >= 2) &
        (df["prcp_total"] <= 60)
    ).astype(int)

    model = FoliageLSTMModel(num_epochs=3, hidden_dim=16, seq_len=4, batch_size=8)
    metrics = model.fit(df)
    assert "accuracy" in metrics

    pred_df = model.predict(df)
    assert "peak_prob_lstm" in pred_df.columns
