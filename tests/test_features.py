# tests/test_features.py

import pandas as pd

from maple.climate_data import ClimateData
from maple.features import WeeklyClimateDataset


def test_weekly_features_basic():
    """
    Basic test that weekly feature generation works
    and required columns are present.
    """
    df = pd.DataFrame({
        "date": pd.date_range("2023-10-01", periods=14, freq="D"),
        "tmin": [10] * 14,
        "tmax": [20] * 14,
        "tavg": [15] * 14,
        "prcp": [0, 1] * 7,
    })
    climate = ClimateData(station_id="TEST", region_name="TestRegion", df=df)
    weekly_dataset = WeeklyClimateDataset.from_climate(climate)
    weekly = weekly_dataset.weekly_df

    assert len(weekly) >= 2
    for col in ["tavg_mean", "prcp_total", "rainy_day_count", "dry_fraction"]:
        assert col in weekly.columns
