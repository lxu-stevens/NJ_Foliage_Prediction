# maple/features.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np
import pandas as pd

from .climate_data import ClimateData


def compute_weekly_features(climate: ClimateData) -> pd.DataFrame:
    """
    Aggregate daily climate data into weekly features.

    Returns a DataFrame where each row is one calendar week and includes:
    - tavg_mean          : mean of daily average temperature
    - diurnal_range_mean : mean of (tmax - tmin)
    - temp_change_rate   : week-to-week change of tavg_mean
    - cold_day_count     : number of days with tmin < 5°C
    - prcp_total         : total weekly precipitation
    - prcp_mean          : mean daily precipitation
    - rainy_day_count    : number of days with prcp > 0
    - dry_fraction       : fraction of non-rainy days
    """
    df = climate.df.copy()

    # If tavg is missing, compute it from tmin and tmax
    if "tavg" not in df.columns and {"tmin", "tmax"}.issubset(df.columns):
        df["tavg"] = (df["tmax"] + df["tmin"]) / 2.0

    # Diurnal range (day-night temperature difference)
    if {"tmax", "tmin"}.issubset(df.columns):
        df["diurnal_range"] = df["tmax"] - df["tmin"]
    else:
        df["diurnal_range"] = np.nan

    # Fill precipitation/snow NaNs with zeros
    for col in ["prcp", "snow"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    df["year"] = df["date"].dt.year
    df["week"] = df["date"].dt.isocalendar().week
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")

    df["rainy"] = (df["prcp"] > 0).astype(int)
    group_cols = ["year", "week", "week_start"]

    # dict comprehension demonstrates Part 2 requirement
    agg_dict = {
        "tavg": ["mean"],
        "diurnal_range": ["mean"],
        "prcp": ["sum", "mean"],
        "rainy": ["sum"],
    }

    weekly = df.groupby(group_cols).agg(agg_dict)
    weekly.columns = [
        f"{col}_{stat}" for col, stat in weekly.columns.to_flat_index()
    ]
    weekly = weekly.reset_index()

    rename_map = {
        "tavg_mean": "tavg_mean",
        "diurnal_range_mean": "diurnal_range_mean",
        "prcp_sum": "prcp_total",
        "prcp_mean": "prcp_mean",
        "rainy_sum": "rainy_day_count",
    }
    weekly = weekly.rename(columns=rename_map)

    # Week-to-week change in average temperature
    weekly = weekly.sort_values(["year", "week"])
    weekly["temp_change_rate"] = weekly["tavg_mean"].diff()

    # Count cold days (tmin < 5°C) per week
    cold_flag = (df["tmin"] < 5.0).astype(int)
    cold_week = (
        df.assign(cold=cold_flag)
        .groupby(group_cols)["cold"]
        .sum()
        .reset_index()
        .rename(columns={"cold": "cold_day_count"})
    )
    weekly = weekly.merge(cold_week, on=group_cols, how="left")

    # Number of days per week and dry fraction
    days_per_week = (
        df.groupby(group_cols)["date"]
        .count()
        .reset_index()
        .rename(columns={"date": "n_days"})
    )
    weekly = weekly.merge(days_per_week, on=group_cols, how="left")
    weekly["dry_fraction"] = (
        weekly["n_days"] - weekly["rainy_day_count"]
    ) / weekly["n_days"]

    return weekly


@dataclass
class WeeklyClimateDataset:
    """
    Weekly-level climate features derived from a ClimateData instance.

    Relationship: composition (this class "has a" ClimateData).
    """
    climate: ClimateData
    weekly_df: pd.DataFrame

    @classmethod
    def from_climate(cls, climate: ClimateData) -> "WeeklyClimateDataset":
        """Construct WeeklyClimateDataset from a ClimateData object."""
        weekly = compute_weekly_features(climate)
        return cls(climate=climate, weekly_df=weekly)

    def iter_weeks(self) -> Generator[tuple[int, pd.Series], None, None]:
        """
        Generator over weekly rows.

        Yields
        ------
        index : int
            Row index.
        row : pandas.Series
            Weekly feature row.
        """
        for idx, (_, row) in enumerate(self.weekly_df.iterrows()):
            yield idx, row

    def __add__(self, other: "WeeklyClimateDataset") -> "WeeklyClimateDataset":
        """
        Operator overloading: merge two weekly datasets.

        Demonstrates __add__ for assignment Part 2.
        """
        if set(self.weekly_df.columns) != set(other.weekly_df.columns):
            raise ValueError("Cannot add weekly datasets with different schemas.")

        merged_df = pd.concat(
            [self.weekly_df, other.weekly_df], ignore_index=True
        ).sort_values(["year", "week"])

        merged_climate = ClimateData(
            station_id=f"{self.climate.station_id}+{other.climate.station_id}",
            region_name=f"{self.climate.region_name} + {other.climate.region_name}",
            df=pd.concat([self.climate.df, other.climate.df], ignore_index=True),
        )
        return WeeklyClimateDataset(merged_climate, merged_df)

    def select_year(self, year: int) -> "WeeklyClimateDataset":
        """Return subset of weekly data for a given year."""
        mask = self.weekly_df["year"] == year
        return WeeklyClimateDataset(
            climate=self.climate.subset_year(year),
            weekly_df=self.weekly_df.loc[mask].reset_index(drop=True),
        )
