# maple/labeling.py

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class LabelingConfig:
    """
    Rule-based configuration for defining peak foliage weeks.

    Parameters
    ----------
    min_tavg : float
        Minimum weekly average temperature (°C) for good foliage color.
    max_tavg : float
        Maximum weekly average temperature (°C) for good foliage color.
    min_cold_days : int
        Minimum number of cold days (tmin < cold_threshold) in the week.
    cold_threshold : float
        Temperature threshold (°C) to define a "cold day".
    max_prcp : float
        Maximum weekly total precipitation (mm).
    """
    min_tavg: float = 8.0
    max_tavg: float = 18.0
    min_cold_days: int = 2
    cold_threshold: float = 5.0
    max_prcp: float = 60.0


def assign_peak_labels_rule_based(
    weekly_df: pd.DataFrame,
    config: LabelingConfig,
) -> pd.DataFrame:
    """
    Assign a binary label 'is_peak' to each week based purely on climate features.

    A week is labeled as peak (1) if all of the following hold:
    - tavg_mean is between [min_tavg, max_tavg]
    - cold_day_count >= min_cold_days
    - prcp_total <= max_prcp

    Parameters
    ----------
    weekly_df : pandas.DataFrame
        Weekly features with columns 'tavg_mean', 'cold_day_count', 'prcp_total'.
    config : LabelingConfig
        Thresholds for the rule-based labeling.

    Returns
    -------
    pandas.DataFrame
        Same as input with an extra 'is_peak' column (0 or 1).
    """
    df = weekly_df.copy()

    cond_temp = (df["tavg_mean"] >= config.min_tavg) & (
        df["tavg_mean"] <= config.max_tavg
    )
    cond_cold = df["cold_day_count"] >= config.min_cold_days
    cond_prcp = df["prcp_total"] <= config.max_prcp

    df["is_peak"] = (cond_temp & cond_cold & cond_prcp).astype(int)
    return df

def assign_peak_labels_trend_based(
    weekly_df: pd.DataFrame,
    smooth_window: int = 3,
    offset: int = 1,
) -> pd.DataFrame:
    """
    Assign a binary label 'is_peak' based on the temperature trend for each year.

    Idea:
    - For each calendar year, we look at the weekly average temperature series.
    - We smooth the series using a rolling mean.
    - We find the week where the temperature drops the fastest
      (i.e., the largest negative first difference).
    - The foliage peak is assumed to occur `offset` weeks BEFORE that sharp drop.

    Parameters
    ----------
    weekly_df : pandas.DataFrame
        Weekly climate features including at least 'week_start' and 'tavg_mean'.
    smooth_window : int
        Window size for rolling mean smoothing.
    offset : int
        Number of weeks before the sharpest drop to mark as the peak week.

    Returns
    -------
    pandas.DataFrame
        Copy of weekly_df with an 'is_peak' column (0 or 1).
    """
    df = weekly_df.copy()

    if "week_start" not in df.columns:
        raise ValueError("weekly_df must contain a 'week_start' column.")
    if "tavg_mean" not in df.columns:
        raise ValueError("weekly_df must contain a 'tavg_mean' column.")

    # Ensure we have a year column
    if "year" not in df.columns:
        df["year"] = df["week_start"].dt.year

    # Initialize all weeks as non-peak
    df["is_peak"] = 0

    # Process each year separately
    for year, sub_orig in df.groupby("year"):
        # Sort by week_start and keep original index
        sub = sub_orig.sort_values("week_start").reset_index()  # original index in 'index' column

        if len(sub) < smooth_window + 2:
            # Not enough data to detect a trend
            continue

        # Smooth the temperature series
        smooth = sub["tavg_mean"].rolling(
            window=smooth_window,
            center=True,
            min_periods=1,
        ).mean()

        # First difference: week-to-week change
        diffs = smooth.diff()

        # Skip the first NaN and find the most negative change
        diffs_valid = diffs.iloc[1:]
        if diffs_valid.empty:
            continue

        # Index (in 'sub') of the fastest temperature drop
        i_min = diffs_valid.idxmin()

        # Choose the peak week as `offset` weeks before the sharpest drop
        i_peak = max(i_min - offset, 0)

        # Map back to original index in df
        orig_idx = sub.loc[i_peak, "index"]
        df.loc[orig_idx, "is_peak"] = 1

    return df

