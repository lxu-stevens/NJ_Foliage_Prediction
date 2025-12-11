# maple/climate_data.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class InvalidClimateDataError(Exception):
    """Raised when the climate CSV is missing required columns."""
    pass


@dataclass
class ClimateData:
    """
    Daily climate data for one station / region.

    Attributes
    ----------
    station_id : str
        Station identifier, e.g. 'MY_STATION'.
    region_name : str
        Human readable region, e.g. 'Newark, DE'.
    df : pandas.DataFrame
        Daily data with at least: 'date', 'tavg' or ('tmin','tmax'), 'prcp'.
    """
    station_id: str
    region_name: str
    df: pd.DataFrame

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        station_id: str,
        region_name: str,
    ) -> "ClimateData":
        """
        Load daily climate data from CSV.

        Expected columns (matching your dataset):
        - date, tavg, tmin, tmax, prcp, snow, wdir, wspd, wpgt, pres, tsun

        Only some of them are required:
        - 'date'
        - 'prcp'
        - 'tavg' or both 'tmin' and 'tmax'
        """
        path = Path(path)
        if not path.exists():
            # Exception handling 1: raise FileNotFoundError
            raise FileNotFoundError(f"Climate data file not found: {path}")

        df = pd.read_csv(path)

        if "date" not in df.columns:
            # Exception handling 2: custom error
            raise InvalidClimateDataError("CSV must contain a 'date' column.")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        has_tavg = "tavg" in df.columns
        has_tmin = "tmin" in df.columns
        has_tmax = "tmax" in df.columns
        has_prcp = "prcp" in df.columns

        if not has_prcp:
            raise InvalidClimateDataError("CSV must contain a 'prcp' column.")
        if not (has_tavg or (has_tmin and has_tmax)):
            raise InvalidClimateDataError(
                "CSV must contain 'tavg' or both 'tmin' and 'tmax'."
            )

        return cls(station_id=station_id, region_name=region_name, df=df)

    @property
    def start_date(self) -> pd.Timestamp:
        """Return the first available date in the dataset."""
        return self.df["date"].min()

    @property
    def end_date(self) -> pd.Timestamp:
        """Return the last available date in the dataset."""
        return self.df["date"].max()

    def subset_year(self, year: int) -> "ClimateData":
        """
        Return a new ClimateData object containing only one calendar year.
        """
        mask = self.df["date"].dt.year == year
        df_year = self.df.loc[mask].copy()
        return ClimateData(self.station_id, f"{self.region_name} {year}", df_year)

    def __len__(self) -> int:
        """Allow len(climate_data) to return number of daily records."""
        return len(self.df)

    def __str__(self) -> str:
        """Human-readable description (satisfies __str__ requirement)."""
        return (
            f"ClimateData(station_id={self.station_id}, "
            f"region={self.region_name}, "
            f"n_days={len(self)}, "
            f"{self.start_date.date()}–{self.end_date.date()})"
        )
