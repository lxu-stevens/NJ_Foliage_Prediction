# maple/utils.py

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_file_exists(path: str | Path) -> Path:
    """
    Ensure that a file exists; raise a clear error otherwise.

    Parameters
    ----------
    path : str or Path
        Path to a file.

    Returns
    -------
    Path
        The same path, converted to a Path object.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")
    return p


def interactive_year_filter(df: pd.DataFrame) -> None:
    """
    Simple interactive year filter using a while-loop.

    Demonstrates:
    - while loop
    - if statement
    - basic user input handling
    """
    print("Interactive year filter. Type 'q' to quit.")
    while True:
        year_str = input("Enter a year (e.g., 2019), or 'q' to quit: ")
        if year_str.lower() == "q":
            print("Exiting interactive filter.")
            break

        if not year_str.isdigit():
            print("Please enter a valid year.")
            continue

        year = int(year_str)
        subset = df[df["year"] == year]
        if subset.empty:
            print(f"No data for year {year}.")
        else:
            print(f"Found {len(subset)} rows for year {year}. Showing head:")
            print(subset.head())
            