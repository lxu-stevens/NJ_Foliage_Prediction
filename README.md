# Maple Foliage Peak Week Prediction with LSTM

## Project Title
Predicting Peak Autumn Foliage Weeks Using Weather Data and a PyTorch LSTM Model

## Student names and emails
Heyuan Zhuang hzhuang2@stevens.edu

Lei Xu lxu44@stevens.edu

## Problem Description

In many regions, autumn foliage (especially maple trees) attracts a large number of visitors.
However, the timing of the "peak foliage week" depends on the weather pattern in the weeks
leading up to autumn: temperature, cold days, and precipitation.

This project aims to **predict which weeks are peak foliage weeks** for a given location,
based only on daily weather data (temperature and precipitation).

We:

1. Collect daily weather data for one station (temperature and precipitation).
2. Aggregate daily data into weekly features.
3. Use simple rule-based criteria to define whether a given week is a "peak foliage" week.
4. Train a **PyTorch LSTM model** to learn the relationship between weather patterns over several weeks and the peak-week label.
5. Predict the probability that each week is a peak foliage week.

## Data Source

Daily weather data was downloaded from [Meteostat](https://meteostat.net/) (or your actual provider),
for the city: **MyCity**.

Columns used:

- `date`  : daily date
- `tavg`  : average daily temperature (°C)
- `tmin`  : minimum daily temperature (°C)
- `tmax`  : maximum daily temperature (°C)
- `prcp`  : daily precipitation (mm)

We store the data in `data/weather_export.csv`.

## Program Structure

The code is organized as follows:

- `main.ipynb`  
  Jupyter Notebook that runs the full pipeline: load data, aggregate weekly features,
  label peak weeks, train the LSTM model, and visualize the results.

- `maple_foliage/climate_data.py`  
  Defines the `ClimateData` class, which loads and validates daily weather data from CSV,
  and provides basic utilities such as subset by year. This module includes exception
  handling for invalid or missing files.

- `maple_foliage/features.py`  
  Defines the `compute_weekly_features` function and the `WeeklyClimateDataset` class.
  It aggregates daily data into weekly features, such as weekly average temperature,
  diurnal temperature range, total precipitation, number of rainy days, and cold days.
  `WeeklyClimateDataset` uses **composition** because it contains a `ClimateData`
  instance and the corresponding weekly DataFrame. It also overloads `__add__`
  to merge two weekly datasets.

- `maple_foliage/labeling.py`  
  Implements a **rule-based labeling** function `assign_peak_labels_rule_based`
  and a `LabelingConfig` dataclass. A week is considered a peak week if:
  - The weekly average temperature is between a lower and upper bound.
  - The number of cold days (Tmin < threshold) is large enough.
  - Total weekly precipitation is below a threshold.

- `maple_foliage/model_lstm.py`  
  Implements the `FoliageLSTMNet` PyTorch model (an LSTM network) and the
  `FoliageLSTMModel` wrapper class. This class:
  - Builds input sequences from weekly features (past `seq_len` weeks → current week label).
  - Trains the LSTM using PyTorch.
  - Evaluates the model using accuracy, F1-score, and ROC-AUC (via scikit-learn).
  - Predicts the peak probability for each week and adds a `peak_prob_lstm` column.

- `maple_foliage/utils.py`  
  Contains utility functions:
  - `ensure_file_exists` to check file existence (exception handling).
  - `interactive_year_filter` demonstrates a simple `while` loop and `if` statements
    for filtering weekly data by year based on user input.

- `tests/`  
  Contains two pytest tests:
  - `test_features.py` tests weekly feature generation.
  - `test_model_lstm.py` tests that the LSTM model can train on synthetic data and
    produce valid peak probabilities.
