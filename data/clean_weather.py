import pandas as pd
from pathlib import Path
class NewarkWeatherCleaner:
   
    def __init__(
        self,
        input_file = "newark_hourly_weather.csv",
        output_file = "clean_weather.csv",
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.df = None
    
    def _load(self):
        df = pd.read_csv(
            self.input_file,
            parse_dates=["time"],
            index_col="time",
        )
        df = df.sort_index()
        df = df.replace("<NA>",pd.NA)
        return df

    def clean(self):
        df = self._load()
        cols = ["temp", "dwpt", "rhum", "wdir", "wspd", "pres","prcp"]
        df = df[cols].astype(float)
        df['prcp']=df['prcp'].fillna(0)
        interp_cols = ["temp", "dwpt", "rhum", "wdir", "wspd", "pres"]
        df[interp_cols] = df[interp_cols].interpolate(method="time")
        self.df = df
        print(f"Missing value ratio:{df.isna().mean()}")
        return df

    def save_to_csv(self):
        if self.df is None:
           self.clean()
        self.df.to_csv(self.output_file)
        print('cleaned data saved successfully to clean_weather.csv')

if __name__ == "__main__":
    cleaner = NewarkweatherCleaner(
        input_file="newark_hourly_weather.csv",
        output_file="clean_weather.csv",
    )
    clean_df = cleaner.clean()
    cleaner.save_to_csv()