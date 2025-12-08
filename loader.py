"""
Loading the data of Newark International Airport weather station using built-in Meteostat library
"""
from datetime import datetime
from meteostat import Stations, Hourly
import pandas as pd

class NewarkWeatherLoader:
   
    def __init__(self,start=datetime(2014,1,1),end=datetime(2024,12,31)):
      if start > end:
         raise RuntimeError("Start date must be earlier than end date")
          
      self.start=start
      self.end=end
      self.kewr_id=self._get_kewr_station_id()
      self.data=None
   
    def _get_kewr_station_id(self):
       NJ_stations=Stations().region('US',"NJ").fetch() 
       kewr_id=NJ_stations.loc[NJ_stations['icao'] == 'KEWR'].index[0]
       return kewr_id

    def fetch_data(self):
        df=Hourly(self.kewr_id, self.start, self.end).fetch()
        self.data=df
        return df
    
    def save_to_csv(self,filename="newark_hourly_weather.csv"):
        if self.data is not None:
           self.data.to_csv(filename)
           print(f"data saved successfully to {filename}")
        else:
           print("No data available.run fetch() data first")

loader=NewarkWeatherLoader()
df=loader.fetch_data()
loader.save_to_csv()

print(df.head(10))
print("Min timestamp:", df.index.min())
print("Max timestamp:", df.index.max())