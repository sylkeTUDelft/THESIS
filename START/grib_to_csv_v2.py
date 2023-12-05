# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:13:29 2023

@author: sylke
"""

import xarray as xr
import pandas as pd
from collections import defaultdict
import numpy as np

#create list of paths to gribfiles
paths = [
          'C:/path/to/folder/ERA5/gribfiles/1_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/2_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/3_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/4_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/5_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/6_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/7_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/8_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/9_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/10_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/11_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/12_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/13_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/14_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/15_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/16_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/17_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/18_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/19_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/20_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/21_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/22_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/23_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/24_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/25_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/26_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/27_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/28_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/29_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/30_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/31_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/32_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/33_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/34_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/35_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/36_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/37_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/38_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/39_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/40_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/41_1_era5.grib',
          'C:/path/to/folder/ERA5/gribfiles/42_1_era5.grib',
         ]

#create target actual locations dict
targets = {
    '1_1_era5': [54.67, -3.53],
    '2_1_era5': [54.07, -3.21],
    '3_1_era5': [51.91, -4.58],
    '4_1_era5': [51.65, -3.45],
    '5_1_era5': [50.14, -5.20],
    '6_1_era5': [53.41, -4.42],
    '7_1_era5': [50.65, -4.31],
    '8_1_era5': [50.48, -4.86],
    '9_1_era5': [50.91, -4.49],
    '10_1_era5': [52.41, -3.88],
    '11_1_era5': [51.94, -4.94],
    '12_1_era5': [51.96, -5.03],
    '13_1_era5': [58.60, -3.60],
    '14_1_era5': [50.33, -5.03],
    '15_1_era5': [50.35, -5.03],
    '16_1_era5': [52.46, -3.42],
    '17_1_era5': [52.3, 4.8],
    '18_1_era5': [51.2, 3.9],
    '19_1_era5': [52.1, 6.7],
    '20_1_era5': [52.0, 4.9],
    '21_1_era5': [51.9, 4.3],
    '22_1_era5': [53.5, 5.9],
    '23_1_era5': [51.71, 3.03],
    '24_1_era5': [52.41, 4.15],
    '25_1_era5': [53.98, -3.67],
    '26_1_era5': [53.87, -3.20],
    '27_1_era5': [53.48, -3.51],
    '28_1_era5': [54.01, 5.33],
    '29_1_era5': [51.98, 2.02],
    '30_1_era5': [51.75, 1.26],
    '31_1_era5': [51.75, 1.26],
    '32_1_era5': [53.48, -3.51],
    '33_1_era5': [53.87, -3.20],
    '34_1_era5': [51.93, 3.67],
    '35_1_era5': [53.16, 3.37],
    '36_1_era5': [53.8, 2.9],
    '37_1_era5': [51.71, 3.03],
    '38_1_era5': [51.71, 3.03],
    '39_1_era5': [52.34, 3.43],
    '40_1_era5': [52.41, 4.15],
    '41_1_era5': [54.01, 5.33],
    '42_1_era5': [55.10, 2.70]
}

chosenlatlon = {}
df = []
#create for loop:
for idx, path in enumerate(paths):
    #open dataset for each station
    ds = xr.open_dataset(path, engine='cfgrib')
    print('latitidues =', ds.latitude.values)
    print('longitudes =', ds.longitude.values)
    station = list(targets.keys())[idx]
    #create subset for each station
    subset = ds.sel(latitude=targets[station][0], longitude=targets[station][1], method='nearest')
    #create dict of latitudes and longitudes for each station, where keys are the stations
    chosenlatlon[station] = [subset.latitude.values.item(), subset.longitude.values.item()]
    #make df of subset
    df.append(subset.to_dataframe())
    #check columns remove unnecesary, rename time to ob_time
    df[idx].drop(['number', 'step', 'surface'], axis=1, inplace=True)
    df[idx].reset_index(drop=True, inplace=True)
    df[idx].rename(columns={'valid_time' : 'ob_time'}, inplace=True)
    df[idx]['ob_time'] = df[idx]['ob_time'].astype('datetime64[ns]')
    #create actual wind from u and v winds
    df[idx]['ref_wind_speed'] = np.sqrt(df[idx]['u10']**2 + df[idx]['v10']**2)
    #add wind_direction from u and v winds
    df[idx]['ref_wind_direction'] = np.arctan2(df[idx]['v10'], df[idx]['u10']) * 180 / np.pi
    df[idx]['ref_wind_direction'] = (df[idx]['ref_wind_direction'] + 180) % 360
    #df to csv
    # df[idx].to_csv(f'C:/path/to/folder/ERA5/csvfiles/{station}.csv', index=False)
    
ds.close()


