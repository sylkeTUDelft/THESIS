# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:19:37 2023

@author: sylke
"""

import numpy as np
import pandas as pd
import mysql.connector as mysql
from sqlalchemy import create_engine
import datetime

# Connect to the MySQL database
DB = mysql.connect(user='user', password='password', host='host', database='database')

# Create the SQLAlchemy engine using the MySQL connection
user = 'add user'
password = 'add password'
host = 'add host'
database = 'add database'
engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{database}')

# Define the table names as a list
table_names = [
              #7x inland simple
              "3_1_era5", 
              "7_1_era5",
              "8_1_era5",
              "17_1_era5",
              "18_1_era5",
              "19_1_era5",
              "20_1_era5",
              #4x inland complex
              "4_1_era5",
              "10_1_era5",
              "16_1_era5",
              "21_1_era5",
              #11x coastal
              "1_1_era5",
              "2_1_era5",
              "5_1_era5",
              "6_1_era5",
              "13_1_era5",
              "9_1_era5",
              "11_1_era5",
              "12_1_era5",
              "14_1_era5",
              "15_1_era5",
              "22_1_era5",
              #13x offshore
              "25_1_era5",
              "29_1_era5",
              "31_1_era5",
              "32_1_era5",
              "33_1_era5",
              "34_1_era5",
              "35_1_era5",
              "36_1_era5",
              "37_1_era5",
              "39_1_era5",
              "40_1_era5",
              "41_1_era5",
              "42_1_era5",
              #8x offshore with onshore reference
              "23_1_era5",
              "24_1_era5",
              "28_1_era5",
              "25_1_era5",
              "29_1_era5",
              "30_1_era5",
              "27_1_era5",
              "26_1_era5",
              #7x inland simple
              "3_2_target",
              "7_2_target",
              "8_2_target",
              "17_2_target",
              "18_2_target",
              "19_2_target",
              "20_2_target",
              #4x inland complex
              "4_2_target",
              "10_2_target",
              "16_2_target",
              "21_2_target",
              #11x coastal
              "1_2_target",
              "2_2_target",
              "5_2_target",
              "6_2_target",
              "13_2_target",
              "9_2_target",
              "11_2_target",
              "12_2_target",
              "14_2_target",
              "15_2_target",
              "22_2_target",
              #13x offshore
              "25_2_target",
              "29_2_target",
              "31_2_target",
              "32_2_target",
              "33_2_target",
              "34_2_target",
              "35_2_target",
              "36_2_target",
              "37_2_target",
              "39_2_target",
              "40_2_target",
              "41_2_target",
              "42_2_target",
              #8x offshore with onshore reference
              "23_2_target",
              "24_2_target",
              "28_2_target",
              "25_2_target",
              "29_2_target",
              "30_2_target",
              "27_2_target",
              "26_2_target"   
]


# Define a generator function to yield DataFrames
def dataframe_generator():
    for table_name in table_names:
        # Query and retrieve data from each table using pandas and SQLAlchemy engine
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, con=engine)
        df = df.reset_index(drop=True)
        
        # Yield the DataFrame
        yield df

# Iterate over the generator and process each DataFrame one at a time
mylist = list(dataframe_generator())

# Close the MySQL connection
DB.close()

#%% set correct wind direction for era5
for i in range(43): 
    mylist[i]['ref_wind_direction'] = (180 + (( 180 / np.pi ) * (np.arctan2(mylist[i]['u10'], mylist[i]['v10'])))) % 360

#%% drop all rows without a direction, or a direction less than 0 or more than 360 degrees
for i in range(43):
    mylist[i] = mylist[i].loc[mylist[i]['ref_wind_direction'] >= 0]
    mylist[i+ 43] = mylist[i + 43].loc[mylist[i + 43]['target_wind_direction'] >= 0]
    mylist[i] = mylist[i].loc[mylist[i]['ref_wind_direction'] <= 360]
    mylist[i+ 43] = mylist[i + 43].loc[mylist[i + 43]['target_wind_direction'] <= 360]
    mylist[i] = mylist[i].dropna(subset = ['ref_wind_direction'])
    mylist[i+ 43] = mylist[i + 43].dropna(subset = ['target_wind_direction'])
    
#reset index    
for i in range(86):
    mylist[i].reset_index(drop=True, inplace=True)
    
#remove erroneous data from rhyd y goes
mylist[57].drop(mylist[57][(mylist[57]['target_wind_direction'] == 80) & (mylist[57]['target_wind_speed'] == 0.04)].index, inplace=True)
# remove 0 values from treculliacks
mylist[56].drop(mylist[56][(mylist[56]['target_wind_direction'] == 0) & (mylist[56]['target_wind_speed'] == 0)].index, inplace=True)
# shift timeseries target siddick back 24 hours
mylist[54]['ob_time'] = mylist[54]['ob_time'] - datetime.timedelta(hours=24)

# select needed columns
for i in range(43): 
    mylist[i] = mylist[i][['ob_time', 'ref_wind_speed', 'ref_wind_direction']]
    mylist[i + 43] = mylist[i + 43][['ob_time', 'target_wind_speed', 'target_wind_direction']]
#%%
# remove duplicates
for k in range(86): 
    mylist[k].drop_duplicates(subset='ob_time', keep='first', inplace= True)
    mylist[k].reset_index(drop=True, inplace=True)

# check for dubplicated values
for k in range(86):
    duplicate_mask = mylist[k]['ob_time'].duplicated(keep='first')  # Keep the first occurrence of each duplicate
    duplicate_indices = duplicate_mask[duplicate_mask].index  # Get the indices of duplicates
    duplicate_values = mylist[k].loc[duplicate_indices, 'ob_time']
    print(f'for {k} duplicates: ', duplicate_values.to_list())
#%%    
#save datasets to csv
for k in range(86): 
    mylist[k].to_csv(f'C:/folder/path/mylist_full/{k}_dataset.csv', index=False)

