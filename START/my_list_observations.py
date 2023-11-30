# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:50:15 2023

@author: sylke
"""

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
# table_names = [
#           #simple
#     "1_1_st_beeshead", 
#               "2_1_walneyisland",
#               "3_1_pendine",
#               "4_1_rhoose",
#               "5_1_culdrose",
#               "6_1_valley",
#               "7_1_cardinham_bodmin",
#               "8_1_cardinham_bodmin",
#               "9_1_chivenor",
#               "10_1_trawsgoed",
#               "11_1_aberporth",
#               "12_1_aberporth",
#               "13_1_wick_airport",
#               "14_1_st_mawgan",
#               "15_1_st_mawgan",
#               "16_1_shawbury",
#               "17_1_ijmuiden",
#               "18_1_vlissingen",
#               "19_1_twenthe",
#               "20_1_debilt",
#               "21_1_rotterdam",
#               "22_1_hoorn_terschelling",
#               "23_1_vlissingen",
#               "24_1_ijmuiden",
#               "25_1_walney",
#               "26_1_walney",
#               "27_1_crosby",
#               "28_1_hoorn_terschelling",
#               "29_1_walton",
#               "30_1_walton",
#               "31_1_greatergabbard",
#               "32_1_celticzone9",
#               "33_1_celticzone9",
#               "34_1_europlatform",
#               "35_1_k13",
#               "36_1_d15fa1",
#               "37_1_europlatform",
#               "38_1_legoeree",
#               "39_1_europlatform",
#               "40_1_legoeree",
#               "41_1_buitengaats",
#               "42_1_d15fa1",
#               "1_2_siddick",
#               "2_2_haverigg",
#               "3_2_dyfbrod",
#               "4_2_penrhys",
#               "5_2_treculliacks",
#               "6_2_rhydygb",
#               "7_2_liftonb",
#               "8_2_stbreock",
#               "9_2_crimpb",
#               "10_2_rheidol",
#               "11_2_ysgubora",
#               "12_2_jordansa",
#               "13_2_forss2",
#               "14_2_truthana",
#               "15_2_carcross",
#               "16_2_altyhdb",
#               "17_2_schiphol",
#               "18_2_westdorpe",
#               "19_2_hupsel",
#               "20_2_cabauw",
#               "21_2_rotterdam_geulhaven",
#               "22_2_platform_awg1",
#               "23_2_borssele",
#               "24_2_hknb",
#               "25_2_celticzone9",
#               "26_2_shellflats",
#               "27_2_gymmast1",
#               "28_2_tnwb",
#               "29_2_greatergabbard",
#               "30_2_gunfleetsands",
#               "31_2_gunfleetsands",
#               "32_2_gymmast1",
#               "33_2_shellflats",
#               "34_2_legoeree",
#               "35_2_k14fa1c",
#               "36_2_j6a",
#               "37_2_borssele",
#               "38_2_borssele",
#               "39_2_hkwa",
#               "40_2_hknb",
#               "41_2_tnwb",
#               "42_2_doggerbank"
# ]
table_names = [
              #simple
              "3_1_ref",
              "7_1_ref",
              "8_1_ref",
              "17_1_ref",
              "18_1_ref",
              "19_1_ref",
              "20_1_ref",
              #complex
              "4_1_ref",
              "10_1_ref",
              "16_1_ref",
              "21_1_ref",
              #coastal
              "1_1_ref", 
              "2_1_ref",              
              "5_1_ref",
              "6_1_ref",
              "13_1_ref",
              "9_1_ref",              
              "11_1_ref",
              "12_1_ref",              
              "14_1_ref",
              "15_1_ref",
              "22_1_ref",
              #offshore
              "43_1_ref",
              "44_1_ref",
              "31_1_ref",
              "32_1_ref",
              "33_1_ref",
              "34_1_ref",
              "35_1_ref",
              "36_1_ref",
              "37_1_ref",
              "39_1_ref",
              "40_1_ref",
              "41_1_ref",
              "42_1_ref",
              #on-off
              "23_1_ref",
              "24_1_ref",
              "28_1_ref",
              "25_1_ref",
              "29_1_ref",
              "30_1_ref",
              "27_1_ref",
              "26_1_ref",
              #simple
              "3_2_target",
              "7_2_target",
              "8_2_target",
              "17_2_target",
              "18_2_target",
              "19_2_target",
              "20_2_target",
              #complex
              "4_2_target",
              "10_2_target",
              "16_2_target",
              "21_2_target",
              #coastal
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
              #offshore
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
              #on-off
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
for k in range(86): 
    mylist[k].to_csv(f'C:/folder/path/mylist_full_OBS/{k}_dataset.csv', index=False)


