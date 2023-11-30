# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 08:05:24 2023

@author: sylke
"""

import matplotlib.pyplot as plt
from windrose import WindroseAxes
from matplotlib import cm

import pandas as pd
import os
import glob
import natsort


# folder_path_obs_conc = folder path concurrent observations
# folder_path_era5_conc = folder path concurrent era5

csv_files_obs_conc = glob.glob(os.path.join(folder_path_obs_conc, '*.csv'))
csv_files_sorted_obs_conc = natsort.natsorted(csv_files_obs_conc)

obs_conc = []
for csv_file in csv_files_sorted_obs_conc: 
    df = pd.read_csv(csv_file)
    df['ob_time'] = pd.to_datetime(df['ob_time'])
    obs_conc.append(df)
    
csv_files_era5_conc = glob.glob(os.path.join(folder_path_era5_conc, '*.csv'))
csv_files_sorted_era5_conc = natsort.natsorted(csv_files_era5_conc)

era5_conc = []
for csv_file in csv_files_sorted_era5_conc: 
    df = pd.read_csv(csv_file)
    df['ob_time'] = pd.to_datetime(df['ob_time'])
    era5_conc.append(df)

#%%
#remove calm values (zero-values)
target = obs_conc.copy()
met = obs_conc.copy()
era5 = era5_conc.copy()
datasets = 43
for k in range(datasets): 
    target_condition = (target[k]['target_wind_speed'] == 0) 
    met_condition = (met[k]['ref_wind_speed'] == 0) 
    era5_condition = (era5[k]['ref_wind_speed'] == 0) 
    # Drop rows based on the conditions
    target[k] = target[k].loc[~target_condition].reset_index(drop=True)
    met[k] = met[k].loc[~met_condition].reset_index(drop=True)
    era5[k] = era5[k].loc[~era5_condition].reset_index(drop=True)

#%% # change number dataset to plot wind roses
dataset = 11
target_name = 'UK7. Siddick'
reference_name = 'UK7. St. Bees Head No. 2'
era5_name = 'UK7.'

# Define custom legend labels and bin ranges
bin_ranges = [0, 2, 4, 8, 10, 12]  # Define your bin ranges
y_ticks = [0,300,600,900,1200,1500]
#%%
# Define the wind speed and wind direction data
target_wind_speed = target[dataset]['target_wind_speed']
target_wind_direction = target[dataset]['target_wind_direction']

# Create a wind rose plot
fig, ax = plt.subplots(subplot_kw={'projection': 'windrose'})

# Plot the wind data on the wind rose
# ax.bar(target_wind_direction, target_wind_speed, normed=True, opening=0.8, edgecolor='white')
ax.bar(target_wind_direction, target_wind_speed, bins=bin_ranges, opening=0.8, edgecolor='white', nsector = 12)
# Customize the wind rose plot
ax.set_title(f'Wind rose at target location {target_name}')
ax.yaxis.grid(linestyle='dotted', alpha=0.7)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks)
# Adjust the position of the legend
ax.set_legend(title='Wind Speed (m/s)', bbox_to_anchor=(1, -0.2))

# Show the wind rose plot
plt.show()
# Define the wind speed and wind direction data
ref_wind_speed = met[dataset]['ref_wind_speed']
ref_wind_direction = met[dataset]['ref_wind_direction']

# Create a wind rose plot
fig, ax = plt.subplots(subplot_kw={'projection': 'windrose'})

# Plot the wind data on the wind rose
ax.bar(ref_wind_direction, ref_wind_speed, bins=bin_ranges, opening=0.8, edgecolor='white', nsector = 12)

# Customize the wind rose plot
ax.set_title(f'Wind rose at MET station {reference_name}')
ax.yaxis.grid(linestyle='dotted', alpha=0.7)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks)
# Adjust the position of the legend
ax.set_legend(title='Wind Speed (m/s)', bbox_to_anchor=(1, -0.2))

# Show the wind rose plot
plt.show()

# Define the wind speed and wind direction data
ref_wind_speed = era5[dataset]['ref_wind_speed']
ref_wind_direction = era5[dataset]['ref_wind_direction']

# Create a wind rose plot
fig, ax = plt.subplots(subplot_kw={'projection': 'windrose'})

# Plot the wind data on the wind rose
ax.bar(ref_wind_direction, ref_wind_speed, bins=bin_ranges, opening=0.8, edgecolor='white', nsector = 12)

# Customize the wind rose plot
ax.set_title(f'Wind rose ERA5 reference {era5_name}')
ax.yaxis.grid(linestyle='dotted', alpha=0.7)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks)
# Adjust the position of the legend
ax.set_legend(title='Wind Speed (m/s)', bbox_to_anchor=(1, -0.2))

# Show the wind rose plot
plt.show()