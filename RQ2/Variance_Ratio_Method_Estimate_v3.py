# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:08:15 2023

@author: sylke
"""

import pandas as pd
from scipy import stats
import numpy as np

import os
import glob
import natsort


# get mylist, validate_df and concurrent_df
folder_path_mylist = 'C:/folder/path/mylist_full'
folder_path_concurrent = 'C:/folder/path/concurrent_full'
folder_path_validate = 'C:/folder/path/validate_full'

csv_files_mylist = glob.glob(os.path.join(folder_path_mylist, '*.csv'))
csv_files_sorted_mylist = natsort.natsorted(csv_files_mylist)

csv_files_concurrent = glob.glob(os.path.join(folder_path_concurrent, '*.csv'))
csv_files_sorted_concurrent = natsort.natsorted(csv_files_concurrent)

csv_files_validate = glob.glob(os.path.join(folder_path_validate, '*.csv'))
csv_files_sorted_validate = natsort.natsorted(csv_files_validate)

mylist = []
for csv_file in csv_files_sorted_mylist: 
    df = pd.read_csv(csv_file)
    mylist.append(df)
concurrent = []
for csv_file in csv_files_sorted_concurrent: 
    df = pd.read_csv(csv_file)
    concurrent.append(df)
validate = []
for csv_file in csv_files_sorted_validate: 
    df = pd.read_csv(csv_file)
    validate.append(df)

# set number of datasets
datasets = 35

#%% direction sector bins of 30 degrees
sector_labels = ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180', '180-210', '210-240', '240-270', '270-300', '300-330', '330-360']

mu_x = []
mu_y = []
sigma_x = []
sigma_y = []

for k in range(datasets): 
    mux = np.zeros(12)
    muy = np.zeros(12)
    sigmax = np.zeros(12)
    sigmay = np.zeros(12)
    for i, label in enumerate(sector_labels): 
        sectordata = concurrent[k].loc[concurrent[k]['target_bin'] == label] # variance ratio method uses target wind direction to determine sectors
        mux[i] = np.mean(sectordata['target_wind_speed'])
        muy[i] = np.mean(sectordata['ref_wind_speed'])
        sigmax[i] = np.std(sectordata['target_wind_speed'])
        sigmay[i] = np.std(sectordata['ref_wind_speed'])
    mu_x.append(mux)
    mu_y.append(muy)
    sigma_x.append(sigmax)
    sigma_y.append(sigmay)

for k in range(datasets): 
    for i, label in enumerate(sector_labels): 
        validate[k].loc[validate[k]['bin'] == label, 'longterm_target_estimate'] = (mu_x[k][i] - (sigma_x[k][i] / sigma_y[k][i]) * mu_y[k][i]) + ((sigma_x[k][i] / sigma_y[k][i]) * validate[k]['ref_wind_speed'])


VRM_estimate = validate
#%%

# import matplotlib.pyplot as plt
# from windrose import WindroseAxes
# from matplotlib import cm


# # Define the wind speed and wind direction data
# ref_wind_speed = concurrent[13]['ref_wind_speed']
# ref_wind_direction = concurrent[13]['ref_wind_direction']

# # Create a wind rose plot
# fig, ax = plt.subplots(subplot_kw={'projection': 'windrose'})


# # Plot the wind data on the wind rose
# ax.bar(ref_wind_direction, ref_wind_speed, normed=True, opening=0.8, edgecolor='white')

# # Customize the wind rose plot
# ax.set_legend(title='Wind Speed (m/s)')
# ax.set_title('Wind rose for ERA5 reference, UK9. Treculliaks')
# ax.yaxis.set_label_coords(-0.15, 0.5)
# ax.yaxis.grid(linestyle='dotted', alpha=0.7)
# ax.yaxis.set_units('Frequency')

# # Adjust the position of the legend
# ax.set_legend(title='Wind Speed (m/s)', bbox_to_anchor=(1, -0.2))

# # Show the wind rose plot
# plt.show()
    
#%%

#variable with number of datapoints for each concurrent and validation sector
datapoints_concurrent = []
datapoints_validation = []
datapoints_concurrent_sector = []
datapoints_validate_sector = []
for k in range(datasets): 
    datapoints_concurrent.append(concurrent[k]['target_wind_speed'].count())
    datapoints_validation.append(validate[k]['target_wind_speed'].count())
    sector = np.zeros(12)
    val_sector = np.zeros(12)
    for i, label in enumerate(sector_labels):
        count = concurrent[k][concurrent[k]['target_bin'] == label]['target_wind_speed'].count()
        count_validate = validate[k][validate[k]['target_bin'] == label]['target_wind_speed'].count()
        sector[i] = count
        val_sector[i] = count_validate
    datapoints_concurrent_sector.append(sector)
    datapoints_validate_sector.append(val_sector)
    
#%%
count_targetbin = []
count_refbin = []
for k in range(datasets): 
    count = np.zeros(12)
    count_ref = np.zeros(12)
    for i, label in enumerate(sector_labels): 
        count[i] = concurrent[k][concurrent[k]['target_bin'] == label]['target_wind_speed'].count()
        count_ref[i] = concurrent[k][concurrent[k]['bin'] == label]['target_wind_speed'].count()
    count_targetbin.append(count.sum())
    count_refbin.append(count_ref.sum())
    
#conclusion: number of data points is the same for refernce bin and target bin


#%%
print(np.mean(concurrent[12]['ref_wind_speed']), np.mean(concurrent[12]['target_wind_speed']))
print(np.mean(validate[12]['longterm_target_estimate']), np.mean(validate[12]['target_wind_speed']))
#%%
pearson = np.zeros(datasets)
# # #check pearson in validation period
for k in range(datasets): 
    pearson[k] = stats.linregress(validate[k]['target_wind_speed'], validate[k]['longterm_target_estimate'])[2]
#%%

for k in range(datasets): 
    validate[k].to_csv(f'C:/folder/path/VRM_full/{k}_vrm_estimate.csv', index=False)
    


