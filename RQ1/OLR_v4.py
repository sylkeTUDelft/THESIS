# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 12:13:34 2023

@author: sylke
"""


import pandas as pd
from scipy import stats
import numpy as np
import os
import glob
import natsort
from sklearn.metrics import r2_score


# get mylist, validate_df and concurrent_df

# folder_path_mylist = add folder path mylist
# folder_path_concurrent = add folder path concurrent
# folder_path_validate = add folder path validate

csv_files_mylist = glob.glob(os.path.join(folder_path_mylist, '*.csv'))
csv_files_sorted_mylist = natsort.natsorted(csv_files_mylist)
# csv_files_mylist2 = glob.glob(os.path.join(folder_path_mylist2, '*.csv'))
# csv_files_sorted_mylist2 = natsort.natsorted(csv_files_mylist2)

csv_files_concurrent = glob.glob(os.path.join(folder_path_concurrent, '*.csv'))
csv_files_sorted_concurrent = natsort.natsorted(csv_files_concurrent)

csv_files_validate = glob.glob(os.path.join(folder_path_validate, '*.csv'))
csv_files_sorted_validate = natsort.natsorted(csv_files_validate)

mylist = []
for csv_file in csv_files_sorted_mylist: 
    df = pd.read_csv(csv_file)
    mylist.append(df)
# mylist2 = []
# for csv_file in csv_files_sorted_mylist2: 
#     df = pd.read_csv(csv_file)
#     mylist2.append(df)
concurrent_df = []
for csv_file in csv_files_sorted_concurrent: 
    df = pd.read_csv(csv_file)
    concurrent_df.append(df)
validate_df = []
for csv_file in csv_files_sorted_validate: 
    df = pd.read_csv(csv_file)
    validate_df.append(df)

# #%%
# test = pd.merge(mylist[38], mylist2[38], on='ob_time', how='outer')
# test2 = pd.merge(mylist[38], mylist2[38], on='ob_time', how='inner')
#%%
# concurrent_long = []
# long_dataset_indices = []
# for k in range(35): 
#     if len(concurrent_df[k]) == 8760:
#         concurrent_long.append(concurrent_df[k].copy())
#         long_dataset_indices.append(k)
#%%

# set number of datasets
# concurrent = []
# validate = []
# for k in range(len(concurrent_long)): 
#     concurrent_dataset = []
#     validate_dataset = []
#     for i in range(12):
#         concurrent_dataset.append(concurrent_long[k][0: (i + 1) * 730].copy())
#         validate_dataset.append(validate_df[long_dataset_indices[k]].copy())
#     concurrent.extend(concurrent_dataset)
#     validate.extend(validate_dataset)

concurrent = concurrent_df.copy()
validate = validate_df.copy()
datasets = len(concurrent)    
#%% direction sector bins of 30 degrees
sector_labels = ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180', '180-210', '210-240', '240-270', '270-300', '300-330', '330-360']

#%%
import warnings 

slope = []
offset = []

for k in range(datasets): 
    m = np.zeros(12)
    c = np.zeros(12)
    for i, label in enumerate(sector_labels): 
        with warnings.catch_warnings():
            warnings.filterwarnings("error")  # Turn warnings into errors
            try:
                data = concurrent[k].loc[concurrent[k]['bin'] == label]
                # if len(data) < 2:
                #     m[i] = 1  # Set to 1 when data has length less than 1
                #     c[i] = 0  # Set to 0 when data has length less than 1
                # else:
                m[i] = stats.linregress(data['ref_wind_speed'], data['target_wind_speed'])[0]
                c[i] = stats.linregress(data['ref_wind_speed'], data['target_wind_speed'])[1]
            except Warning as w:
                print(f"Warning occurred in dataset {k}, label {label}: {w}")
            except Exception as e:
                print(f"Error occurred in dataset {k}, label {label}: {e}")
            finally:
                warnings.resetwarnings()  # Reset warnings to their original state
    slope.append(m)
    offset.append(c)

#%%
reconstructed = []
for k in range(datasets): 
    slope_set = slope[k]
    offset_set = offset[k]
    validate_copy = validate[k].copy()
    for i, label in enumerate(sector_labels):
        # validate_copy = validate_set.copy()
        validate_copy.loc[validate_copy['bin'] == label, 'longterm_target_estimate'] = slope_set[i] * validate_copy['ref_wind_speed'] + offset_set[i]
    reconstructed.append(validate_copy)
    if k == 0:
        print(slope[k], offset[k])
        print(reconstructed[k].head(5))


#%%
for k in range(datasets): 
    reconstructed[k].to_csv(f'C:/path/to/folder/OLR_full/{k}_olr_estimate.csv', index=False)
print('saved to csv')    

#%%        
pearson = np.zeros(datasets)
r2score = np.zeros(datasets)
# # #check pearson in validation period
for k in range(datasets): 
    pearson[k] = stats.linregress(reconstructed[k]['target_wind_speed'], reconstructed[k]['longterm_target_estimate'])[2]
    r2score[k] = r2_score(reconstructed[k]['target_wind_speed'], reconstructed[k]['longterm_target_estimate'])

print(r2score)
#%%
test = validate_df[15].loc[validate[15]['bin'] == validate[15]['target_bin']]
test2 = concurrent_df[15].loc[concurrent[15]['bin'] == concurrent[15]['target_bin']]
print(len(validate_df[15]), len(test), 100/len(validate_df[15]) * len(test))
print(len(concurrent_df[15]), len(test2), 100/len(concurrent_df[15]) * len(test2))

test = validate_df[0].loc[validate[0]['bin'] == validate[0]['target_bin']]
test2 = concurrent_df[0].loc[concurrent[0]['bin'] == concurrent[0]['target_bin']]
print(len(validate_df[0]), len(test), 100/len(validate_df[0]) * len(test))
print(len(concurrent_df[0]), len(test2), 100/len(concurrent_df[0]) * len(test2))
#%%
# validate_OLR = validate.copy()
# OLR_estimate = validate.copy()
# # calculate sector counts OLR and set to percentage of total population
# OLR_sector_counts = []
# total_population_OLR = []
# OLR_sector_counts_percent = []
# for k in range(43): 
#     total_population_OLR.append(validate_OLR[k]['bin'].count())
#     sector_count = np.zeros(12)
#     for i in range(12): 
#         sector_count[i] = np.count_nonzero(validate_OLR[k]['bin'] == labels[i])
#     OLR_sector_counts.append(sector_count)
#     OLR_sector_counts_percent.append(np.round(OLR_sector_counts[k] / total_population_OLR[k] * 100, 2))
    
# sector_mean_OLR = []
# mean_OLR = []
# normalized_mean_OLR = []
# for k in range(43): 
#     mean = np.zeros(12)
#     for i, label in enumerate(labels):
#         mean[i] = np.mean(validate_OLR[k]['longterm_target_estimate'].loc[validate_OLR[k]['bin'] == label])
#     sector_mean_OLR.append(mean)
#     # mean_OLR.append(np.round(np.mean(validate_OLR[k]['longterm_target_estimate']),2))
#     mean_OLR.append(np.sum(OLR_sector_counts_percent[k] / 100 * sector_mean_OLR[k]))
#     # normalized_mean_OLR.append(mean_OLR[k] / mean_actual[k])

# #%%
# import seaborn as sns
# import matplotlib.pyplot as plt
# dataset = concurrent[0].loc[concurrent[0]['bin'] == '0-30']
# plt.figure()
# sns.regplot(data = dataset , x = 'ref_wind_speed', y = 'target_wind_speed')
# #%% create scatter plot for outliers
# dataset = concurrent[7]
# plt.figure()
# sns.regplot(data = dataset , x='ref_wind_speed', y = 'target_wind_speed')
# plt.title('UK4. Penrhys concurrent dataset scatter')
# plt.xlim(0, 20)
# plt.ylim(0,20)
# prediction_dataset = validate[7]
# # plt.figure()
# # sns.regplot(data = dataset, x='ref_wind_direction', y = 'target_wind_direction')
# # plt.title('UK10. Rhyd Y Goes wind direction concurrent period scatter')
# plt.figure()
# sns.regplot(data=prediction_dataset, x = 'target_wind_speed', y = 'longterm_target_estimate')
# plt.title('UK4. Penrhys target vs prediction dataset scatter')
# plt.xlim(0, 20)
# plt.ylim(0,20)
# #%%
# #variable with number of datapoints for each concurrent and validation sector
# datapoints_concurrent = []
# datapoints_validation = []
# datapoints_concurrent_sector = []
# datapoints_validate_sector = []
# for k in range(35): 
#     datapoints_concurrent.append(concurrent[k]['target_wind_speed'].count())
#     datapoints_validation.append(validate[k]['target_wind_speed'].count())
#     sector = np.zeros(12)
#     val_sector = np.zeros(12)
#     for i, label in enumerate(labels):
#         count = concurrent[k][concurrent[k]['bin'] == label]['target_wind_speed'].count()
#         count_validate = validate[k][validate[k]['bin'] == label]['target_wind_speed'].count()
#         sector[i] = count
#         val_sector[i] = count_validate
#     datapoints_concurrent_sector.append(sector)
#     datapoints_validate_sector.append(val_sector)
        
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
# from sklearn.metrics import r2_score
# import math
# from sklearn.metrics import mean_squared_error

# r2_olr_original = np.zeros(35)
# rmse_olr_original = np.zeros(35) #normalized
# for k in range(35): 
#     r2_olr_original[k] = r2_score(validate[k]['target_wind_speed'], validate[k]['longterm_target_estimate'])
#     rmse_olr_original[k] = math.sqrt(mean_squared_error(validate[k]['target_wind_speed'], validate[k]['longterm_target_estimate'])) / np.mean(validate[k]['target_wind_speed'])


