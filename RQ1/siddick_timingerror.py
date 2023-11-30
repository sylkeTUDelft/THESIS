# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:05:08 2023

@author: sylke
"""


import pandas as pd
import os
import glob
import natsort
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
from sklearn.metrics import r2_score

# folder_path_mcp = folder path olr results
csv_files_mcp = glob.glob(os.path.join(folder_path_mcp, '*.csv'))
csv_files_sorted_mcp = natsort.natsorted(csv_files_mcp)

OLR = []
for csv_file in csv_files_sorted_mcp: 
    df = pd.read_csv(csv_file)
    df['ob_time'] = pd.to_datetime(df['ob_time'])
    OLR.append(df)

#%%
siddick = OLR[11]
siddick.drop(siddick.tail(1).index, inplace=True)
#%%
actual = siddick[['ob_time', 'target_wind_speed', 'target_wind_direction']]
reference = siddick[['ob_time', 'ref_wind_speed', 'ref_wind_direction']]
prediction = siddick[['ob_time', 'longterm_target_estimate', 'ref_wind_speed']]

#%%
# actual['ob_time'] = actual['ob_time'] - datetime.timedelta(hours=24)
#%%
#plot timeseries
font = 20
plt.figure(figsize=(24,10))
sns.lineplot(data=actual, x='ob_time', y='target_wind_speed', color='tab:blue', label='Actual wind speed')
# plt.axhline(y = np.mean(obs[28]['target_wind_speed']), color = 'tab:blue', linestyle = '-')
sns.lineplot(data=prediction, x='ob_time', y='longterm_target_estimate', color='tab:orange', label='Prediction')
# plt.axhline(y = np.mean(obs[28]['longterm_target_estimate']), color = 'tab:orange', linestyle = '-')
plt.xlabel('date [yyyy-mm-dd]', fontsize=font)
plt.ylabel('Wind Speed [m/s]', fontsize=font)
plt.title('Actual wind speed v.s predicted wind  with MET-station timeseries, actual values shifted by 24 h', fontsize=font)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=font)
plt.show()
#plot timseries actual vs reference
font = 20
plt.figure(figsize=(24,10))
sns.lineplot(data=actual, x='ob_time', y='target_wind_speed', color='tab:blue', label='Actual wind speed')
# plt.axhline(y = np.mean(obs[28]['target_wind_speed']), color = 'tab:blue', linestyle = '-')
sns.lineplot(data=reference, x='ob_time', y='ref_wind_speed', color='tab:orange', label='Reference')
# plt.axhline(y = np.mean(obs[28]['longterm_target_estimate']), color = 'tab:orange', linestyle = '-')
plt.xlabel('date [yyyy-mm-dd]', fontsize=font)
plt.ylabel('Wind Speed [m/s]', fontsize=font)
plt.title('Actual wind speed v.s reference wind  with MET-station timeseries', fontsize=font)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=font)
plt.show()

#%%
#shifting target 48 hours backwards
target = []
for k in range(48):
    onetime = actual.copy()
    onetime['ob_time'] = onetime['ob_time'] - datetime.timedelta(hours=k)
    target.append(onetime)

period = []
for k in range(48): 
    period.append(pd.merge(target[k], prediction, how='inner', on='ob_time'))
    
r2 = np.zeros(48)
for k in range(48): 
    r2[k] = r2_score(period[k]['target_wind_speed'], period[k]['longterm_target_estimate'])

print(r2)