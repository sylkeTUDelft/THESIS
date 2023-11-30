# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 09:35:53 2023

@author: sylke
"""

import pandas as pd
import os
import glob
import natsort
import numpy as np
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy import stats
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

folder_path_obs = 'C:/Users/sylke/OneDrive/Documenten/THESIS/DATA/1_CSVS/new_validate_concurrent/OLR_OBS_RQ1'
folder_path_era5 = 'C:/Users/sylke/OneDrive/Documenten/THESIS/DATA/1_CSVS/new_validate_concurrent/OLR_ERA5_RQ1'

csv_files_obs = glob.glob(os.path.join(folder_path_obs, '*.csv'))
csv_files_sorted_obs = natsort.natsorted(csv_files_obs)

obs = []
for csv_file in csv_files_sorted_obs: 
    df = pd.read_csv(csv_file)
    df['ob_time'] = pd.to_datetime(df['ob_time'])
    obs.append(df)
    
csv_files_era5 = glob.glob(os.path.join(folder_path_era5, '*.csv'))
csv_files_sorted_era5 = natsort.natsorted(csv_files_era5)

era5 = []
for csv_file in csv_files_sorted_era5: 
    df = pd.read_csv(csv_file)
    df['ob_time'] = pd.to_datetime(df['ob_time'])
    era5.append(df)

folder_path_obs_conc = 'C:/Users/sylke/OneDrive/Documenten/THESIS/DATA/1_CSVS/new_validate_concurrent/concurrent_OBS_RQ1'
folder_path_era5_conc = 'C:/Users/sylke/OneDrive/Documenten/THESIS/DATA/1_CSVS/new_validate_concurrent/concurrent_ERA5_RQ1'

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
datasets = 35
pearson_conc_obs = np.zeros(datasets)
pearson_conc_era5 = np.zeros(datasets)
r2_obs = np.zeros(datasets)
r2_era5 = np.zeros(datasets)
rmse_obs = np.zeros(datasets)
rmse_era5 = np.zeros(datasets)
mean_obs = np.zeros(datasets)
mean_era5 = np.zeros(datasets)
mae_obs = np.zeros(datasets)
mae_era5 = np.zeros(datasets)
mbe_obs = np.zeros(datasets)
mbe_era5 = np.zeros(datasets)
for k in range(datasets): 
    pearson_conc_obs[k] = (stats.linregress(obs_conc[k]['target_wind_speed'], obs_conc[k]['ref_wind_speed'])[2])
    pearson_conc_era5[k] = (stats.linregress(era5_conc[k]['target_wind_speed'], era5_conc[k]['ref_wind_speed'])[2])
    r2_obs[k] = (r2_score(obs[k]['target_wind_speed'], obs[k]['longterm_target_estimate']))
    r2_era5[k] = r2_score(era5[k]['target_wind_speed'], era5[k]['longterm_target_estimate'])
    # rmse
    rmse_obs[k] = (math.sqrt(mean_squared_error(obs[k]['target_wind_speed'], obs[k]['longterm_target_estimate'])) / obs[k]['target_wind_speed'].mean(axis=0, skipna = True))
    rmse_era5[k] = (math.sqrt(mean_squared_error(era5[k]['target_wind_speed'], era5[k]['longterm_target_estimate'])) / era5[k]['target_wind_speed'].mean(axis=0, skipna = True))
    # normalized mean
    mean_obs[k] = (obs[k]['longterm_target_estimate'].mean(axis=0, skipna = True) / obs[k]['target_wind_speed'].mean(axis=0, skipna = True))
    mean_era5[k] = (era5[k]['longterm_target_estimate'].mean(axis=0, skipna = True) / era5[k]['target_wind_speed'].mean(axis=0, skipna = True))
    #mae
    mae_obs[k] = (mean_absolute_error(obs[k]['target_wind_speed'], obs[k]['longterm_target_estimate']) / obs[k]['target_wind_speed'].mean(axis=0, skipna = True))
    mae_era5[k] = (mean_absolute_error(era5[k]['target_wind_speed'], era5[k]['longterm_target_estimate']) / era5[k]['target_wind_speed'].mean(axis=0, skipna = True))
    # mean bias error normalized
    mbe_obs[k] = ((np.mean(obs[k]['longterm_target_estimate']) - np.mean(obs[k]['target_wind_speed'])) / np.mean(obs[k]['target_wind_speed']))
    mbe_era5[k] = ((np.mean(era5[k]['longterm_target_estimate']) - np.mean(era5[k]['target_wind_speed'])) / np.mean(era5[k]['target_wind_speed']))
 
pearson_conc_obs_rounded = np.round(pearson_conc_obs, 2)
#%%

df_obs = pd.DataFrame({'pearson': pearson_conc_obs, 'r2': r2_obs, 'rmse': rmse_obs, 'mean': mean_obs, 'mae': mae_obs, 'mbe': mbe_obs})
df_obs.loc[df_obs['r2'] <= 0.6, 'class'] = 'very poor'
df_obs.loc[(df_obs['r2'] > 0.6) & (df_obs['r2'] <= 0.7), 'class'] = 'poor'
df_obs.loc[(df_obs['r2'] > 0.7) & (df_obs['r2'] <= 0.8), 'class'] = 'moderate'
df_obs.loc[(df_obs['r2'] > 0.8) * (df_obs['r2'] <= 0.9), 'class'] = 'good'
df_obs.loc[(df_obs['r2'] > 0.9), 'class'] = 'very good'

data_obs = round(df_obs, 2)

df_era5 = pd.DataFrame({'pearson': pearson_conc_era5, 'r2': r2_era5, 'rmse': rmse_era5, 'mean': mean_era5, 'mae': mae_era5, 'mbe': mbe_era5})
df_era5.loc[df_era5['r2'] <= 0.6, 'class'] = 'very poor'
df_era5.loc[(df_era5['r2'] > 0.6) & (df_era5['r2'] <= 0.7), 'class'] = 'poor'
df_era5.loc[(df_era5['r2'] > 0.7) & (df_era5['r2'] <= 0.8), 'class'] = 'moderate'
df_era5.loc[(df_era5['r2'] > 0.8) * (df_era5['r2'] <= 0.9), 'class'] = 'good'
df_era5.loc[(df_era5['r2'] > 0.9), 'class'] = 'very good'

data_era5 = round(df_era5, 2)
#%%

custom_colors = {
    'very poor': '#8B0000',   # Dark Red
    'poor': 'red',
    'moderate': 'orange',
    'good': 'green',
    'very good': '#90EE90'   # Light Green
}
hue_order = [ 'very poor', 'poor', 'moderate', 'good', 'very good']

plt.figure(figsize=(7,5))
sns.scatterplot(data = df_obs, x='pearson', y='r2', hue='class', palette = custom_colors, hue_order = hue_order)
plt.legend(title='Classification', loc='lower right')
plt.xlim(0.6,1)
plt.ylim(-0.2, 1)
plt.xlabel('Pearson correlation concurrent period [-]')
plt.ylabel('R^2 [-]')
plt.title('Pearson correlation and resulting R^2 with MET station reference')
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(data = df_era5, x='pearson', y='r2', hue='class', palette = custom_colors, hue_order = hue_order)
legend_order = ['poor', 'moderate', 'good', 'very good']
plt.legend(title='Classification', loc='lower right')
plt.xlim(0.6,1)
plt.ylim(-0.2, 1)
plt.xlabel('Pearson correlation concurrent period [-]')
plt.ylabel('R^2 [-]')
plt.title('Pearson correlation and resulting R^2 with ERA5 reference')
plt.show()

#%%
# Create a single figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatterplot for the first subplot
sns.scatterplot(data=df_obs, x='pearson', y='r2', hue='class', palette=custom_colors, hue_order=hue_order, ax=axes[0])
axes[0].set_xlim(0.6, 1)
axes[0].set_ylim(-0.2, 1)
axes[0].set_xlabel('Pearson correlation concurrent period [-]')
axes[0].set_ylabel('R^2 [-]')
axes[0].set_title('MET station reference')
axes[0].legend(title='Classification', loc='lower right')

# Scatterplot for the second subplot
sns.scatterplot(data=df_era5, x='pearson', y='r2', hue='class', palette=custom_colors, hue_order=hue_order, ax=axes[1])
axes[1].set_xlim(0.6, 1)
axes[1].set_ylim(-0.2, 1)
axes[1].set_xlabel('Pearson correlation concurrent period [-]')
axes[1].set_ylabel('R^2 [-]')
axes[1].set_title('ERA5 reference')
axes[1].legend(title='Classification', loc='lower right')

# Adjust the layout of subplots
plt.tight_layout()

# Show the combined figure with subplots
plt.show()

#


#%%
keys = ["UK1. Dyffryn Brodyn", "UK2. Lifton Down", "UK3. St. Breock", "NL1. Schiphol", "NL2. Westdorpe", "NL3. Hupsel", "NL4. Cabauw",
        "UK4. Penrhys", "UK5. Rheidol", "UK6. Alt-Yr-Hende", "NL5. Rotterdam Geulhaven", "UK7. Siddick", "UK8. Haverigg", "UK9. Treculliacks",
        "UK10. Rhyd-Y-Goes", "UK11. Hill of Forss", "UK12. Crimp", "UK13. Ysgubor", "UK14. Jordanston", "UK15. Truthan", "UK16. Carland Cross",
        "NL6. Platform AWG-1", "UK17. Celtic Array Zone 9", "UK18. Greater Gabbard", "UK19. Gunfleet Sands", "UK20. Gwynt Y Mor", "UK21. Shell Flats",
        "NL7. Lichteiland Goeree", "NL8. K14FA1C", "NL9. J6-A", "NL10. Borssele 1", "NL11. Hollandse Kust West (HKWA)", "NL12. Hollandse Kust Noord (HKNB)",
        "NL13. Ten Noorden van de Wadden (TNWB)", "NL14. Dogger Bank zone 3", "on/off: NL10. Borssele 1", 'on/off: NL12. Hollandse Kust Noord (HKNB)', 
        'on/off: NL13. Ten Noorden van de Wadden (TNWB)', 'on/off: UK17. Celtic Array Zone 9', 'on/off: UK18. Greater Gabbard', 'on/off: UK19. Gunfleet Sands',
        'on/off: UK20. Gwynt Y Mor', 'on/off: UK21. Shell Flats']
keys = ["UK1. Dyffryn Brodyn", "UK2. Lifton Down", "UK3. St. Breock", "NL1. Schiphol", "NL2. Westdorpe", "NL3. Hupsel", "NL4. Cabauw",
        "UK4. Penrhys", "UK5. Rheidol", "UK6. Alt-Yr-Hende", "NL5. Rotterdam Geulhaven", "UK7. Siddick", "UK8. Haverigg", "UK9. Treculliacks",
        "UK10. Rhyd-Y-Goes", "UK11. Hill of Forss", "UK12. Crimp", "UK13. Ysgubor", "UK14. Jordanston", "UK15. Truthan", "UK16. Carland Cross",
        "NL6. Platform AWG-1", "UK17. Celtic Array Zone 9", "UK18. Greater Gabbard", "UK19. Gunfleet Sands", "UK20. Gwynt Y Mor", "UK21. Shell Flats",
        "NL7. Lichteiland Goeree", "NL8. K14FA1C", "NL9. J6-A", "NL10. Borssele 1", "NL11. Hollandse Kust West (HKWA)", "NL12. Hollandse Kust Noord (HKNB)",
        "NL13. Ten Noorden van de Wadden (TNWB)", "NL14. Dogger Bank zone 3"]

# df_obs['target'] = keys
# df_era5['target'] = keys
df_obs.insert(0, 'target', keys)
df_era5.insert(0, 'target', keys)
#%%
# df_obs[['pearson', 'r2', 'rmse', 'mean', 'mae', 'mbe']] = round(df_obs[['pearson', 'r2', 'rmse', 'mean', 'mae', 'mbe']], 3)
# df_era5[['pearson', 'r2', 'rmse', 'mean', 'mae', 'mbe']] = round(df_era5[['pearson', 'r2', 'rmse', 'mean', 'mae', 'mbe']], 3)

best_obs = 0
for k in range(len(df_obs)): 
    if df_obs['r2'].iloc[k] > df_era5['r2'].iloc[k] : 
        best_obs = best_obs + 1
        print(df_obs['target'][k])
print(len(df_obs), best_obs)

best_era5 = 0
for k in range(len(df_era5)): 
    if df_obs['r2'].iloc[k] < df_era5['r2'].iloc[k] : 
        best_era5 = best_era5 + 1
        print(df_era5['target'][k])
print(len(df_era5), best_era5)

best_same = 0
for k in range(len(df_era5)): 
    if df_obs['r2'].iloc[k] == df_era5['r2'].iloc[k] : 
        best_era5 = best_same + 1
        print(df_era5['target'][k])
print(len(df_era5), best_same)
#%%


high_pearson_obs = df_obs.loc[df_obs['pearson'] >= 0.9]
high_pearson_era5 = df_era5.loc[df_era5['pearson'] >= 0.9]
high_pearson_obs_era5 = df_era5.iloc[[6, 10, 27, 28, 30], :]

high_pearson_obs['datatype'] = 'MET'
high_pearson_obs_era5['datatype'] = 'ERA5'

high = pd.concat([high_pearson_obs.copy(), high_pearson_obs_era5.copy()])

poor_pearson_obs = df_obs.loc[df_obs['pearson'] <= 0.8]
poor_pearson_era5 = df_era5.loc[ df_era5['pearson'] <= 0.8]
poor_pearson_obs_era5 = df_era5.iloc[[1, 7, 8, 9, 11, 22, 25, 26], :]
poor_pearson_obs['datatype'] = 'MET'
poor_pearson_obs_era5['datatype'] = 'ERA5'

poor = pd.concat([poor_pearson_obs.copy(), poor_pearson_obs_era5.copy()])


good_pearson_obs = df_obs.loc[(df_obs['pearson'] > 0.8) & (df_obs['pearson'] < 0.9)]
good_pearson_era5 = df_era5.loc[(df_era5['pearson'] > 0.8) & (df_era5['pearson'] < 0.9)]
good_pearson_obs_era5 = df_era5.iloc[[0, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 29, 31, 32, 33, 34], : ]

good_pearson_obs['datatype'] = 'MET'
good_pearson_obs_era5['datatype'] = 'ERA5'

good = pd.concat([good_pearson_obs.copy(), good_pearson_obs_era5.copy()])
#%%
plt.figure(figsize=(8,6))
sns.barplot(data=high, x= 'target', y='r2', hue='datatype')
plt.xticks(rotation=90)
plt.legend(loc='lower right')

high_obs_best = 0
for k in range(int(len(high) / 2)): 
    if high_pearson_obs['r2'].iloc[k] > high_pearson_obs_era5['r2'].iloc[k] : 
        high_obs_best = high_obs_best + 1
print('total high obs: ', len(high) / 2, ' best obs: ', high_obs_best, 100/(len(high)/2) * high_obs_best)

plt.figure(figsize=(8,6))
sns.barplot(data=poor, x= 'target', y='r2', hue='datatype')
plt.xticks(rotation=90)
plt.legend(loc='lower right')

poor_obs_best = 0
for k in range(int(len(poor) / 2)): 
    if poor_pearson_obs['r2'].iloc[k] > poor_pearson_obs_era5['r2'].iloc[k] : 
        poor_obs_best = poor_obs_best + 1
print('total poor obs: ', len(poor) / 2, ' best obs: ', poor_obs_best, 100 / (len(poor)/2) * poor_obs_best)

plt.figure(figsize=(8,6))
sns.barplot(data=good, x= 'target', y='r2', hue='datatype')
plt.xticks(rotation=90)
plt.legend(loc='lower right')

good_obs_best = 0
for k in range(int(len(good) / 2)): 
    if good_pearson_obs['r2'].iloc[k] > good_pearson_obs_era5['r2'].iloc[k] : 
        good_obs_best = good_obs_best + 1
print('total good obs: ', len(good) / 2, ' best obs: ', good_obs_best, 100 / (len(good)/2) * good_obs_best)
#%%
#plot a graph scattering the r2 values against the pearson correlation with trend line
# plt.figure(figsize=(8,6))
# sns.regplot(data=df_obs, x='pearson', y='r2')
# sns.regplot(data=df_era5, x='pearson', y='r2')

barwidth = 0.6
font = 20
fontaxes = 16
# Create a single figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 12))

# Plot for "poor" data
sns.barplot(data=poor, x='target', y='r2', hue='datatype', ax=axes[0], width=barwidth)
axes[0].set_title('Pearson r < 0.8', fontsize=font)
axes[0].set_ylim(-0.1, 1)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, fontsize=fontaxes)
axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=fontaxes)
axes[0].legend(loc='lower right', fontsize=font)
axes[0].set_xlabel('Target site', fontsize=font)
axes[0].set_ylabel('R^2', fontsize=font)


# Plot for "good" data
sns.barplot(data=good, x='target', y='r2', hue='datatype', ax=axes[1], width=barwidth)
axes[1].set_title('0.8 < Pearson r < 0.9', fontsize=font)
axes[1].set_ylim(-0.1, 1)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90, fontsize=fontaxes)
axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=fontaxes)
axes[1].legend(loc='lower right', fontsize=font)
axes[1].set_xlabel('Target site', fontsize=font) 
axes[1].set_ylabel('R^2', fontsize=font)

# Plot for "high" data
sns.barplot(data=high, x='target', y='r2', hue='datatype', ax=axes[2], width=barwidth)
axes[2].set_title('Pearson r > 0.9', fontsize=font)
axes[2].set_ylim(-0.1, 1)
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=90, fontsize=fontaxes)
axes[2].set_yticklabels(axes[2].get_yticklabels(), fontsize=fontaxes)
axes[2].legend(loc='lower right', fontsize=font)
axes[2].set_xlabel('Target site', fontsize=font)
axes[2].set_ylabel('R^2', fontsize=font)

# Adjust spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()

#%%
# figure for sorted by era5 pearson correlation


high_pearson_obs = df_obs.loc[df_obs['pearson'] >= 0.9]
high_pearson_era5 = df_era5.loc[df_era5['pearson'] >= 0.9]
high_pearson_era5_obs = df_obs.iloc[[0, 13, 14, 18, 27, 28, 29, 30, 31, 32, 33, 34], :]

high_pearson_era5['datatype'] = 'ERA5'
high_pearson_era5_obs['datatype'] = 'MET'

high = pd.concat([high_pearson_era5_obs.copy(),high_pearson_era5.copy()])

poor_pearson_obs = df_obs.loc[df_obs['pearson'] <= 0.8]
poor_pearson_era5 = df_era5.loc[ df_era5['pearson'] <= 0.8]
poor_pearson_era5_obs = df_obs.iloc[[7, 11, 26], :]
poor_pearson_era5['datatype'] = 'ERA5'
poor_pearson_era5_obs['datatype'] = 'MET'

poor = pd.concat([poor_pearson_era5_obs.copy(), poor_pearson_era5.copy()])


good_pearson_obs = df_obs.loc[(df_obs['pearson'] > 0.8) & (df_obs['pearson'] < 0.9)]
good_pearson_era5 = df_era5.loc[(df_era5['pearson'] > 0.8) & (df_era5['pearson'] < 0.9)]
good_pearson_era5_obs = df_obs.iloc[[1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25], : ]

good_pearson_era5['datatype'] = 'ERA5'
good_pearson_era5_obs['datatype'] = 'MET'

good = pd.concat([good_pearson_era5_obs.copy(), good_pearson_era5.copy()])
#%%
plt.figure(figsize=(8,6))
sns.barplot(data=high, x= 'target', y='r2', hue='datatype')
plt.xticks(rotation=90)
plt.legend(loc='lower right')

high_era5_best = 0
for k in range(int(len(high) / 2)): 
    if high_pearson_era5['r2'].iloc[k] > high_pearson_era5_obs['r2'].iloc[k] : 
        high_era5_best = high_era5_best + 1
print('total high era5: ', len(high) / 2, ' best era5: ', high_era5_best, 100/(len(high)/2) * high_era5_best)

plt.figure(figsize=(8,6))
sns.barplot(data=poor, x= 'target', y='r2', hue='datatype')
plt.xticks(rotation=90)
plt.legend(loc='lower right')

poor_era5_best = 0
for k in range(int(len(poor) / 2)): 
    if poor_pearson_era5['r2'].iloc[k] > poor_pearson_era5_obs['r2'].iloc[k] : 
        poor_era5_best = poor_era5_best + 1
print('total poor era5: ', len(poor) / 2, ' best era5: ', poor_era5_best, 100 /( len(poor)/2) * poor_era5_best)

plt.figure(figsize=(8,6))
sns.barplot(data=good, x= 'target', y='r2', hue='datatype')
plt.xticks(rotation=90)
plt.legend(loc='lower right')

good_era5_best = 0
for k in range(int(len(good) / 2)): 
    if good_pearson_era5['r2'].iloc[k] > good_pearson_era5_obs['r2'].iloc[k] : 
        good_era5_best = good_era5_best + 1
print('total good era5: ', len(good) / 2, ' best era5: ', good_era5_best, 100 / (len(good)/2) * good_era5_best)
#%%
#plot a graph scattering the r2 values against the pearson correlation with trend line
# plt.figure(figsize=(8,6))
# sns.regplot(data=df_obs, x='pearson', y='r2')
# sns.regplot(data=df_era5, x='pearson', y='r2')

barwidth = 0.6
font = 20
fontaxes = 16
# Create a single figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 12))

# Plot for "poor" data
sns.barplot(data=poor, x='target', y='r2', hue='datatype', ax=axes[0], width=barwidth)
axes[0].set_title('Pearson r < 0.8', fontsize=font)
axes[0].set_ylim(-0.1, 1)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, fontsize=fontaxes)
axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=fontaxes)
axes[0].legend(loc='lower right', fontsize=font)
axes[0].set_xlabel('Target site', fontsize=font)
axes[0].set_ylabel('R^2', fontsize=font)


# Plot for "good" data
sns.barplot(data=good, x='target', y='r2', hue='datatype', ax=axes[1], width=barwidth)
axes[1].set_title('0.8 < Pearson r < 0.9', fontsize=font)
axes[1].set_ylim(-0.1, 1)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90, fontsize=fontaxes)
axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=fontaxes)
axes[1].legend(loc='lower right', fontsize=font)
axes[1].set_xlabel('Target site', fontsize=font) 
axes[1].set_ylabel('R^2', fontsize=font)

# Plot for "high" data
sns.barplot(data=high, x='target', y='r2', hue='datatype', ax=axes[2], width=barwidth)
axes[2].set_title('Pearson r > 0.9', fontsize=font)
axes[2].set_ylim(-0.1, 1)
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=90, fontsize=fontaxes)
axes[2].set_yticklabels(axes[2].get_yticklabels(), fontsize=fontaxes)
axes[2].legend(loc='lower right', fontsize=font)
axes[2].set_xlabel('Target site', fontsize=font)
axes[2].set_ylabel('R^2', fontsize=font)

# Adjust spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()

#%%
siddick_obs = obs_conc[11]
siddick_era5 = era5_conc[11]
full_siddick_length = len(siddick_obs)
correct_bin_obs = len(siddick_obs.loc[siddick_obs['bin'] == siddick_obs['target_bin']])
correct_bin_era5 = len(siddick_era5.loc[siddick_era5['bin'] == siddick_era5['target_bin']])
percentage_obs = 100 / full_siddick_length * correct_bin_obs
percentage_era5 = 100 / full_siddick_length * correct_bin_era5
print(percentage_obs, percentage_era5)
print(correct_bin_obs, correct_bin_era5)

mean_target = np.mean(siddick_obs['target_wind_speed'])
mean_met = np.mean(siddick_obs['ref_wind_speed'])
mean_era5 = np.mean(siddick_era5['ref_wind_speed'])
mean_val = np.mean(obs[11]['target_wind_speed'])
mean_pred_met = np.mean(obs[11]['longterm_target_estimate'])
mean_pred_era5 = np.mean(era5[11]['longterm_target_estimate'])   
print(mean_target, mean_met, mean_era5)
print(mean_val, mean_pred_met, mean_pred_era5)

#%%
# tryout rheidol without zero values
                    
# rheidol_obs = obs_conc[8].loc[(obs_conc[8]['ref_wind_speed'] != 0) & (obs_conc[8]['ref_wind_direction'] != 0) ]
# rheidol_era5 = era5_conc[8].loc[era5_conc[8]['ob_time'].isin(rheidol_obs['ob_time'])]
# full_rheidol_length = len(rheidol_obs)
# correct_bin_obs = len(rheidol_obs.loc[rheidol_obs['bin'] == rheidol_obs['target_bin']])
# correct_bin_era5 = len(rheidol_era5.loc[rheidol_era5['bin'] == rheidol_era5['target_bin']])
# percentage_obs = 100 / full_rheidol_length * correct_bin_obs
# percentage_era5 = 100 / full_rheidol_length * correct_bin_era5
# print(percentage_obs, percentage_era5)
# print(correct_bin_obs, correct_bin_era5)

# mean_target = np.mean(rheidol_obs['target_wind_speed'])
# mean_met = np.mean(rheidol_obs['ref_wind_speed'])
# mean_era5 = np.mean(rheidol_era5['ref_wind_speed'])
# mean_val = np.mean(obs[8]['target_wind_speed'])
# mean_pred_met = np.mean(obs[8]['longterm_target_estimate'])
# mean_pred_era5 = np.mean(era5[8]['longterm_target_estimate'])   
# print(mean_target, mean_met, mean_era5)
# print(mean_val, mean_pred_met, mean_pred_era5)

# plt.figure(figsize=(8,6))
# sns.scatterplot(data=obs_conc[8], x='ref_wind_speed', y='target_wind_speed')

#%%
percentage_obs = []
percentage_era5 = []
for k in range(35): 
    full_length = len(obs_conc[k])
    correct_bin_obs = len(obs_conc[k].loc[obs_conc[k]['bin'] == obs_conc[k]['target_bin']])
    correct_bin_era5 = len(era5_conc[k].loc[era5_conc[k]['bin'] == era5_conc[k]['target_bin']])
    percentage_obs.append(100 / full_length * correct_bin_obs)
    percentage_era5.append(100 / full_length * correct_bin_era5)
avg_percentage_obs = np.median(percentage_obs)
avg_percentage_era5 = np.median(percentage_era5)
print(avg_percentage_obs, avg_percentage_era5)
#%%
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from matplotlib import cm

# Define custom legend labels and bin ranges
legend_labels = ['0-3 m/s', '3-6 m/s', '6-9 m/s', '9-12 m/s','12-15 m/s', '> 15 m/s']  # Add more labels if needed
bin_ranges = [0, 3, 6, 9, 12, 15]  # Define your bin ranges
y_ticks = [0,200,400,600,800,1000]

# Define the wind speed and wind direction data
target_wind_speed = obs_conc[12]['target_wind_speed']
target_wind_direction = obs_conc[12]['target_wind_direction']

# Create a wind rose plot
fig, ax = plt.subplots(subplot_kw={'projection': 'windrose'})

# Plot the wind data on the wind rose
# ax.bar(target_wind_direction, target_wind_speed, normed=True, opening=0.8, edgecolor='white')
ax.bar(target_wind_direction, target_wind_speed, bins=bin_ranges, opening=0.8, edgecolor='white', nsector = 12)
# Customize the wind rose plot
ax.set_title('Wind rose at target location UK8. Haverigg')
ax.yaxis.grid(linestyle='dotted', alpha=0.7)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks)
# Adjust the position of the legend
ax.set_legend(title='Wind Speed (m/s)', bbox_to_anchor=(1, -0.2))

# Show the wind rose plot
plt.show()
#%%
# Define the wind speed and wind direction data
ref_wind_speed = obs_conc[12]['ref_wind_speed']
ref_wind_direction = obs_conc[12]['ref_wind_direction']

# Create a wind rose plot
fig, ax = plt.subplots(subplot_kw={'projection': 'windrose'})

# Plot the wind data on the wind rose
ax.bar(ref_wind_direction, ref_wind_speed, bins=bin_ranges, opening=0.8, edgecolor='white', nsector = 12, calm_limit = 0.01)

# Customize the wind rose plot
ax.set_title('Wind rose at MET station UK8. Walney Island')
ax.yaxis.grid(linestyle='dotted', alpha=0.7)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks)
# Adjust the position of the legend
ax.set_legend(title='Wind Speed (m/s)', bbox_to_anchor=(1, -0.2))

# Show the wind rose plot
plt.show()

# Define the wind speed and wind direction data
ref_wind_speed = era5_conc[12]['ref_wind_speed']
ref_wind_direction = era5_conc[12]['ref_wind_direction']

# Create a wind rose plot
fig, ax = plt.subplots(subplot_kw={'projection': 'windrose'})

# Plot the wind data on the wind rose
ax.bar(ref_wind_direction, ref_wind_speed, bins=bin_ranges, opening=0.8, edgecolor='white', nsector = 12, calm_limit = 0.01)

# Customize the wind rose plot
ax.set_title('Wind rose ERA5 reference UK8.')
ax.yaxis.grid(linestyle='dotted', alpha=0.7)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks)
# Adjust the position of the legend
ax.set_legend(title='Wind Speed (m/s)', bbox_to_anchor=(1, -0.2))

# Show the wind rose plot
plt.show()

#%%
# scatter haverigg
plt.figure(figsize=(8,6))
sns.scatterplot(data = obs[9], x='target_wind_speed', y='longterm_target_estimate', color='tab:blue')
plt.plot([0, 20], [0, 20], color='tab:blue')
plt.xlim(0,20)
plt.ylim(0,20)
plt.xlabel('Target wind speed [m/s]')
plt.ylabel('Prediction [m/s]')
plt.title('Actual wind speed vs Prediction with MET-station')
plt.show()
plt.figure(figsize=(8,6))
sns.regplot(data=era5[9], x='target_wind_speed', y='longterm_target_estimate', color='tab:orange')
plt.plot([0, 20], [0, 20], color='tab:orange')
plt.xlim(0,20)
plt.ylim(0,20)
plt.xlabel('Target wind speed [m/s]')
plt.ylabel('Prediction [m/s]')
plt.title('Actual wind speed vs Prediction with ERA5')
plt.show()
#%%
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot for MET-station data
sns.scatterplot(data=obs[11], x='target_wind_speed', y='longterm_target_estimate', ax=axes[0], color='tab:blue')
axes[0].plot([0, 25], [0, 25], color='tab:blue')
axes[0].set_xlim(0, 25)
axes[0].set_ylim(0, 25)
axes[0].set_xlabel('Target wind speed [m/s]')
axes[0].set_ylabel('Prediction [m/s]')
axes[0].set_title('Actual wind speed vs Prediction with MET-station')

# Plot for ERA5 data
sns.scatterplot(data=era5[11], x='target_wind_speed', y='longterm_target_estimate', color='tab:orange', ax=axes[1])
axes[1].plot([0, 25], [0, 25], color='tab:orange')
axes[1].set_xlim(0, 25)
axes[1].set_ylim(0, 25)
axes[1].set_xlabel('Target wind speed [m/s]')
axes[1].set_ylabel('Prediction [m/s]')
axes[1].set_title('Actual wind speed vs Prediction with ERA5')

# Adjust spacing between subplots
plt.tight_layout()

# Show the figure with both subplots
plt.show()

#%%
#plot timeseries
font = 20
plt.figure(figsize=(24,10))
sns.lineplot(data=obs[11], x='ob_time', y='target_wind_speed', color='tab:blue', label='Actual wind speed')
plt.axhline(y = np.mean(obs[11]['target_wind_speed']), color = 'tab:blue', linestyle = '-')
sns.lineplot(data=obs[11], x='ob_time', y='longterm_target_estimate', color='tab:orange', label='Prediction')
plt.axhline(y = np.mean(obs[11]['longterm_target_estimate']), color = 'tab:orange', linestyle = '-')
plt.xlabel('date [yyyy-mm-dd]', fontsize=font)
plt.ylabel('Wind Speed [m/s]', fontsize=font)
plt.title('Actual wind speed v.s predicted wind  with MET-station timeseries', fontsize=font)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=font)
plt.show()
#%%
plt.figure(figsize=(24,10))
sns.lineplot(data=era5[11], x='ob_time', y='target_wind_speed', color='tab:blue', label='Actual wind speed')
plt.axhline(y = np.mean(era5[11]['target_wind_speed']), color = 'tab:blue', linestyle = '-')
sns.lineplot(data=era5[11], x='ob_time', y='longterm_target_estimate', color='tab:orange', label='Prediction')
plt.axhline(y = np.mean(era5[11]['longterm_target_estimate']), color = 'tab:orange', linestyle = '-')
plt.xlabel('date [yyyy-mm-dd]', fontsize=font)
plt.ylabel('Wind Speed [m/s]', fontsize=font)
plt.title('Actual wind speed v.s predicted wind speed with ERA5 timeseries', fontsize=font)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=font)
plt.show()


plt.figure(figsize=(24,10))
sns.lineplot(data=era5_conc[11], x='ob_time', y='target_wind_speed', color='tab:blue', label='Actual wind speed')
plt.axhline(y = np.mean(era5_conc[11]['target_wind_speed']), color = 'tab:blue', linestyle = '-')
sns.lineplot(data=era5_conc[11], x='ob_time', y='ref_wind_speed', color='tab:orange', label='Prediction')
plt.axhline(y = np.mean(era5_conc[11]['ref_wind_speed']), color = 'tab:orange', linestyle = '-')
plt.xlabel('date [yyyy-mm-dd]', fontsize=font)
plt.ylabel('Wind Speed [m/s]', fontsize=font)
plt.title('Concurrent period actual vs ERA5 values', fontsize=font)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=font)
plt.show()
#%%
zeros_ref_obs = []
zeros_target_obs = []
for k in range(43): 
    zeros_ref_obs.append(len(obs_conc[k].loc[(obs_conc[k]['ref_wind_speed'] == 0) & (obs_conc[k]['ref_wind_direction'] == 0)]))
    zeros_target_obs.append(len(obs_conc[k].loc[(obs_conc[k]['target_wind_speed'] == 0) & (obs_conc[k]['target_wind_direction'] == 0)]))
zeros = []
for k in range(43): 
    zeros.append([zeros_ref_obs[k], zeros_target_obs[k]])

keys = ["UK1. Dyffryn Brodyn", "UK2. Lifton Down", "UK3. St. Breock", "NL1. Schiphol", "NL2. Westdorpe", "NL3. Hupsel", "NL4. Cabauw",
        "UK4. Penrhys", "UK5. Rheidol", "UK6. Alt-Yr-Hende", "NL5. Rotterdam Geulhaven", "UK7. Siddick", "UK8. Haverigg", "UK9. Treculliacks",
        "UK10. Rhyd-Y-Goes", "UK11. Hill of Forss", "UK12. Crimp", "UK13. Ysgubor", "UK14. Jordanston", "UK15. Truthan", "UK16. Carland Cross",
        "NL6. Platform AWG-1", "UK17. Celtic Array Zone 9", "UK18. Greater Gabbard", "UK19. Gunfleet Sands", "UK20. Gwynt Y Mor", "UK21. Shell Flats",
        "NL7. Lichteiland Goeree", "NL8. K14FA1C", "NL9. J6-A", "NL10. Borssele 1", "NL11. Hollandse Kust West (HKWA)", "NL12. Hollandse Kust Noord (HKNB)",
        "NL13. Ten Noorden van de Wadden (TNWB)", "NL14. Dogger Bank zone 3", "on/off: NL10. Borssele 1", 'on/off: NL12. Hollandse Kust Noord (HKNB)', 
        'on/off: NL13. Ten Noorden van de Wadden (TNWB)', 'on/off: UK17. Celtic Array Zone 9', 'on/off: UK18. Greater Gabbard', 'on/off: UK19. Gunfleet Sands',
        'on/off: UK20. Gwynt Y Mor', 'on/off: UK21. Shell Flats']

mydf = pd.DataFrame({'target site': keys, 'zeros target': zeros_target_obs, 'zeros ref': zeros_ref_obs})

#%%
check = era5_conc[12]
check2 = era5[12]
conc_15 = check['target_wind_speed'][check['target_wind_speed'] <=1].count()
val_15 = check2['target_wind_speed'][check2['target_wind_speed'] <=1].count()
print(conc_15, len(check), 100/len(check) * conc_15)
print(val_15, len(check2), 100/len(check2) * val_15)

#%%
df_onoff_obs = round(df_obs[35:], 2)
df_onoff_ERA5 = round(df_era5[35:], 2)