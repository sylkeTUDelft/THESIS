# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:07:20 2023

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

# folder_path_obs = add folder path olr observations
# folder_path_era5 = add folder path olr era5

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

#%%
datasets = len(obs)

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
pearson_ref = np.zeros(datasets)
pearson_obs = np.zeros(datasets)
pearson_era5 = np.zeros(datasets)
for k in range(datasets): 
    # determine R^2
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
 
    # pearson correlation of reconstructed period
    pearson_obs[k] = (stats.linregress(obs[k]['target_wind_speed'], obs[k]['longterm_target_estimate'])[2])
    pearson_era5[k] = (stats.linregress(era5[k]['target_wind_speed'], era5[k]['longterm_target_estimate'])[2])

#%%
data_obs = {'r2score': r2_obs, 'rmse': rmse_obs, 'mean': mean_obs, 'mbe': mbe_obs, 'mae': mae_obs, 'pearson': pearson_obs}
df_obs = pd.DataFrame(data_obs)
df_obs['datatype'] = 'Observations'
data_era5 = {'r2score': r2_era5, 'rmse': rmse_era5, 'mean': mean_era5, 'mbe': mbe_era5, 'mae': mae_era5, 'pearson': pearson_era5}
df_era5 = pd.DataFrame(data_era5)
df_era5['datatype'] = 'ERA5'
df = pd.concat([df_obs, df_era5], ignore_index=True)

#%%
# for each terrain type
df_simple = pd.concat([df[0:7].copy(), df[43:50].copy()], ignore_index=True)
df_complex = pd.concat([df[7:11].copy(), df[50:54].copy()], ignore_index=True)
df_coastal = pd.concat([df[11:22].copy(), df[54:65].copy()], ignore_index=True)
df_offshore = pd.concat([df[22:35].copy(), df[65:78].copy()], ignore_index=True)
df_onoff = pd.concat([df[35:43].copy(), df[78:86].copy()], ignore_index=True)

df_simple['terrain'] = 'simple'
df_complex['terrain'] = 'complex'
df_coastal['terrain'] = 'coastal'
df_offshore['terrain'] = 'offshore'
df_onoff['terrain'] = 'onshore ref, offshore target'

df_terrain = pd.concat([df_simple, df_complex, df_coastal, df_offshore], ignore_index=True)


#%%
print("R2 SCORE")
best_era5 = 0 
best_obs = 0
N = 35
for k in range(N):
    if r2_era5[k] > r2_obs[k] : 
        best_era5 = best_era5 + 1
    if r2_era5[k] < r2_obs[k] : 
        best_obs = best_obs + 1
print('total: best r2 era5: ', best_era5, ', best r2 obs: ', best_obs,', num datasets: ', best_era5 + best_obs)

# determine best r^2 score for the 35 targets per terrain type
n_simple = int(len(df_simple) / 2)
n_complex = int(len(df_complex) / 2)
n_coastal = int(len(df_coastal) / 2)
n_offshore = int(len(df_offshore) / 2)
n_onoff = int(len(df_onoff) / 2)

simple_best_era5 = 0
simple_best_obs = 0
for k in range(n_simple):
    if df_simple['r2score'].iloc[k] > df_simple['r2score'].iloc[k + n_simple] :
        simple_best_obs = simple_best_obs + 1 
    if df_simple['r2score'].iloc[k] < df_simple['r2score'].iloc[k + n_simple] :
        simple_best_era5 = simple_best_era5 + 1
print('simple: best era5: ', simple_best_era5, ', best obs: ', simple_best_obs, ', num datasets: ', simple_best_era5 + simple_best_obs)

complex_best_era5 = 0
complex_best_obs = 0
for k in range(n_complex):
    if df_complex['r2score'].iloc[k] > df_complex['r2score'].iloc[k + n_complex] :
        complex_best_obs = complex_best_obs + 1 
    if df_complex['r2score'].iloc[k] < df_complex['r2score'].iloc[k + n_complex] :
        complex_best_era5 = complex_best_era5 + 1
print('complex: best era5: ', complex_best_era5, ', best obs: ', complex_best_obs, ', num datasets: ', complex_best_era5 + complex_best_obs)

coastal_best_era5 = 0
coastal_best_obs = 0
for k in range(n_coastal):
    if df_coastal['r2score'].iloc[k] > df_coastal['r2score'].iloc[k + n_coastal] :
        coastal_best_obs = coastal_best_obs + 1 
    if df_coastal['r2score'].iloc[k] < df_coastal['r2score'].iloc[k + n_coastal] :
        coastal_best_era5 = coastal_best_era5 + 1
        # print('index best coastal obs: ', k)
print('coastal: best era5: ', coastal_best_era5, ', best obs: ', coastal_best_obs, ', num datasets: ', coastal_best_era5 + coastal_best_obs)

offshore_best_era5 = 0
offshore_best_obs = 0
for k in range(n_offshore):
    if df_offshore['r2score'].iloc[k] > df_offshore['r2score'].iloc[k + n_offshore] :
        offshore_best_obs = offshore_best_obs + 1 
    if df_offshore['r2score'].iloc[k] < df_offshore['r2score'].iloc[k + n_offshore] :
        offshore_best_era5 = offshore_best_era5 + 1
print('offshore: best era5: ', offshore_best_era5, ', best obs: ', offshore_best_obs, ', num datasets: ', offshore_best_era5 + offshore_best_obs)

onoff_best_era5 = 0
onoff_best_obs = 0
for k in range(n_onoff):
    if df_onoff['r2score'].iloc[k] > df_onoff['r2score'].iloc[k + n_onoff] :
        onoff_best_obs = onoff_best_obs + 1 
    if df_onoff['r2score'].iloc[k] < df_onoff['r2score'].iloc[k + n_onoff] :
        onoff_best_era5 = onoff_best_era5 + 1
print('onoff: best era5: ', onoff_best_era5, ', best obs: ', onoff_best_obs, ', num datasets: ', onoff_best_era5 + onoff_best_obs)


# Create a figure with two subplots
fig = plt.figure(figsize=(16, 6))

gs = gridspec.GridSpec(1, 2, width_ratios=[3, 7])  # Divide the figure into 2 parts, with 30% for the first subplot

# Subplot 1 - Boxplot for 'method' data
ax1 = plt.subplot(gs[0])  # 1st subplot
sns.boxplot(data=df, x='datatype', y='r2score', ax=ax1)
ax1.set_xlabel('Data type', fontsize=12)
ax1.set_ylabel('R^2 [-]', fontsize=12)
ax1.set_ylim(0, 1)
ax1.set_title('R^2 with different reference data', fontsize=14)

# Subplot 2 - Boxplot for 'terrain' data
ax2 = plt.subplot(gs[1])  # 2nd subplot
sns.boxplot(data=df_terrain, x='terrain', y='r2score', hue='datatype', dodge=True, ax=ax2)
ax2.set_xlabel('Terrain type', fontsize=12)
ax2.set_ylabel('R^2 [-]', fontsize=12)
ax2.set_ylim(0, 1)
ax2.set_title('R^2 obtained with observed or ERA5 reference data, for different terrain types', fontsize=14)

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()
#%%
#onoff
plt.figure(figsize=(8,6))
ax = sns.boxplot(data = df_onoff, x='datatype', y='r2score')
ax.set_xlabel('Reference datatype', fontsize=12)
ax.set_ylabel('R^2', fontsize=12)
ax.set_ylim(0, 1)
plt.title('R^2 obtained with different reference datatype for offshore targets with onshore reference', fontsize=14)
plt.show()

#%%
#RMSE
print('RMSE')
best_era5 = 0 
best_obs = 0
N = 35
for k in range(N):
    if rmse_era5[k] < rmse_obs[k] : 
        best_era5 = best_era5 + 1
    if rmse_era5[k] > rmse_obs[k] : 
        best_obs = best_obs + 1
print('total: best rmse era5: ', best_era5, ', best rmse obs: ', best_obs,', num datasets: ', best_era5 + best_obs)


simple_best_era5 = 0
simple_best_obs = 0
for k in range(n_simple):
    if df_simple['rmse'].iloc[k] < df_simple['rmse'].iloc[k + n_simple] :
        simple_best_obs = simple_best_obs + 1 
    if df_simple['rmse'].iloc[k] > df_simple['rmse'].iloc[k + n_simple] :
        simple_best_era5 = simple_best_era5 + 1
print('simple: best era5: ', simple_best_era5, ', best obs: ', simple_best_obs, ', num datasets: ', simple_best_era5 + simple_best_obs)

complex_best_era5 = 0
complex_best_obs = 0
for k in range(n_complex):
    if df_complex['rmse'].iloc[k] < df_complex['rmse'].iloc[k + n_complex] :
        complex_best_obs = complex_best_obs + 1 
    if df_complex['rmse'].iloc[k] > df_complex['rmse'].iloc[k + n_complex] :
        complex_best_era5 = complex_best_era5 + 1
print('complex: best era5: ', complex_best_era5, ', best obs: ', complex_best_obs, ', num datasets: ', complex_best_era5 + complex_best_obs)

coastal_best_era5 = 0
coastal_best_obs = 0
for k in range(n_coastal):
    if df_coastal['rmse'].iloc[k] < df_coastal['rmse'].iloc[k + n_coastal] :
        coastal_best_obs = coastal_best_obs + 1 
    if df_coastal['rmse'].iloc[k] > df_coastal['rmse'].iloc[k + n_coastal] :
        coastal_best_era5 = coastal_best_era5 + 1
        # print('index best coastal obs: ', k)
print('coastal: best era5: ', coastal_best_era5, ', best obs: ', coastal_best_obs, ', num datasets: ', coastal_best_era5 + coastal_best_obs)

offshore_best_era5 = 0
offshore_best_obs = 0
for k in range(n_offshore):
    if df_offshore['rmse'].iloc[k] < df_offshore['rmse'].iloc[k + n_offshore] :
        offshore_best_obs = offshore_best_obs + 1 
    if df_offshore['rmse'].iloc[k] > df_offshore['rmse'].iloc[k + n_offshore] :
        offshore_best_era5 = offshore_best_era5 + 1
print('offshore: best era5: ', offshore_best_era5, ', best obs: ', offshore_best_obs, ', num datasets: ', offshore_best_era5 + offshore_best_obs)

onoff_best_era5 = 0
onoff_best_obs = 0
for k in range(n_onoff):
    if df_onoff['rmse'].iloc[k] < df_onoff['rmse'].iloc[k + n_onoff] :
        onoff_best_obs = onoff_best_obs + 1 
    if df_onoff['rmse'].iloc[k] > df_onoff['rmse'].iloc[k + n_onoff] :
        onoff_best_era5 = onoff_best_era5 + 1
print('onoff: best era5: ', onoff_best_era5, ', best obs: ', onoff_best_obs, ', num datasets: ', onoff_best_era5 + onoff_best_obs)

# Create a figure with two subplots
fig = plt.figure(figsize=(16, 6))

gs = gridspec.GridSpec(1, 2, width_ratios=[3, 7])  # Divide the figure into 2 parts, with 30% for the first subplot

# Subplot 1 - Boxplot for 'method' data
ax1 = plt.subplot(gs[0])  # 1st subplot
sns.boxplot(data=df, x='datatype', y='rmse', ax=ax1)
ax1.set_xlabel('Data type', fontsize=12)
ax1.set_ylabel('NRMSE [-]', fontsize=12)
ax1.set_ylim(0, 0.6)
ax1.set_title('Normalized RMSE with different reference data', fontsize=14)

# Subplot 2 - Boxplot for 'terrain' data
ax2 = plt.subplot(gs[1])  # 2nd subplot
sns.boxplot(data=df_terrain, x='terrain', y='rmse', hue='datatype', dodge=True, ax=ax2)
ax2.set_xlabel('Terrain type', fontsize=12)
ax2.set_ylabel('NRMSE [-]', fontsize=12)
ax2.set_ylim(0, 0.6)
ax2.set_title('Normalized RMSE obtained with observed or ERA5 reference data, for different terrain types', fontsize=14)

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

#%%
#Normalized Mean
print('Normalzied MEAN')
best_era5 = 0 
best_obs = 0
N = 35
for k in range(N):
    if abs(1 - mean_era5[k]) < abs(1 - mean_obs[k]) : 
        best_era5 = best_era5 + 1
    if abs(1 - mean_era5[k]) > abs(1 - mean_obs[k]) : 
        best_obs = best_obs + 1
print('total: best mean era5: ', best_era5, ', best mean obs: ', best_obs,', num datasets: ', best_era5 + best_obs)


simple_best_era5 = 0
simple_best_obs = 0
for k in range(n_simple):
    if abs(1 - df_simple['mean'].iloc[k]) < abs(1 - df_simple['mean'].iloc[k + n_simple]) :
        simple_best_obs = simple_best_obs + 1 
    if abs(1 - df_simple['mean'].iloc[k]) > abs(1 - df_simple['mean'].iloc[k + n_simple]) :
        simple_best_era5 = simple_best_era5 + 1
print('simple: best era5: ', simple_best_era5, ', best obs: ', simple_best_obs, ', num datasets: ', simple_best_era5 + simple_best_obs)

complex_best_era5 = 0
complex_best_obs = 0
for k in range(n_complex):
    if abs(1 - df_complex['mean'].iloc[k]) < abs(1 - df_complex['mean'].iloc[k + n_complex]) :
        complex_best_obs = complex_best_obs + 1 
    if abs(1 - df_complex['mean'].iloc[k]) > abs(1 - df_complex['mean'].iloc[k + n_complex]) :
        complex_best_era5 = complex_best_era5 + 1
print('complex: best era5: ', complex_best_era5, ', best obs: ', complex_best_obs, ', num datasets: ', complex_best_era5 + complex_best_obs)

coastal_best_era5 = 0
coastal_best_obs = 0
for k in range(n_coastal):
    if abs(1 - df_coastal['mean'].iloc[k]) < abs(1 - df_coastal['mean'].iloc[k + n_coastal]) :
        coastal_best_obs = coastal_best_obs + 1 
    if abs(1 - df_coastal['mean'].iloc[k]) > abs(1 - df_coastal['mean'].iloc[k + n_coastal]) :
        coastal_best_era5 = coastal_best_era5 + 1
        # print('index best coastal obs: ', k)
print('coastal: best era5: ', coastal_best_era5, ', best obs: ', coastal_best_obs, ', num datasets: ', coastal_best_era5 + coastal_best_obs)

offshore_best_era5 = 0
offshore_best_obs = 0
for k in range(n_offshore):
    if abs(1 - df_offshore['mean'].iloc[k]) < abs(1 - df_offshore['mean'].iloc[k + n_offshore]) :
        offshore_best_obs = offshore_best_obs + 1 
    if abs(1 - df_offshore['mean'].iloc[k]) > abs(1 - df_offshore['mean'].iloc[k + n_offshore]) :
        offshore_best_era5 = offshore_best_era5 + 1
print('offshore: best era5: ', offshore_best_era5, ', best obs: ', offshore_best_obs, ', num datasets: ', offshore_best_era5 + offshore_best_obs)

onoff_best_era5 = 0
onoff_best_obs = 0
for k in range(n_onoff):
    if abs(1 - df_onoff['mean'].iloc[k]) < abs(1 - df_onoff['mean'].iloc[k + n_onoff]) :
        onoff_best_obs = onoff_best_obs + 1 
    if abs(1 - df_onoff['mean'].iloc[k]) > abs(1 - df_onoff['mean'].iloc[k + n_onoff]) :
        onoff_best_era5 = onoff_best_era5 + 1
print('onoff: best era5: ', onoff_best_era5, ', best obs: ', onoff_best_obs, ', num datasets: ', onoff_best_era5 + onoff_best_obs)

# Create a figure with two subplots
fig = plt.figure(figsize=(16, 6))

gs = gridspec.GridSpec(1, 2, width_ratios=[3, 7])  # Divide the figure into 2 parts, with 30% for the first subplot

# Subplot 1 - Boxplot for 'method' data
ax1 = plt.subplot(gs[0])  # 1st subplot
sns.boxplot(data=df, x='datatype', y='mean', ax=ax1)
plt.axhline(y = 1, color = 'r', linestyle = '--', linewidth=0.8) 
ax1.set_xlabel('Data type', fontsize=12)
ax1.set_ylabel('NMean [-]', fontsize=12)
ax1.set_ylim(0.75, 1.25)
ax1.set_title('Normalized Mean with different reference data', fontsize=14)

# Subplot 2 - Boxplot for 'terrain' data
ax2 = plt.subplot(gs[1])  # 2nd subplot
sns.boxplot(data=df_terrain, x='terrain', y='mean', hue='datatype', dodge=True, ax=ax2)
plt.axhline(y = 1, color = 'r', linestyle = '--', linewidth=0.8) 
ax2.set_xlabel('Terrain type', fontsize=12)
ax2.set_ylabel('NMean [-]', fontsize=12)
ax2.set_ylim(0.75, 1.25)
ax2.set_title('Normalized Mean obtained with observed or ERA5 reference data, for different terrain types', fontsize=14)

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

#%%
#MAE
print('NMAE')
best_era5 = 0 
best_obs = 0
N = 35
for k in range(N):
    if mae_era5[k] < mae_obs[k] : 
        best_era5 = best_era5 + 1
    if mae_era5[k] > mae_obs[k] : 
        best_obs = best_obs + 1
print('total: best mae era5: ', best_era5, ', best mae obs: ', best_obs,', num datasets: ', best_era5 + best_obs)


simple_best_era5 = 0
simple_best_obs = 0
for k in range(n_simple):
    if df_simple['mae'].iloc[k] < df_simple['mae'].iloc[k + n_simple] :
        simple_best_obs = simple_best_obs + 1 
    if df_simple['mae'].iloc[k] > df_simple['mae'].iloc[k + n_simple] :
        simple_best_era5 = simple_best_era5 + 1
print('simple: best era5: ', simple_best_era5, ', best obs: ', simple_best_obs, ', num datasets: ', simple_best_era5 + simple_best_obs)

complex_best_era5 = 0
complex_best_obs = 0
for k in range(n_complex):
    if df_complex['mae'].iloc[k] < df_complex['mae'].iloc[k + n_complex] :
        complex_best_obs = complex_best_obs + 1 
    if df_complex['mae'].iloc[k] > df_complex['mae'].iloc[k + n_complex] :
        complex_best_era5 = complex_best_era5 + 1
print('complex: best era5: ', complex_best_era5, ', best obs: ', complex_best_obs, ', num datasets: ', complex_best_era5 + complex_best_obs)

coastal_best_era5 = 0
coastal_best_obs = 0
for k in range(n_coastal):
    if df_coastal['mae'].iloc[k] < df_coastal['mae'].iloc[k + n_coastal] :
        coastal_best_obs = coastal_best_obs + 1 
    if df_coastal['mae'].iloc[k] > df_coastal['mae'].iloc[k + n_coastal] :
        coastal_best_era5 = coastal_best_era5 + 1
        # print('index best coastal obs: ', k)
print('coastal: best era5: ', coastal_best_era5, ', best obs: ', coastal_best_obs, ', num datasets: ', coastal_best_era5 + coastal_best_obs)

offshore_best_era5 = 0
offshore_best_obs = 0
for k in range(n_offshore):
    if df_offshore['mae'].iloc[k] < df_offshore['mae'].iloc[k + n_offshore] :
        offshore_best_obs = offshore_best_obs + 1 
    if df_offshore['mae'].iloc[k] > df_offshore['mae'].iloc[k + n_offshore] :
        offshore_best_era5 = offshore_best_era5 + 1
print('offshore: best era5: ', offshore_best_era5, ', best obs: ', offshore_best_obs, ', num datasets: ', offshore_best_era5 + offshore_best_obs)

onoff_best_era5 = 0
onoff_best_obs = 0
for k in range(n_onoff):
    if df_onoff['mae'].iloc[k] < df_onoff['mae'].iloc[k + n_onoff] :
        onoff_best_obs = onoff_best_obs + 1 
    if df_onoff['mae'].iloc[k] > df_onoff['mae'].iloc[k + n_onoff] :
        onoff_best_era5 = onoff_best_era5 + 1
print('onoff: best era5: ', onoff_best_era5, ', best obs: ', onoff_best_obs, ', num datasets: ', onoff_best_era5 + onoff_best_obs)

# Create a figure with two subplots
fig = plt.figure(figsize=(16, 6))

gs = gridspec.GridSpec(1, 2, width_ratios=[3, 7])  # Divide the figure into 2 parts, with 30% for the first subplot

# Subplot 1 - Boxplot for 'method' data
ax1 = plt.subplot(gs[0])  # 1st subplot
sns.boxplot(data=df, x='datatype', y='mae', ax=ax1)
ax1.set_xlabel('Data type', fontsize=12)
ax1.set_ylabel('NMAE [-]', fontsize=12)
ax1.set_ylim(0, 0.45)
ax1.set_title('Normalized MAE with different reference data', fontsize=14)

# Subplot 2 - Boxplot for 'terrain' data
ax2 = plt.subplot(gs[1])  # 2nd subplot
sns.boxplot(data=df_terrain, x='terrain', y='mae', hue='datatype', dodge=True, ax=ax2)
ax2.set_xlabel('Terrain type', fontsize=12)
ax2.set_ylabel('NMAE [-]', fontsize=12)
ax2.set_ylim(0, 0.45)
ax2.set_title('Normalized MAE obtained with observed or ERA5 reference data, for different terrain types', fontsize=14)

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

#%%
#MBE
print('MBE')
best_era5 = 0 
best_obs = 0
N = 35
for k in range(N):
    if abs( mbe_era5[k]) < abs( mbe_obs[k]) : 
        best_era5 = best_era5 + 1
    if abs( mbe_era5[k]) > abs( mbe_obs[k]) : 
        best_obs = best_obs + 1
print('total: best mbe era5: ', best_era5, ', best mbe obs: ', best_obs,', num datasets: ', best_era5 + best_obs)


simple_best_era5 = 0
simple_best_obs = 0
for k in range(n_simple):
    if abs( df_simple['mbe'].iloc[k]) < abs( df_simple['mbe'].iloc[k + n_simple]) :
        simple_best_obs = simple_best_obs + 1 
    if abs( df_simple['mbe'].iloc[k]) > abs( df_simple['mbe'].iloc[k + n_simple]) :
        simple_best_era5 = simple_best_era5 + 1
print('simple: best era5: ', simple_best_era5, ', best obs: ', simple_best_obs, ', num datasets: ', simple_best_era5 + simple_best_obs)

complex_best_era5 = 0
complex_best_obs = 0
for k in range(n_complex):
    if abs( df_complex['mbe'].iloc[k]) < abs( df_complex['mbe'].iloc[k + n_complex]) :
        complex_best_obs = complex_best_obs + 1 
    if abs( df_complex['mbe'].iloc[k]) > abs( df_complex['mbe'].iloc[k + n_complex]) :
        complex_best_era5 = complex_best_era5 + 1
print('complex: best era5: ', complex_best_era5, ', best obs: ', complex_best_obs, ', num datasets: ', complex_best_era5 + complex_best_obs)

coastal_best_era5 = 0
coastal_best_obs = 0
for k in range(n_coastal):
    if abs( df_coastal['mbe'].iloc[k]) < abs( df_coastal['mbe'].iloc[k + n_coastal]) :
        coastal_best_obs = coastal_best_obs + 1 
    if abs( df_coastal['mbe'].iloc[k]) > abs( df_coastal['mbe'].iloc[k + n_coastal]) :
        coastal_best_era5 = coastal_best_era5 + 1
        # print('index best coastal obs: ', k)
print('coastal: best era5: ', coastal_best_era5, ', best obs: ', coastal_best_obs, ', num datasets: ', coastal_best_era5 + coastal_best_obs)

offshore_best_era5 = 0
offshore_best_obs = 0
for k in range(n_offshore):
    if abs( df_offshore['mbe'].iloc[k]) < abs( df_offshore['mbe'].iloc[k + n_offshore]) :
        offshore_best_obs = offshore_best_obs + 1 
    if abs( df_offshore['mbe'].iloc[k]) > abs( df_offshore['mbe'].iloc[k + n_offshore]) :
        offshore_best_era5 = offshore_best_era5 + 1
print('offshore: best era5: ', offshore_best_era5, ', best obs: ', offshore_best_obs, ', num datasets: ', offshore_best_era5 + offshore_best_obs)

onoff_best_era5 = 0
onoff_best_obs = 0
for k in range(n_onoff):
    if abs( df_onoff['mbe'].iloc[k]) < abs( df_onoff['mbe'].iloc[k + n_onoff]) :
        onoff_best_obs = onoff_best_obs + 1 
    if abs( df_onoff['mbe'].iloc[k]) > abs( df_onoff['mbe'].iloc[k + n_onoff]) :
        onoff_best_era5 = onoff_best_era5 + 1
print('onoff: best era5: ', onoff_best_era5, ', best obs: ', onoff_best_obs, ', num datasets: ', onoff_best_era5 + onoff_best_obs)

# Create a figure with two subplots
fig = plt.figure(figsize=(16, 6))

gs = gridspec.GridSpec(1, 2, width_ratios=[3, 7])  # Divide the figure into 2 parts, with 30% for the first subplot

# Subplot  Boxplot for 'method' data
ax1 = plt.subplot(gs[0])  # 1st subplot
sns.boxplot(data=df, x='datatype', y='mbe', ax=ax1)
plt.axhline(y = 0, color = 'r', linestyle = '--', linewidth=0.8) 
ax1.set_xlabel('Data type', fontsize=12)
ax1.set_ylabel('NMBE [-]', fontsize=12)
ax1.set_ylim(-0.15, 0.3)
ax1.set_title('Normalized MBE with different reference data', fontsize=14)

# Subplot 2 - Boxplot for 'terrain' data
ax2 = plt.subplot(gs[1])  # 2nd subplot
sns.boxplot(data=df_terrain, x='terrain', y='mbe', hue='datatype', dodge=True, ax=ax2)
plt.axhline(y = 0, color = 'r', linestyle = '--', linewidth=0.8) 
ax2.set_xlabel('Terrain type', fontsize=12)
ax2.set_ylabel('NMBE [-]', fontsize=12)
ax2.set_ylim(-0.15, 0.3)
ax2.set_title('Normalized MBE obtained with observed or ERA5 reference data, for different terrain types', fontsize=14)

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

#%%
#PEARSON
print('PEARSON')
best_era5 = 0 
best_obs = 0
for k in range(N):
    if pearson_era5[k] > pearson_obs[k] : 
        best_era5 = best_era5 + 1
    if pearson_era5[k] < pearson_obs[k] : 
        best_obs = best_obs + 1
print('total: best pearson era5: ', best_era5, ', best pearson obs: ', best_obs,', num datasets: ', best_era5 + best_obs)

simple_best_era5 = 0
simple_best_obs = 0
for k in range(n_simple):
    if df_simple['pearson'].iloc[k] > df_simple['pearson'].iloc[k + n_simple] :
        simple_best_obs = simple_best_obs + 1 
    if df_simple['pearson'].iloc[k] < df_simple['pearson'].iloc[k + n_simple] :
        simple_best_era5 = simple_best_era5 + 1
print('simple: best era5: ', simple_best_era5, ', best obs: ', simple_best_obs, ', num datasets: ', simple_best_era5 + simple_best_obs)

complex_best_era5 = 0
complex_best_obs = 0
for k in range(n_complex):
    if df_complex['pearson'].iloc[k] > df_complex['pearson'].iloc[k + n_complex] :
        complex_best_obs = complex_best_obs + 1 
    if df_complex['pearson'].iloc[k] < df_complex['pearson'].iloc[k + n_complex] :
        complex_best_era5 = complex_best_era5 + 1
print('complex: best era5: ', complex_best_era5, ', best obs: ', complex_best_obs, ', num datasets: ', complex_best_era5 + complex_best_obs)

coastal_best_era5 = 0
coastal_best_obs = 0
for k in range(n_coastal):
    if df_coastal['pearson'].iloc[k] > df_coastal['pearson'].iloc[k + n_coastal] :
        coastal_best_obs = coastal_best_obs + 1 
    if df_coastal['pearson'].iloc[k] < df_coastal['pearson'].iloc[k + n_coastal] :
        coastal_best_era5 = coastal_best_era5 + 1
        # print('index best coastal obs: ', k)
print('coastal: best era5: ', coastal_best_era5, ', best obs: ', coastal_best_obs, ', num datasets: ', coastal_best_era5 + coastal_best_obs)

offshore_best_era5 = 0
offshore_best_obs = 0
for k in range(n_offshore):
    if df_offshore['pearson'].iloc[k] > df_offshore['pearson'].iloc[k + n_offshore] :
        offshore_best_obs = offshore_best_obs + 1 
    if df_offshore['pearson'].iloc[k] < df_offshore['pearson'].iloc[k + n_offshore] :
        offshore_best_era5 = offshore_best_era5 + 1
print('offshore: best era5: ', offshore_best_era5, ', best obs: ', offshore_best_obs, ', num datasets: ', offshore_best_era5 + offshore_best_obs)

onoff_best_era5 = 0
onoff_best_obs = 0
for k in range(n_onoff):
    if df_onoff['pearson'].iloc[k] > df_onoff['pearson'].iloc[k + n_onoff] :
        onoff_best_obs = onoff_best_obs + 1 
    if df_onoff['pearson'].iloc[k] < df_onoff['pearson'].iloc[k + n_onoff] :
        onoff_best_era5 = onoff_best_era5 + 1
print('onoff: best era5: ', onoff_best_era5, ', best obs: ', onoff_best_obs, ', num datasets: ', onoff_best_era5 + onoff_best_obs)


# Create a figure with two subplots
fig = plt.figure(figsize=(16, 6))

gs = gridspec.GridSpec(1, 2, width_ratios=[3, 7])  # Divide the figure into 2 parts, with 30% for the first subplot

# Subplot 1 - Boxplot for 'method' data
ax1 = plt.subplot(gs[0])  # 1st subplot
sns.boxplot(data=df, x='datatype', y='pearson', ax=ax1)
ax1.set_xlabel('Data type', fontsize=12)
ax1.set_ylabel('Pearson Corr [-]', fontsize=12)
ax1.set_ylim(0, 1)
ax1.set_title('Pearson correlation with different reference data', fontsize=14)

# Subplot 2 - Boxplot for 'terrain' data
ax2 = plt.subplot(gs[1])  # 2nd subplot
sns.boxplot(data=df_terrain, x='terrain', y='pearson', hue='datatype', dodge=True, ax=ax2)
ax2.set_xlabel('Terrain type', fontsize=12)
ax2.set_ylabel('Pearson Corr [-]', fontsize=12)
ax2.set_ylim(0, 1)
ax2.set_title('Pearson correlation obtained with observed or ERA5 reference data, for different terrain types', fontsize=14)

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

#%%
# print outliers
print('obs lower than 0.6: ')
for k in range(N): 
    if r2_obs[k] <= 0.6:
        print(k, 'r2 obs: ', r2_obs[k], 'r2 era5: ', r2_era5[k])
print('era5 lower than 0.6: ')
for k in range(N): 
    if r2_era5[k] <= 0.6:
        print(k, 'r2 obs: ', r2_obs[k], 'r2 era5: ', r2_era5[k])
        

#%%
#look into outliers
