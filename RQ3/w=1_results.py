# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:39:23 2023

@author: sylke
"""

import os 
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import r2_score
import math 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy import stats
import natsort
import matplotlib.pyplot as plt
import seaborn as sns

keys = ["UK1. Dyffryn Brodyn", "UK2. Lifton Down", "UK3. St. Breock", "NL1. Schiphol", "NL2. Westdorpe", "NL3. Hupsel", "NL4. Cabauw",
        "UK4. Penrhys", "UK5. Rheidol", "UK6. Alt-Yr-Hende", "NL5. Rotterdam Geulhaven", "UK7. Siddick", "UK8. Haverigg", "UK9. Treculliacks",
        "UK10. Rhyd-Y-Goes", "UK11. Hill of Forss", "UK12. Crimp", "UK13. Ysgubor", "UK14. Jordanston", "UK15. Truthan", "UK16. Carland Cross",
        "NL6. Platform AWG-1", "UK17. Celtic Array Zone 9", "UK18. Greater Gabbard", "UK19. Gunfleet Sands", "UK20. Gwynt Y Mor", "UK21. Shell Flats",
        "NL7. Lichteiland Goeree", "NL8. K14FA1C", "NL9. J6-A", "NL10. Borssele 1", "NL11. Hollandse Kust West (HKWA)", "NL12. Hollandse Kust Noord (HKNB)",
        "NL13. Ten Noorden van de Wadden (TNWB)", "NL14. Dogger Bank zone 3"]


# folder_path = 'C:/Users/sylke/OneDrive/Documenten/THESIS/DATA/1_CSVS/new_validate_concurrent/AE_full'
folder_path = 'C:/Users/sylke/OneDrive/Documenten/THESIS/DATA/1_CSVS/new_validate_concurrent/AE_OBS'

csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
csv_files_sorted = natsort.natsorted(csv_files)

reconstructed = []
for csv_file in csv_files_sorted: 
    df = pd.read_csv(csv_file)
    reconstructed.append(df)

# folder_path = 'C:/Users/sylke/OneDrive/Documenten/THESIS/DATA/1_CSVS//new_validate_concurrent/OLR_full'
# folder_path = 'C:/Users/sylke/OneDrive/Documenten/THESIS/DATA/1_CSVS/OLR_OBS_results_Q1_v3'
folder_path = 'C:/Users/sylke/OneDrive/Documenten/THESIS/DATA/1_CSVS//new_validate_concurrent/OLR_OBS_RQ1'

csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
csv_files_sorted = natsort.natsorted(csv_files)

OLR = []
for csv_file in csv_files_sorted: 
    df = pd.read_csv(csv_file)
    OLR.append(df)

# folder_path = 'C:/Users/sylke/OneDrive/Documenten/THESIS/DATA/1_CSVS/new_validate_concurrent/concurrent_full'
folder_path = 'C:/Users/sylke/OneDrive/Documenten/THESIS/DATA/1_CSVS/new_validate_concurrent/concurrent_OBS_RQ1'

csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
csv_files_sorted = natsort.natsorted(csv_files)

concurrent_df = []
for csv_file in csv_files_sorted: 
    df = pd.read_csv(csv_file)
    concurrent_df.append(df)


datasets = 35
#%%
# # make sure there are no NaN values
for k in range(datasets): 
    reconstructed[k].dropna(ignore_index = True, inplace = True)

r2_AE = np.zeros(datasets)
r2_olr = np.zeros(datasets)
rmse_AE = np.zeros(datasets)
rmse_olr = np.zeros(datasets)
mean_actual = np.zeros(datasets)
mean_AE = np.zeros(datasets)
mean_olr = np.zeros(datasets)
mae_AE = np.zeros(datasets)
mae_olr = np.zeros(datasets)
mbe_AE = np.zeros(datasets)
mbe_olr = np.zeros(datasets)
pearson_ref = np.zeros(datasets)
pearson_AE = np.zeros(datasets)
pearson_olr = np.zeros(datasets)
for k in range(datasets): 
    # determine R^2

    r2_AE[k] = (r2_score(reconstructed[k]['target_wind_speed'], reconstructed[k]['prediction']))
    r2_olr[k] = r2_score(OLR[k]['target_wind_speed'], OLR[k]['longterm_target_estimate'])
    # root mean squared error
    rmse_AE[k] = (math.sqrt(mean_squared_error(reconstructed[k]['target_wind_speed'], reconstructed[k]['prediction'])) / reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True))
    rmse_olr[k] = (math.sqrt(mean_squared_error(OLR[k]['target_wind_speed'], OLR[k]['longterm_target_estimate'])) / OLR[k]['target_wind_speed'].mean(axis=0, skipna = True))
    # normalized mean
    mean_actual[k] = (reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True) / reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True))
    mean_AE[k] = (reconstructed[k]['prediction'].mean(axis=0, skipna = True) / reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True))
    mean_olr[k] = (OLR[k]['longterm_target_estimate'].mean(axis=0, skipna = True) / OLR[k]['target_wind_speed'].mean(axis=0, skipna = True))
    #mean absolute error normalized
    mae_AE[k] = (mean_absolute_error(reconstructed[k]['target_wind_speed'], reconstructed[k]['prediction']) / reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True))
    mae_olr[k] = (mean_absolute_error(OLR[k]['target_wind_speed'], OLR[k]['longterm_target_estimate']) / OLR[k]['target_wind_speed'].mean(axis=0, skipna = True))
    # mean bias error normalized
    mbe_AE[k] = ((np.mean(reconstructed[k]['prediction']) - np.mean(reconstructed[k]['target_wind_speed'])) / np.mean(reconstructed[k]['target_wind_speed']))
    mbe_olr[k] = ((np.mean(OLR[k]['longterm_target_estimate']) - np.mean(OLR[k]['target_wind_speed'])) / np.mean(OLR[k]['target_wind_speed']))
    # pearson correlation of reconstructed period
    pearson_ref[k] = (stats.linregress(reconstructed[k]['target_wind_speed'], reconstructed[k]['ref_wind_speed'])[2])
    pearson_AE[k] = (stats.linregress(reconstructed[k]['target_wind_speed'], reconstructed[k]['prediction'])[2])
    pearson_olr[k] = (stats.linregress(OLR[k]['target_wind_speed'], OLR[k]['longterm_target_estimate'])[2])

#%%
data_olr = {'r2score': r2_olr, 'rmse': rmse_olr, 'mean': mean_olr, 'mbe': mbe_olr, 'mae': mae_olr, 'pearson': pearson_olr}
df_olr = pd.DataFrame(data_olr)
df_olr['method'] = 'MCP'
data_AE = {'r2score': r2_AE, 'rmse': rmse_AE, 'mean': mean_AE, 'mbe': mbe_AE, 'mae': mae_AE, 'pearson': pearson_AE}
df_AE = pd.DataFrame(data_AE)
df_AE['method'] = 'Analog Ensembles'
df = pd.concat([df_olr, df_AE], ignore_index=True)
#%%
# create table appendix results method of analogs
mcp_data = round(df_olr, 2)
ae_data = round(df_AE, 2)
#%%

# plt.figure(figsize=(8,6))
# ax = sns.boxplot(data = df, x='method', y='rmse')
# ax.set_xlabel('Method', fontsize=12)
# ax.set_ylabel('NRMSE', fontsize=12)
# ax.set_ylim(0, 0.6)
# plt.title('Normalized Root Mean Squared Error MCP vs Analog Ensembles', fontsize=14)
# plt.show()

plt.figure(figsize=(8,6))
ax = sns.boxplot(data = df, x='method', y='r2score')
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('R^2', fontsize=12)
ax.set_ylim(-0.4, 1)
plt.title('Coefficient of Determination MCP vs Analog Ensembles', fontsize=14)
plt.show()

# plt.figure(figsize=(8,6))
# ax = sns.boxplot(data = df, x='method', y='mean')
# ax.set_xlabel('Method', fontsize=12)
# ax.set_ylabel('Normalized Mean Wind Speed [m/s]', fontsize=12)
# ax.set_ylim(0.9, 1.2)
# plt.title('Normalized Mean Wind Speed MCP vs Analog Ensembles', fontsize=14)
# plt.show()

# plt.figure(figsize=(8,6))
# ax = sns.boxplot(data = df, x='method', y='mbe')
# ax.set_xlabel('Method', fontsize=12)
# ax.set_ylabel('Normalized Mean Bias Error', fontsize=12)
# ax.set_ylim(-0.1, 0.2)
# plt.title('Normalized Mean Bias Error MCP vs Analog Ensembles', fontsize=14)
# plt.show()

# plt.figure(figsize=(8,6))
# ax = sns.boxplot(data = df, x='method', y='pearson')
# ax.set_xlabel('Method', fontsize=12)
# ax.set_ylabel('Pearson Correlation', fontsize=12)
# ax.set_ylim(0, 1)
# plt.title('Pearson Correlation Prediction MCP vs Analog Ensembles', fontsize=14)
# plt.show()

#%%
# for each terrain type
df_simple = pd.concat([df[0:7].copy(), df[35:42].copy()], ignore_index=True)
df_complex = pd.concat([df[7:11].copy(), df[42:46].copy()], ignore_index=True)
df_coastal = pd.concat([df[11:22].copy(), df[46:57].copy()], ignore_index=True)
df_offshore = pd.concat([df[22:35].copy(), df[57:70].copy()], ignore_index=True)

df_simple['terrain'] = 'simple'
df_complex['terrain'] = 'complex'
df_coastal['terrain'] = 'coastal'
df_offshore['terrain'] = 'offshore'

df_terrain = pd.concat([df_simple, df_complex, df_coastal, df_offshore], ignore_index=True)

#%%
#r2 score
plt.figure(figsize=(16, 6))
ax = sns.boxplot(data=df_terrain, x='terrain', y='r2score', hue='method', dodge=True)
ax.set_xlabel('Terrain', fontsize=12)
ax.set_ylabel('R^2', fontsize=12)
ax.set_ylim(-0.4, 1)
ax.set_title('Coefficient of Determination for Different Terrains, MCP & Method of Analogs', fontsize=14)

# # Show the combined plot
# plt.show()

# #rmse
# plt.figure(figsize=(16, 6))
# ax = sns.boxplot(data=df_terrain, x='terrain', y='rmse', hue='method', dodge=True)
# ax.set_xlabel('Terrain', fontsize=12)
# ax.set_ylabel('NRMSE', fontsize=12)
# ax.set_ylim(0, 0.6)
# ax.set_title('Normalized RMSE for Different Terrains, MCP & Method of Analogs', fontsize=14)

# # Show the combined plot
# plt.show()

# #mbe
# plt.figure(figsize=(16, 6))
# ax = sns.boxplot(data=df_terrain, x='terrain', y='mbe', hue='method', dodge=True)
# plt.axhline(y = 0, color = 'r', linestyle = '--', linewidth=1)
# ax.set_xlabel('Terrain', fontsize=12)
# ax.set_ylabel('MBE', fontsize=12)
# ax.set_ylim(-0.1, 0.2)
# ax.set_title('Normalized Mean Bias Error for Different Terrains, MCP & Method of Analogs', fontsize=14)

# # Show the combined plot
# plt.show()


# #mbe
# plt.figure(figsize=(16, 6))
# ax = sns.boxplot(data=df_terrain, x='terrain', y='mean', hue='method', dodge=True)
# plt.axhline(y = 1, color = 'r', linestyle = '--', linewidth=1)
# ax.set_xlabel('Terrain', fontsize=12)
# ax.set_ylabel('Normalized Mean', fontsize=12)
# ax.set_ylim(0.9, 1.2)
# ax.set_title('Normalized Mean Wind Speed for Different Terrains, MCP & Method of Analogs', fontsize=14)

# # Show the combined plot
# plt.show()

#%%
print('avg values w=1: r2:', np.mean(df_AE['r2score']), 'nrmse: ', np.mean(df_AE['rmse']), 'norm mean: ', np.mean(df_AE['mean']), 'nmbe: ', np.mean(df_AE['mbe']), 'nmae: ', np.mean(df_AE['mae']), 'pearson: ', np.mean(df_AE['pearson']))
print('OLR: avg values w=1: r2:', np.mean(df_olr['r2score']), 'nrmse: ', np.mean(df_olr['rmse']), 'norm mean: ', np.mean(df_olr['mean']), 'nmbe: ', np.mean(df_olr['mbe']), 'nmae: ', np.mean(df_olr['mae']), 'pearson: ', np.mean(df_olr['pearson']))
#%%
# determine percentage best r^2 score for the 35 targets
# best_olr = 0 
# best_AE = 0
# for k in range(datasets):
#     if min(mae_olr[k], mae_AE[k], key=abs) == mae_olr[k] : 
#         best_olr = best_olr + 1
#     if min(mae_olr[k], mae_AE[k], key=abs) == mae_AE[k] : 
#         best_AE = best_AE + 1
# print('best mae olr: ', best_olr, ', best mae ae: ', best_AE,', num datasets: ', best_olr + best_AE)

best_olr = 0 
best_AE = 0
for k in range(datasets):
    if r2_olr[k] > r2_AE[k] : 
        best_olr = best_olr + 1
    if r2_olr[k] < r2_AE[k] : 
        best_AE = best_AE + 1
print('best r2 olr: ', best_olr, ', best r2 ae: ', best_AE,', num datasets: ', best_olr + best_AE)

# best_olr = 0 
# best_AE = 0
# for k in range(datasets):
#     if rmse_olr[k] < rmse_AE[k] : 
#         best_olr = best_olr + 1
#     if rmse_olr[k] > rmse_AE[k] : 
#         best_AE = best_AE + 1
# print('best rmse olr: ', best_olr, ', best rmse ae: ', best_AE,', num datasets: ', best_olr + best_AE)

# best_olr = 0 
# best_AE = 0
# for k in range(datasets):
#     if min((mean_olr[k] - 1), (mean_AE[k] - 1), key=abs) == (mean_olr[k]-1) : 
#         best_olr = best_olr + 1
#     if min((mean_olr[k] - 1), (mean_AE[k] - 1), key=abs) == (mean_AE[k]-1) :
#         best_AE = best_AE + 1
# print('best mean olr: ', best_olr, ', best mean ae: ', best_AE,', num datasets: ', best_olr + best_AE)

# best_olr = 0 
# best_AE = 0
# for k in range(datasets):
#     if min((mbe_olr[k]), (mbe_AE[k]), key=abs) == (mbe_olr[k]) : 
#         best_olr = best_olr + 1
#     if min((mbe_olr[k]), (mbe_AE[k]), key=abs) == (mbe_AE[k]) :
#         best_AE = best_AE + 1
# print('best mbe olr: ', best_olr, ', best mbe ae: ', best_AE,', num datasets: ', best_olr + best_AE)

# best_olr = 0 
# best_AE = 0
# for k in range(datasets):
#     if pearson_olr[k] > pearson_AE[k] : 
#         best_olr = best_olr + 1
#     if pearson_olr[k] < pearson_AE[k] : 
#         best_AE = best_AE + 1
# print('best pearson olr: ', best_olr, ', best pearson ae: ', best_AE,', num datasets: ', best_olr + best_AE)

#%%
# determine best r^2 score for the 35 targets per terrain type
n_simple = int(len(df_simple) / 2)
n_complex = int(len(df_complex) / 2)
n_coastal = int(len(df_coastal) / 2)
n_offshore = int(len(df_offshore) / 2)

simple_best_olr = 0
simple_best_ae = 0
for k in range(n_simple):
    if df_simple['r2score'].iloc[k] > df_simple['r2score'].iloc[k + n_simple] :
        simple_best_olr = simple_best_olr + 1 
    if df_simple['r2score'].iloc[k] < df_simple['r2score'].iloc[k + n_simple] :
        simple_best_ae = simple_best_ae + 1
print('simple: best olr: ', simple_best_olr, ', best ae: ', simple_best_ae, ', num datasets: ', simple_best_olr + simple_best_ae)

complex_best_olr = 0
complex_best_ae = 0
for k in range(n_complex):
    if df_complex['r2score'].iloc[k] > df_complex['r2score'].iloc[k + n_complex] :
        complex_best_olr = complex_best_olr + 1 
    if df_complex['r2score'].iloc[k] < df_complex['r2score'].iloc[k + n_complex] :
        complex_best_ae = complex_best_ae + 1
print('complex: best olr: ', complex_best_olr, ', best ae: ', complex_best_ae, ', num datasets: ', complex_best_olr + complex_best_ae)

coastal_best_olr = 0
coastal_best_ae = 0
for k in range(n_coastal):
    if df_coastal['r2score'].iloc[k] > df_coastal['r2score'].iloc[k + n_coastal] :
        coastal_best_olr = coastal_best_olr + 1 
    if df_coastal['r2score'].iloc[k] < df_coastal['r2score'].iloc[k + n_coastal] :
        coastal_best_ae = coastal_best_ae + 1
        # print('index best coastal ae: ', k)
print('coastal: best olr: ', coastal_best_olr, ', best ae: ', coastal_best_ae, ', num datasets: ', coastal_best_olr + coastal_best_ae)

offshore_best_olr = 0
offshore_best_ae = 0
for k in range(n_offshore):
    if df_offshore['r2score'].iloc[k] > df_offshore['r2score'].iloc[k + n_offshore] :
        offshore_best_olr = offshore_best_olr + 1 
    if df_offshore['r2score'].iloc[k] < df_offshore['r2score'].iloc[k + n_offshore] :
        offshore_best_ae = offshore_best_ae + 1
print('offshore: best olr: ', offshore_best_olr, ', best ae: ', offshore_best_ae, ', num datasets: ', offshore_best_olr + offshore_best_ae)

#%%
import matplotlib.gridspec as gridspec

# Create a figure with two subplots
fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 7])  # Divide the figure into 2 parts, with 30% for the first subplot

# Subplot 1 - Boxplot for 'method' data
ax1 = plt.subplot(gs[0])  # 1st subplot
sns.boxplot(data=df, x='method', y='r2score', ax=ax1)
ax1.set_xlabel('Method', fontsize=12)
ax1.set_ylabel('R^2 [-]', fontsize=12)
ax1.set_ylim(0, 1)
ax1.set_title('Coefficient of Determination per method', fontsize=14)

# Subplot 2 - Boxplot for 'terrain' data
ax2 = plt.subplot(gs[1])  # 2nd subplot
sns.boxplot(data=df_terrain, x='terrain', y='r2score', hue='method', dodge=True, ax=ax2)
ax2.set_xlabel('Terrain', fontsize=12)
ax2.set_ylabel('R^2 [-]', fontsize=12)
ax2.set_ylim(0, 1)
ax2.set_title('Coefficient of Determination for different terrain types per method', fontsize=14)
ax2.legend(loc='lower right', fontsize=12)

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

#%%
simple_best_olr = 0
simple_best_ae = 0
for k in range(n_simple):
    if min(df_simple['mbe'].iloc[k], df_simple['mbe'].iloc[k + n_simple], key=abs) == df_simple['mbe'].iloc[k] :
        simple_best_olr = simple_best_olr + 1 
    if min(df_simple['mbe'].iloc[k], df_simple['mbe'].iloc[k + n_simple], key=abs) == df_simple['mbe'].iloc[k + n_simple] :
        simple_best_ae = simple_best_ae + 1
print('simple: best olr: ', simple_best_olr, ', best ae: ', simple_best_ae, ', num datasets: ', simple_best_olr + simple_best_ae)

complex_best_olr = 0
complex_best_ae = 0
for k in range(n_complex):
    if min(df_complex['mbe'].iloc[k], df_complex['mbe'].iloc[k + n_complex], key=abs) == df_complex['mbe'].iloc[k] :
        complex_best_olr = complex_best_olr + 1 
    if min(df_complex['mbe'].iloc[k], df_complex['mbe'].iloc[k + n_complex], key=abs) == df_complex['mbe'].iloc[k + n_complex] :
        complex_best_ae = complex_best_ae + 1
print('complex: best olr: ', complex_best_olr, ', best ae: ', complex_best_ae, ', num datasets: ', complex_best_olr + complex_best_ae)

coastal_best_olr = 0
coastal_best_ae = 0
for k in range(n_coastal):
    if min(df_coastal['mbe'].iloc[k], df_coastal['mbe'].iloc[k + n_coastal],key=abs) == df_coastal['mbe'].iloc[k] :
        coastal_best_olr = coastal_best_olr + 1 
    if min(df_coastal['mbe'].iloc[k], df_coastal['mbe'].iloc[k + n_coastal],key=abs) == df_coastal['mbe'].iloc[k + n_coastal] :
        coastal_best_ae = coastal_best_ae + 1
        print('index best coastal ae: ', k)
print('coastal: best olr: ', coastal_best_olr, ', best ae: ', coastal_best_ae, ', num datasets: ', coastal_best_olr + coastal_best_ae)

offshore_best_olr = 0
offshore_best_ae = 0
for k in range(n_offshore):
    if min(df_offshore['mbe'].iloc[k], df_offshore['mbe'].iloc[k + n_offshore], key=abs) == df_offshore['mbe'].iloc[k] :
        offshore_best_olr = offshore_best_olr + 1 
    if min(df_offshore['mbe'].iloc[k], df_offshore['mbe'].iloc[k + n_offshore], key=abs) == df_offshore['mbe'].iloc[k + n_offshore] :
        offshore_best_ae = offshore_best_ae + 1
print('offshore: best olr: ', offshore_best_olr, ', best ae: ', offshore_best_ae, ', num datasets: ', offshore_best_olr + offshore_best_ae)

#%%
df_rounded = df.round(2)

for k in range(datasets + datasets):
    if (df['r2score'][k] <= 0.6):
        print(k)
        
#%%
print(np.mean(r2_AE))

#%%
print(np.mean(df_offshore['r2score'].loc[df_offshore['method'] == 'Analog Ensembles']))

#%%
diff_r2 = abs(r2_olr - r2_AE)
print(np.mean(diff_r2))
#%%
print('r2_olr < 0.6 : ')
for k in range(datasets): 
    if r2_olr[k] < 0.6: 
        print(k, 'olr: ', 'r2 : ', round(r2_olr[k],2), 'rmse: ', round(rmse_olr[k],2), 'mean: ', round(mean_olr[k],2), 'mae: ', round(mae_olr[k],2), 'mbe: ', round(mbe_olr[k],2), 'pearson: ', round(pearson_olr[k],2))
        print(k, 'ae: ', 'r2 : ', round(r2_AE[k],2), 'rmse: ', round(rmse_AE[k],2), 'mean: ', round(mean_AE[k],2), 'mae: ', round(mae_AE[k],2), 'mbe: ', round(mbe_AE[k],2), 'pearson: ', round(pearson_AE[k],2))

print('r2_AE < 0.6 : ')
for k in range(datasets): 
    if r2_AE[k] < 0.6: 
        print(k, 'olr: ', 'r2 : ', round(r2_olr[k],2), 'rmse: ', round(rmse_olr[k],2), 'mean: ', round(mean_olr[k],2), 'mae: ', round(mae_olr[k],2), 'mbe: ', round(mbe_olr[k],2), 'pearson: ', round(pearson_olr[k],2))
        print(k, 'ae: ', 'r2 : ', round(r2_AE[k],2), 'rmse: ', round(rmse_AE[k],2), 'mean: ', round(mean_AE[k],2), 'mae: ', round(mae_AE[k],2), 'mbe: ', round(mbe_AE[k],2), 'pearson: ', round(pearson_AE[k],2))
        
#%%
pearson_wind_speed = np.zeros(datasets)
for k in range(datasets): 
    pearson_wind_speed[k] = (stats.linregress(concurrent_df[k]['target_wind_speed'], concurrent_df[k]['ref_wind_speed'])[2])
# lowest 17 pearson indices
idx = np.argsort(pearson_wind_speed)
r2_lowest_r_conc_olr = r2_olr[idx[:17]]
r2_lowest_r_conc_ae = r2_AE[idx[:17]]

best_olr = 0
best_ae = 0  
for k in range(len(r2_lowest_r_conc_olr)):
    if r2_lowest_r_conc_olr[k] > r2_lowest_r_conc_ae[k]: 
        best_olr = best_olr + 1 
    if r2_lowest_r_conc_olr[k] < r2_lowest_r_conc_ae[k]: 
        best_ae = best_ae + 1 
print(best_olr, best_ae, len(r2_lowest_r_conc_olr))
print(np.mean(pearson_wind_speed))