# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:46:30 2023

@author: sylke
"""''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score

import method_comparison_OLR_VRM_MM as methods #only use the 35 targets, on/off are double
normalized_mean_actual = methods.normalized_mean_actual[:35]
normalized_mean_OLR = methods.normalized_mean_OLR[:35]
normalized_mean_VRM = methods.normalized_mean_VRM[:35]
normalized_mean_MM1 = methods.normalized_mean_MM_opt1[:35] 
normalized_mean_MM2 = methods.normalized_mean_MM_opt2[:35]
keys = methods.keys[:35]
normalized_sector_mean_actual = methods.normalized_sector_mean_actual[:35]
normalized_sector_mean_OLR = methods.normalized_sector_mean_OLR[:35]
normalized_sector_mean_VRM = methods.normalized_sector_mean_VRM[:35]
normalized_sector_mean_MM1 = methods.normalized_sector_mean_MM1[:35]
normalized_sector_mean_MM2 = methods.normalized_sector_mean_MM2[:35]

validate_df_OLR = methods.validate_df_OLR
validate_df_VRM = methods.validate_df_VRM
mean_MM1 = methods.mean_MM_opt1
mean_MM2 = methods.mean_MM_opt2
actual_mean = methods.mean_actual

#%% create box plot to show performance of the data 
# use the normalized data, so that all sets are comparable. (one site could have a mean of 5 while another might be 8)

# create dataframe to containe the method and the normalized overall mean
df_actual = pd.DataFrame({'normalized_mean': normalized_mean_actual})
df_actual['method'] = 'actual'
df_actual['key'] = keys
df_OLR = pd.DataFrame({'normalized_mean': normalized_mean_OLR})
df_OLR['method'] = 'OLR'
df_OLR['key'] = keys
df_VRM = pd.DataFrame({'normalized_mean': normalized_mean_VRM})
df_VRM['method'] = 'VRM'
df_VRM['key'] = keys
df_MM1 = pd.DataFrame({'normalized_mean': normalized_mean_MM1})
df_MM1['method'] = 'MM1'
df_MM1['key'] = keys
df_MM2 = pd.DataFrame({'normalized_mean': normalized_mean_MM2})
df_MM2['method'] = 'MM2'
df_MM2['key'] = keys
df = pd.concat([df_actual, df_OLR, df_VRM, df_MM1, df_MM2], ignore_index=True)

# split df by terrain type
df_inlandsimple = pd.concat([df_actual[:7], df_OLR[:7], df_VRM[:7], df_MM1[:7], df_MM2[:7]], ignore_index=True)
df_inlandcomplex = pd.concat([df_actual[7:11], df_OLR[7:11], df_VRM[7:11], df_MM1[7:11], df_MM2[7:11]], ignore_index=True)
df_coastal = pd.concat([df_actual[11:22], df_OLR[11:22], df_VRM[11:22], df_MM1[11:22], df_MM2[11:22]], ignore_index=True)
df_offshore = pd.concat([df_actual[22:], df_OLR[22:], df_VRM[22:], df_MM1[22:], df_MM2[22:]])

#all
plt.figure(figsize=(8,6))
ax = sns.boxplot(data=df, x='method', y='normalized_mean')
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Normalized overall mean [-]', fontsize=12)
ax.set_ylim(0.8, 1.4)
plt.title('normalized_overall_mean prediction with different methods', fontsize=14)
plt.show()
#inlandsimple
plt.figure(figsize=(8,6))
ax = sns.boxplot(data=df_inlandsimple, x='method', y='normalized_mean')
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Normalized overall mean [-]', fontsize=12)
ax.set_ylim(0.8, 1.4)
plt.title('Inland Simple Terrain: Normalized_overall_mean prediction with different methods', fontsize=14)
plt.show()
#inlandcomplex
plt.figure(figsize=(8,6))
ax = sns.boxplot(data=df_inlandcomplex, x='method', y='normalized_mean')
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Normalized overall mean [-]', fontsize=12)
ax.set_ylim(0.8, 1.4)
plt.title('Inland Complex Terrain: Normalized_overall_mean prediction with different methods', fontsize=14)
plt.show()
#coastal
plt.figure(figsize=(8,6))
ax = sns.boxplot(data=df_coastal, x='method', y='normalized_mean')
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Normalized overall mean [-]', fontsize=12)
ax.set_ylim(0.8, 1.4)
plt.title('Coastal Terrain: Normalized_overall_mean prediction with different methods', fontsize=14)
plt.show()
#offshore
plt.figure(figsize=(8,6))
ax = sns.boxplot(data=df_offshore, x='method', y='normalized_mean')
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Normalized overall mean [-]', fontsize=12)
ax.set_ylim(0.8, 1.4)
plt.title('Offshore Terrain: Normalized_overall_mean prediction with different methods', fontsize=14)
plt.show()
#%% create boxplot of normalized sector means 1 for each sector

# df_1_OLR = pd.DataFrame({'normalized_sector_mean_1' : [arr[0] for arr in normalized_sector_mean_OLR], 'method' : 'OLR', 'key': keys})
# df_2_OLR = pd.DataFrame({'normalized_sector_mean_2' : [arr[1] for arr in normalized_sector_mean_OLR], 'method' : 'OLR', 'key': keys})

df_list_OLR = []
df_list_VRM = []
df_list_MM1 = []
df_list_MM2 = []

# Iterate through the indices 0 to 11 to create DataFrames for each sector
for sector_index in range(12):
    # Create a DataFrame for each sector and append it to the list
    df_sector_OLR = pd.DataFrame({
        'normalized_sector_mean': [arr[sector_index] for arr in normalized_sector_mean_OLR],
        'method': 'OLR',
        'key': keys
    })
    df_sector_VRM = pd.DataFrame({
        'normalized_sector_mean': [arr[sector_index] for arr in normalized_sector_mean_VRM],
        'method': 'VRM',
        'key': keys
    })
    df_sector_MM1 = pd.DataFrame({
        'normalized_sector_mean': [arr[sector_index] for arr in normalized_sector_mean_MM1],
        'method': 'MM1',
        'key': keys
    })
    df_sector_MM2 = pd.DataFrame({
        'normalized_sector_mean': [arr[sector_index] for arr in normalized_sector_mean_MM2],
        'method': 'MM2',
        'key': keys
    })
    df_list_OLR.append(df_sector_OLR)
    df_list_VRM.append(df_sector_VRM)
    df_list_MM1.append(df_sector_MM1)
    df_list_MM2.append(df_sector_MM2)
    
result_dataframes = []
for sector_index in range(12):
    df_concatenated = pd.concat([df_list_OLR[sector_index], df_list_VRM[sector_index], df_list_MM1[sector_index], df_list_MM2[sector_index]], axis=0, ignore_index=True)
    result_dataframes.append(df_concatenated)

#%%    
for i in range(12): 
    plt.figure()
    ax = sns.boxplot(data = result_dataframes[i], x='method', y='normalized_sector_mean')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Normalized sector mean [-]', fontsize=12)
    plt.title(f'normalized_sector_mean prediction with different methods for sector {i + 1}', fontsize=14)
    plt.show()
#%% determine upper and lower quartile, interquartile range (IQR) to determine outliers
# OLR
q3_OLR = np.percentile(df_OLR['normalized_mean'], 75)
q1_OLR = np.percentile(df_OLR['normalized_mean'], 25)
IQR_OLR = q3_OLR - q1_OLR
upper_whisker_OLR = q3_OLR + (1.5 * IQR_OLR)
lower_whisker_OLR = q1_OLR - (1.5 * IQR_OLR)
# VRM
q3_VRM = np.percentile(df_VRM['normalized_mean'], 75)
q1_VRM = np.percentile(df_VRM['normalized_mean'], 25)
IQR_VRM = q3_VRM - q1_VRM
upper_whisker_VRM = q3_VRM + (1.5 * IQR_VRM)
lower_whisker_VRM = q1_VRM - (1.5 * IQR_VRM)
# MM1
q3_MM1 = np.percentile(df_MM1['normalized_mean'], 75)   
q1_MM1 = np.percentile(df_MM1['normalized_mean'], 25)
IQR_MM1 = q3_MM1 - q1_MM1
upper_whisker_MM1 = q3_MM1 + (1.5 * IQR_MM1)
lower_whisker_MM1 = q1_MM1 - (1.5 * IQR_MM1)
# MM2
q3_MM2 = np.percentile(df_MM2['normalized_mean'], 75)
q1_MM2 = np.percentile(df_MM2['normalized_mean'], 25)
IQR_MM2 = q3_MM2 - q1_MM2
upper_whisker_MM2 = q3_MM2 + (1.5 * IQR_MM2)
lower_whisker_MM2 = q1_MM2 - (1.5 * IQR_MM2)

print(df_OLR.loc[(df_OLR['normalized_mean'] < lower_whisker_OLR) | (df_OLR['normalized_mean'] > upper_whisker_OLR)])
print(df_VRM.loc[(df_VRM['normalized_mean'] < lower_whisker_VRM) | (df_VRM['normalized_mean'] > upper_whisker_VRM)])
print(df_MM1.loc[(df_MM1['normalized_mean'] < lower_whisker_MM1) | (df_MM1['normalized_mean'] > upper_whisker_MM1)])
print(df_MM2.loc[(df_MM2['normalized_mean'] < lower_whisker_MM2) | (df_MM2['normalized_mean'] > upper_whisker_MM2)])

#%%
# box plots r^2
#%% calculate R^2 for OLR and VRM
r2_OLR = []
r2_VRM = []

for k in range(35): 
    r2_OLR.append(r2_score(validate_df_OLR[k]['target_wind_speed'], validate_df_OLR[k]['longterm_target_estimate']))
    r2_VRM.append(r2_score(validate_df_VRM[k]['target_wind_speed'], validate_df_VRM[k]['longterm_target_estimate']))


df_r2_olr = pd.DataFrame({'r2_score': r2_OLR, 'method': 'OLR', 'key': keys})
df_r2_vrm = pd.DataFrame({'r2_score' : r2_VRM, 'method' : 'VRM', 'key' : keys})

df_r2 = pd.concat([df_r2_olr, df_r2_vrm], ignore_index = True)

plt.figure(figsize=(8,6))
ax = sns.boxplot(data=df_r2, x='method', y='r2_score')
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('R^2 score [-]', fontsize=12)
plt.title('R^2 score for prediction with OLR or VRM', fontsize=14)
plt.show()

print(df_r2_olr.loc[df_r2_olr['r2_score'] < 0.6])
print(df_r2_vrm.loc[df_r2_vrm['r2_score'] < 0.6])

print(np.sum(df_r2_olr['r2_score']/ len(df_r2_olr['r2_score'])))
print(np.sum(df_r2_vrm['r2_score']/ len(df_r2_vrm['r2_score'])))
print('median olr', np.median(df_r2_olr['r2_score']))
print('median olr', np.median(df_r2_vrm['r2_score']))

#%%
