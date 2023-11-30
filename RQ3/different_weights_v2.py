# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 17:36:20 2023

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
from collections import Counter


folder_path = 'C:/folder/path/weights_v2'


csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
csv_files_sorted = natsort.natsorted(csv_files)


reconstructed = []
for csv_file in csv_files_sorted: 
    df = pd.read_csv(csv_file)
    reconstructed.append(df)

datasets = 35 * 10

#%%
# weight01 = reconstructed[0::10]
# weight02 = reconstructed[1::10]
# weight03 = reconstructed[2::10]
# weight04 = reconstructed[3::10]
# weight05 = reconstructed[4::10]
# weight06 = reconstructed[5::10]
# weight07 = reconstructed[6::10]
# weight08 = reconstructed[7::10]
# weight09 = reconstructed[8::10]
# weight1 = reconstructed[9::10]
r2_AE = np.zeros(datasets)
for k in range(datasets): 
    r2_AE[k] = (r2_score(reconstructed[k]['target_wind_speed'], reconstructed[k]['prediction']))
    
per_site = np.array_split(r2_AE, 35)

best_indices = np.zeros(35)
for k in range(35): 
    best_indices[k] = np.argmax(per_site[k])

weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
best_weights = np.zeros(35)
for k in range(35): 
    best_weights[k] = weights[int(best_indices[k])]

frequency = np.zeros(len(weights))
for i, weight in enumerate(weights): 
    count = 0
    for k in range(35): 
        if weight == best_weights[k]: 
            count = count + 1
    frequency[i] = count
#%% create bar plot of number of occurrances weights
str_weights = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
fig, ax = plt.subplots()
ax.bar(str_weights, frequency)
ax.set_xlabel('wind direction weight [-]')
ax.set_ylabel('Frequency [-]')
ax.set_title('Best wind direction weights frequency distribution')
#%% create bar plot of number of occurrances weights
w_simple = best_weights[0:7]
w_complex = best_weights[7:11]
w_coastal = best_weights[11:22]
w_offshore = best_weights[22:]

# simple counter
frequency_simple = np.zeros(len(weights))
for i, weight in enumerate(weights): 
    count = 0
    for k in range(len(w_simple)): 
        if weight == w_simple[k]: 
            count = count + 1
    frequency_simple[i] = count
# complex counter
frequency_complex = np.zeros(len(weights))
for i, weight in enumerate(weights): 
    count = 0
    for k in range(len(w_complex)): 
        if weight == w_complex[k]: 
            count = count + 1
    frequency_complex[i] = count
# coastal counter
frequency_coastal = np.zeros(len(weights))
for i, weight in enumerate(weights): 
    count = 0
    for k in range(len(w_coastal)): 
        if weight == w_coastal[k]: 
            count = count + 1
    frequency_coastal[i] = count
#offshore counter
frequency_offshore = np.zeros(len(weights))
for i, weight in enumerate(weights): 
    count = 0
    for k in range(len(w_offshore)): 
        if weight == w_offshore[k]: 
            count = count + 1
    frequency_offshore[i] = count

#%% terrain figures
fig, ax = plt.subplots()
ax.bar(str_weights, frequency_simple)
ax.set_xlabel('wind direction weight [-]')
ax.set_ylabel('Frequency [-]')
ax.set_title('Best wind direction weights frequency distribution Inland; Simple')

fig, ax = plt.subplots()
ax.bar(str_weights, frequency_complex)
ax.set_xlabel('wind direction weight [-]')
ax.set_ylabel('Frequency [-]')
ax.set_title('Best wind direction weights frequency distribution Inland; Complex')

fig, ax = plt.subplots()
ax.bar(str_weights, frequency_coastal)
ax.set_xlabel('wind direction weight [-]')
ax.set_ylabel('Frequency [-]')
ax.set_title('Best wind direction weights frequency distribution Coastal')

fig, ax = plt.subplots()
ax.bar(str_weights, frequency_offshore)
ax.set_xlabel('wind direction weight [-]')
ax.set_ylabel('Frequency [-]')
ax.set_title('Best wind direction weights frequency distribution Offshore')

#%%
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Plot for frequency_simple
axes[0, 0].bar(str_weights, frequency_simple)
axes[0, 0].set_xlabel('wind direction weight [-]')
axes[0, 0].set_ylabel('Frequency [-]')
axes[0, 0].set_title('Inland; Simple')
axes[0, 0].set_yticks([0, 1, 2, 3, 4, 5, 6])

# Plot for frequency_complex
axes[0, 1].bar(str_weights, frequency_complex)
axes[0, 1].set_xlabel('wind direction weight [-]')
axes[0, 1].set_ylabel('Frequency [-]')
axes[0, 1].set_title('Inland; Complex')
axes[0, 1].set_yticks([0, 1, 2, 3, 4, 5, 6])

# Plot for frequency_coastal
axes[1, 0].bar(str_weights, frequency_coastal)
axes[1, 0].set_xlabel('wind direction weight [-]')
axes[1, 0].set_ylabel('Frequency [-]')
axes[1, 0].set_title('Coastal')
axes[1, 0].set_yticks([0, 1, 2, 3, 4, 5, 6])

# Plot for frequency_offshore
axes[1, 1].bar(str_weights, frequency_offshore)
axes[1, 1].set_xlabel('wind direction weight [-]')
axes[1, 1].set_ylabel('Frequency [-]')
axes[1, 1].set_title('Offshore')
axes[1, 1].set_yticks([0, 1, 2, 3, 4, 5, 6])

fig.suptitle('Frequency Distributions of Best Wind Direction Weights per Terrain Type')
# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
#%%
print(np.mean(best_weights))

# df = pd.DataFrame({'weights': best_weights})
# df_sorted = df.sort_values(by='weights')
# fig, ax = plt.subplots()
# df['weights'].value_counts().sort_index().plot(ax=ax, kind='bar', xlabel='wind direction weight [-]', ylabel='Frequency', title='Best wind direction weights frequency distribution')

# #%% create bar plots for each terrain type
# df_simple = df[0:7]
# df_simple_sorted = df_simple.sort_values(by='weights')
# df_complex = df[7:11]
# df_complex_sorted = df_complex.sort_values(by='weights')
# df_coastal = df[11:22]
# df_coastal_sorted = df_coastal.sort_values(by='weights')
# df_offshore = df[22:]
# df_offshore_sorted = df_offshore.sort_values(by='weights')

# fig, ax = plt.subplots()
# df_simple['weights'].value_counts().sort_index().plot(ax =ax, kind='bar', xlabel='wind direction weight [-]', ylabel='Frequency')