# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:25:01 2023

@author: sylke
"""

import pandas as pd
from scipy import stats
import numpy as np
import math

import os
import glob
import natsort


# get mylist, validate_df and concurrent_df
folder_path_mylist = 'C:/folder/path/mylist'
folder_path_concurrent = 'C:/folder/path/concurrent'
folder_path_validate = 'C:/folder/path/validate'


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

E = []
for k in range(datasets):
    matrix = np.zeros((12,12), dtype = float)
    for i in range(12):
        target_label = sector_labels[i]
        for j in range(12):
            reference_label = sector_labels[j]
            count = np.count_nonzero((concurrent[k]['target_bin'] == target_label) & (concurrent[k]['bin'] == reference_label))
            matrix[i,j] = count
    E.append(matrix)

#%% create a 12 x 12 matrix W and Z (j x i) which rejects non-significant measurements
# W: percentage populations sum to 100% for each met sector (column = 100%)
# Z: percentage populations sum to 100% for each target sector (row = 100%)

#cut-off
delta = 5 #percent

#calculate included sum
includedsum = []
for k in range(datasets): 
    includedsum.append(E[k].sum(axis=1))

# exclude cut-off bins from E in Esignificant
Esignificant = []
for k in range(datasets): 
    matrix = np.zeros((12,12), dtype = float)
    for i in range(12):
        for j in range(12): 
            if (E[k][i,j] / includedsum[k][i] * 100 > delta) : 
                matrix[i,j] = E[k][i,j]
    Esignificant.append(matrix)

# create W : columns Esignificant = 100%
Wsectorsum = []
W = []
for k in range(datasets): 
    Wsectorsum.append(Esignificant[k].sum(axis=0))
    matrix = np.zeros((12,12), dtype = float)
    for i in range(12): 
        matrix[:,i] = Esignificant[k][:,i] / Wsectorsum[k][i] * 100
    W.append(matrix)

# create Z : rows Esignificant = 100%
Zsectorsum = []
Z = []
for k in range(datasets): 
    Zsectorsum.append(Esignificant[k].sum(axis=1))
    matrix = np.zeros((12,12), dtype = float)
    for i in range(12): 
        matrix[i,:] = Esignificant[k][i,:] / Zsectorsum[k][i] *100
    Z.append(matrix)

#%% option 1 derive mean wind speed from reference site sector mean wind speeds
# derive regression relations based on reference sectors in concurrent period
m_ref = []
c_ref = []
for k in range(datasets):
    # calculate the linear regression parameters for the met sectors i: m = slope, c = offset
    m = np.zeros(12)
    c = np.zeros(12)
    for i, label in enumerate(sector_labels):
        reference_data = concurrent[k].loc[concurrent[k]['bin'] == label]
        m[i] = stats.linregress(reference_data['ref_wind_speed'], reference_data['target_wind_speed'])[0]
        c[i] = stats.linregress(reference_data['ref_wind_speed'], reference_data['target_wind_speed'])[1]
    m_ref.append(m)
    c_ref.append(c)          
                              

#%% option 2 derive mean wind speed from target site sector mean wind speeds
# derive regression relations based on target sectors in concurrent period
    
m_target = []
c_target = []
for k in range(datasets):
    # calculate the linear regression parameters for the met sectors i: m = slope, c = offset
    m = np.zeros(12)
    c = np.zeros(12)
    for i, label in enumerate(sector_labels):
        target_data = concurrent[k].loc[concurrent[k]['target_bin'] == label]
        m[i] = stats.linregress(target_data['ref_wind_speed'], target_data['target_wind_speed'])[0]
        c[i] = stats.linregress(target_data['ref_wind_speed'], target_data['target_wind_speed'])[1]
    m_target.append(m)
    c_target.append(c)


#%%
v_ref_mean = []
for k in range(datasets): 
    mean_ref_wind_speed = np.zeros(12)
    for i, label in enumerate(sector_labels): 
        mean_ref_wind_speed[i] = np.mean(validate[k]['ref_wind_speed'].loc[validate[k]['bin'] == label])
    v_ref_mean.append(mean_ref_wind_speed)

# determine long term target wind speed method 1
v_target_mean_1 = []
step1 = []
step2 = []

for k in range(datasets): 
    v = np.zeros(12)
    for i in range(12): 
        v[i] = (m_ref[k][i]*v_ref_mean[k][i] + c_ref[k][i])
    step1.append(v)

for k in range(datasets): 
    v = np.zeros(12)
    for i in range(12): 
        v[i] = np.sum(step1[k] * Z[k][i])
    step2.append(v)
    v_target_mean_1.append(step2[k] / 100)

 # determine long term target windspeed method 2 
v_target_mean_2 = []
step1_method2 = []
step2_method2 = []
for k in range(datasets): 
    v = np.zeros(12)
    for i in range(12):
        v[i] = np.sum(v_ref_mean[k] * Z[k][i])
        #v[i] = m_target[k][i] * ((np.sum(Z[k][i,:] * v_ref_mean[k][i])) / 100) + c_target[k][i]
    step1_method2.append(v)
    step2_method2.append(step1_method2[k] / 100)

for k in range(datasets): 
    v = np.zeros(12)
    for i in range(12): 
        v[i] = m_target[k][i] * step2_method2[k][i] + c_target[k][i]
    v_target_mean_2.append(v)
    

actual_mean = []
for k in range(datasets): 
    mean_wind_speed = np.zeros(12)
    for i, label in enumerate(sector_labels): 
        mean_wind_speed[i] = np.mean(validate[k]['target_wind_speed'].loc[validate[k]['bin'] == label])
    actual_mean.append(mean_wind_speed)
        
#%%
#count reference population per sector
p_ref = []
for k in range(datasets): 
    count = np.zeros(12)
    for i, label in enumerate(sector_labels):
        count[i] = np.count_nonzero((validate[k]['bin'] == label))
    p_ref.append(count)
           
    
# determine sector populations at target
p_target = []
for k in range(datasets): 
    p = np.zeros(12)
    for i in range(12): 
        p[i] = np.sum(p_ref[k] * W[k][i]) / 100
    p_target.append(p)


p_act = []
for k in range(datasets): 
    count = np.zeros(12)
    for i, label in enumerate(sector_labels):
        count[i] = np.count_nonzero((validate[k]['target_bin'] == label))
    p_act.append(count)
    
p_act_conc = []
for k in range(datasets): 
    count = np.zeros(12)
    for i, label in enumerate(sector_labels):
        count[i] = np.count_nonzero((concurrent[k]['target_bin'] == label))
    p_act_conc.append(count)
    
p_ref_conc = []
for k in range(datasets): 
    count = np.zeros(12)
    for i, label in enumerate(sector_labels):
        count[i] = np.count_nonzero((concurrent[k]['bin'] == label))
    p_ref_conc.append(count)
#%%

import matplotlib.pyplot as plt

labels = ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180', '180-210', '210-240', '240-270', '270-300', '300-330', '330-360']

actual_target_wind_speed = actual_mean[0]
estimated_target_mean_opt1 = v_target_mean_1[0]
estimated_target_mean_opt2 = v_target_mean_2[0]

width_size = 10
height_size = 6
bins = 12


df = pd.DataFrame({'actual mean wind speed': actual_target_wind_speed, 'option 1: bin by reference dir': estimated_target_mean_opt1, 'option 2: bin by target dir': estimated_target_mean_opt2})
df.plot.bar(figsize=(width_size, height_size))

# Add labels and title
plt.xlabel('direction sector')
plt.ylabel('mean wind speed [-]')
plt.title('Mean per direction sector Matrix method: dyffryn brodyn')
positions = np.arange(len(labels))
plt.xticks(positions, labels, rotation='vertical')
plt.show()

#%%
MM_estimate = validate.copy()
import seaborn as sns
dataset = concurrent[0].loc[concurrent[0]['bin'] == '0-30']
plt.figure()
sns.regplot(data = dataset , x = 'ref_wind_speed', y = 'target_wind_speed')

