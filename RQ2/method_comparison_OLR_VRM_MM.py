# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:32:53 2023

@author: sylke
"""

import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from sklearn.metrics import r2_score
import pandas as pd
import os
import glob
import natsort

folder_path_mcp = 'C:/folder/path/OLR_full'
folder_path_vrm = 'C:/folder/path/VRM_full'


csv_files_mcp = glob.glob(os.path.join(folder_path_mcp, '*.csv'))
csv_files_sorted_mcp = natsort.natsorted(csv_files_mcp)

validate_df_OLR = []
for csv_file in csv_files_sorted_mcp: 
    df = pd.read_csv(csv_file)
    df['ob_time'] = pd.to_datetime(df['ob_time'])
    validate_df_OLR.append(df)

csv_files_vrm = glob.glob(os.path.join(folder_path_vrm, '*.csv'))
csv_files_sorted_vrm = natsort.natsorted(csv_files_vrm)

validate_df_VRM = []
for csv_file in csv_files_sorted_vrm: 
    df = pd.read_csv(csv_file)
    df['ob_time'] = pd.to_datetime(df['ob_time'])
    validate_df_VRM.append(df)
    
folder_path_conc = 'C:/folder/path/concurrent_full'

csv_files_conc = glob.glob(os.path.join(folder_path_conc, '*.csv'))
csv_files_sorted_conc = natsort.natsorted(csv_files_conc)

concurrent_df = []
for csv_file in csv_files_sorted_conc: 
    df = pd.read_csv(csv_file)
    df['ob_time'] = pd.to_datetime(df['ob_time'])
    concurrent_df.append(df)

# import Ordinary_Linear_Regression_Estimate_v2 as OLR
# validate_df_OLR = OLR.OLR_estimate
# import Variance_Ratio_Method_Estimate_v2 as VRM
# validate_df_VRM = VRM.VRM_estimate
import Matrix_method_Estimate_v3 as MM
validate_df_MM = MM.MM_estimate

keys = ["UK1. Dyffryn Brodyn", "UK2. Lifton Down", "UK3. St. Breock", "NL1. Schiphol", "NL2. Westdorpe", "NL3. Hupsel", "NL4. Cabauw",
        "UK4. Penrhys", "UK5. Rheidol", "UK6. Alt-Yr-Hende", "NL5. Rotterdam Geulhaven", "UK7. Siddick", "UK8. Haverigg", "UK9. Treculliacks",
        "UK10. Rhyd-Y-Goes", "UK11. Hill of Forss", "UK12. Crimp", "UK13. Ysgubor", "UK14. Jordanston", "UK15. Truthan", "UK16. Carland Cross",
        "NL6. Platform AWG-1", "UK17. Celtic Array Zone 9", "UK18. Greater Gabbard", "UK19. Gunfleet Sands", "UK20. Gwynt Y Mor", "UK21. Shell Flats",
        "NL7. Lichteiland Goeree", "NL8. K14FA1C", "NL9. J6-A", "NL10. Borssele 1", "NL11. Hollandse Kust West (HKWA)", "NL12. Hollandse Kust Noord (HKNB)",
        "NL13. Ten Noorden van de Wadden (TNWB)", "NL14. Dogger Bank zone 3", "[on/off] NL10. Borssele 1", "[on/off] NL12. Hollandse Kust Noord (HKNB)", "[on/off] NL13. Ten Noorden van de Wadden (TNWB)", 
        "[on/off] UK17. Celtic Array Zone 9", "[on/off] UK18. Greater Gabbard", "[on/off] UK19. Gunfleet Sands","[on/off] UK20. Gwynt Y Mor", "[on/off] UK21. Shell Flats"]

datasets = 35

#%%

#%% calculate R^2 for OLR and VRM
r2_OLR = []
r2_VRM = []
r2_MM1 = []
r2score = defaultdict(list)
for k in range(datasets): 
    r2_OLR.append(r2_score(validate_df_OLR[k]['target_wind_speed'], validate_df_OLR[k]['longterm_target_estimate']))
    r2_VRM.append(r2_score(validate_df_VRM[k]['target_wind_speed'], validate_df_VRM[k]['longterm_target_estimate']))
    r2score[keys[k]].append(r2_OLR[k])
    r2score[keys[k]].append(r2_VRM[k])

#%%
labels = ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180', '180-210', '210-240', '240-270', '270-300', '300-330', '330-360']
# calculate sector counts actual target and set to percentage of total population
actual_sector_counts = []
total_actual_population = []
for k in range(datasets): 
    total_actual_population.append(validate_df_OLR[k]['target_bin'].count())
    sector_count = np.zeros(12)
    for i in range(12): 
        sector_count[i] = np.count_nonzero(validate_df_OLR[k]['target_bin'] == labels[i])
    actual_sector_counts.append(sector_count)

actual_sector_counts_percent = []
for k in range(datasets): 
    percent = np.zeros(12)
    for i in range(12): 
        percent[i] = actual_sector_counts[k][i] / total_actual_population[k] * 100
    actual_sector_counts_percent.append(np.round(percent, 2))
    
# calculate sector counts OLR and set to percentage of total population
OLR_sector_counts = []
total_population_OLR = []
OLR_sector_counts_percent = []
for k in range(datasets): 
    total_population_OLR.append(validate_df_OLR[k]['bin'].count())
    sector_count = np.zeros(12)
    for i in range(12): 
        sector_count[i] = np.count_nonzero(validate_df_OLR[k]['bin'] == labels[i])
    OLR_sector_counts.append(sector_count)
    OLR_sector_counts_percent.append(np.round(OLR_sector_counts[k] / total_population_OLR[k] * 100, 2))

    
# calculate sector counts VRM and set to percentage of total population
# VRM uses target bin as reference bins, but in the validation period target bin 'is not known'
VRM_sector_counts = []
total_population_VRM = []
for k in range(datasets): 
    total_population_VRM.append(validate_df_VRM[k]['bin'].count())
    sector_count = np.zeros(12)
    for i in range(12): 
        sector_count[i] = np.count_nonzero(validate_df_VRM[k]['bin'] == labels[i])
    VRM_sector_counts.append(sector_count)

VRM_sector_counts_percent = []
for k in range(datasets): 
    percent = np.zeros(12)
    for i in range(12): 
        percent[i] = VRM_sector_counts[k][i] / total_population_VRM[k] * 100
    VRM_sector_counts_percent.append(np.round(percent, 2))

# calculate sector counts MM and set to percentage of total population
MM_sector_counts = MM.p_target
total_population_MM = []
MM_sector_counts_percent = []
for k in range(datasets): 
    total_population_MM.append(validate_df_MM[k]['bin'].count())
    percent = np.zeros(12)
    for i in range(12): 
        percent [i] = MM_sector_counts[k][i] / total_population_MM[k] * 100
    MM_sector_counts_percent.append(np.round(percent,2))


    
#%% WEIGH OVERALL MEAN PER SECTOR: SUM (SECTOR POPULATION [%] * SECTOR MEAN) = OVERALL MEAN 
# actual target sector mean and overall actual target mean
sector_mean_actual = []
mean_actual = []
normalized_mean_actual = []
for k in range(datasets): 
    mean = np.zeros(12)
    for i, label in enumerate(labels):
        mean[i] = np.mean(validate_df_OLR[k]['target_wind_speed'].loc[validate_df_OLR[k]['bin'] == label])
    sector_mean_actual.append(np.round(mean,2))
    # mean_actual.append(np.round(np.mean(validate_df_MM[k]['target_wind_speed']),2))
    mean_actual.append(np.sum(actual_sector_counts_percent[k] / 100 * sector_mean_actual[k]))
    normalized_mean_actual.append(mean_actual[k] / mean_actual[k])

# calculate sector mean OLR and normalized sector mean OLR
sector_mean_OLR = []
mean_OLR = []
normalized_mean_OLR = []
for k in range(datasets): 
    mean = np.zeros(12)
    for i, label in enumerate(labels):
        mean[i] = np.mean(validate_df_OLR[k]['longterm_target_estimate'].loc[validate_df_OLR[k]['bin'] == label])
    sector_mean_OLR.append(mean)
    # mean_OLR.append(np.round(np.mean(validate_df_OLR[k]['longterm_target_estimate']),2))
    mean_OLR.append(np.sum(OLR_sector_counts_percent[k] / 100 * sector_mean_OLR[k]))
    normalized_mean_OLR.append(mean_OLR[k] / mean_actual[k])
  
# calculate sector mean VRM and normalized sector mean VRM    
sector_mean_VRM = []
mean_VRM = []
normalized_mean_VRM = []
for k in range(datasets): 
    mean = np.zeros(12)
    for i, label in enumerate(labels):
        mean[i] = np.mean(validate_df_VRM[k]['longterm_target_estimate'].loc[validate_df_VRM[k]['bin'] == label])
    sector_mean_VRM.append(mean)
    # mean_VRM.append(np.round(np.mean(validate_df_VRM[k]['longterm_target_estimate']),2))
    mean_VRM.append(np.sum(VRM_sector_counts_percent[k] / 100 * sector_mean_VRM[k]))
    normalized_mean_VRM.append(mean_VRM[k] / mean_actual[k])
    
# calculate sector mean MM and normalized sector mean MM, MM has to options to predict the mean
sector_mean_MM_opt1 = MM.v_target_mean_1
sector_mean_MM_opt2 = MM.v_target_mean_2
mean_MM_opt1 = []
mean_MM_opt2 = []
normalized_mean_MM_opt1 = []
normalized_mean_MM_opt2 = []

for k in range(datasets): 
    # mean_MM_opt1.append(np.round(np.mean(sector_mean_MM_opt1[k]),2))
    # mean_MM_opt2.append(np.round(np.mean(sector_mean_MM_opt2[k]),2))
    mean_MM_opt1.append(np.sum(MM_sector_counts_percent[k] / 100 * sector_mean_MM_opt1[k]))
    mean_MM_opt2.append(np.sum(MM_sector_counts_percent[k] / 100 * sector_mean_MM_opt2[k]))
    normalized_mean_MM_opt1.append(mean_MM_opt1[k] / mean_actual[k])
    normalized_mean_MM_opt2.append(mean_MM_opt2[k] / mean_actual[k])


#%%
# create tables

# table 1: overview of all 35 sites: overall mean of all methods
overall_mean = defaultdict(list)
for i in range(datasets): 
    overall_mean[keys[i]].append(mean_actual[i])
    overall_mean[keys[i]].append(mean_OLR[i])
    overall_mean[keys[i]].append(mean_VRM[i])
    overall_mean[keys[i]].append(mean_MM_opt1[i])
    overall_mean[keys[i]].append(mean_MM_opt2[i])

# create a df from table 1
overall_mean = []
for k in range(datasets): 
    table = pd.DataFrame(0, index=range(1), columns = ['actual', 'OLR', 'VRM', 'MM opt1', 'MM opt2'])
    overall_mean.append(table)
    overall_mean[k]['actual'] = mean_actual[k]
    overall_mean[k]['OLR'] = mean_OLR[k]
    overall_mean[k]['VRM'] = mean_VRM[k]
    overall_mean[k]['MM opt1'] = mean_MM_opt1[k]
    overall_mean[k]['MM opt2'] = mean_MM_opt2[k]

#table 2: overview of all 35 sites: normalized overall mean of all methods
normalized_overall_mean = defaultdict(list)
for i in range(datasets): 
    normalized_overall_mean[keys[i]].append(mean_actual[i] / mean_actual[i])
    normalized_overall_mean[keys[i]].append(mean_OLR[i] / mean_actual[i])
    normalized_overall_mean[keys[i]].append(mean_VRM[i] / mean_actual[i])
    normalized_overall_mean[keys[i]].append(mean_MM_opt1[i] / mean_actual[i])
    normalized_overall_mean[keys[i]].append(mean_MM_opt2[i] / mean_actual[i])
    
# create a df from table 2
normalized_overall_mean = []
for k in range(datasets): 
    table = pd.DataFrame(0, index=range(1), columns = ['actual', 'OLR', 'VRM', 'MM opt1', 'MM opt2'])
    normalized_overall_mean.append(table)
    normalized_overall_mean[k]['actual'] = mean_actual[k]  / mean_actual[k]
    normalized_overall_mean[k]['OLR'] = mean_OLR[k]  / mean_actual[k]
    normalized_overall_mean[k]['VRM'] = mean_VRM[k]  / mean_actual[k]
    normalized_overall_mean[k]['MM opt1'] = mean_MM_opt1[k]  / mean_actual[k]
    normalized_overall_mean[k]['MM opt2'] = mean_MM_opt2[k]  / mean_actual[k]    

#table 3: overview per sector: % of population per sector and sector mean for all methods
overview = []
for k in range(datasets): 
    table = pd.DataFrame(0, index=range(12), columns = ['sector', 'act. pop', 'act. u', 'OLR pop', 'OLR u', 'VRM pop', 'VRM u', 'MM pop', 'MM u opt1', 'MM u opt2'])
    overview.append(table)
    overview[k]['sector'] = list(range(1,13))
    overview[k]['act. pop'] = actual_sector_counts_percent[k]
    overview[k]['act. u'] = sector_mean_actual[k]
    overview[k]['OLR pop'] = OLR_sector_counts_percent[k]
    overview[k]['OLR u'] = sector_mean_OLR[k]
    overview[k]['VRM pop'] = VRM_sector_counts_percent[k]
    overview[k]['VRM u'] = sector_mean_VRM[k]
    overview[k]['MM pop'] = MM_sector_counts_percent[k]
    overview[k]['MM u opt1'] = sector_mean_MM_opt1[k]
    overview[k]['MM u opt2'] = sector_mean_MM_opt2[k]
    
#table 3: overview per sector: % of population per sector and normalized sector mean for all methods
normalized_overview = []
for k in range(datasets): 
    table = pd.DataFrame(0, index=range(12), columns = ['sector', 'act. pop', 'act. u', 'OLR pop', 'OLR u', 'VRM pop', 'VRM u', 'MM pop', 'MM u opt1', 'MM u opt2'])
    normalized_overview.append(table)
    normalized_overview[k]['sector'] = list(range(1,13))
    normalized_overview[k]['act. pop'] = actual_sector_counts_percent[k]
    normalized_overview[k]['act. u'] = sector_mean_actual[k] / sector_mean_actual[k]
    normalized_overview[k]['OLR pop'] = OLR_sector_counts_percent[k]
    normalized_overview[k]['OLR u'] = sector_mean_OLR[k] / sector_mean_actual[k]
    normalized_overview[k]['VRM pop'] = VRM_sector_counts_percent[k]
    normalized_overview[k]['VRM u'] = sector_mean_VRM[k] / sector_mean_actual[k]
    normalized_overview[k]['MM pop'] = MM_sector_counts_percent[k]
    normalized_overview[k]['MM u opt1'] = sector_mean_MM_opt1[k] / sector_mean_actual[k]
    normalized_overview[k]['MM u opt2'] = sector_mean_MM_opt2[k] / sector_mean_actual[k]

normalized_sector_mean_actual = []
normalized_sector_mean_OLR = []
normalized_sector_mean_VRM = []
normalized_sector_mean_MM1 = []
normalized_sector_mean_MM2 = []
for k in range(datasets): 
    actual = np.zeros(12)
    OLR = np.zeros(12)
    VRM = np.zeros(12)
    MM1 = np.zeros(12)
    MM2 = np.zeros(12)
    for i in range(12): 
        actual[i] = sector_mean_actual[k][i] / sector_mean_actual[k][i]
        OLR[i] = sector_mean_OLR[k][i] / sector_mean_actual[k][i]
        VRM[i] = sector_mean_VRM[k][i] / sector_mean_actual[k][i]
        MM1[i] = sector_mean_MM_opt1[k][i] / sector_mean_actual[k][i]
        MM2[i] = sector_mean_MM_opt2[k][i] / sector_mean_actual[k][i]
    normalized_sector_mean_actual.append(actual)
    normalized_sector_mean_OLR.append(OLR)
    normalized_sector_mean_VRM.append(VRM)
    normalized_sector_mean_MM1.append(MM1)
    normalized_sector_mean_MM2.append(MM2)
#%% create some figures
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
width_size = 10
height_size = 6
bins = 12
# Define a custom greyscale colormap
colors = [(0, 0, 0), (0.2, 0.2, 0.2), (0.4, 0.4, 0.4), (0.6, 0.6, 0.6), (0.8, 0.8, 0.8)]
cmap_name = 'custom_greyscale'
cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=5)

#%%
# for k in range(datasets): 
#     df = overview[k][['act. u', 'OLR u', 'VRM u', 'MM u opt1', 'MM u opt2']]
#     ax = df.plot.bar(figsize=(width_size, height_size), colormap= cm)
    
#     plt.xlabel('sectors')
#     plt.ylabel('mean wind speed [m/s]')
#     plt.title(keys[k])
#     positions = np.arange(len(labels))
#     plt.xticks(positions, labels, rotation='vertical')
#     ax.legend(loc='lower left')
#     plt.show()

#%%
# test = overview[0][['act. u', 'OLR u', 'VRM u', 'MM u opt1', 'MM u opt2']]
# ax = test.plot.bar(figsize=(width_size, height_size), colormap= cm)
# plt.xlabel('sectors')
# plt.ylabel('mean wind speed [m/s]')
# plt.title(keys[0])
# positions = np.arange(len(labels))
# plt.xticks(positions, labels, rotation='vertical')
# ax.legend(loc='lower left')
# plt.show()
# #%%

# for k in range(datasets):
#     values = overall_mean[k][['actual', 'OLR', 'VRM', 'MM opt1', 'MM opt2']].values[0]
#     methods = ['actual', 'OLR', 'VRM', 'MM opt1', 'MM opt2']

#     fig, ax = plt.subplots(figsize=(width_size, height_size))

#     positions = np.arange(len(values))  # Generate positions for each method
#     colors = [cm(i) for i in np.linspace(0, 1, len(values))]
#     ax.bar(positions, values, width=0.4, align='center', color = colors)

#     ax.set_xlabel('Methods')
#     ax.set_ylabel('weighted mean wind speed [m/s]')
#     ax.set_title(keys[k])
#     ax.set_xticks(positions)
#     ax.set_xticklabels(methods, rotation='vertical')

#     plt.tight_layout()
#     plt.show()


# #%% 35 plots of normalized overall mean for each method

# for k in range(datasets):
#     values = normalized_overall_mean[k][['actual', 'OLR', 'VRM', 'MM opt1', 'MM opt2']].values[0]
#     methods = ['actual', 'OLR', 'VRM', 'MM opt1', 'MM opt2']

#     fig, ax = plt.subplots(figsize=(width_size, height_size))
#     plt.axhline(y = 1, color = 'r', linestyle = '-')
#     positions = np.arange(len(values))  # Generate positions for each method
#     colors = [cm(i) for i in np.linspace(0, 1, len(values))]
#     ax.bar(positions, values, width=0.4, align='center', color = colors)

#     ax.set_xlabel('Methods')
#     ax.set_ylabel('weighted normalized mean wind speed [-]')
#     ax.set_title(keys[k])
#     ax.set_xticks(positions)
#     ax.set_xticklabels(methods, rotation='vertical')

#     plt.tight_layout()
#     plt.show()

#%% 35 plots of normalized sector means for each method

# for k in range(datasets): 
#     df = normalized_overview[k][['act. u', 'OLR u', 'VRM u', 'MM u opt1', 'MM u opt2']]
#     ax = df.plot.bar(figsize=(width_size, height_size), colormap= cm)
#     plt.axhline(y = 1, color = 'r', linestyle = '-')
    
#     plt.xlabel('sectors')
#     plt.ylabel('normalized mean wind speed [-]')
#     plt.title(keys[k])
#     positions = np.arange(len(labels))
#     #ax.set_prop_cycle('color', cm.tab20.colors) 
#     plt.xticks(positions, labels, rotation='vertical')
#     ax.legend(loc='lower left')
#     plt.show()
    
#%%
# test figure
# import matplotlib.pyplot as plt
# width_size = 10
# height_size = 6
# bins = 12
 
# df = overview[21][['act. u', 'OLR u', 'VRM u', 'MM u opt1', 'MM u opt2']]
# df.plot.bar(figsize=(width_size, height_size))

# plt.xlabel('sectors')
# plt.ylabel('mean wind speed [m/s]')
# plt.title(keys[21])
# positions = np.arange(len(labels))
# plt.xticks(positions, labels, rotation='vertical')
# plt.show()

#%%
# determine relationship between number of datapoints in the sector and the normalized sector wind speed VRM
sector_labels = ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180', '180-210', '210-240', '240-270', '270-300', '300-330', '330-360']
datapoints_concurrent = []
datapoints_validation = []
datapoints_concurrent_sector = []
datapoints_validate_sector = []
for k in range(datasets): 
    datapoints_concurrent.append(concurrent_df[k]['target_wind_speed'].count())
    datapoints_validation.append(validate_df_VRM[k]['target_wind_speed'].count())
    sector = np.zeros(12)
    val_sector = np.zeros(12)
    for i, label in enumerate(sector_labels):
        count = concurrent_df[k][concurrent_df[k]['target_bin'] == label]['target_wind_speed'].count()
        count_validate = validate_df_VRM[k][validate_df_VRM[k]['target_bin'] == label]['target_wind_speed'].count()
        sector[i] = count
        val_sector[i] = count_validate
    datapoints_concurrent_sector.append(sector)
    datapoints_validate_sector.append(val_sector)

normalized_sector_mean_VRM_check = normalized_sector_mean_VRM.copy()[:35]
normalized_mean_VRM_check = normalized_mean_VRM.copy()[:35]

plt.figure()
plt.scatter(datapoints_concurrent, normalized_mean_VRM_check)


datapoints_conc_sector = np.concatenate(datapoints_concurrent_sector, axis=0)
sector_mean_list = np.concatenate(normalized_sector_mean_VRM_check, axis=0)

plt.figure()
plt.scatter(datapoints_conc_sector, sector_mean_list)
# plt.xlim(0, 2500)
plt.ylim(0, 1.8)
plt.xlim(0, 2500)
plt.title('VRM number of datapoints vs. normalized sector mean')
plt.xlabel('Number of datapoints in a sector [-]')
plt.ylabel('Normalized sector mean [-]')

#%% same for olr
datapoints_concurrent = np.zeros(datasets)
datapoints_validation = np.zeros(datasets)
for k in range(datasets):
    datapoints_concurrent[k] = len(concurrent_df[k])
    datapoints_validation[k] = len(validate_df_OLR[k])
normalized_sector_mean_OLR= normalized_sector_mean_OLR.copy()[:35]
normalized_mean_OLR_check = normalized_mean_OLR.copy()[:35]

plt.figure()
plt.scatter(datapoints_concurrent, normalized_mean_OLR_check)


# datapoints_conc_sector_olr = np.concatenate(datapoints_concurrent_sector_OLR, axis=0)
sector_mean_list_olr = np.concatenate(normalized_sector_mean_OLR, axis=0)

plt.figure()
plt.scatter(datapoints_conc_sector, sector_mean_list_olr)
# plt.xlim(0, 2500)
plt.ylim(0, 1.8)
plt.xlim(0, 2500)
plt.title('OLR number of datapoints vs. normalized sector mean')
plt.xlabel('Number of datapoints in a sector [-]')
plt.ylabel('Normalized sector mean [-]')

#%% check bias in boxplot
import seaborn as sns

actual =[]
vrm_prediction = []
olr_prediction = []
mbe_vrm = []
mbe_olr = []
for k in range(35): 
    actual.append(validate_df_OLR[k]['target_wind_speed'].copy())
    vrm_prediction.append(validate_df_VRM[k]['longterm_target_estimate'].copy())
    olr_prediction.append(validate_df_OLR[k]['longterm_target_estimate'].copy())
    mbe_vrm.append(np.mean(vrm_prediction[k] - actual[k]))
    mbe_olr.append(np.mean(olr_prediction[k] - actual[k] ))
  
df_mbe_vrm = pd.DataFrame({'MBE [m/s]' : mbe_vrm, 'Method': 'VRM'})
df_mbe_olr = pd.DataFrame({'MBE [m/s]': mbe_olr, 'Method': 'OLR'})
    
df_mbe = pd.concat([df_mbe_olr, df_mbe_vrm], ignore_index = True)

plt.figure()
sns.boxplot(data = df_mbe, x = 'Method', y = 'MBE [m/s]')
plt.axhline(y=0, color='red', linestyle='--', label='y = 0')
plt.title('Mean Bias Error in predicted values')

#%% check bias in sector mean boxplot
actual_sector = np.concatenate(sector_mean_actual.copy())
vrm_prediction = np.concatenate(sector_mean_VRM.copy())
olr_prediction = np.concatenate(sector_mean_OLR.copy())
mbe_vrm =  vrm_prediction - actual_sector
mbe_olr = olr_prediction - actual_sector
mbe_mm1 = np.concatenate(sector_mean_MM_opt1.copy()) - actual_sector 
mbe_mm2 = np.concatenate(sector_mean_MM_opt2.copy()) - actual_sector 
  
df_mbe_vrm = pd.DataFrame({'Bias [m/s]' : mbe_vrm, 'Method': 'VRM'})
df_mbe_olr = pd.DataFrame({'Bias [m/s]': mbe_olr, 'Method': 'OLR'})
df_mbe_mm2 = pd.DataFrame({'Bias [m/s]':mbe_mm2, 'Method': 'MM opt 2'})
df_mbe_mm1 = pd.DataFrame({'Bias [m/s]':mbe_mm1, 'Method': 'MM opt 1'})
df_mbe = pd.concat([df_mbe_olr, df_mbe_vrm], ignore_index = True)

plt.figure()
sns.boxplot(data = df_mbe, x = 'Method', y = 'Bias [m/s]')
plt.axhline(y=0, color='red', linestyle='--', label='y = 0')
plt.title('Bias in predicted sector mean')

#%% check bias overall mean boxplot
actual_overall = np.array(mean_actual.copy())
vrm_overall = mean_VRM.copy()
olr_overall = mean_OLR.copy()

bias_vrm = vrm_overall - actual_overall
bias_olr = olr_overall - actual_overall
bias_mm1 = mean_MM_opt1 - actual_overall
bias_mm2 = mean_MM_opt2 - actual_overall

df_mbe_vrm = pd.DataFrame({'Bias [m/s]' : bias_vrm, 'Method': 'VRM'})
df_mbe_olr = pd.DataFrame({'Bias [m/s]': bias_olr, 'Method': 'OLR'})
df_mbe_mm1 = pd.DataFrame({'Bias [m/s]': bias_mm1, 'Method': 'MM opt 1'})
df_mbe_mm2 = pd.DataFrame({'Bias [m/s]': bias_mm2, 'Method': 'MM opt 2'})
    
df_mbe = pd.concat([df_mbe_olr, df_mbe_vrm, df_mbe_mm1, df_mbe_mm2], ignore_index = True)

plt.figure()
sns.boxplot(data = df_mbe, x = 'Method', y = 'Bias [m/s]')
plt.axhline(y=0, color='red', linestyle='--', label='y = 0')
plt.title('Bias in predicted overall mean')

#%%
#wind direction sectors populations
#table 3: overview per sector: % of population per sector and sector mean for all methods


populations = []
for k in range(datasets): 
    table = pd.DataFrame(0, index=range(12), columns = ['sector', 'act. pop', 'OLR pop', 'MM pop'])
    populations.append(table)
    populations[k]['sector'] = list(range(1,13))
    populations[k]['act. pop'] = actual_sector_counts_percent[k]
    populations[k]['OLR pop'] = OLR_sector_counts_percent[k]
    populations[k]['MM pop'] = MM_sector_counts_percent[k]
    
#normalized sector counts
norm_populations = []
for k in range(datasets): 
    table = pd.DataFrame(0, index=range(1, 13), columns=['sector', 'act. pop', 'OLR pop', 'MM pop'])
    norm_populations.append(table)
    norm_populations[k]['sector'] = list(range(1, 13))
    
    for i in range(1, 13):     
        sector = actual_sector_counts[k][i - 1]
        OLR_sector = OLR_sector_counts[k][i - 1]
        MM_sector = MM_sector_counts[k][i - 1]
        
        norm_populations[k].loc[i, 'act. pop'] = sector / sector
        norm_populations[k].loc[i, 'OLR pop'] = OLR_sector / sector
        norm_populations[k].loc[i, 'MM pop'] = MM_sector / sector
        
        if (OLR_sector / sector) > 10: 
            print(k)
                
del norm_populations[1]
del populations[1]

all_data = pd.concat(populations, ignore_index=True)  
indices_to_concat = [8, 11, 15, 19, 26]
# all_data = pd.concat([populations[i] for i in indices_to_concat], ignore_index=True)

sector_data = all_data[['act. pop', 'OLR pop', 'MM pop']]  

# complex_sites = pd.concat(populations[7:16], ignore_index=True)

diff_olr_act = abs(sector_data['OLR pop'] - sector_data['act. pop'])
diff_mm_act = abs(sector_data['MM pop'] - sector_data['act. pop'])

# Check how many times OLR pop is closer to act. pop than MM pop
count_closer_olr = (diff_olr_act < diff_mm_act).sum()
count_closer_mm = (diff_mm_act < diff_olr_act).sum()
count_equal = (diff_mm_act == diff_olr_act).sum()

print(f'The count of rows where OLR pop is closer to act. pop than MM pop: {count_closer_olr} out of ', len(sector_data))
print(f'The count of rows where MM pop is closer to act. pop than OLR pop: {count_closer_mm} out of ', len(sector_data))
print(f'equal: {count_equal} out of ', len(sector_data))
#%%
from scipy import stats

best_MM_indices = []
worst_MM_indices = []
number_best_mm_sectors = []
for k in range(datasets-1): 
    distribution = populations[k]
    actual_sectors = distribution['act. pop'].to_numpy()
    olr_sectors = distribution['OLR pop'].to_numpy()
    mm_sectors = distribution['MM pop'].to_numpy()
    diff_olr_act = abs(actual_sectors - olr_sectors)
    diff_mm_act = abs(actual_sectors - mm_sectors)
    best_mm = 0
    for i in range(12): 
        if diff_olr_act[i] > diff_mm_act[i]: 
            best_mm = best_mm + 1
    number_best_mm_sectors.append(best_mm)
    if best_mm >= 6: 
        best_MM_indices.append(k)
    if best_mm < 6: 
        worst_MM_indices.append(k)
best_MM_indices = np.array(best_MM_indices)
number_best_mm_sectors = np.array(number_best_mm_sectors)
best_mm_indices_sectors = number_best_mm_sectors[best_MM_indices]
print(best_mm_indices_sectors)
print(best_MM_indices)
pearson_wind_dir = np.zeros(datasets)
pearson_wind_speed = np.zeros(datasets)
for k in range(datasets): 
    pearson_wind_dir[k] = (stats.linregress(concurrent_df[k]['target_wind_direction'], concurrent_df[k]['ref_wind_direction'])[2])
    pearson_wind_speed[k] = (stats.linregress(concurrent_df[k]['target_wind_speed'], concurrent_df[k]['ref_wind_speed'])[2])

idx = np.argpartition(pearson_wind_speed, 17)
print(idx)

MM_pearson = pearson_wind_dir[best_MM_indices]
print(MM_pearson)

worst_mm_pearson = pearson_wind_dir[np.array(worst_MM_indices)]
print(worst_mm_pearson)

sorted_pearson_wind_dir = np.sort(pearson_wind_dir)
#%%
#check
awg = overview[21]
diff_act_olr_pop = np.zeros(12)
diff_act_mm_pop = np.zeros(12)
for i in range(12): 
    diff_act_olr_pop[i] = abs(awg['act. pop'][i] - awg['OLR pop'][i])
    diff_act_mm_pop[i] = abs(awg['act. pop'][i] - awg['MM pop'][i])
    
print(np.sum(diff_act_olr_pop) / 12)
print(np.sum(diff_act_mm_pop) / 12)



#%% find best method for each site based on overall_mean
# best_methods = []
# for k in range(datasets): 
#     best = overall_mean[k].idxmax(axis=1)
#     best_methods.append(str(best[0]))
df = overall_mean.copy()
best_methods = []
for k in range(datasets): 
    actual_value = df[k]['actual'][0]
    differences = df[k].drop(columns=['actual']).apply(lambda x: abs(x - actual_value))

    # Find the column with the minimum absolute difference
    closest_column = differences.idxmin(axis=1).values[0]
    best_methods.append(closest_column)
    
#%% find average difference between prediction mm1 and mm2
diff = np.zeros(35)
for k in range(35): 
    diff[k] = abs(mean_MM_opt1[k] - mean_MM_opt2[k])

print(np.mean(diff))

#%%
all_means = pd.DataFrame()
for k in range(datasets): 
    all_means = pd.concat([all_means, overall_mean[k]], ignore_index=True)

all_means = round(all_means,2)

#%%
pearson_conc = np.zeros(datasets)
for k in range(datasets): 
    pearson_conc[k] = (stats.linregress(concurrent_df[k]['target_wind_speed'], concurrent_df[k]['ref_wind_speed'])[2])
# lowest 17 pearson indices
idx = np.argsort(pearson_conc)
mm1 = np.array(mean_MM_opt1)
mm2 = np.array(mean_MM_opt2)
actual = np.array(mean_actual)
mm1_lowest_r = mm1[idx[:35]]
mm2_lowest_r = mm2[idx[:35]]
actual_lowest_r = actual[idx[:35]]
diff_mm1 = abs(actual_lowest_r - mm1_lowest_r)
diff_mm2 = abs(actual_lowest_r - mm2_lowest_r)
best_mm1 = 0
best_mm2 = 0  
for k in range(len(mm1_lowest_r)):
    if diff_mm1[k] < diff_mm2[k]: 
        best_mm1 = best_mm1 + 1 
    if diff_mm1[k] > diff_mm2[k]: 
        best_mm2 = best_mm2 + 1 
print(best_mm1, best_mm2, len(mm1_lowest_r))
print(np.mean(pearson_conc))

pearson_rounded = np.round(pearson_conc, 2)