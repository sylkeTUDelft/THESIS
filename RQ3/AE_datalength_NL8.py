# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:42:13 2023

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
folder_path_olr = 'C:/folder/path/OLR_full'

csv_files_mylist = glob.glob(os.path.join(folder_path_mylist, '*.csv'))
csv_files_sorted_mylist = natsort.natsorted(csv_files_mylist)

csv_files_concurrent = glob.glob(os.path.join(folder_path_concurrent, '*.csv'))
csv_files_sorted_concurrent = natsort.natsorted(csv_files_concurrent)

csv_files_validate = glob.glob(os.path.join(folder_path_validate, '*.csv'))
csv_files_sorted_validate = natsort.natsorted(csv_files_validate)

csv_files_olr = glob.glob(os.path.join(folder_path_olr, '*.csv'))
csv_files_sorted_olr = natsort.natsorted(csv_files_olr)

mylist = []
for csv_file in csv_files_sorted_mylist: 
    df = pd.read_csv(csv_file)
    df['ob_time'] = pd.to_datetime(df['ob_time'])
    mylist.append(df)
concurrent_df = []
for csv_file in csv_files_sorted_concurrent: 
    df = pd.read_csv(csv_file)
    df['ob_time'] = pd.to_datetime(df['ob_time'])
    concurrent_df.append(df)
validate_df = []
for csv_file in csv_files_sorted_validate: 
    df = pd.read_csv(csv_file)
    df['ob_time'] = pd.to_datetime(df['ob_time'])
    validate_df.append(df)
OLR_all = []
for csv_file in csv_files_sorted_olr: 
    df = pd.read_csv(csv_file, usecols = ['ob_time','longterm_target_estimate'])
    df['ob_time'] = pd.to_datetime(df['ob_time'])
    OLR_all.append(df)

#%%
# use only one dataset
datasets = 43
# start by merging full reference and target dataframes
count = []
df = []
for k in range(datasets): 
    # merge reference and target
    full_set = pd.merge(mylist[k], mylist[k + datasets], on = 'ob_time', how = 'inner')
    # count amount of datapoints for each set
    count.append(len(full_set))
    # append full sets to a list
    df.append(full_set)
#%%
print('only computing for 30')
# validate = [validate_df[28]].copy() #highest historical pearson correlation
# concurrent = [concurrent_df[28]].copy()
OLR = [OLR_all[30]].copy()
datasets = 24

n_rows = np.linspace(720, 8052, num=24, dtype=int)
concurrent = [df[30].iloc[:n_row] for n_row in n_rows]
validate = [df[30][n_rows[23]:] for _ in range(24)]
# print('computing for normal datasets')
# number of datasets to run
# datasets = 35


# set weights
w_wind_speed = 1
w_wind_direction = 1

# find standard deviation wind speed and wind direction
std_wind_speed = []
std_wind_direction = []
for k in range(datasets): 
    std_wind_speed.append(np.std(validate[k]['ref_wind_speed']))
    std_wind_direction.append(np.std(validate[k]['ref_wind_direction']))


#%%
analog_indices = []
analogs = []
predictions = []
ob_time_predictions = []
indices_v6 = []
wind_speed_trend = []
wind_direction_trend = []
predicted_ob_time = []
for k in range(datasets): 
    # print start new k
    print('indices: start new dataset: ', k)
    # clear dataset lists for each iteration
    indices_dataset = []
    wind_speed_dataset = []
    wind_direction_dataset = []
    predicted_hours_dataset = []
    # select the kth dataset
    df = validate[k]
    # find the number of hours in the concurrent period
    training = concurrent[k]
    training_ob_time = training['ob_time'].to_list()
    training_hours = [hour.hour for hour in training_ob_time]
    # create new variables
    best_analogs = []
    training_best = []
    predictions_dataset = []
    predicted_hours_dataset = []
    # create analog trend window 
    for window in df.rolling(window=5, center = True):
        # check how far we are
        # if window.index.start%500 == 0:
            # print(f'dataset = {k}, window = {window.index.start}')
            
        # window should always contain 5 elements: t + - 2 hours
        if window.shape[0] == 5: 
            # select wind speed, wind direction and times for analog trend window
            wind_speed = window['ref_wind_speed'].values
            wind_direction = window['ref_wind_direction'].values
            # t + - 2 hours: ob_time[2] == hour t
            ob_time = window['ob_time'].tolist()
            #determine the hours present in ob_time
            ob_time_hours = {analog_trend_hour.hour for analog_trend_hour in ob_time}
            # find the indices of the analog search window
            # indices_dataset = []
            indices_hour = np.zeros(len(training_hours) - 3, dtype=int)
            index = 0
            for i in range(3, len(training_hours) - 3):  
                if training_hours[i] in ob_time_hours : 
                    indices_hour[index] = i
                    index += 1
            indices_dataset.append(indices_hour[indices_hour != 0])
            wind_speed_dataset.append(wind_speed)
            wind_direction_dataset.append(wind_direction)
            predicted_hours_dataset.append(ob_time[2])
    wind_speed_trend.append(wind_speed_dataset)
    wind_direction_trend.append(wind_direction_dataset)
    indices_v6.append(indices_dataset)
    predicted_ob_time.append(predicted_hours_dataset)
print('indices found')

m_values = []
for k in range(datasets): 
    print('m_values: start new dataset: ', k)
    # clear dataset variables
    m_dataset = []
    # get dataset indices
    indices_dataset = indices_v6[k]
    # get analog trend wind speeds
    wind_speed_trends = wind_speed_trend[k]
    # get analog trend wind directions
    wind_direction_trends = wind_direction_trend[k]
    # set standard deviations
    sigma_wind_speed = std_wind_speed[k]
    sigma_wind_direction = std_wind_direction[k]
    # select the kth concurrent dataset
    training = concurrent[k]
    training_wind_speed = training['ref_wind_speed'].values
    training_wind_direction = training['ref_wind_direction'].values
    training_ob_time = training['ob_time'].to_list()
    training_hours = [hour.hour for hour in training_ob_time]
    # print where we are in the loop
    print('m_values: start new dataset: ', k, 'number of iterations', len(indices_dataset))
    for j, indices_hour in enumerate(indices_dataset) :
        m_hour = np.zeros(len(indices_hour))
        wind_speed = wind_speed_trends[j]
        wind_direction = wind_direction_trends[j]
        # print how far we are in iterations for k
        if j%500 == 0: 
            print(f'number of iterations done: {j}')
        for i, index in enumerate(indices_hour): 
            if (index > 2) and (index < len(training) - 3): 
                search_wind_speed = training_wind_speed[index - 2 : index + 3]
                search_wind_direction = training_wind_direction[index - 2 : index + 3]
                diff_wind_speed = wind_speed - search_wind_speed
                diff_wind_direction = wind_direction - search_wind_direction
                # Calculate the terms separately for readability
                term1 = w_wind_speed / sigma_wind_speed * np.sqrt(np.sum(diff_wind_speed ** 2))
                term2 = w_wind_direction / sigma_wind_direction * np.sqrt(np.sum(diff_wind_direction ** 2))
                m_value = term1 + term2
                # if (j==267) and (i == 61): 
                #     print('analog trend wind speed: ', wind_speed)
                #     print('analog search wind speed: ', search_wind_speed)
                #     print('diff wind speed: ', diff_wind_speed )
                #     print('analog trend wind direction: ', wind_direction)
                #     print('analog_search wind direction: ', search_wind_direction)
                #     print('diff wind direction: ', diff_wind_direction)
                #     print('term1: ', term1)
                #     print('term2: ', term2)
                #     print('m value: ', m_value)
                # if (j > 266) and (j < 268) and (index > 0) and (index < 100):
                #     print(f'{j}, {index}')
                
                m_hour[i] = m_value
        m_dataset.append(m_hour)
    m_values.append(m_dataset)
#%%
analog_indices = []
analog_m_values = []
for k in range(datasets):           
    #reset dataset variables
    analog_indices_dataset = []
    analog_m_values_dataset = []
    # get dataset values best 25 m and asociated indices
    m_values_dataset = m_values[k]
    indices_dataset = indices_v6[k]
    for i, m_hour in enumerate(m_values_dataset): # for each hour
        n = 25  #number of K best analogs
        indices_hour = indices_dataset[i]
        # get the best m values
        sorted_indices = np.argsort(m_hour)[:n]
        m_best = m_hour[sorted_indices]
        best_indices = indices_hour[sorted_indices]
        analog_indices_dataset.append(best_indices)
        analog_m_values_dataset.append(m_best)
    analog_indices.append(analog_indices_dataset)
    analog_m_values.append(analog_m_values_dataset)
print('finished second loop')
#%%
gamma = []
for k in range(datasets):
    # reset dataset variable
    gamma_dataset = []
    # get correct indices
    analog_indices_dataset = analog_indices[k]
    analog_m_dataset = analog_m_values[k]
    for analogs_hour_m in analog_m_dataset: 
        m_sum = np.sum([1/x for x in analogs_hour_m])
        gamma_hour = [m_t**(-1) / m_sum for m_t in analogs_hour_m]
        gamma_dataset.append(gamma_hour)
    gamma.append(gamma_dataset)
print('gamma found')
#%%
best_observations = []
for k in range(datasets): 
    # reset dataset variable
    best_observations_dataset = []
    # get correct observation data
    training = concurrent[k]
    training_observations = training['target_wind_speed'].values
    analog_indices_dataset = analog_indices[k]
    for analog_indices_hour in analog_indices_dataset: 
        training_best_observations = training_observations[analog_indices_hour]
        best_observations_dataset.append(training_best_observations)
    best_observations.append(best_observations_dataset)
print('best observations found')
#%%
prediction = []
for k in range(datasets):
    Ok = best_observations[k]
    gamma_k = gamma[k]
    prediction_dataset = []  # Reset the prediction dataset for each dataset
    for i in range(len(Ok)):
        prediction_hour = np.sum(gamma_k[i] * Ok[i])
        prediction_dataset.append(prediction_hour)
    prediction.append(prediction_dataset)
print('predictions found')

#%%
# create pandas dataframe from prediction
prediction_df = []
for k in range(datasets): 
    prediction_series = pd.Series(prediction[k], name='prediction')
    ob_time_series = pd.Series(predicted_ob_time[k], name = 'ob_time')
    prediction_dataframe = pd.concat([prediction_series, ob_time_series], axis=1)
    prediction_df.append(prediction_dataframe)

#%%
# determine OLR estimate
# labels = ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180', '180-210', '210-240', '240-270', '270-300', '300-330', '330-360']

# slope = []
# offset = []

# for k in range(datasets): 
#     m = np.zeros(12)
#     c = np.zeros(12)
#     for i, label in enumerate(labels): 
#         data = concurrent[k].loc[concurrent[k]['bin'] == label]
#         m[i] = stats.linregress(data['ref_wind_speed'], data['target_wind_speed'])[0]
#         c[i] = stats.linregress(data['ref_wind_speed'], data['target_wind_speed'])[1]
#     slope.append(m)
#     offset.append(c)

# for k in range(datasets): 
#     for i, label in enumerate(labels): 
#         validate[k].loc[validate[k]['bin'] == label, 'longterm_target_estimate'] = slope[k][i] * validate[k]['ref_wind_speed'] + offset[k][i]

# print('olr carried out')
#%%
# merge validate df with prediction AE
reconstructed = []
for k in range(datasets): 
    data = validate[k].copy()
    toadd = prediction_df[k].copy()
    # olr = OLR[k].copy()
    result1 = pd.merge(data, toadd, left_on="ob_time", right_on = "ob_time", how="inner")
    # result = pd.merge(result1, olr, on='ob_time', how= 'inner')
    reconstructed.append(result1)
print('reconstruced datasets and predictions')
# #%% create time series plot
# import matplotlib.pyplot as plt

# # Create a figure and axis
# fig, ax = plt.subplots(figsize=(14, 6))

# # Plot the three lines on the same figure
# reconstructed[0].plot(x='ob_time', y='target_wind_speed', ax=ax, label='Target Wind Speed', linewidth=0.7)
# reconstructed[0].plot(x='ob_time', y='longterm_target_estimate', ax=ax, label='Ordinary Linear Regression', linewidth=0.7)
# reconstructed[0].plot(x='ob_time', y='prediction', ax=ax, label='Analog Ensembles', linewidth=0.7)

# # Rotate x-axis labels for better readability
# plt.xticks(rotation=45)  # Adjust the rotation angle as needed

# # Add labels and a legend
# ax.set_xlabel('Date')
# ax.set_ylabel('Wind speed [m/s]')
# ax.set_title('Time series wind speed at Cabauw')
# ax.legend(loc='upper right')

# # Show the plot
# plt.tight_layout()  # Ensure all labels are visible
# plt.show()

# #%%
# # determine rmse, coefficient of correlation, mean
# from collections import defaultdict
# from sklearn.metrics import r2_score
# import math 
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# keys = ["UK1. Dyffryn Brodyn", "UK2. Lifton Down", "UK3. St. Breock", "NL1. Schiphol", "NL2. Westdorpe", "NL3. Hupsel", "NL4. Cabauw",
#         "UK4. Penrhys", "UK5. Rheidol", "UK6. Alt-Yr-Hende", "NL5. Rotterdam Geulhaven", "UK7. Siddick", "UK8. Haverigg", "UK9. Treculliacks",
#         "UK10. Rhyd-Y-Goes", "UK11. Hill of Forss", "UK12. Crimp", "UK13. Ysgubor", "UK14. Jordanston", "UK15. Truthan", "UK16. Carland Cross",
#         "NL6. Platform AWG-1", "UK17. Celtic Array Zone 9", "UK18. Greater Gabbard", "UK19. Gunfleet Sands", "UK20. Gwynt Y Mor", "UK21. Shell Flats",
#         "NL7. Lichteiland Goeree", "NL8. K14FA1C", "NL9. J6-A", "NL10. Borssele 1", "NL11. Hollandse Kust West (HKWA)", "NL12. Hollandse Kust Noord (HKNB)",
#         "NL13. Ten Noorden van de Wadden (TNWB)", "NL14. Dogger Bank zone 3"]

# # # make sure there are no NaN values
# for k in range(datasets): 
#     reconstructed[k].dropna(subset=['prediction', 'longterm_target_estimate', 'target_wind_speed'], inplace=True, ignore_index = True)
# #%%
# r2score = defaultdict(list)
# rmse = defaultdict(list)
# mean_norm = defaultdict(list)
# mae = defaultdict(list)
# mbe = defaultdict(list)
# pearson = defaultdict(list)
# r2_AE = np.zeros(datasets)
# rmse_AE = np.zeros(datasets)
# rmse_olr = np.zeros(datasets)
# mean_AE = np.zeros(datasets)
# mae_AE = np.zeros(datasets)
# mbe_AE = np.zeros(datasets)
# pearson_AE = np.zeros(datasets)
# for k in range(datasets): 
#     # determine R^2
#     r2score[keys[k]].append(r2_score(reconstructed[k]['target_wind_speed'], reconstructed[k]['longterm_target_estimate']))
#     r2score[keys[k]].append(r2_score(reconstructed[k]['target_wind_speed'], reconstructed[k]['prediction']))
#     r2_AE[k] = (r2_score(reconstructed[k]['target_wind_speed'], reconstructed[k]['prediction']))
#     # root mean squared error
#     rmse[keys[k]].append(round(math.sqrt(mean_squared_error(reconstructed[k]['target_wind_speed'], reconstructed[k]['longterm_target_estimate'])) / reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True), 2))
#     rmse[keys[k]].append(round(math.sqrt(mean_squared_error(reconstructed[k]['target_wind_speed'], reconstructed[k]['prediction'])) / reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True), 2))
#     rmse_AE[k] = round(math.sqrt(mean_squared_error(reconstructed[k]['target_wind_speed'], reconstructed[k]['prediction'])) / reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True), 2)
#     rmse_olr[k] = round(math.sqrt(mean_squared_error(reconstructed[k]['target_wind_speed'], reconstructed[k]['longterm_target_estimate'])) / reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True), 2)
#     # normalized mean
#     mean_norm[keys[k]].append(round(reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True) / reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True), 2))
#     mean_norm[keys[k]].append(round(reconstructed[k]['longterm_target_estimate'].mean(axis=0, skipna = True) / reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True), 2))
#     mean_norm[keys[k]].append(round(reconstructed[k]['prediction'].mean(axis=0, skipna = True) / reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True), 2))
#     mean_AE[k] = round(reconstructed[k]['prediction'].mean(axis=0, skipna = True) / reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True), 2)
#     #mean absolute error
#     mae[keys[k]].append(round(mean_absolute_error(reconstructed[k]['target_wind_speed'], reconstructed[k]['longterm_target_estimate']) / reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True), 2))
#     mae[keys[k]].append(round(mean_absolute_error(reconstructed[k]['target_wind_speed'], reconstructed[k]['prediction']) / reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True), 2))
#     mae_AE[k] = round(mean_absolute_error(reconstructed[k]['target_wind_speed'], reconstructed[k]['prediction']) / reconstructed[k]['target_wind_speed'].mean(axis=0, skipna = True), 2)
#     # mean bias error normalized
#     mbe[keys[k]].append((np.mean(reconstructed[k]['longterm_target_estimate']) - np.mean(reconstructed[k]['target_wind_speed'])) / np.mean(reconstructed[k]['target_wind_speed']))
#     mbe[keys[k]].append((np.mean(reconstructed[k]['prediction']) - np.mean(reconstructed[k]['target_wind_speed'])) / np.mean(reconstructed[k]['target_wind_speed']))
#     # not normalized
#     mbe_AE[k] = round(np.mean(reconstructed[k]['prediction']) - np.mean(reconstructed[k]['target_wind_speed']), 2)
#     # pearson correlation of reconstructed period
#     pearson[keys[k]].append(round(stats.linregress(reconstructed[k]['target_wind_speed'], reconstructed[k]['ref_wind_speed'])[2], 2))
#     pearson[keys[k]].append(round(stats.linregress(reconstructed[k]['target_wind_speed'], reconstructed[k]['longterm_target_estimate'])[2], 2))
#     pearson[keys[k]].append(round(stats.linregress(reconstructed[k]['target_wind_speed'], reconstructed[k]['prediction'])[2], 2))
#     pearson_AE[k] = round(stats.linregress(reconstructed[k]['target_wind_speed'], reconstructed[k]['prediction'])[2], 2)

# print('weights used: wind_speed: ', w_wind_speed, ', wind_direction: ', w_wind_direction)
# print('cabauw: r2: ', r2score[keys[0]], ', rmse: ', rmse[keys[0]], ', norm mean: ', mean_norm[keys[0]], ', mae: ', mae[keys[0]], ', mbe: ', mbe[keys[0]], ', pearson: ', pearson[keys[0]] )
# print('avg R2: ', np.mean(r2_AE), ', avg rmse: ', np.mean(rmse_AE), ', avg norm mean: ', np.mean(mean_AE), ', avg mae: ', np.mean(mae_AE), ', avg mbe: ', np.mean(mbe_AE), ', avg pearson: ', np.mean(pearson_AE))
# #%%
# import seaborn as sns

# # create box plots
# r2_OLR = pd.DataFrame({'r2_score': [r2score[key][0] for key in r2score], 'method': 'OLR'})
# r2_AE = pd.DataFrame({'r2_score' : [r2score[key][1] for key in r2score], 'method': 'AE'})

# df_r2 = pd.concat([r2_OLR, r2_AE])

# plt.figure(figsize=(8,6))
# ax = sns.boxplot(data = df_r2, x='method', y='r2_score')
# ax.set_xlabel('Method', fontsize=12)
# ax.set_ylabel('R^2 score [-]', fontsize=12)
# plt.title('R^2 score MCP vs Analog Ensembles', fontsize=14)
# plt.show()
# #%%
# # normalized mean
# mean_target = pd.DataFrame({'mean': [mean_norm[key][0] for key in mean_norm], 'method': 'target'})
# mean_olr = pd.DataFrame({'mean': [mean_norm[key][1] for key in mean_norm], 'method': 'OLR'})
# mean_ae = pd.DataFrame({'mean': [mean_norm[key][2] for key in mean_norm], 'method': 'AE'})

# df_mean = pd.concat([mean_target, mean_olr, mean_ae])

# plt.figure(figsize=(8,6))
# ax = sns.boxplot(data = df_mean, x='method', y='mean')
# ax.set_xlabel('Method', fontsize=12)
# ax.set_ylabel('Normalized mean [-]', fontsize=12)
# plt.title('Normalized mean MCP vs Analog Ensembles', fontsize=14)
# plt.show()

# #%%
# # pearson correlation
# pearson_ref = pd.DataFrame({'pearson': [pearson[key][0] for key in pearson], 'method': 'ref'})
# pearson_olr = pd.DataFrame({'pearson': [pearson[key][1] for key in pearson], 'method': 'OLR'})
# pearson_ae = pd.DataFrame({'pearson': [pearson[key][2] for key in pearson], 'method': 'AE'})

# df_pearson = pd.concat([pearson_ref, pearson_olr, pearson_ae])

# plt.figure(figsize=(8,6))
# ax = sns.boxplot(data = df_pearson, x='method', y='pearson')
# ax.set_xlabel('Method', fontsize=12)
# ax.set_ylabel('Pearson Correlation [-]', fontsize=12)
# plt.title('Pearson Correlation MCP vs Analog Ensembles', fontsize=14)
# plt.show()

# #%%
# # mean bias error
# mbe_olr = pd.DataFrame({'mbe': [mbe[key][0] for key in mbe], 'method': 'OLR'})
# mbe_ae = pd.DataFrame({'mbe': [mbe[key][1] for key in mbe], 'method': 'AE'})

# df_mbe = pd.concat([mbe_olr, mbe_ae])

# plt.figure(figsize=(8,6))
# ax = sns.boxplot(data = df_mbe, x='method', y='mbe')
# ax.set_xlabel('Method', fontsize=12)
# ax.set_ylabel('MBE Correlation [-]', fontsize=12)
# plt.title('Normalized Mean Bias Error MCP vs Analog Ensembles', fontsize=14)
# plt.show()

# #%%
# # root mean square error
# rmse_olr = pd.DataFrame({'rmse': [rmse[key][0] for key in rmse], 'method': 'OLR'})
# rmse_ae = pd.DataFrame({'rmse': [rmse[key][1] for key in rmse], 'method': 'AE'})

# df_rmse = pd.concat([rmse_olr, rmse_ae])

# plt.figure(figsize=(8,6))
# ax = sns.boxplot(data = df_rmse, x='method', y='rmse')
# ax.set_xlabel('Method', fontsize=12)
# ax.set_ylabel('RMSE Correlation [-]', fontsize=12)
# plt.title('Root Mean Squared Error MCP vs Analog Ensembles', fontsize=14)
# plt.show()



#%%
# for k in range(datasets): 
#     reconstructed[k].to_csv(f'C:/Users/sylke/OneDrive/Documenten/THESIS/DATA/1_CSVS/Analogs_results/w=0.5/{k}_w_0.5_reconstructed.csv', index=False)
    



for k in range(datasets): 
    reconstructed[k].to_csv(f'C:/folder/path/AE_DATA_LENGTH/{k}_w_1_reconstructed.csv', index=False)
    






















