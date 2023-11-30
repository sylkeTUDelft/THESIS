# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:17:23 2023

@author: sylke
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:51:55 2023

@author: sylke
"""
import pandas as pd
import os
import glob
import natsort

# get mylist

folder_path = 'C:/folder/path/mylist_full'

csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
csv_files_sorted = natsort.natsorted(csv_files)

mylist = []
for csv_file in csv_files_sorted: 
    df = pd.read_csv(csv_file)
    df['ob_time'] = pd.to_datetime(df['ob_time'])
    mylist.append(df)

    #%%
datasets = 43
# start by merging full reference and target dataframes
count = []
df = []
for k in range(datasets): 
    # merge reference and target
    full_set = pd.merge(mylist[k], mylist[k + datasets], on = 'ob_time', how = 'inner')
    print(full_set['ob_time'].iloc[0])
    if k == 9: 
        myset = full_set
    # count amount of datapoints for each set
    count.append(len(full_set))
    # append full sets to a list
    df.append(full_set)
#checked: all full sets have the same length as the full period of the target set
#%%
# set number of hours per time frame
hours_in_month = 30 * 24
hours_in_9_months = 274 * 24
hours_in_10_months = 304 * 24
hours_in_13_months = 396 * 24
hours_in_year = 365 * 24

# separate concurrent and validation period from df
concurrent_df = []
validate_df = []
for k in range(datasets): 
    n = count[k]
    full_df = df[k]
    if n < hours_in_10_months : # if the dataset has less than 10 months as data (one month for validation, 9 months for training)
        conc_set = full_df[0 : n - hours_in_month]
        concurrent_df.append(conc_set)
        val_set = full_df[n - hours_in_month : ]
        validate_df.append(val_set)
    if (n > hours_in_10_months) and (n < hours_in_13_months): # use 9 months for training, the rest for validation
        conc_set = full_df[0 : hours_in_9_months]
        concurrent_df.append(conc_set)
        val_set = full_df[hours_in_9_months : ]
        validate_df.append(val_set)
    if n > hours_in_13_months : # use 1 year for training and the rest for validation
        conc_set = full_df[0 : hours_in_year]
        concurrent_df.append(conc_set)
        val_set = full_df[hours_in_year : ]
        validate_df.append(val_set)
    concurrent_df[k].reset_index(drop=True, inplace=True)
    validate_df[k].reset_index(drop=True, inplace=True)
    # print(f' {k} : full_set n: {n}, conc: {len(conc_set)}, val: {len(val_set)}, sum : {len(conc_set) + len(val_set)}')
 
# check first and last dates 

for k in range(datasets): 
    last_date = concurrent_df[k]['ob_time'].iloc[-1]
    first_df_date = validate_df[k]['ob_time'].iloc[0]
    print(f'dataset: {k}: conc last date: {last_date}, validate first date: {first_df_date}')
#%%
testdf = mylist[9].loc[(mylist[9]['ob_time'] >= '1991-03-14 15:00:00') & (mylist[9]['ob_time'] <= '1992-01-27 09:00:00')]
print(len(testdf))

#%% create direction sector bins of 30 degrees
sector_bins = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
sector_labels = ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180', '180-210', '210-240', '240-270', '270-300', '300-330', '330-360']

concurrent = []
validate = []
for i in range(datasets): 
    conc_set = concurrent_df[i].copy()
    val_set = validate_df[i].copy()
    conc_set.loc[: , 'target_bin'] = pd.cut(conc_set.loc[:, 'target_wind_direction'], bins = sector_bins, include_lowest=True,  labels = sector_labels)
    conc_set.loc[:, 'bin'] = pd.cut(conc_set.loc[: , 'ref_wind_direction'], bins = sector_bins, include_lowest=True, labels = sector_labels)
    val_set.loc[:,'bin'] = pd.cut(val_set.loc[: , 'ref_wind_direction'], bins = sector_bins, include_lowest = True, labels = sector_labels)
    val_set.loc[:,'target_bin'] = pd.cut(val_set.loc[:,'target_wind_direction'], bins = sector_bins, include_lowest = True, labels = sector_labels)
    concurrent.append(conc_set.copy())
    validate.append(val_set.copy())

# #%% create csvs
for k in range(datasets): 
    validate[k].to_csv(f'C:/folder/path/validate_full/{k}_validate.csv', index=False)
    concurrent[k].to_csv(f'C:/folder/path/concurrent_full/{k}_concurrent.csv', index=False)



    