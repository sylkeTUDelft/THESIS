# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:34:55 2023

@author: sylke
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import method_comparison_OLR_VRM_MM as methods
overview = methods.overview #sector overview (mean per sector)
normalized_overview = methods.normalized_overview #normalized sector overview (mean per sector)
overall_mean = methods.overall_mean #overall mean calculated from weighted sector means
normalized_overall_mean = methods.normalized_overall_mean #normalized overall mean calculated from weighted sector means

datasets = 35
#%% calculate the number of times each method performs best --> number of times closest to 1 in normalized overview and normalized overall mean
diffs = {
    'OLR': [],
    'VRM': [],
    'MM1': [],
    'MM2': []
}

best = []
counters = {
    'OLR': 0,
    'VRM': 0,
    'MM1': 0,
    'MM2': 0
}
count_multiple_equal = 0  # Counter for cases with multiple equal values

for k in range(datasets):
    diffs['OLR'].append(normalized_overall_mean[k]['actual'] - normalized_overall_mean[k]['OLR'])
    diffs['VRM'].append(normalized_overall_mean[k]['actual'] - normalized_overall_mean[k]['VRM'])
    diffs['MM1'].append(normalized_overall_mean[k]['actual'] - normalized_overall_mean[k]['MM opt1'])
    diffs['MM2'].append(normalized_overall_mean[k]['actual'] - normalized_overall_mean[k]['MM opt2'])
    
    # min_diff = min(abs(diffs[method][k].iloc[0]) for method in diffs.keys())
    # best.append(k, min_diff)
best = []  # Initialize the 'best' list
list_best = []
methods = list(diffs.keys())  # Get a list of method names

for k in range(datasets):
    method_diffs = {}
    for method in methods:
        diff = abs(diffs[method][k].iloc[0])
        method_diffs[method] = diff
    
    min_method = min(method_diffs, key=method_diffs.get)
    min_diff = method_diffs[min_method]
    
    best.append((k, min_method, min_diff))  # Append a tuple of (index, method, min_diff) to 'best'
    list_best.append(min_method)
    
    for method in counters.keys():
        is_best = all(abs(diffs[method][k].iloc[0]) == best[k] for m, d in diffs.items() if m != method)
        is_other_best = any(abs(diffs[m][k].iloc[0]) == best[k] for m in diffs.keys() if m != method)
        
        if is_best and not is_other_best:
            counters[method] += 1
    
    equal_count = sum(1 for val in diffs.values() if abs(val[k].iloc[0]) == best[k])
    if equal_count >= 2:
        count_multiple_equal += 1

# Convert the Series in diffs to floats
for method in diffs.keys():
    diffs[method] = [d.iloc[0] for d in diffs[method]]
    
#%% determine the number of times each method performs best per sector (from normalized overview)
diffs_sector = {
    'OLR': [],
    'VRM': [],
    'MM1': [],
    'MM2': []
}

best_sector = []
counters_sector = {
    'OLR': 0,
    'VRM': 0,
    'MM1': 0,
    'MM2': 0
}
count_multiple_equal_sector = 0  # Counter for cases with multiple equal values

for k in range(datasets): 
    olr = np.zeros(12)
    vrm = np.zeros(12)
    mm1 = np.zeros(12)
    mm2 = np.zeros(12)
    for i in range(12): 
        olr[i] = normalized_overview[k]['act. u'][i] - normalized_overview[k]['OLR u'][i]
        vrm[i] = normalized_overview[k]['act. u'][i] - normalized_overview[k]['VRM u'][i]
        mm1[i] = normalized_overview[k]['act. u'][i] - normalized_overview[k]['MM u opt1'][i]
        mm2[i] = normalized_overview[k]['act. u'][i] - normalized_overview[k]['MM u opt2'][i]
    diffs_sector['OLR'].append(olr)
    diffs_sector['VRM'].append(vrm)
    diffs_sector['MM1'].append(mm1)
    diffs_sector['MM2'].append(mm2)

best_sector = [[] for _ in range(datasets)]  # Initialize a list of 12 empty lists

for i in range(12):
    for k in range(datasets): 
        min_diff = min(
            abs(diffs_sector['OLR'][k][i]),
            abs(diffs_sector['VRM'][k][i]),
            abs(diffs_sector['MM1'][k][i]),
            abs(diffs_sector['MM2'][k][i])
        )
        best_method_index = np.argmin([
            abs(diffs_sector['OLR'][k][i]),
            abs(diffs_sector['VRM'][k][i]),
            abs(diffs_sector['MM1'][k][i]),
            abs(diffs_sector['MM2'][k][i])
        ])
        best_method = methods[best_method_index]
        best_sector[k].append((i, best_method, min_diff))
        
        
# Initialize counts
count_OLR = 0
count_VRM = 0
count_MM1 = 0
count_MM2 = 0

for sublist in best_sector:
    for item in sublist:
        method = item[1]  # Get the method from the tuple
        if method == 'OLR':
            count_OLR += 1
        elif method == 'VRM':
            count_VRM += 1
        elif method == 'MM1':
            count_MM1 += 1
        elif method == 'MM2':
            count_MM2 += 1

# Print the counts
print("Count of OLR:", count_OLR)
print("Count of VRM:", count_VRM)
print("Count of MM1:", count_MM1)
print("Count of MM2:", count_MM2)




