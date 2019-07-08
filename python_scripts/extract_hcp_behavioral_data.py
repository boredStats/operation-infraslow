# -*- coding: utf-8 -*-
"""
Extract HCP behavioral data from the subset of subjects for this project

Created on Fri Mar  1 09:10:12 2019
"""
import numpy as np
import pandas as pd

import sys
sys.path.append("..")
import proj_utils as pu

def data_check(data):
    float_transformed = []
    for d in data:
        try:
            float_transformed.append(float(str(d)))
        except ValueError: 
            float_transformed.append('s')
    
    return all([isinstance(f, float) for f in float_transformed])

print('%s: Getting metadata' % pu.ctime())
pdir = pu._get_proj_dir()
pdObj = pu.proj_data()
meg_subj, _ = pdObj.get_meg_metadata()
mri_subj, _ = pdObj.get_mri_metadata()
subj_overlap = [int(s) for s in mri_subj if s in meg_subj]

print('%s: Preparing first filter to clean data' % pu.ctime())
with open(pdir + '/data/behavioral_data_variables.xlsx', 'rb') as varfile:
    variable_table_raw = pd.read_excel(varfile)
    
category_list = list(variable_table_raw['category'].values)
categories = list(pd.unique(category_list))
colvars = list(variable_table_raw['columnHeader'].values)

categories_of_interest = ['Subject Information',
                          'Alertness',
                          'Cognition',
                          'Emotion',
                          'Motor',
                          'Personality',
                          'Sensory']

categories_dull = [c for c in categories if c not in categories_of_interest]
dull_vars = [] 
for c, cat in enumerate(category_list):
    if cat in categories_dull:
        dull_vars.append(colvars[c])
first_var_filter = list(pd.unique(dull_vars))

print('%s: Loading in raw behavioral data' % pu.ctime())
with open(pdir + '/data/behavioral_data.xlsx', 'rb') as datafile:
    behavior_data_raw = pd.read_excel(datafile, index_col=0)

print('%s: Filtering data, removing non-numeric variables' % pu.ctime())
behavior_df = behavior_data_raw.loc[subj_overlap]
behavior_vars = list(behavior_df)
non_string_vars = [b for b in behavior_vars if data_check(behavior_df[b])]
filtered_vars = [v for v in non_string_vars if v not in first_var_filter]

final_df = pd.DataFrame()
final_df['Gender'] = behavior_df['Gender']
for fv in filtered_vars:
    final_df[fv] = behavior_df[fv]

missing_data_indices = np.where(pd.isnull(final_df))
subj_mdata = missing_data_indices[0]
var_mdata = missing_data_indices[1]
for d in range(len(subj_mdata)):
    missing_subj = final_df.index[subj_mdata[d]]
    missing_var = list(final_df)[var_mdata[d]]
    print('%s %s' % (missing_subj, missing_var))
    
final_df_path = pdir + '/data/hcp_behavioral.xlsx'
final_df.to_excel(final_df_path, sheet_name='extracted')