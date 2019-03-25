# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:39:26 2019
"""

import os
import nilearn
import numpy as np
import pandas as pd
import pickle as pkl
import mPLSC_functions as mf

import sys
sys.path.append("..")
import proj_utils as pu

nilearn.EXPAND_PATH_WILDCARDS = False

def _load_cfc_subtables(fpaths, keynames):
    #Support function - load x tables
    subtable_data = {}
    for f, file in enumerate(fpaths):
        table = pd.read_excel(file, sheet_name=None, index_col=0) 
        subtable_data[keynames[f]] = dict(table)
        
    return subtable_data
def _load_behavior_subtables(behavior_raw, variable_metadata):
    #Support function - load y tables
    names = list(variable_metadata['name'].values)
    
    overlap = [b for b in names if b in list(behavior_raw)]
    to_drop = [b for b, n in enumerate(names) if n not in list(behavior_raw)]
    
    
    variable_metadata.drop(to_drop, inplace=True)
    categories = list(variable_metadata['category'].values)
    
    behavior_data = behavior_raw.loc[:, overlap]
    
    
    subtable_data = {}
    for c in list(pd.unique(categories)):
        blist = [beh for b, beh in enumerate(overlap) if categories[b] == c]
        subtable_data[c] = behavior_data.loc[:, blist]
        
    return subtable_data

print('%s: Loading data' % pu.ctime())
pdir = pu._get_proj_dir()

# Creating list of x tables
flist = []
for f in os.listdir(pdir + '/data/phase_amplitude_tables'):
    if f.endswith('.xlsx'):
        flist.append(os.path.join(pdir + '/data/phase_amplitude_tables', f))

sessions = ['session1', 'session2', 'session3']
cfc_tables = _load_cfc_subtables(flist, sessions)
x_tables = []
for sess in sessions:
    for t in list(cfc_tables[sess]):
        x_tables.append(cfc_tables[sess][t])


# Creating list of y tables
behavior_metadata = pd.read_csv(pdir + '/data/b_variables_mPLSC.txt',
                                   delimiter='\t', header=None)

behavior_metadata.rename(dict(zip([0, 1], ['category','name'])),
                            axis='columns', inplace=True)

behavior_raw = pd.read_excel(pdir + '/data/hcp_behavioral.xlsx',
                              index_col=0, sheet_name='cleaned')

behavior_tables = _load_behavior_subtables(behavior_raw, behavior_metadata)
y_tables = [behavior_tables[t] for t in list(behavior_tables)]

# calculating mPLSC
res_perm = mf.permutation_pls(y_tables, x_tables) 
res_boot = mf.boostrap_sal(y_tables, x_tables)

# saving results
outfile = pdir + '/data/mPLSC_full.pkl'
with open(outfile, 'wb') as file:
    pkl.dump(res_perm, file)
    pkl.dump(res_boot, file)
#
#with open(outfile, 'rb') as file:
#    res_perm = pkl.load(file)
#    res_boot = pkl.load(file)
#
#num_latent_vars = len(np.where(res_perm['p_values'] < .05)[0])
#latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]
#
#y_sals = res_boot['y_saliences'][:, :num_latent_vars]
#bootstrap_y = pd.DataFrame(y_sals, index=list(y_table), columns=latent_names)
#bootstrap_y.to_excel(data_path + 'mPLSC_MEG_session1_behavior_saliences.xlsx')
#
#bootstrap_x = {}
#start_index = 0
#for x_index, x_table in enumerate(cfc_tables):
#    end_index = start_index + num_vars_in_x[x_index]
#    x_saliences = res_boot['x_saliences'][start_index:end_index, :6]
#    df = pd.DataFrame(x_saliences,
#                      index=list(cfc_tables[x_table]),
#                      columns=latent_names)
#    bootstrap_x[x_table] = df
#    start_index = start_index + num_vars_in_x[x_index]
#
#with pd.ExcelWriter(data_path + 'mPLSC_MEG_session1_pac_saliences.xlsx') as wr:
#    for table_name in bootstrap_x:
#        bootstrap_x[table_name].to_excel(wr, table_name)
#    wr.save()
#
##plotting radar plots
#for latent_var in latent_names:
#    fname = fig_path + '/' + '%s_behavior_top10' % latent_var
#    series = bootstrap_y[latent_var]
#    mf.plot_radar2(series, fname)

#plotting brains
#roi_path = pdir + '/data/glasser_atlas/'
#for latent_var in latent_names:
#    for roi in list(bootstrap_x):
#        print('%s: Plotting %s - %s' % (pu.ctime(), latent_var, roi))
#        test_df = bootstrap_x[roi]
#        sal = test_df[latent_var]
#        mags = sal.values
#        minval = np.min(mags[np.nonzero(mags)])
#        custom_roi = mf.create_custom_roi(roi_path, list(test_df.index), mags)
#        
#        figname = fig_path + '/' + '%s_%s' % (latent_var, roi)
#        mf.plot_brain_saliences(custom_roi, minval, figname)