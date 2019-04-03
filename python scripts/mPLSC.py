# -*- coding: utf-8 -*-
"""
Multitable PLS-C attempt with session1 MEG cross-frequency coupling data

Created on Mon Mar 18 13:56:17 2019
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

##### Load data ######
print('%s: Getting metadata, parameters' % pu.ctime())
pdir = pu._get_proj_dir()
fig_path = pdir + '/figures/mPLSC/'

data_path = pdir + '/data/mPLSC/'
cfc_path = pdir + '/data/phase_amplitude_tables'
cfc_files = sorted([os.path.join(cfc_path, f) for f in os.listdir(cfc_path)])
cfc_tables = pd.read_excel(cfc_files[0], sheet_name=None, index_col=0) 

x_tables = [cfc_tables[t].values for t in cfc_tables]
num_vars_in_x = [cfc_tables[x].values.shape[1] for x in cfc_tables]

with open(data_path + 'cog_emotion_variables.txt', 'r') as boi:
    behavior_of_interest = [b.replace('\n', '') for b in boi]
    
with open(data_path + 'cog_emotion_short.txt', 'r') as f:
    short_behavior = [b.replace('\n','') for b in f]

behavior_path = pdir + '/data/hcp_behavioral.xlsx'
behavior_raw = pd.read_excel(behavior_path, index_col=0, sheet_name='cleaned')
y_table = behavior_raw.loc[:, behavior_of_interest]
behavior_dict = {o:n for o, n in zip(behavior_of_interest, short_behavior)}
y_table.rename(columns=behavior_dict, inplace=True)

##Calculating mPLSC
#print('%s: Running eigenvalue permutations' % pu.ctime())
#res_perm = mf.permutation_pls(y_table, x_tables, fig_path)
#print('%s: Bootstrapping saliences' % pu.ctime())
#res_boot = mf.boostrap_sal(y_table, x_tables)

outfile = data_path + 'mPLSC_MEG_session1.pkl'
##saving results
#with open(outfile, 'wb') as file:
#    pkl.dump(res_perm, file)
#    pkl.dump(res_boot, file)

with open(outfile, 'rb') as file:
    res_perm = pkl.load(file)
    res_boot = pkl.load(file)

num_latent_vars = len(np.where(res_perm['p_values'] < .05)[0])
latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]

y_sals = res_boot['y_saliences'][:, :num_latent_vars]
bootstrap_y = pd.DataFrame(y_sals, index=list(y_table), columns=latent_names)
bootstrap_y.to_excel(data_path + 'mPLSC_MEG_session1_behavior_saliences.xlsx')

bootstrap_x = {}
start_index = 0
for x_index, x_table in enumerate(cfc_tables):
    end_index = start_index + num_vars_in_x[x_index]
    x_saliences = res_boot['x_saliences'][start_index:end_index, :6]
    df = pd.DataFrame(x_saliences,
                      index=list(cfc_tables[x_table]),
                      columns=latent_names)
    bootstrap_x[x_table] = df
    start_index = start_index + num_vars_in_x[x_index]

with pd.ExcelWriter(data_path + 'mPLSC_MEG_session1_pac_saliences.xlsx') as wr:
    for table_name in bootstrap_x:
        bootstrap_x[table_name].to_excel(wr, table_name)
    wr.save()

#plotting radar plots
for latent_var in latent_names:
    fname = fig_path + '/' + '%s_behavior_top10' % latent_var
    series = bootstrap_y[latent_var]
    mf.plot_radar2(series, fname)

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