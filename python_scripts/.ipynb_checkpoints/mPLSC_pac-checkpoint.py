# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:39:26 2019
"""

import os
import numpy as np
import pandas as pd
import pickle as pkl
import mPLSC_functions as mf
from boredStats import pls_tools

import sys
sys.path.append("..")
import proj_utils as pu

print('%s: Loading data' % pu.ctime())
pdir = pu._get_proj_dir()
pdObj = pu.proj_data()
rois = pdObj.roiLabels
roi_path = pdir + '/data/glasser_atlas/'
fig_path = pdir + '/figures/mPLSC_cfc/'

# Creating list of x tables
def _load_cfc_subtables(fpath):
    #Support function - load x tables
    
    flist = []
    for f in os.listdir(fpath):
        if f.endswith('.xlsx'):
            flist.append(os.path.join(fpath, f))
    
    sessions = ['session1', 'session2', 'session3']
    subtable_data = {}
    for f, file in enumerate(flist):
        table = pd.read_excel(file, sheet_name=None, index_col=0) 
        subtable_data[sessions[f]] = dict(table)
        
    return subtable_data

meg_sess = ['session1', 'session2', 'session3']
fpath = pdir + '/data/phase_amplitude_tables'
cfc_tables = _load_cfc_subtables(fpath)
x_tables = []
for sess in meg_sess:
    for t in list(cfc_tables[sess]):
        x_tables.append(cfc_tables[sess][t])

# Creating list of y tables
behavior_metadata = pd.read_csv(pdir + '/data/b_variables_mPLSC.txt',
                                   delimiter='\t', header=None)

behavior_metadata.rename(dict(zip([0, 1], ['category','name'])),
                            axis='columns', inplace=True)

behavior_raw = pd.read_excel(pdir + '/data/hcp_behavioral.xlsx',
                              index_col=0, sheet_name='cleaned')

behavior_tables = mf.load_behavior_subtables(behavior_raw, behavior_metadata)
y_tables = [behavior_tables[t] for t in list(behavior_tables)]

p = pls_tools.MultitablePLSC(n_iters=1000)
print('%s: Running permutation testing on latent variables' % pu.ctime())
res_perm = p.mult_plsc_eigenperm(y_tables, x_tables)
print('%s: Running bootstrap testing on saliences' % pu.ctime())
res_boot = p.mult_plsc_bootstrap_saliences(y_tables, x_tables, 3)

print('%s: Plotting scree' % pu.ctime())   
mf.plotScree(res_perm['true_eigenvalues'],
             res_perm['p_values'],
             alpha=.001,
             fname=fig_path + '/scree.png')

num_latent_vars = 6#len(np.where(res_perm['p_values'] < .001)[0])
latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]

y_saliences = mf.create_salience_subtables(
        sals=res_boot['y_saliences'][:, :num_latent_vars],
        dataframes=y_tables,
        subtable_names=list(behavior_tables),
        latent_names=latent_names)

print('%s: Plotting behavior saliences' % pu.ctime())
res_behavior = mf.average_behavior_scores(y_saliences, latent_names)
for latent_var in latent_names:
    fname = fig_path + '/behavior_saliences_%s' % latent_var
    series = y_saliences[latent_var]
    mf.plot_radar2(series, choose=False, fname=fname)
    
x_table_names = []
for sess in list(cfc_tables):
    session_dict = cfc_tables[sess]
    full_table_names = ["%s %s" % (sess, cfc) for cfc in list(session_dict)]
    x_table_names = x_table_names + full_table_names

x_saliences = mf.create_salience_subtables(
        sals=res_boot['x_saliences'][:, :num_latent_vars],
        dataframes=x_tables,
        subtable_names=x_table_names,
        latent_names=latent_names)

def _x_conjunctions_cfc(x_saliences, latent_names, rois, return_avg=True):
    """Do conjunction analysis on multiple latent variables, cfc version
    Wrapper for mPLSC_functions.conjunction_analysis
    
    Returns a dictionary of conjunctions corresponding to each freq band
    """
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    keys = list(x_saliences)
    band_dataframes = {}
    for band in bands:
        sess_keys = [s for s in keys if band in s]
        conjunctions = []
        for name in latent_names:
            brains = []
            for s in sess_keys:
                brains.append(x_saliences[s][name].values)
            
            conj_data = pd.DataFrame(np.asarray(brains).T, index=rois)
            res = mf.conjunction_analysis(conj_data, 'sign', return_avg=return_avg)
            conjunctions.append(np.ndarray.flatten(res.values))
            
        band_df = pd.DataFrame(np.asarray(conjunctions).T,
                               index=res.index,
                               columns=latent_names)
        band_dataframes[band] = band_df
    return band_dataframes

res_conj = _x_conjunctions_cfc(x_saliences, latent_names, rois, True)
print('%s: Creating brain figures' % pu.ctime())
for band in list(res_conj):
    df = res_conj[band]
    for name in latent_names:
        mags = df[name].values
        
        custom_roi = mf.create_custom_roi(roi_path, rois, mags)
        
        fname = fig_path + '/brain_%s.png' % name
        minval = np.min(mags[np.nonzero(mags)])
        if len(np.nonzero(mags)) == 1:
            minval = None
        fig = mf.plot_brain_saliences(custom_roi, minval)
        fig.savefig(fname, bbox_inches='tight')

output = {'permutation_tests':res_perm,
          'bootstrap_tests':res_boot,
          'y_saliences':y_saliences,
          'x_saliences':x_saliences,
          'behaviors':res_behavior,
          'conjunctions':res_conj}

with open(pdir + '/data/mPLSC_cfc.pkl', 'wb') as file:
    pkl.dump(output, file)