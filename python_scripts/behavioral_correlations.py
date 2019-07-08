# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:47:08 2019
"""

import os
import h5py
import numpy as np
import pandas as pd

import sys
sys.path.append("..")
import proj_utils as pu

def center_matrix(a):
    col_means = a.mean(0)
    n_rows = a.shape[0]
    rep_mean = np.reshape(np.repeat(col_means, n_rows), a.shape, order="F")
    
    return np.subtract(a, rep_mean)

def cross_corr(x, y):
    s = x.shape[0] #num subjects/timepoints
    if s != y.shape[0]:
        raise ValueError ("x and y must have the same number of subjects")
    
    std_x = x.std(0, ddof=s - 1)
    std_y = y.std(0, ddof=s - 1)
    
    cov = np.dot(center_matrix(x).T, center_matrix(y))
    
    return cov/np.dot(std_x[:, np.newaxis], std_y[np.newaxis, :])

def r_to_p_matrix(rmat, n):
    from scipy.stats import t as tfunc
    denmat = (1 - rmat**2) / (n - 2)
    tmat = rmat / np.sqrt(denmat)
    
    tvect = np.ndarray.flatten(tmat)
    pvect = np.ndarray(shape=tvect.shape)
    for ti, tval in enumerate(tvect):
        pvect[ti] = tfunc.sf(np.abs(tval), n-1) * 2
    
    return np.reshape(pvect, rmat.shape)
    
def r_to_se_matrix(rmat, n):
    a = (1 - r**2) / (n - 2)
    return np.sqrt(a)

def fdr_p(pmat):
    from statsmodels.stats import multitest as mt
    pvect = np.ndarray.flatten(pmat)
    _, fdr_p = mt.fdrcorrection(pvect)
    return np.reshape(fdr_p, pmat.shape)
    
def _extract_meg_pac(subj_list, sess, rois, hdf5_path):
    between_subj_data = []
    for subj in subj_list:
        within_subj_data = []
        
        for roi in rois:
            hdf5 = h5py.File(hdf5_path, 'r')
            rval_path = sess  + '/' + subj + '/' + roi + '/' + 'r_vals'
            
            dset = hdf5.get(rval_path).value
            within_subj_data.append(dset[:, :])
            hdf5.close()
            
        within_subj_array = np.arctanh(np.asarray(within_subj_data))
        between_subj_data.append(within_subj_array)
        
    between_subj_array = np.asarray(between_subj_data)
    
    return between_subj_array

print('%s: Getting metadata, parameters' % pu.ctime())
pdir = pu._get_proj_dir()

pdObj = pu.proj_data()
meg_subj, meg_sess = pdObj.get_meg_metadata()
mri_subj, mri_sess = pdObj.get_mri_metadata()
subj_overlap = [s for s in mri_subj if s in meg_subj]

slow_bands = ['BOLD', 'Slow 4', 'Slow 3', 'Slow 2', 'Slow 1'] #rows
reg_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'] #cols

pData = pdObj.get_data()
rois = pData['roiLabels']

print('%s: Getting behavior data' % pu.ctime())
with open(pdir + '/data/cog_emotion_variables.txt', 'r') as boi:
    behavior_of_interest = [b.replace('\n', '') for b in boi]

behavior_path = pdir + '/data/hcp_behavioral.xlsx'
behavior_raw = pd.read_excel(behavior_path, index_col=0, sheet_name='cleaned')
behavior_df = behavior_raw.loc[:, behavior_of_interest]

data = pdir + '/data/MEG_pac_first_level_slow_with_supra.hdf5'
pac_outdir = pdir + '/data/phase_amplitude_excelsheets/'
if not os.path.isdir(pac_outdir):
    os.mkdir(pac_outdir)

check = input('Extract phase-amp data to excel? Y/N ')
if check == 'Y':
    for sess in meg_sess:
        print('%s: Getting PAC data %s' % (pu.ctime(), sess))
        meg_infraslow_data = _extract_meg_pac(subj_overlap, sess, rois, data)
        
        outpath = pac_outdir + 'MEG_pac_%s.xlsx' % sess
        writer = pd.ExcelWriter(outpath)
        bold_corr_data = meg_infraslow_data[:, :, 0, :]
        band_tables = {}
        for r, reg in enumerate(reg_bands):
            band_data = bold_corr_data[:, :, r]
            band_df = pd.DataFrame(band_data, columns=rois, index=subj_overlap)
            
            key_name = 'MEG-BOLD with %s' % reg    
            band_df.to_excel(writer, sheet_name=key_name)
            band_tables[key_name] = band_df
        writer.save()

bcor_outdir = pac_outdir + '/behavioral_correlations/'
if not os.path.isdir(bcor_outdir):
    os.mkdir(bcor_outdir)

print('%s: Running behavioral correlations' % pu.ctime())
x = center_matrix(behavior_df.values)
for sess in meg_sess:
    pac_path = pac_outdir + 'MEG_pac_%s.xlsx' % sess
    band_tables = {}
    for reg in reg_bands:
        key = 'MEG-BOLD with %s' % reg
        band_tables[key] = pd.read_excel(pac_path, sheet_name=key, index_col=0)
    
#    outpath = bcor_outdir + 'MEG_pac_%s_with_behavior_raw.xlsx' % sess
        outpath = bcor_outdir + 'MEG_pac_%s_with_behavior_fdr.xlsx' % sess
    writer = pd.ExcelWriter(outpath)
    for k, key in enumerate(list(band_tables)):
        pac_data = band_tables[key]
        y = center_matrix(pac_data.values)
        
        rmat = cross_corr(x, y)
#        r_df = pd.DataFrame(rmat, index=behavior_of_interest, columns=rois)
#        r_df.to_excel(writer, sheet_name='BOLD-%s' % reg_bands[k])
        
        pmat = r_to_p_matrix(rmat, y.shape[0])
        fdr_pmat = fdr_p(pmat)
        rmat[fdr_pmat > .05] = 0
        r_df = pd.DataFrame(rmat, index=behavior_of_interest, columns=rois)
        r_df.to_excel(writer, sheet_name='BOLD-%s' % reg_bands[k])
        
    writer.save()