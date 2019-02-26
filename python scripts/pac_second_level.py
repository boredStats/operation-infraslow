# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:01:21 2019
"""

import h5py
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
import proj_utils as pu

def extract_meg_pac(subj_list, sess, rois, data, out_path=None, early=True):
    between_subj_data = []
    
    for subj in subj_list:
        within_subj_data = []
        
        for roi in rois:
            hdf5 = h5py.File(data, 'r')
            if early: #Early analyses had subj and session hierarchy switched
                rval_path = subj  + '/' + sess + '/' + roi + '/' + 'r_vals'
            else:
                rval_path = sess  + '/' + subj + '/' + roi + '/' + 'r_vals'
            
            dset = hdf5.get(rval_path).value
            within_subj_data.append(dset[:, :])
            hdf5.close()
            
        within_subj_array = np.arctanh(np.asarray(within_subj_data))
        between_subj_data.append(within_subj_array)
        
    between_subj_array = np.asarray(between_subj_data)
    
    if out_path is not None:
        with open(out_path, 'wb') as out_file:
            pkl.dump(between_subj_array, out_file)
    
    return between_subj_array

def extract_mri_pac(subj_list, sess_combo, bands, hdf5_path, out_path=None):
    between_subj_mri_meg_list = []
    
    for subj in subj_list:
        rval_list = []
        
        for band in bands:
            hdf5 = h5py.File(hdf5_path, 'r')
            rval_path = subj + '/' + sess_combo + '/' + band + '/' + 'r_vals'
            rvals = np.asarray(hdf5.get(rval_path))
            rval_list.append(rvals)
            
        subj_rvals = np.asarray(rval_list).T

        within_subj_mri_meg = subj_rvals[:, 5:] #Getting supraband correlations
        
        between_subj_mri_meg_list.append(within_subj_mri_meg)
    between_subj_mri_meg = np.asarray(between_subj_mri_meg_list)
    
    if out_path is not None:
        with open(out_path, 'wb') as out_file:
            pkl.dump(between_subj_mri_meg, out_file)
            
    return between_subj_mri_meg

def build_violin_dataframe(between_subj_data, rois, row_bands, col_bands):
    mean_roi_data = np.mean(between_subj_data, axis=0)
    
    band_data = np.ndarray(shape=[len(rois) * len(row_bands), len(col_bands)])
    
    col_band_ref, roi_ref = [], []
    for b, band in enumerate(col_bands):
        band_to_plot = mean_roi_data[:, :, b]
        band_data[len(rois) * b: len(rois) * (b + 1), :] = band_to_plot
        
        col_band_ref = col_band_ref + [band] * len(rois)
        roi_ref = roi_ref + rois
    
    band_df = pd.DataFrame(band_data, index=roi_ref, columns=row_bands)
    
    vectorized_data_list = []
    df_band_ref = []
    reg_band_full_ref = []
    roi_full_rep = []
    for band_in_dataframe in list(band_df):
        data_to_grab = band_df[band_in_dataframe].values
        vectorized_data_list = vectorized_data_list + list(data_to_grab)
        
        #creating more lists for final dataframe 
        df_band_ref = df_band_ref + [band_in_dataframe]*band_df.values.shape[0]
        reg_band_full_ref = reg_band_full_ref + col_band_ref
        roi_full_rep = roi_full_rep + roi_ref
        
    vectorized_data = np.asarray(vectorized_data_list)
    
    final_df = pd.DataFrame(vectorized_data,
                            columns=['Cross-Frequency Coupling'],
                            index=roi_full_rep) 
    
    final_df['Phase bands'] = df_band_ref
    final_df['Amplitude bands'] = reg_band_full_ref
    return final_df

print('%s: Getting metadata, parameters' % pu.ctime())
pdir = pu._get_proj_dir()

pdObj = pu.proj_data()
meg_subj, meg_sess = pdObj.get_meg_metadata()
mri_subj, mri_sess = pdObj.get_mri_metadata()
subj_overlap = [s for s in mri_subj if s in meg_subj]

pData = pdObj.get_data()
rois = pData['roiLabels']
band_dict = pData['bands']

slow_bands = ['BOLD', 'Slow 4', 'Slow 3', 'Slow 2', 'Slow 1'] #rows
reg_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'] #cols

print('%s: Extracting MEG phase-amplitude correlations' % pu.ctime())
sess = meg_sess[1] #Edit MEG session to plot here

data_path = pdir + '/data/MEG_pac_first_level_slow_with_supra.hdf5'
out_path = pdir + '/data/violin_meg_data_slow_with_supra_%s.pkl' % sess
#test_data = extract_meg_pac(subj_overlap, sess, rois, data_path, out_path)
with open(out_path, 'rb') as file:
    test_data = pkl.load(file)
#test_df = build_violin_dataframe(test_data, rois, slow_bands, reg_bands)

#data_path = pdir + '/data/MEG_pac_first_level_supra_only.hdf5'
#out_path = pdir + '/data/violin_meg_data_suprabands_%s.pkl' % sess
#test_data = extract_meg_pac(subj_overlap, sess, rois, data_path, out_path, early=False)
#with open(out_path, 'rb') as file:
#    test_data = pkl.load(file)
#test_df = build_violin_dataframe(test_data, rois, reg_bands, reg_bands)

print('%s: Plotting data' % pu.ctime())
#sns.set_style('darkgrid')
#sns.set_context('notebook', font_scale=2)
#fig = sns.catplot(x='Phase bands', y='Cross-Frequency Coupling', 
#                   height=15, aspect=1.78, data=test_df,
#                   hue='Amplitude bands', kind='violin')
#fig.set(yscale='log')
#fig.set(ylim=(.001, .1))

    
data_path = pdir + '/data/MEG_pac_first_level_with_MRI.hdf5'
sess_combo = mri_sess[0] + '_' + meg_sess[0]
out_path = pdir + '/data/violin_meg_mri_data_%s.pkl' % sess_combo
test2_data = extract_mri_pac(subj_overlap, sess_combo, list(band_dict), data_path, out_path)
#with open(out_path, 'rb') as file:
#    test2_data = pkl.load(file)

combined_data_shape = [len(subj_overlap), len(rois), 6, 5]
combined_between_subj_data = np.ndarray(shape=combined_data_shape)
for subj_index, subj in enumerate(subj_overlap):
    for roi_index, roi in enumerate(rois):
        meg_with_meg_data = test_data[subj_index, roi_index, :, :]
        mri_with_meg_data = test2_data[subj_index, roi_index, :]
        combined_mat = np.vstack((mri_with_meg_data, meg_with_meg_data))
        combined_between_subj_data[subj_index, roi_index, :, :] = combined_mat

slow_bands_with_mri = ['MRI-BOLD'] + slow_bands

 
mean_roi_data = np.mean(combined_between_subj_data, axis=0)

band_data = np.ndarray(shape=[len(rois) * len(reg_bands),
                              len(slow_bands_with_mri)])
reg_band_ref = []
roi_rep = []
for b, band in enumerate(reg_bands):
    band_to_plot = mean_roi_data[:, :, b]
    band_data[360 * b: 360* (b+1), :] = band_to_plot
    
    #creatings lists to be repeated later
    reg_band_ref = reg_band_ref + [band]*len(rois)
    roi_rep = roi_rep + rois

band_df = pd.DataFrame(band_data, index=roi_rep, columns=slow_bands_with_mri)

vector_list = []
slow_band_ref = []
reg_band_full_ref = []
roi_full_rep = []
for slow_band in list(band_df):
    data_to_grab = band_df[slow_band].values
    vector_list = vector_list + list(data_to_grab)
    
    #creating more lists for final dataframe 
    slow_band_ref = slow_band_ref + [slow_band]*band_df.values.shape[0]
    reg_band_full_ref = reg_band_full_ref + reg_band_ref
    roi_full_rep = roi_full_rep + roi_rep
vector_data = np.asarray(vector_list)

final_df = pd.DataFrame(vector_data, columns=['Cross-Frequency Coupling'], index=roi_full_rep) 
final_df['Phase bands'] = slow_band_ref
final_df['Amplitude bands'] = reg_band_full_ref

sns.set_style('darkgrid')
sns.set_context('notebook', font_scale=2)
fig = sns.catplot(x='Phase bands', y='Cross-Frequency Coupling', 
                   height=15, aspect=1.78, data=final_df,
                   hue='Amplitude bands', kind='violin')

fig.set(ylim=(.001, .1))
fig.set(yscale='log')
sns.axes_style({'axes.grid': True})