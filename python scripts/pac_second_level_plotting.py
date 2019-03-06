# -*- coding: utf-8 -*-
"""
Code for making violin plots

Functions in this script are for data transformation only
Seaborn catplot requires SPSS-like dataframe structure... sigh...
        
Created on Mon Feb 11 09:01:21 2019
"""

import h5py
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.append("..")
import proj_utils as pu

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

def _extract_mri_pac(subj_list, sess_combo, bands, hdf5_path):
    between_subj_mri_meg_list = []
    
    for subj in subj_list:
        rval_list = []
        
        for band in bands:
            hdf5 = h5py.File(hdf5_path, 'r')
            rval_path = subj + '/' + sess_combo + '/' + band + '/' + 'r_vals'
            rvals = np.asarray(hdf5.get(rval_path))
            rval_list.append(rvals)
            
        within_subj_mri_meg = np.asarray(rval_list).T        
        between_subj_mri_meg_list.append(within_subj_mri_meg)
        
    between_subj_mri_meg = np.asarray(between_subj_mri_meg_list)
    return between_subj_mri_meg

def _build_violin_dataframe(between_subj_data, rois, row_bands, column_bands):
    avg_subj_data = np.mean(between_subj_data, axis=0)
    first_transformation = np.ndarray(shape=[len(rois) * len(row_bands),
                                         len(column_bands)])
    first_supraband_repetition, first_roi_repetition = [], []
    for b, band in enumerate(row_bands):
        band_to_plot = avg_subj_data[:, :, b]
        first_transformation[360 * b: 360* (b+1), :] = band_to_plot
        
        #creatings lists to be repeated later
        first_supraband_repetition += [band]*len(rois)
        first_roi_repetition += rois
    
    first_df = pd.DataFrame(first_transformation,
                       index=first_roi_repetition,
                       columns=column_bands)
    
    vectorized_data_as_list, infraslow_repetition = [], []
    second_supraband_repetition, second_roi_repetition = [], []
    for col_label in list(first_df):
        data_to_grab = first_df[col_label].values
        vectorized_data_as_list += list(data_to_grab)
        
        #creating more lists for final dataframe 
        infraslow_repetition += [col_label]*first_df.values.shape[0]
        second_supraband_repetition += first_supraband_repetition    
        second_roi_repetition += first_roi_repetition
        
    vectorized_data = np.asarray(vectorized_data_as_list)
    
    catplot_df = pd.DataFrame(vectorized_data,
                            columns=['Cross-Frequency Coupling'],
                            index=second_roi_repetition) 
    catplot_df['Phase bands'] = infraslow_repetition
    catplot_df['Amplitude bands'] = second_supraband_repetition
    
    return catplot_df

def _plot_violin(dataframe):
    sns.set_style('darkgrid')
    sns.set_context('notebook', font_scale=2)
    fig = sns.catplot(x='Phase bands', y='Cross-Frequency Coupling', 
                       height=15, aspect=1.78, data=dataframe,
                       hue='Amplitude bands', kind='violin')
    fig.set(yscale='log')
    fig.set(ylim=(.001, .1))

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

#--- Infraslow results ---#
sess = meg_sess[1] #Edit MEG session to plot here
print('%s: Plotting infraslow results for MEG %s' % (pu.ctime(), sess))
data_path = pdir + '/data/MEG_pac_first_level_slow_with_supra.hdf5'
meg_infraslow_data = _extract_meg_pac(subj_overlap, sess, rois, data_path)
catplot_df = _build_violin_dataframe(meg_infraslow_data,
                                    rois, slow_bands, reg_bands)
_plot_violin(catplot_df)

#--- Supraband results ---#
sess = meg_sess[2] #Edit MEG session to plot here
print('%s: Plotting supraband results for MEG %s' % (pu.ctime(), sess))
data_path = pdir + '/data/MEG_pac_first_level_supra_only.hdf5'
supra_data = _extract_meg_pac(subj_overlap, sess, rois, data_path)
catplot_df = _build_violin_dataframe(supra_data, rois, reg_bands, reg_bands)
_plot_violin(catplot_df)

#--- fmri + meg scan combo results ---#
mri_session = mri_sess[1]
meg_session = meg_sess[2]
sess_combo = mri_session + '_' + meg_session
print('%s: Plotting combo infraslow results %s' % (pu.ctime(), sess_combo))
data_path = pdir + '/data/MEG_pac_first_level_with_MRI.hdf5'
mri_meg_data = _extract_mri_pac(subj_overlap, sess_combo, reg_bands, data_path)
data_path = pdir + '/data/MEG_pac_first_level_slow_with_supra.hdf5'
meg_data = _extract_meg_pac(subj_overlap, meg_session, rois, data_path)

slow_bands_with_mri = ['MRI-BOLD'] + slow_bands
combined_data_shape = [len(subj_overlap), len(rois),
                       len(slow_bands_with_mri), len(reg_bands)]
combined_between_subj_data = np.ndarray(shape=combined_data_shape)
for subj_index, subj in enumerate(subj_overlap):
    for roi_index, roi in enumerate(rois):
        meg_with_meg_data = meg_data[subj_index, roi_index, :, :]
        mri_with_meg_data = mri_meg_data[subj_index, roi_index, :]
        combined_mat = np.vstack((mri_with_meg_data, meg_with_meg_data))
        combined_between_subj_data[subj_index, roi_index, :, :] = combined_mat

catplot_df = _build_violin_dataframe(combined_between_subj_data,
                               rois, reg_bands, slow_bands_with_mri)
_plot_violin(catplot_df)