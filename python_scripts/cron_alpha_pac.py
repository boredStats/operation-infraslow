# -*- coding: utf-8 -*-
"""
Calculate Cronbach's alpha on cross-frequency coupling across 3 MEG sessions

Created on Wed Feb 20 11:20:35 2019
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

def cron_alpha(array):
    k = array.shape[1] #Columns are the groups
    variances_sum = np.sum(np.var(array, axis=0, ddof=1))
    variances_total = np.var(np.sum(array, axis=1), ddof=1)
    
    return (k / (k-1)) * (1 - (variances_sum / variances_total))

print('%s: Starting' % pu.ctime())   

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
supra_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'] #cols

cron_alpha_path = pdir + '/data/MEG_pac_first_level_slow_reg_dict.pkl'
check = input('Reorganize data for Cronbachs alpha calc? Y/N ')
if check == 'Y':
    pac_first_level = pdir + '/data/MEG_pac_first_level_slow_reg.hdf5'
    sess_dict = {}
    for sess in meg_sess:
        between_subj_data = []
        
        for subj in subj_overlap:
            print('%s: %s %s Getting corr data' % (pu.ctime(), sess, subj))
            within_subj_data = []
            
            for roi in rois:
                cfc_file = h5py.File(pac_first_level, 'r')
                rval_path = subj + '/' + sess + '/' + roi + '/' + 'r_vals'
                
                dset = cfc_file.get(rval_path).value
                within_subj_data.append(dset[:, :])
                
                cfc_file.close()
                
            between_subj_data.append(np.arctanh(np.asarray(within_subj_data)))
            del within_subj_data
        
        between_subj_array = np.asarray(between_subj_data)
        del between_subj_data
        
        print('%s: %s Reorganizing data by ROI' % (pu.ctime(), sess))
        roi_dict = {}
        for r, roi in enumerate(rois):
            band_dict = {}
            
            for s, slow in enumerate(slow_bands):
                key_name = slow + ' with supraslow bands'
                data_to_grab = between_subj_array[:, r, s, :]
    
                band_dict[key_name] = np.ndarray.squeeze(data_to_grab)
            
            roi_dict[roi] = band_dict
        
        sess_dict[sess] = roi_dict
        del between_subj_array, roi_dict

    with open(cron_alpha_path, 'wb') as file:
        pkl.dump(sess_dict, file)
        
else:
    with open(cron_alpha_path, 'rb') as file:
        sess_dict = pkl.load(file)
    
print('%s: Calculating Cronbachs alpha' % pu.ctime())
key_list = []
for slow in slow_bands:
    key_name = slow + ' with supraslow bands'
    key_list.append(key_name)

cron_alpha_dict = {}
for c_index, cfc in enumerate(key_list):
    
    distribution_list = []
    for r, roi in enumerate(rois):
        cfc_over_sessions_list = []
        
        for sess in list(sess_dict):
            sess_data = sess_dict[sess]
            roi_data = sess_data[roi]
            cfc_data = roi_data[cfc]
            
            cfc_over_sessions_list.append(cfc_data)
            
        cfc_over_sessions = np.asarray(cfc_over_sessions_list)
        calpha_list = []
        for supra_index, supra in enumerate(supra_bands):
            cron_list = []
            
            for sess_index in range(len(list(sess_dict))):
                vector = cfc_over_sessions[sess_index, :, supra_index]
                cron_list.append(vector)
                
            cron_data = np.asarray(cron_list).T
            calpha = cron_alpha(cron_data) 
            calpha_list.append(calpha)
        
        distribution_list.append(calpha_list)
        
    distribution_array = np.asarray(distribution_list)
    dist_df = pd.DataFrame(distribution_array, index=rois, columns=supra_bands)
    
    cron_alpha_dict[slow_bands[c_index]] = dist_df
            
print('%s: Building monster dataframe to plot results' % pu.ctime())
supra_label_list = []
slow_label_list = []
data_list =[]
for slow_band in list(cron_alpha_dict):
    
    for supra_band in supra_bands:
        supra_labels = [supra_band] * len(rois)
        supra_label_list = supra_label_list + supra_labels
    
    slow_labels = [slow_band] * len(rois) * len(supra_bands)
    slow_label_list = slow_label_list + slow_labels
    
    data_to_grab = cron_alpha_dict[slow_band]
    flattened_data = np.ndarray.flatten(data_to_grab.values, order='F')
    data_list.append(flattened_data)

full_data = np.concatenate(data_list)
full_data[full_data < 0] = 0

monster_df = pd.DataFrame(full_data, columns=['Cronbach alpha'])
monster_df['Phase bands'] = slow_label_list
monster_df['Amplitude bands'] = supra_label_list

print('%s: Plotting results' % pu.ctime())
sns.set_style('darkgrid')
sns.set_context('notebook', font_scale=2)
fig = sns.catplot(x='Phase bands', y='Cronbach alpha', data=monster_df,
                  hue='Amplitude bands',
                  height=15, aspect=1.78, kind='violin')