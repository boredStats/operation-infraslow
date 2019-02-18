# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:21:17 2019

@author: jah150330

Calculate coherence of MEG and rfMRI signals for different time windows.
"""
import sys
sys.path.append("..")

import os
import time
import datetime
import proj_utils as pu
from scipy import signal
import h5py as h5
import numpy as np
import pandas as pd
import pac_functions as pac

import matplotlib.pyplot as plt
from scipy.signal import coherence
#from nipype.algorithms import icc

#exec(open('load_downsampledData.py').read())

#Load data
pdir = pu._get_proj_dir()
pdObj = pu.proj_data()
pData = pdObj.get_data()
database = pData['database']
rois = pData['roiLabels']

subj_MEG, sess_MEG = pdObj.get_meg_metadata()
subj_MRI, sess_MRI = pdObj.get_mri_metadata()

#Loading single subject
dset = database['/' + subj_MRI[0] + '/rsfMRI/' + sess_MRI[0] + '/timeseries']
dataMRI = pu.read_database(dset, rois)

meg_downsamp = h5.File(os.path.join(pdir, 'data/resampled_MEG_data.hdf5'))
dset = meg_downsamp['/' + subj_MEG[0] + '/MEG/' + sess_MEG[0] + '/resampled']
dataMEG = pu.read_database(dset, rois)

#Params
fs = 1/.72
max_f = 1/.72/2

#Running coherence on one subject, one ROI
ts_1 = dataMEG[rois[0]].values
ts_2 = dataMRI[rois[0]].values

#freqs, coh = coherence(ts_1, ts_2, fs=fs)
offset=0
freqs, coh = coherence(ts_1[0:407], ts_2[0+offset:407+offset], fs=fs)

f_to_plot = freqs[freqs < max_f]
#plt.semilogy(f_to_plot, coh[:len(f_to_plot)])
#plt.xlabel('Frequency (Hz)')
#plt.ylabel('Coherence')

#Calculating coherence for all subjects, ROIs
subject_overlap = [s for s in subj_MRI if s in subj_MEG]
coh_full = np.ndarray(shape=[len(f_to_plot), len(rois), len(subject_overlap)])
for s, subj in enumerate(subject_overlap):
    t = datetime.datetime.now()
    print('%s Calculating coherence for subject %s' % (t, str(subj)))
    
    dset = database['/' + subj + '/rsfMRI/' + sess_MRI[1] + '/timeseries']
    fmri_data = pu.read_database(dset, rois)
    
    dset = meg_downsamp['/' + subj + '/MEG/' + sess_MEG[2] + '/resampled']
    meg_data = pu.read_database(dset, rois)
    
    coherence_roi_mat = np.ndarray(shape=[len(f_to_plot), len(rois)])
    for r, roi in enumerate(rois):
        ts_1 = meg_data[roi].values
        ts_1_filt = pac.butter_filter(ts_1, fs, max_f, 'lowpass')
        ts_2 = fmri_data[roi].values
        ts_2_filt = pac.butter_filter(ts_2, fs, max_f, 'lowpass')
        
        offset = 600
        
#        freqs, coh = coherence(ts_1, ts_2[0+offset:407+offset], fs=fs)
        freqs, coh = coherence(ts_1_filt[0:407], ts_2_filt[0+offset:407+offset], fs=fs)
        coherence_roi_mat[:, r] = coh[:len(f_to_plot)]
    coh_full[:, :, s] = coherence_roi_mat
    
coh_over_rois = np.mean(coh_full, axis=2)     
#res = icc.ICC_rep_anova(coh_over_rois.T)

#Plotting coherence across all subjects
plt.clf()
fig, ax = plt.subplots(figsize=[8, 8])
for r in range(len(rois)):
    if r == 0:
        continue
    coh = coh_over_rois[:, r] 
    l = ax.semilogy(f_to_plot, coh[:len(f_to_plot)])

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Coherence')
ax.set_title('Coherence for all 360 ROIs across subjects')

#kwargs = {'handlelength':0, 'loc':8, 'frameon':False, 'fontsize':'large'}
#ax.legend(l, ['Intraclass correlation: %.05f' % res[0]], **kwargs)
fig.savefig('coherence_all_subjects_epoched.png')

#Plotting coherence across all subjects and ROIs
full_avg = np.mean(coh_over_rois, axis=1)
plt.clf()
fig, ax = plt.subplots(figsize=[8, 8])
ax.semilogy(f_to_plot, full_avg)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Coherence')
ax.set_title('Coherence averaged over all subjects, ROIs')
ax.set_ylim([.1, 1])
fig.savefig('coherence_full_epoched.png')

df_idx = ['%.03f' % freq for freq in f_to_plot] 
coh_df = pd.DataFrame(coh_over_rois, index=df_idx, columns=rois)
corr_df = coh_df.corr()