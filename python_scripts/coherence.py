# -*- coding: utf-8 -*-
"""
Coherence measures between MEG and fMRI

Created on Mon Jan 28 09:53:41 2019
"""
import sys
sys.path.append("..")

import os
import h5py
import datetime
import numpy as np
import pandas as pd
import proj_utils as pu
import matplotlib.pyplot as plt
from scipy.signal import coherence
from nipype.algorithms import icc

#Load data
pdObj = pu.proj_data()
pdir = pu._get_proj_dir()

pData = pdObj.get_data()
rois = pData['roiLabels']
database = pData['database']

#Getting fmri info
fmri_subj_path = pdir+'/data/timeseries_rs-fMRI'
files = os.listdir(fmri_subj_path)
fmri_subj = sorted(list(set([f.split('_')[0] for f in files])))
fmri_sess = list(set([f.split('_')[2].replace('.mat', '') for f in files]))

badSubjects = ['104012', '125525', '151526', '182840', '200109', '500222']

for subject in badSubjects:
    if subject in fmri_subj: 
        fmri_subj.remove(subject)
        
#Getting MEG info
path = pdir+'/data/timeseries_MEG'
files = os.listdir(path)
meg_files = sorted(files,key=str.lower)

meg_subj = sorted(list(set([f.split('_')[0] for f in meg_files])))
meg_sess = list(set([f.split('_')[-1].replace('.mat', '') for f in meg_files]))
meg_sess = sorted(meg_sess)

#Loading single subject
dset = database['/' + fmri_subj[0] + '/rsfMRI/' + fmri_sess[0] + '/timeseries']
fmri_data = pu.read_database(dset, rois)

meg_downsamp = h5py.File(os.path.join(pdir, 'data/resampled_MEG_data.hdf5'))
dset = meg_downsamp['/' + meg_subj[0] + '/MEG/' + meg_sess[0] + '/resampled']
meg_data = pu.read_database(dset, rois)

#Params
fs = 1/.72
max_f = 1/.72/2

#Running coherence on one subject, one ROI
ts_1 = meg_data[rois[0]].values
ts_2 = fmri_data[rois[0]].values
freqs, coh = coherence(ts_1, ts_2, fs=fs)

f_to_plot = freqs[freqs < max_f]
plt.semilogy(f_to_plot, coh[:len(f_to_plot)])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence')

#Running coherence on one subject, all ROIs
coherence_roi_mat = np.ndarray(shape=[len(f_to_plot), len(rois)])
for r, roi in enumerate(rois):
    ts_1 = meg_data[roi].values
    ts_2 = fmri_data[roi].values
    
    pad_lengths = (0, len(ts_2) - len(ts_1))
    ts_1_padded = np.pad(ts_1, pad_lengths, 'constant', constant_values=0)
    
    freqs, coh = coherence(ts_1_padded, ts_2, fs=fs)
    coherence_roi_mat[:, r] = coh[:len(f_to_plot)]

plt.clf()
fig, ax = plt.subplots(figsize=[8, 8])
for r in range(len(rois)):
    coh = coherence_roi_mat[:, r] 
    l = ax.semilogy(f_to_plot, coh[:len(f_to_plot)])
#res = icc.ICC_rep_anova(coherence_roi_mat.T) 

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Coherence')
ax.set_title('Coherence for subject %s for all 360 ROIs' % str(meg_subj[0]))

#kwargs = {'handlelength':0, 'loc':8, 'frameon':False, 'fontsize':'large'}
#ax.legend(l, ['Intraclass correlation: %.05f' % res[0]], **kwargs)
fig.savefig('coherence_%s.png' % str(meg_subj[0]))

#Averaging across ROIs for one subject
avg_coh = np.mean(coherence_roi_mat, axis=1)

plt.clf()
fig, ax = plt.subplots(figsize=[8, 8])
ax.semilogy(f_to_plot, avg_coh)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Coherence')
ax.set_title('Coherence avearaged across 360 ROIs')

#Calculating coherence for all subjects, ROIs
subject_overlap = [s for s in fmri_subj if s in meg_subj]
coh_full = np.ndarray(shape=[len(f_to_plot), len(rois), len(subject_overlap)])
for s, subj in enumerate(subject_overlap):
    t = datetime.datetime.now()
    print('%s Calculating coherence for subject %s' % (t, str(subj)))
    
    dset = database['/' + subj + '/rsfMRI/' + fmri_sess[0] + '/timeseries']
    fmri_data = pu.read_database(dset, rois)
    
    dset = meg_downsamp['/' + subj + '/MEG/' + meg_sess[0] + '/resampled']
    meg_data = pu.read_database(dset, rois)
    
    coherence_roi_mat = np.ndarray(shape=[len(f_to_plot), len(rois)])
    for r, roi in enumerate(rois):
        ts_1 = meg_data[roi].values
        ts_2 = fmri_data[roi].values
        
        pad_lengths = (0, len(ts_2) - len(ts_1))
        ts_1_padded = np.pad(ts_1, pad_lengths, 'constant', constant_values=0)
        
        freqs, coh = coherence(ts_1_padded, ts_2, fs=fs)
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
fig.savefig('coherence_all_subjects.png')

#Plotting coherence across all subjects and ROIs
full_avg = np.mean(coh_over_rois, axis=1)
plt.clf()
fig, ax = plt.subplots(figsize=[8, 8])
ax.semilogy(f_to_plot, full_avg)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Coherence')
ax.set_title('Coherence averaged over all subjects, ROIs')
ax.set_ylim([.1, .2])
fig.savefig('coherence_full.png')

df_idx = ['%.03f' % freq for freq in f_to_plot] 
coh_df = pd.DataFrame(coh_over_rois, index=df_idx, columns=rois)
corr_df = coh_df.corr()