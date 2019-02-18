# -*- coding: utf-8 -*-
"""
Permutation testing on coherence

Created on Mon Feb 11 14:40:21 2019
"""

import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import pac_functions as pac
from scipy.signal import coherence, resample

import sys
sys.path.append("..")
import proj_utils as pu

pdir = pu._get_proj_dir()
pdObj = pu.proj_data()
pData = pdObj.get_data()
database = pData['database']
rois = pData['roiLabels']
meg_subj, meg_sess = pdObj.get_meg_metadata()
fmri_subj, fmri_sess = pdObj.get_mri_metadata()
subj_overlap = [s for s in fmri_subj if s in meg_subj]

fs = 1/.72
max_time = 328 #maximum timepoints of fmri to grab based on downsampled MEG

#Constants for on the fly downsampling/filtering
max_f = fs/2
trunc_meg = 118088
meg_downsampled_sampling_rate = 500 #from Hye
meg_con = (trunc_meg * fs ) / meg_downsampled_sampling_rate

downsamp_meg_file = os.path.join(pdir, 'data/downsampled_MEG_truncated.hdf5')
meg_timeseries = np.ndarray(shape=[max_time, len(rois), len(subj_overlap)])
fmri_timeseries = np.ndarray(shape=[max_time, len(rois), len(subj_overlap)])

meg_lengths = []

print('%s: Building timeseries matrices' % pu.ctime())
for s, subj in enumerate(subj_overlap):
    dname = database['/' + subj + '/rsfMRI/' + fmri_sess[0] + '/timeseries']
    fmri_data = pu.read_database(dname, rois)
    
    fmri_timeseries[:, :, s] = fmri_data.values[:max_time, :]
    
    meg_h5 = h5.File(downsamp_meg_file, 'r')
    dset = meg_h5['/' + subj + '/MEG/' + meg_sess[0] + '/resampled_truncated']
    meg_data = dset.value
    meg_h5.close()
    meg_timeseries[:, :, s] = meg_data
    
    #Version where MEG filtering occurs before downsampling
#    dset_name = database['/' + subj + '/MEG/' + meg_sess[0] + '/timeseries']
#    meg_data = pu.read_database(dset_name, rois).values
#    filt_data = pac.butter_filter(meg_data[:trunc_meg, :], 500, max_f, 'low')
#    downsamp_data = resample(filt_data, int(meg_con))
#    meg_timeseries[:, :, s] = downsamp_data
    
    del fmri_data, meg_data

freq, coh = coherence(meg_timeseries[:, 0, 0], fmri_timeseries[:, 0, 0], fs=fs)
coh = coh-1
n_iters = 1000
n = 0
coherence_perm = np.ndarray(shape=[n_iters, len(coh)])
while n != n_iters:
    print('%s: Running permutation coherence %d' % (pu.ctime(), n+1))
    rand_subj = np.random.randint(0, len(subj_overlap))
    rand_roi = np.random.randint(0, len(rois))
    fmri_rand_ts = fmri_timeseries[:, rand_roi, rand_subj]
    
    rand_subj = np.random.randint(0, len(subj_overlap))
    rand_roi = np.random.randint(0, len(rois))
    meg_rand_ts = meg_timeseries[:, rand_roi, rand_subj]
    
    _, coh_perm = coherence(fmri_rand_ts, meg_rand_ts, fs=fs)
    coherence_perm[n, :] = coh_perm
    n += 1
    
print('%s: Finished calculating permutation ' % pu.ctime())

#Plotting average coherence of permuted data
avg_coherence_perm = np.mean(coherence_perm, axis=0)    
f_to_plot = freq[freq < max_f]

fig, ax = plt.subplots(figsize=[8, 8])
l = ax.semilogy(f_to_plot, avg_coherence_perm[:len(f_to_plot)])
ax.set_ylim([.1, 2])