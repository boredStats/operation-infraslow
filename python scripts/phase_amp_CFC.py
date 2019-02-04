# -*- coding: utf-8 -*-
"""
Calculate phase-amplitude cross-frequency coupling

Created on Thu Jan 31 09:32:47 2019
"""

import os
import h5py
import numpy as np

import sys
sys.path.append("..")
import proj_utils as pu

pdir = pu._get_proj_dir()
pdObj = pu.proj_data()
pData = pdObj.get_data()
rois = pData['roiLabels']
database = pData['database']

print('%s: Getting metadata' % pu.ctime())
meg_subj_path = pdir + '/data/timeseries_MEG'
files = sorted(os.listdir(meg_subj_path), key=str.lower)

meg_subj = list(set([f.split('_')[0] for f in files]))
meg_subj = sorted(meg_subj)
bad_meg_subj = ['169040', '662551']
for bad in bad_meg_subj:
    if bad in meg_subj:
        meg_subj.remove(bad)

meg_sess = list(set([f.split('_')[-1].replace('.mat', '') for f in files]))
meg_sess = sorted(meg_sess)

freq_limits = ['full_range', 'low_pass_100Hz']
dset_names = ['frequencies', 'amplitude_data', 'phase_data']

band_cutoffs = np.around(np.arange(0, 250, .01), 2) 
bands = []
for c in range(len(band_cutoffs)):
    if band_cutoffs[c] != band_cutoffs[-1]: #if currrent != last
        band = (band_cutoffs[c], band_cutoffs[c + 1])
        bands.append(band)

for b, band in enumerate(bands):
    if band[1] <= 12:
        alpha_index = b
    if band[1] <= 55:
        gamma_index = b

maxf = freq_limits[0]
subj = meg_subj[0]
sess = meg_sess[0]
data_file = pdir + '/data/MEG_phase_amplitude_discrete.hdf5'
h = h5py.File(data_file, 'r')

top_level = h.require_group(maxf)
subj_level = top_level.require_group(subj)
sess_level = subj_level.get(sess)

freqs = np.array(sess_level[dset_names[0]])[:, 0]
amp_data = np.array(sess_level[dset_names[1]])
phase_data = np.array(sess_level[dset_names[2]])
h.close()

print('%s: %s %s %s averaging Fs' % (pu.ctime(), maxf, sess, subj))
avg_matrix_shape = [len(bands), len(rois)]
amp_matrix = np.ndarray(shape=avg_matrix_shape)
phase_matrix = np.ndarray(shape=avg_matrix_shape)
for b, band in enumerate(bands):
    f_range = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    for r in range(len(rois)):
        amp_matrix[b, r] = np.mean(amp_data[f_range, r])
        phase_matrix[b, r] = np.mean(phase_data[f_range, r])
del freqs, amp_data, phase_data
  
print('%s: Starting CFC' % pu.ctime())
for maxf in freq_limits:
    for sess in meg_sess:
        for s, subj in enumerate(meg_subj):
            print('%s: %s %s %s loading data' % (pu.ctime(), maxf, sess, subj))
            data_file = pdir + '/data/MEG_phase_amplitude_discrete_001Hz_bin.hdf5'
            h = h5py.File(data_file, 'r')
            
            top_level = h.require_group(maxf)
            subj_level = top_level.require_group(subj)
            sess_level = subj_level.get(sess)
            
            freqs = np.array(sess_level[dset_names[0]])[:, 0]
            amp_data = np.array(sess_level[dset_names[1]])
            phase_data = np.array(sess_level[dset_names[2]])
            h.close()
        
            print('%s: %s %s %s averaging Fs' % (pu.ctime(), maxf, sess, subj))
            avg_matrix_shape = [len(bands), len(rois)]
            amp_matrix = np.ndarray(shape=avg_matrix_shape)
            phase_matrix = np.ndarray(shape=avg_matrix_shape)
            for b, band in enumerate(bands):
                f_range = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
                for r in range(len(rois)):
                    amp_matrix[b, r] = np.mean(amp_data[f_range, r])
                    phase_matrix[b, r] = np.mean(phase_data[f_range, r])
            del freqs, amp_data, phase_data
                
            print('%s: %s %s %s calc CFC' % (pu.ctime(), maxf, sess, subj))
            cfc = pu.super_corr(phase_matrix.T[:, :alpha_index],
                                amp_matrix.T[:, :gamma_index])
            
            out_file = pdir + '/data/MEG_phase_amplitude_discrete_CFC.hdf5'
            out = h5py.File(out_file, 'a')
            
            grp = out.require_group(maxf + '/' + sess)
            grp.create_dataset(subj, data=cfc)
            out.close()
            
            del amp_matrix, phase_matrix, cfc
        break #Single session
    break #Full range
print('%s: Finished' % pu.ctime())