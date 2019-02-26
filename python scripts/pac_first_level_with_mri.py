# -*- coding: utf-8 -*-
"""
Get phase, amplitude data for the fMRI signal
Run CFC with circular correlation (fMRI phase band with MEG amp bands)

Created on Tue Feb 19 09:24:38 2019
"""
import h5py
import numpy as np
import pandas as pd
import pac_functions as pac
from scipy.signal import hilbert, resample, butter, sosfilt

def butter_filter(timeseries, fs, cutoffs, btype='band', order=4):
    #Scipy v1.2.0
    #Note - copied to proj_utils
    nyquist = fs/2
    butter_cut = np.divide(cutoffs, nyquist) #butterworth param (digital)
    sos = butter(order, butter_cut, output='sos', btype=btype)
    return sosfilt(sos, timeseries)

def get_phase_amp_no_filter(data):
    phase_hilbert = hilbert(data)
    phase_data = np.angle(phase_hilbert)
    
    amp_hilbert = hilbert(data)
    amp_data = np.absolute(amp_hilbert)
    
    return phase_data, amp_data
    
def build_output(subj_data, rois, mri_check=False, band=None, fs=None):
    #Extract phase/amplitude data by ROI
    timeseries_length = len(subj_data[rois[0]])
    phase_mat = np.ndarray(shape=[timeseries_length, len(rois)])
    amp_mat = np.ndarray(shape=[timeseries_length, len(rois)])
    
    for r, roi in enumerate(rois):
        if mri_check is False:
            phase, amp = pac.get_phase_amp_data(subj_data[roi], fs, band, band)
        else:
            phase, amp = get_phase_amp_no_filter(subj_data[roi])
        phase_mat[:, r] = phase
        amp_mat[:, r] = amp
    
    return phase_mat, amp_mat

def cir_corr(mri_dataset, meg_dataset, roi_index, mri_band, reg_bands):
    #Modified corr_bands function from pac_first_level
    #Takes in two datasets, as well as lim (to truncate MRI signal)
    
    r_mat = np.ndarray(shape=[len(reg_bands)])
    p_mat = np.ndarray(shape=[len(reg_bands)])

    slow_group = mri_dataset.get(mri_band)
    slow_ts = slow_group.get('phase_data')[:, roi_index]
    
    for reg_index, reg in enumerate(reg_bands):
        reg_group = meg_dataset.get(reg)
        reg_ts = reg_group.get('amplitude_data')[:, roi_index]
        
        r_val, p_val = pac.circCorr(slow_ts, reg_ts)
        r_mat[reg_index] = r_val
        p_mat[reg_index] = p_val

    return r_mat, p_mat

import sys
sys.path.append("..")
import proj_utils as pu
start = pu.ctime()
print('%s: Starting' % pu.ctime())

print('%s: Getting metadata, parameters' % pu.ctime())
pdir = pu._get_proj_dir()

pdObj = pu.proj_data()
meg_subj, meg_sess = pdObj.get_meg_metadata()
mri_subj, mri_sess = pdObj.get_mri_metadata()
subj_overlap = [s for s in mri_subj if s in meg_subj]

pData = pdObj.get_data()
rois = pData['roiLabels']
database = pData['database']
band_dict = pData['bands']
min_meg_length = 111980 #118088

fs = 1/.72
band = band_dict['BOLD']
reg_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

output_path = pdir + '/data/rsfMRI_phase_amp_data.hdf5'
print('%s: Extracting MRI phase/amp data' % pu.ctime())
for sess in mri_sess:
    for subj in mri_subj:
        print('%s: %s %s' % (pu.ctime(), sess, str(subj)))
        dset = database['/'+ subj +'/rsfMRI/'+ sess +'/timeseries']
        mri_data = pu.read_database(dset, rois)
        upsampled = resample(mri_data.values, num=min_meg_length)
        upsampled_df = pd.DataFrame(upsampled, columns=rois)

        phase_mat, amp_mat = build_output(upsampled_df, rois, mri_check=True)
        
        h5 = h5py.File(output_path)
        grp = h5.require_group('/' + subj + '/' + sess + '/' + 'BOLD')
        grp.create_dataset('phase_data', data=phase_mat)
        grp.create_dataset('amplitude_data', data=amp_mat)
        h5.close()

meg_fs = 500
output_path = pdir + '/data/MEG_truncated_phase_amp_data.hdf5'
print('%s: Extracting MEG resampled phase/amp data' % pu.ctime())
#list_of_timeseries_lengths = []
for sess in meg_sess:
    for subj in meg_subj:
        for b in band_dict:
            band = band_dict[b]
            print('%s: %s %s' % (pu.ctime(), sess, str(subj)))#, b))
            dset_to_load = database[subj + '/MEG/' + sess + '/timeseries']
            meg_data = pu.read_database(dset_to_load, rois)
            ts_length = meg_data.values.shape[0]
    #        list_of_timeseries_lengths.append(ts_length)
            
            meg_trunc = meg_data.values[:min_meg_length, :]
            meg_data_trunc = pd.DataFrame(meg_trunc, columns=rois)
            
            phase_mat, amp_mat = build_output(meg_data_trunc, rois,
                                              band=band, fs=meg_fs)
            
            h5 = h5py.File(output_path)
            grp = h5.require_group('/' + subj + '/' + sess + '/' + b)
            grp.create_dataset('phase_data', data=phase_mat)
            grp.create_dataset('amplitude_data', data=amp_mat)
            h5.close()

meg_phase_amp = pdir + '/data/MEG_truncated_phase_amp_data.hdf5'
mri_phase_amp = pdir + '/data/rsfMRI_phase_amp_data.hdf5'
out_path = pdir + '/data/MEG_pac_first_level_with_MRI_.hdf5'

print('%s: Calculating phase-amplitude coupling' % pu.ctime())
for mris in mri_sess:
    for megs in meg_sess:
        session_combo = mris + '_' + megs
        
        for subj in subj_overlap:
            mri_file = h5py.File(mri_phase_amp, 'r')
            mri_dataset = mri_file.require_group('/' + subj + '/' + mris)
            mri_group = mri_dataset.get('BOLD')
            
            phase_data = mri_group.get('phase_data')
            phase_matrix =  phase_data[:, :]
            
            meg_file = h5py.File(meg_phase_amp, 'r')
            meg_dataset = meg_file.require_group('/' + subj + '/' + megs)
            
            for b in reg_bands: #list(band_dict):
                print('%s: %s %s %s' % (pu.ctime(), session_combo, subj, b))
                meg_group = meg_dataset.get(b)
                amp_data = meg_group.get('amplitude_data')
                amp_matrix = amp_data[:, :]
                    
                r_list, p_list  = [], []
                
                for roi_index, roi in enumerate(rois):
                    phase_vector = phase_matrix[:, roi_index]
                    amp_vector = amp_matrix[:, roi_index]
                    r, p = pac.circCorr(phase_vector, amp_vector)
                    
                    r_list.append(r)
                    p_list.append(p)

                first_level_file = h5py.File(out_path)
                
                current_group = subj + '/' + session_combo + '/' + b
                grp = first_level_file.require_group(current_group)
                
                grp.create_dataset('r_vals', data=np.asarray(r_list))
                grp.create_dataset('p_vals', data=np.asarray(p_list))
                
                first_level_file.close()
                
            mri_file.close()
            meg_file.close()
    
print('%s: Finished' % pu.ctime())
print('Started at %s' % start)