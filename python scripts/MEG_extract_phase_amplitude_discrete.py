# -*- coding: utf-8 -*-
"""
Calculate phase, amplitude for MEG data

Created on Wed Jan 30 09:52:15 2019
"""

import os
import h5py
import numpy as np
from scipy.signal import butter, sosfilt, spectrogram

import sys
sys.path.append("..")
import proj_utils as pu

def butter_filter(timeseries, fs, low_pass_cutoff, order=4):
    nyquist = fs/2
    butter_cut = low_pass_cutoff/nyquist #butterworth filter param (digital)
    sos = butter(order, butter_cut, output='sos')
    return sosfilt(sos, timeseries)
    
def freq_analysis(timeseries, sampling_rate):
    amp = np.absolute(np.fft.rfft(timeseries))
    phase = np.angle(np.fft.rfft(timeseries))
    freqs = np.fft.rfftfreq(len(timeseries), 1.0/sampling_rate)
    return amp, phase, freqs

def freq_analysis_sliding_window(ts, fs, win_len):
    _, _, amp_wins = spectrogram(ts, fs, nperseg=win_len, mode='magnitude')
    freqs , _, phase_wins = spectrogram(ts, fs, nperseg=win_len, mode='angle')
    amp = np.mean(amp_wins, axis=1)
    phase = np.mean(phase_wins, axis=1)
    return amp, phase, freqs
    
pdir = pu._get_proj_dir()
pdObj = pu.proj_data()
pData = pdObj.get_data()
rois = pData['roiLabels']
database = pData['database']

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

fs = 500 #Sampling rate
low_pass_cutoff = 100 #lowpass cutoff
window = 10*fs #window length for 0.1 Hz bin

print('%s: Single subject test' % pu.ctime())
subj = meg_subj[0]
sess = meg_sess[0]
roi = rois[0]

dset = database['/'+ subj +'/MEG/'+ sess +'/timeseries']
meg_data = pu.read_database(dset, rois)
timeseries = meg_data[roi]

f, _, amp_wins = spectrogram(timeseries, fs, nperseg=window, mode='magnitude')
amp = np.mean(amp_wins, 1)

print('%s: Starting' % pu.ctime())
for sess in meg_sess:
    for subj in meg_subj:
        phase_amp_file = pdir + '/data/MEG_phase_amplitude_discrete.hdf5'
        out_file = h5py.File(phase_amp_file)
        
        print('%s: %s %s' % (pu.ctime(), sess, str(subj)))
        dset = database['/'+ subj +'/MEG/'+ sess +'/timeseries']
        meg_data = pu.read_database(dset, rois)
        
        ts_length = meg_data[rois[0]].size
        out_matrix_shape = [int(ts_length / 2), len(rois)]
        
        amp_list, phase_list, freqs_list = [], [], []
        for roi in rois:
            timeseries = meg_data[roi]
            amp, phase, freqs = freq_analysis(timeseries, fs)
            
            amp_list.append(amp)
            phase_list.append(phase)
            freqs_list.append(freqs)
 
        grp = out_file.require_group('/full_range/' + subj + '/' + sess)
        grp.create_dataset('amplitude_data', data=np.asarray(amp_list).T)
        grp.create_dataset('phase_data', data=np.asarray(phase_list).T)
        grp.create_dataset('frequencies', data= np.asarray(freqs_list).T)
        
        amp_list, phase_list, freqs_list = [], [], []
        for roi in rois:
            timeseries = meg_data[roi]
            filtered = butter_filter(timeseries, fs, low_pass_cutoff)
            amp, phase, freqs = freq_analysis(filtered, low_pass_cutoff)
            
            amp_list.append(amp)
            phase_list.append(phase)
            freqs_list.append(freqs)

        lowpass_gname = '/low_pass_%dHz/' % low_pass_cutoff
        grp = out_file.require_group(lowpass_gname + subj + '/' + sess)
        grp.create_dataset('amplitude_data', data=np.asarray(amp_list).T)
        grp.create_dataset('phase_data', data=np.asarray(phase_list).T)
        grp.create_dataset('frequencies', data= np.asarray(freqs_list).T)

        out_file.close()
        
print('%s: Finished' % pu.ctime())