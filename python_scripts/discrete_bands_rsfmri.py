# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:16:43 2019
"""

import sys
sys.path.append("..")

import os
import h5py
import time
import datetime
import numpy as np
import proj_utils as pu

#Load data
pdObj = pu.proj_data()
pData = pdObj.get_data()
labels = pData['roiLabels']
database = pData['database']

pdir = pu._get_proj_dir()
path = pdir+'/data/timeseries_rs-fMRI'

filelist = os.listdir(path)
subj_list = sorted(list(set([f.split('_')[0] for f in filelist])))
sess_list = list(set([f.split('_')[2].replace('.mat', '') for f in filelist]))

badSubjects = ['104012', '125525', '151526', '182840', '200109', '500222']

for subject in badSubjects:
    if subject in subj_list: subj_list.remove(subject)

#Define discrete frequency ranges
#Note: rounding errors in np.arange --> trailing 1's, therefore np.around(x, 4)
band_cutoffs = np.around(np.arange(.0005, (1/.72/2), .01), 4) 
bands = []
for c in range(len(band_cutoffs)):
    if band_cutoffs[c] != band_cutoffs[-1]: #if currrent != last
        bands.append(tuple(band_cutoffs[c], band_cutoffs[c + 1]))

#Get spectral data from discrete frequencies
fs = 1/.72
for sess in sess_list:
    session_file = h5py.File("rsfmri_discrete_spectra_level1_%s.hdf5" % sess)
    
    for roi in labels:
        roi_group = session_file.create_group(roi)
        
        amplitude_array = np.ndarrray(shape=[len(subj_list), len(bands)])
        power_array = np.ndarray(shape=[len(subj_list), len(bands)])
        phase_array = np.ndarray(shape=[len(subj_list), len(bands)])
        
        for s, subj in enumerate(subj_list):
            dset = database['/' + subj + '/rsfMRI/' + sess + '/timeseries']

            rsfmri_data = pu.read_database(dset, labels)
            timeseries = rsfmri_data[roi].values
            
            fft_amp = np.absolute(np.fft.rfft(timeseries))
            fft_power = np.absolute(np.fft.rfft(timeseries))**2
            fft_phase = np.angle(np.fft.rfft(timeseries))
            
            fft_freq = np.fft.rfftfreq(len(timeseries), 1.0/fs)
            
            for b, band in enumerate(bands):
                freq_ix = np.where((fft_freq >= band[0]) &
                                   (fft_freq <= band[1]))[0]
                
                amplitude_array[s, b] = np.mean(fft_amp[freq_ix])
                power_array[s, b] = np.mean(fft_power[freq_ix])
                phase_array[s, b] = np.mean(fft_phase[freq_ix])
                
            st = datetime.datetime.fromtimestamp(time.time())
            print('Subject ' + subj + ' complete at ' + st)
        
        roi_group.create_dataset('Amplitude', data=amplitude_array)
        roi_group.create_dataset('Power', data=power_array)
        roi_group.create_dataset('Phase', data=phase_array)
        
    session_file.close()
    
    st = datetime.datetime.fromtimestamp(time.time())
    print('Session ' + sess + ' complete at: ' + st)