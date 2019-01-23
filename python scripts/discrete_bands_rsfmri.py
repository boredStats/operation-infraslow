# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:16:43 2019
"""

import sys
sys.path.append("..")

import os
import h5py
import numpy as np
import pandas as pd
import proj_utils as pu
import time
import datetime

pdObj = pu.proj_data()
pData = pdObj.get_data()
pdir = pu._get_proj_dir()

path = pdir+'/data/timeseries_rs-fMRI'

labels = pData['roiLabels']
database = pData['database']

filelist = os.listdir(path)
filelist = sorted(filelist,key=str.lower)

all_files = [x.split('_')[0] for x in filelist]
raw_sess_list = [x.split('_')[2].replace('.mat', '') for x in filelist]
subjectList = sorted(list(set(all_files)))[0: 2] #get unique in subj codes, to list 
sessionList = sorted(list(set(raw_sess_list)))[0: 2] #get unique session labels

badSubjects = ['104012', '125525', '151526', '182840', '200109', '500222']

for subject in badSubjects:
    if subject in subjectList: subjectList.remove(subject)

#Define discrete EEG bands
#Note: rounding errors in np.arange == trailing 1's, therefore np.around(x, 4)
band_cutoffs = np.around(np.arange(.0005, (1/.72/2), .01), 4) 
bands = []
for c in range(len(band_cutoffs)):
    if band_cutoffs[c] != band_cutoffs[-1]: #if currrent != last
        bands.append(tuple((band_cutoffs[c], band_cutoffs[c+1])))

band_labels = [str(band[0]) + " to " + str(band[1]) for band in bands]

#Generate dictionary key/values iteratively 
eeg_bands = {band_labels[x]: bands[x] for x in range(len(band_labels))}

#iterables = [sessionList,subjectList,band_labels,['Amplitude','Power','Phase']]
#names = ['Session','Subject','Band','Spectrum']
#
#dfIdx = pd.MultiIndex.from_product(iterables,names=names)
#df = pd.DataFrame(index=dfIdx,columns=labels)

#Get ROI labels
subject = subjectList[0]
session = sessionList[0]
dset = database['/'+subject+'/rsfMRI/'+session+'/timeseries']
MEGdata = pu.read_database(dset, labels)
rois = list(MEGdata.columns.values)

fs = 1/.72
session_data = {}
for sess in sessionList:
    session_file = h5py.File("rsfmri_discrete_spectra_level1_%s.hdf5" % sess)
    
    for roi in rois:
        roi_group = session_file.create_group(roi)
        
        array_shape = [len(subjectList), len(band_labels)]
        amplitude_array = np.ndarrray(shape=array_shape)
        power_array = np.ndarray(shape=array_shape)
        phase_array = np.ndarray(shape=array_shape)
#        amplitude_array = pd.DataFrame(index=subjectList, columns=band_labels)
#        power_array = pd.DataFrame(index=subjectList, columns=band_labels)
#        phase_array = pd.DataFrame(index=subjectList, columns=band_labels)
        
        for s, subj in enumerate(subjectList):
            dset = database['/' + subj + '/rsfMRI/' + sess + '/timeseries']

            rsfmri_data = pu.read_database(dset, labels)
            timeseries = rsfmri_data[roi].values
            
            fft_amp = np.absolute(np.fft.rfft(timeseries))
            fft_power = np.absolute(np.fft.rfft(timeseries))**2
            fft_phase = np.angle(np.fft.rfft(timeseries))
            
            fft_freq = np.fft.rfftfreq(len(timeseries), 1.0/fs)
            
            for b, band in enumerate(eeg_bands):
                freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                   (fft_freq <= eeg_bands[band][1]))[0]
                
                amplitude_array[s, b] = np.mean(fft_amp[freq_ix])
                power_array[s, b] = np.mean(fft_power[freq_ix])
                phase_array[s, b] = np.mean(fft_phase[freq_ix])
                
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            print('Subject ' + subj + ' complete at ' + st)
        
        roi_group.create_dataset('Amplitude', data=amplitude_array)
        roi_group.create_dataset('Power', data=power_array)
        roi_group.create_dataset('Phase', data=phase_array)
        
    session_file.close()
    
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print('Session ' + sess + ' complete at: ' + st)