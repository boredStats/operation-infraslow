# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:16:03 2018

@author: jah150330
"""
import sys
sys.path.append("..")

import os
import numpy as np
import pandas as pd
import proj_utils as pu

pdObj = pu.proj_data()
pData = pdObj.get_data()
pdir = pu._get_proj_dir()

path = pdir+'/data/timeseries_rs-fMRI'
#os.chdir(path)

labels = pData['roiLabels']
database = pData['database']

filelist = os.listdir(path)
filelist = sorted(filelist,key=str.lower)

subjectList = [filelist[x].split('_') for x in range(0,66960)]
sessionList = [subjectList[x][2].split('.') for x in range(0,66960)]
subjectList = [subjectList[x][0] for x in range(0,66960,720)]
sessionList = [sessionList[x][0] for x in range(0,2)]

badSubjects = list(['104012','125525','151526','182840','200109','500222'])

for subject in badSubjects:
    if subject in subjectList: subjectList.remove(subject)

# Define EEG bands
eeg_bands = {'BOLD': (.0005, 1/.72/2)}   # Bandpass range for HCP rs-fMRI

iterables = [sessionList,subjectList,['Amplitude','Power','Phase']]
names = ['Session','Subject','Spectrum']

dfIdx = pd.MultiIndex.from_product(iterables,names=names)
df = pd.DataFrame(index=dfIdx,columns=labels)

fs = 1/.72

import time
import datetime

for session in sessionList:
    for subject in subjectList:
        dset = database['/'+subject+'/rsfMRI/'+
                        session+'/timeseries']

        MEGdata = pu.read_database(dset, labels)
        MEGdata = MEGdata.values
        
        for ROIindex in range(0,360):        
            data = MEGdata[:,ROIindex]
            label = labels[ROIindex]
    
            # Get real amplitudes of FFT (only in postive frequencies)
            # Squared to get power
            fft_amp = np.absolute(np.fft.rfft(data))
            fft_power = np.absolute(np.fft.rfft(data))**2
            fft_phase = np.angle(np.fft.rfft(data))
            
            # Get frequencies for amplitudes in Hz
            fft_freq = np.fft.rfftfreq(len(data), 1.0/fs)
            
            # Take the mean of the fft amplitude for each EEG band
            eeg_band_fft = dict()
            for band in eeg_bands:  
                freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                                   (fft_freq <= eeg_bands[band][1]))[0]
                
                eeg_band_fft[band] = dict()
                eeg_band_fft[band]['Amp'] = np.mean(fft_amp[freq_ix])
                eeg_band_fft[band]['Power'] = np.mean(fft_power[freq_ix])
                eeg_band_fft[band]['Phase'] = np.mean(fft_phase[freq_ix])
                
                df.at[(session,subject,'Amplitude',band),label] = eeg_band_fft[band]['Amp']
                df.at[(session,subject,'Power',band),label] = eeg_band_fft[band]['Power']
                df.at[(session,subject,'Phase',band),label] = eeg_band_fft[band]['Phase']
                
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            
        print('Subject '+subject+' complete at '+st)
        
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    
    print('Session '+session+' complete at: '+st)

#def plot_bands(data, label):
#    # Plot the data (using pandas here cause it's easy)
#    df = pd.DataFrame(columns=['band', 'val'])
#    df['band'] = eeg_bands.keys()
#    df['val'] = [eeg_band_fft[band] for band in eeg_bands]
#    ax = df.plot.bar(x='band', y='val', legend=False)
#    ax.set_xlabel("EEG band")
#    ax.set_ylabel("Mean band power")
#    ax.set_title('Subject '+subjectList[0]+', ROI: '+label)
# 
#    return None

#plot_bands(data,label)