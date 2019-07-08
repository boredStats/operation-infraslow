# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:00:10 2018

@author: jah150330
"""
import sys
sys.path.append("..")

import numpy as np
import pandas as pd

import proj_utils as pu
pdObj = pu.proj_data()
pData = pdObj.get_data()
pdir = pu._get_proj_dir()

import os
path = pdir+'/data/timeseries_MEG'
os.chdir(path)

labels = pData['roiLabels']
database = pData['database']

filelist = os.listdir(path)
filelist = sorted(filelist,key=str.lower)

subjectList = [filelist[x].split('_') for x in range(0,267)]
sessionList = [subjectList[x][2].split('.') for x in range(0,267)]
subjectList = [subjectList[x][0] for x in range(0,267,3)]
sessionList = [sessionList[x][0] for x in range(0,3)]

if '169040' in subjectList: subjectList.remove('169040')
if '662551' in subjectList: subjectList.remove('662551')

# Define MEG bands
MEG_bands = {'BOLD': (.0005, 1/.72/2),   # Bandpass range for HCP rs-fMRI
             'Slow 4': (.02, .06),
             'Slow 3': (.06, .2),
             'Slow 2': (.2, .5),
             'Slow 1': (.5, 1.5),
             'Delta': (1.5, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 55)}

iterables = [sessionList,subjectList,['Amplitude','Power','Phase'],MEG_bands]
names = ['Session','Subject','Spectrum','Freq. Band']

dfIdx = pd.MultiIndex.from_product(iterables,names=names)
df = pd.DataFrame(index=dfIdx,columns=labels)

fs = 2034.51

import time
import datetime

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

def processInput(band,temp):
    freq_ix = np.where((fft_freq >= MEG_bands[band][0]) & 
                       (fft_freq <= MEG_bands[band][1]))[0]
    
    
    temp['Amp'] = np.mean(fft_amp[freq_ix])
    temp['Power'] = np.mean(fft_power[freq_ix])
    temp['Phase'] = np.mean(fft_phase[freq_ix])
    
    return temp

for session in sessionList:
    for subject in subjectList:
        dset = database['/'+subject+'/MEG/'+
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
            
            # Take the mean of the fft amplitude for each MEG band
            MEG_band_fft = dict()
            
            for band in MEG_bands:
                MEG_band_fft[band] = dict()
                
#            Parallel(n_jobs=num_cores)(delayed(processInput)(band,MEG_band_fft[band]) for band in MEG_bands)
#            
#            df.at[(session,subject,'Amplitude',band),label] = temp['Amp']
#            df.at[(session,subject,'Power',band),label] = temp['Power']
#            df.at[(session,subject,'Phase',band),label] = temp['Phase']
#                    
#            ts = time.time()
#            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
#                
#            print('Subject '+subject+' complete at '+st)
#        
#    ts = time.time()
#    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
#    
#    print(session+' complete at: '+st)

