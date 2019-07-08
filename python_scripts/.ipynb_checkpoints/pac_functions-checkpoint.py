# -*- coding: utf-8 -*-
"""
Cognitive and Neural Dynamics Lab Tutorials
Phase-Amplitude Coupling (PAC)
Torben Noto 2015

https://github.com/voytekresearch/tutorials

Functionized for her pleasure

Created on Mon Feb  4 15:57:35 2019
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, hilbert, sosfilt

def butter_filter(timeseries, fs, cutoffs, btype='band', order=4):
    #Scipy v1.2.0
    #Note - copied to proj_utils
    nyquist = fs/2
    butter_cut = np.divide(cutoffs, nyquist) #butterworth param (digital)
    sos = butter(order, butter_cut, output='sos', btype=btype)
    return sosfilt(sos, timeseries)

def get_phase_amp_data(data, fs, phase_band=None, amp_band=None):
    phase_banded = butter_filter(data, fs, phase_band)
    phase_hilbert = hilbert(phase_banded)
    phase_data = np.angle(phase_hilbert)

    amp_banded = butter_filter(data, fs, amp_band)
    amp_hilbert = hilbert(amp_banded)
    amp_data = np.absolute(amp_hilbert)

    return phase_data, amp_data

def plot_raw_phase_amp(timeseries, phase_data, amp_data, fs):
    plt.figure(figsize = (15,6))
    ts_to_plot = timeseries[1:int(fs)*2]
    amp_to_plot = amp_data[1:int(fs)*2]
    
    normalized_amp = (amp_to_plot - np.mean(amp_to_plot)) / np.std(amp_to_plot)
    normalized_ts = (ts_to_plot - np.mean(ts_to_plot)) / np.std(ts_to_plot)
    
    plt.plot(normalized_ts, label= 'Raw Data') #normalized raw data
    plt.plot(phase_data[1:int(fs)*2], label='Phase')
    plt.plot(normalized_amp, label='Amplitude') 
    plt.xlabel('Two Seconds')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def circCorr(ang,line):
    #Correlate periodic data with linear data
    n = len(ang)
    rxs = sp.stats.pearsonr(line,np.sin(ang))
    rxs = rxs[0]
    rxc = sp.stats.pearsonr(line,np.cos(ang))
    rxc = rxc[0]
    rcs = sp.stats.pearsonr(np.sin(ang),np.cos(ang))
    rcs = rcs[0]
    rho = np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs)/(1-rcs**2)) #r
    #r_2 = rho**2 #r squared
    pval = 1- sp.stats.chi2.cdf(n*(rho**2),1)
    #standard_error = np.sqrt((1-r_2)/(n-2))

    return rho, pval#, r_2,standard_error

if __name__ == "__main__":
    import os
    import sys
    sys.path.append("..")
    import proj_utils as pu
     
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
    
    print('%s: Single subject test' % pu.ctime())
    subj = meg_subj[0]
    sess = meg_sess[0]
    roi = rois[50]
    
    dset = database['/'+ subj +'/MEG/'+ sess +'/timeseries']
    meg_data = pu.read_database(dset, rois)
    timeseries = meg_data[roi]

    phase_data, amp_data = get_phase_amp_data(timeseries, fs)
    plot_raw_phase_amp(timeseries, phase_data, amp_data, fs)
    res = circCorr(phase_data, amp_data)