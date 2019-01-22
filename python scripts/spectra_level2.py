# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:51:10 2018

@author: jah150330
"""
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

import proj_utils as pu
pdObj = pu.proj_data()
pData = pdObj.get_data()
pdir = pu._get_proj_dir()

dfSpectra = pd.read_pickle(pdir+'\data\MEG_AmpPowPha_spectra.pkl')
dfMeanStd = pd.read_pickle(pdir+'\data\MEG_AmpPowPha_spectra_level2.pkl')

import os
path = pdir+'/data/timeseries_MEG'

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

# Define EEG bands
eeg_bands = {'BOLD': (.0005, 1/.72/2),   # Bandpass range for HCP rs-fMRI
             'Slow 4': (.02, .06),
             'Slow 3': (.06, .2),
             'Slow 2': (.2, .5),
             'Slow 1': (.5, 1.5),
             'Delta': (1.5, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 55)}

iterables = [sessionList,eeg_bands,['Amplitude','Power','Phase'],['mean','std']]
names = ['Session','Freq. Band','Spectrum','Statistic']

dfIdx = pd.MultiIndex.from_product(iterables,names=names)
df2 = pd.DataFrame(index=dfIdx,columns=labels)

#def plot_phase_amplitude(df,label,session='Session3'):
#    rMatrix = np.zeros((10,10))
#    pMatrix = np.zeros((10,10))
#    
#    keyList = list(eeg_bands.keys())
#    
#    for bandX in eeg_bands:
#        x = df.loc[pd.IndexSlice[session,:,'Phase',bandX],label]
#        
#        for bandY in eeg_bands:
#            y = df.loc[pd.IndexSlice[session,:,'Amplitude',bandY],label]
#            
#            (rMatrix[keyList.index(bandX),keyList.index(bandY)],
#            pMatrix[keyList.index(bandX),keyList.index(bandY)])= scipy.stats.pearsonr(x,y)
#    
##    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
#    plt.figure()
#    sns.set(rc={'figure.figsize':(9.75,8.27)})
#    sns.set_context("talk")
#    
#    ax = sns.heatmap(rMatrix, 
#                xticklabels = eeg_bands.keys(),
#                yticklabels = eeg_bands.keys(),
#                vmin=0,vmax=1,annot=True)
#    
#    for tick in ax.get_xticklabels():
#        tick.set_rotation(0)
#        tick.set_horizontalalignment('center')
#        
#    fsize = 24
#    lpad = 15
#    ax.set_title('ROI: '+label, fontsize=fsize,pad=lpad)
#    ax.set_xlabel("Phase Spectrum",fontsize=fsize,labelpad=lpad)
#    ax.set_ylabel("Amplitude Spectrum",fontsize=fsize,labelpad=lpad)
#    
#    fig = ax.get_figure()
#    fig.savefig(pdir+'/figures/PAC/Session 3/PAC_'+label+'.png')
#        
#    return None
#
#for ROIindex in range(0,360):
#    plot_phase_amplitude(dfSpectra,label=labels[ROIindex])

def plot_power_power(df,label,session):
    rMatrix = np.zeros((10,10))
    pMatrix = np.zeros((10,10))
    
    keyList = list(eeg_bands.keys())
    
    for bandX in eeg_bands:
        x = df.loc[pd.IndexSlice[session,:,'Power',bandX],label]
        
        for bandY in eeg_bands:
            y = df.loc[pd.IndexSlice[session,:,'Power',bandY],label]
            
            (rMatrix[keyList.index(bandX),keyList.index(bandY)],
            pMatrix[keyList.index(bandX),keyList.index(bandY)])= scipy.stats.pearsonr(x,y)
    
#    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    plt.figure()
    sns.set(rc={'figure.figsize':(9.75,8.27)})
    sns.set_context("talk")
    
    ax = sns.heatmap(rMatrix, 
                xticklabels = eeg_bands.keys(),
                yticklabels = eeg_bands.keys(),
                vmin=0,vmax=1,annot=True)
    
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_horizontalalignment('center')
        
    fsize = 24
    lpad = 15
    ax.set_title('ROI: '+label, fontsize=fsize,pad=lpad)
    ax.set_xlabel("Power Spectrum",fontsize=fsize,labelpad=lpad)
    ax.set_ylabel("Power Spectrum",fontsize=fsize,labelpad=lpad)
    
    fig = ax.get_figure()
    fig.savefig(pdir+'/figures/Power-Power/'+session+'/PowerPower_'+label+'.png')
        
    return None

for session in sessionList:
    for ROIindex in range(0,360):
        plot_power_power(dfSpectra,label=labels[ROIindex],session=session)
