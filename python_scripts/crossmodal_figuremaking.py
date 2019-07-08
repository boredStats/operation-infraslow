# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:25:44 2018

@author: jah150330
"""
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats

import proj_utils as pu
pdObj = pu.proj_data()
pData = pdObj.get_data()
pdir = pu._get_proj_dir()

dfMEG = pd.read_pickle(pdir+'\data\MEG_AmpPowPha_spectra_level2.pkl')
dfMRI = pd.read_pickle(pdir+'/data/rsfMRI_AmpPowPha_spectra_level2.pkl')

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

badSubjects = list(['104012','125525','151526','169040',
                    '182840','200109','500222','662551'])

for subject in badSubjects:
    if subject in subjectList: subjectList.remove(subject)

# Define EEG bands
meg_bands = {'BOLD': (.0005, 1/.72/2),   # Bandpass range for HCP rs-fMRI
             'Slow 4': (.02, .06),
             'Slow 3': (.06, .2),
             'Slow 2': (.2, .5),
             'Slow 1': (.5, 1.5),
             'Delta': (1.5, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 55)}

mri_bands = {'MRI_BOLD': (.0005, 1/.72/2)}   # Bandpass range for HCP rs-fMRI

def plot_power_power(dfMRI,dfMEG,label):
    rMatrix = np.zeros((3,10))
    pMatrix = np.zeros((3,10))
    
    keyList = list(meg_bands.keys())
    
    for session in ['LR','RL']:
        x = dfMRI.loc[pd.IndexSlice[session,:,bandX],label]
        
        for bandY in meg_bands:
            y = dfMEG.loc[pd.IndexSlice[megSession,:,bandY],label]
            
            (rMatrix[keyList.index(bandX),keyList.index(bandY)],
            pMatrix[keyList.index(bandX),keyList.index(bandY)])= scipy.stats.pearsonr(x,y)
    
#    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set(rc={'figure.figsize':(9.75,8.27)})
    sns.set_context("talk")
    
    ax = sns.heatmap(rMatrix, 
                xticklabels = meg_bands.keys(),
                yticklabels = meg_bands.keys(),
                vmin=0,vmax=1,annot=True)
    
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_horizontalalignment('center')
        
    fsize = 24
    lpad = 15
    ax.set_title('ROI: '+label, fontsize=fsize,pad=lpad)
        
    return None