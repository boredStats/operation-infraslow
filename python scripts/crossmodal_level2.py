# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:29:53 2018

@author: jah150330
"""
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import math
import seaborn as sns
import scipy.stats

import proj_utils as pu
pdObj = pu.proj_data()
pData = pdObj.get_data()
pdir = pu._get_proj_dir()

dfMEG = pd.read_pickle(pdir+'\data\MEG_AmpPowPha_spectra_level2.pkl')
dfMRI = pd.read_pickle(pdir+'/data/rsfMRI_AmpPowPha_spectra.pkl')

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

iterables = [['LR','RL'],['Amplitude','Power','Phase'],['mean','std']]
names = ['MR Session','Spectrum','Statistic']

dfIdx = pd.MultiIndex.from_product(iterables,names=names)
df2 = pd.DataFrame(index=dfIdx,columns=labels)

import time
import datetime

for mriSession in list(['LR','RL']):
    for ROI in labels:
        dataAmp = dfMRI.loc[pd.IndexSlice[mriSession,:,'Amplitude'],ROI]
        dataPow = dfMRI.loc[pd.IndexSlice[mriSession,:,'Power'],ROI]
        dataPha = dfMRI.loc[pd.IndexSlice[mriSession,:,'Phase'],ROI]
        
        asdMu = np.mean(dataAmp.values)
        asdSigma = np.std(dataAmp.values)
        df2.at[(mriSession,'Amplitude','mean'),ROI] = asdMu
        df2.at[(mriSession,'Amplitude','std'),ROI] = asdSigma            
        
        psdMu = np.mean(dataPow.values)
        psdSigma = np.std(dataPow.values)
        df2.at[(mriSession,'Power','mean'),ROI] = psdMu
        df2.at[(mriSession,'Power','std'),ROI] = psdSigma
        
        phsdMu = np.mean(dataPha.values)
        phsdSigma = np.std(dataPha.values)            
        df2.at[(mriSession,'Phase','mean'),ROI] = phsdMu
        df2.at[(mriSession,'Phase','std'),ROI] = phsdSigma
        
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        
    print('Session '+mriSession+' complete at '+st)

#def plot_bands(df2, label):
#    df = pd.DataFrame(columns=['band', 'val', 'error'])
#    df['band'] = meg_bands.keys()
#    df['val'] = [df2.at[(megSession,band,'mean'),label] for band in meg_bands]
#    df['error'] = [df2.at[(megSession,band,'std'),label]/math.sqrt(87) for band in meg_bands]
#    
#    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
#    sns.set(rc={'figure.figsize':(11.7,8.27)})
#    sns.set_context("poster")
#    ax = df.plot.bar(x='band', y='val', yerr='error', legend=False,
#                     color=sns.cubehelix_palette(10, start=4.5, rot=-1,
#                                                 reverse=True))
#    fsize = 24
#    lpad = 15
#    ax.set_xlabel("MEG frequency band",fontsize=fsize,labelpad=lpad)
#    ax.set_ylabel("Power spectral density", fontsize=fsize,labelpad=lpad)
#    ax.set_title('ROI: '+label, fontsize=fsize,pad=lpad)
##    ax.set_ylim([0,25])
#    
#    for tick in ax.get_xticklabels():
#        tick.set_rotation(0)
#        tick.set_horizontalalignment('center')
# 
#    return None

#plot_bands(df2,label='V1_R')

#plot_power_power(df,label='V1_L')