# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:12:45 2018

@author: jah150330
"""
import numpy as np
import pandas as pd
import math
import seaborn as sns
import scipy.stats

import proj_utils as pu
pdObj = pu.proj_data()
pData = pdObj.get_data()
pdir = pu._get_proj_dir()

df = pd.read_pickle(pdir+'\data\MEG_AmpPowPha_spectra.pkl')

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

import time
import datetime

for session in sessionList:
    for band in eeg_bands:
        for ROI in labels:
            dataAmp = df.loc[pd.IndexSlice[session,:,'Amplitude',band],ROI]
            dataPow = df.loc[pd.IndexSlice[session,:,'Power',band],ROI]
            dataPha = df.loc[pd.IndexSlice[session,:,'Phase',band],ROI]
            
            asdMu = np.mean(dataAmp.values)
            asdSigma = np.std(dataAmp.values)
            df2.at[(session,band,'Amplitude','mean'),ROI] = asdMu
            df2.at[(session,band,'Amplitude','std'),ROI] = asdSigma            
            
            psdMu = np.mean(dataPow.values)
            psdSigma = np.std(dataPow.values)
            df2.at[(session,band,'Power','mean'),ROI] = psdMu
            df2.at[(session,band,'Power','std'),ROI] = psdSigma
            
            phsdMu = np.mean(dataPha.values)
            phsdSigma = np.std(dataPha.values)            
            df2.at[(session,band,'Phase','mean'),ROI] = phsdMu
            df2.at[(session,band,'Phase','std'),ROI] = phsdSigma
        
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            
        print('Session '+session+', '+band+' band complete at '+st)

#def plot_bands(df2, label):
#    df = pd.DataFrame(columns=['band', 'val', 'error'])
#    df['band'] = eeg_bands.keys()
#    df['val'] = [df2.at[(session,band,'mean'),label] for band in eeg_bands]
#    df['error'] = [df2.at[(session,band,'std'),label]/math.sqrt(87) for band in eeg_bands]
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

#def plot_power_power(df,label):
#    rMatrix = np.zeros((10,10))
#    pMatrix = np.zeros((10,10))
#    
#    keyList = list(eeg_bands.keys())
#    
#    for bandX in eeg_bands:
#        x = df.loc[pd.IndexSlice[session,:,bandX],label]
#        
#        for bandY in eeg_bands:
#            y = df.loc[pd.IndexSlice[session,:,bandY],label]
#            
#            (rMatrix[keyList.index(bandX),keyList.index(bandY)],
#            pMatrix[keyList.index(bandX),keyList.index(bandY)])= scipy.stats.pearsonr(x,y)
#    
##    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
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
#        
#    return None
#
##plot_power_power(df,label='V1_L')
#
#def plot_cluster(df2,labels,sesh=0):
#    session = sessionList[sesh]
#    data = df2.loc[pd.IndexSlice[session,:,'mean'],labels]
#    
##    data = scipy.stats.zscore(data.values.astype(float),axis=1)
#        
#    data = data.transpose()
#    data = data.values.astype(float)
#    
##    sns.set(rc={'figure.figsize':(8.27,20)})
#    ax = sns.clustermap(data,
#                        figsize=[10,10],
#                        method='ward',
##                        metric='correlation',
#                        col_cluster=False,
##                        z_score=1,
#                        standard_scale=1,
#                        xticklabels=list(eeg_bands.keys()),
#                        yticklabels=False)
##                        cmap=sns.diverging_palette(250, 15, s=75, l=40,
##                                                   n=9, center="dark"))
##                        vmax=3)
#    fsize = 24
#    lpad = 15
#    ax.ax_heatmap.set_title('PSD cluster map ('+session+')', fontsize=fsize,pad=lpad)
#    
#    return None

#plot_cluster(df2,labels,sesh=2)
