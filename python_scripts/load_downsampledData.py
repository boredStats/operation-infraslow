# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:45:43 2019

@author: jah150330
"""

import sys
sys.path.append("..")

import os
import time
import datetime
import proj_utils as pu
from scipy import signal
import h5py as h5

#Load data
pdir = pu._get_proj_dir()
pdObj = pu.proj_data()
pData = pdObj.get_data()
database = pData['database']
rois = pData['roiLabels']

## Load MEG data
path = pdir+'/data/timeseries_MEG'
filelist = os.listdir(path)
filelist = sorted(filelist,key=str.lower)

subj_MEG = sorted(list(set([f.split('_')[0] for f in filelist])))
sess_MEG = sorted(list(set([f.split('_')[-1].replace('.mat', '') for f in filelist])))

badSubjects = ['169040','662551']

for subject in badSubjects:
    if subject in subj_MEG:
        subj_MEG.remove(subject)

## Load rfMRI data
path = pdir+'/data/timeseries_rs-fMRI'
filelist = os.listdir(path)
filelist = sorted(filelist,key=str.lower)

subj_MRI = sorted(list(set([f.split('_')[0] for f in filelist])))
sess_MRI = sorted(list(set([f.split('_')[-1].replace('.mat', '') for f in filelist])))

badSubjects = ['104012', '125525', '151526', '182840', '200109', '500222']

for subject in badSubjects:
    if subject in subj_MRI:
        subj_MRI.remove(subject)