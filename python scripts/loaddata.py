# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:20:05 2018

@author: jah150330
"""

import proj_utils as pu
pdObj = pu.proj_data()
pData = pdObj.get_data()
pdir = pu._get_proj_dir()

import os
import scipy.io as sio
import numpy as np
import pandas as pd
import h5py as h5

path = pdir+'/data/timeseries_MEG'
os.chdir(path)

labels = pData['roiLabels']
database = pData['database']

filelist = os.listdir(path)
filelist = sorted(filelist, key=str.lower)

subjectList = [filelist[x].split('_') for x in range(0, 267)]
sessionList = [subjectList[x][2].split('.') for x in range(0, 267)]

for index in range(0,360):
    subject = filelist[index][0:6]
    session = filelist[index][7:-4]
    
    matFile = sio.loadmat(path+'/'+filelist[index])
    data = matFile[session]['trial'][0,0].flatten()
    data = np.concatenate(data[:],axis=1).astype(float)
    
    df = pd.DataFrame(data)
    df = df.transpose()
    df.columns = labels
    
    # /SUBJ_CODE/MEG/SESSION/TIMESERIES
    grp = database.create_group('/' + subjectList[index][0] +
                                '/MEG/' + sessionList[index][0])
    dset = grp.create_dataset('timeseries', data=df.values,
                              chunks=True, compression='gzip')