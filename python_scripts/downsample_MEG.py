# -*- coding: utf-8 -*-
"""
Resample MEG data to match rsfmri data

Created on Wed Jan 23 10:16:58 2019
"""
import sys
sys.path.append("..")

import os
import time
import h5py
import datetime
import proj_utils as pu
from scipy import signal

#Load data
pdir = pu._get_proj_dir()
pdObj = pu.proj_data()
pData = pdObj.get_data()
database = pData['database']
roi_labels = pData['roiLabels']

path = pdir+'/data/timeseries_MEG'
filelist = os.listdir(path)
filelist = sorted(filelist,key=str.lower)

subj_list = sorted(list(set([f.split('_')[0] for f in filelist])))
sess_list = sorted(list(set([f.split('_')[-1].replace('.mat', '') for f in filelist])))

#Define constants
meg_original_timepoints = 745619
meg_downsampled_timepoints = 146592
min_meg_length = 118088
meg_original_sampling_rate = 2034.51 
#meg_downsampled_sampling_rate = meg_downsampled_timepoints / meg_original_timepoints * meg_original_sampling_rate
meg_downsampled_sampling_rate = 500 #from Hye

rsfmri_sampling_rate = 1/.72
rsfmri_timepoints = 1200
rsfmri_time = rsfmri_timepoints / rsfmri_sampling_rate

meg_con = (min_meg_length * rsfmri_sampling_rate ) / meg_downsampled_sampling_rate

#Load example timeesries
subj = subj_list[0]
sess = sess_list[0]
meg_dset = database['/' + subj + '/MEG/' + sess + '/timeseries']
meg_rois = pu.read_database(meg_dset, roi_labels)

meg_resampled_timeseries = signal.resample(meg_rois.values, int(meg_con))

#Resample MEG data, append to hdf5 file
downsamp_path = pdir + '/data/downsampled_MEG_truncated.hdf5'

for sess in sess_list:
    for subj in subj_list:
        dset_to_load = database['/' + subj + '/MEG/' + sess + '/timeseries']
        meg_data = pu.read_database(dset_to_load, roi_labels)
        
        meg_data_resampled = signal.resample(meg_data.values[:min_meg_length, :], int(meg_con))
        downsamp_file = h5py.File(downsamp_path)
        grp = downsamp_file.require_group('/' + subj + '/MEG/' + sess)
        
        dset = grp.create_dataset('resampled_truncated',
                                  data=meg_data_resampled,
                                  chunks=True, compression='gzip')
        downsamp_file.close()
        st = datetime.datetime.fromtimestamp(time.time())
        print('Subject ' + subj + ' complete at ' + str(st))
    st = datetime.datetime.fromtimestamp(time.time())
    print('Session ' + sess + ' complete at: ' + str(st))