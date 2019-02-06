# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 09:54:17 2019
"""
import h5py
import numpy as np
import pac_functions as pac

import sys
sys.path.append("..")
import proj_utils as pu

print('%s: Starting' % pu.ctime())   
     
print('%s: Getting metadata, parameters' % pu.ctime())
pdir = pu._get_proj_dir()
pdObj = pu.proj_data()

pData = pdObj.get_data()
rois = pData['roiLabels']
database = pData['database']
band_dict = pData['bands']
meg_subj, meg_sess = pdObj.get_meg_metadata()
fs = 500

def extract_phase_amp(meg_subj, meg_sess, database, rois, fs, band_dict):
    def build_output(ts_data, fs, rois, band):
        #Load in a subject, calculate phase/amplitude for each roi
        ts_len = len(ts_data[rois[0]])
        phase_mat = np.ndarray(shape=[ts_len, len(rois)])
        amp_mat = np.ndarray(shape=[ts_len, len(rois)])
        
        for r, roi in rois:
            phase, amp = pac.get_phase_amp_data(ts_data[roi], fs, band, band)
            phase_mat[:, r] = phase
            amp_mat[:, r] = amp
    
        return phase_mat, amp_mat
    
    print('%s: Extracting phase/amp data' % pu.ctime())
    for sess in meg_sess:
        for subj in meg_subj:
            for b in band_dict:
                band = band_dict[b]
                phase_amp_file = pdir + '/data/MEG_pac_first_level.hdf5'
                out_file = h5py.File(phase_amp_file)
                
                print('%s: %s %s %s' % (pu.ctime(), sess, str(subj)), b)
                dset = database['/'+ subj +'/MEG/'+ sess +'/timeseries']
                meg_data = pu.read_database(dset, rois)
                phase_mat, amp_mat = build_output(meg_data, fs, rois, band)
                
                grp = out_file.require_group('/' + subj + '/' + sess + '/' + b)
                grp.create_dataset('phase_data', data=phase_mat)
                grp.create_dataset('amplitude_data', data=amp_mat)
                out_file.close()          
    print('%s: Finished extracting phase/amplitude data' % pu.ctime())

c1 = input("Calculate phase/amp data again? (y/n) ")
if c1 == 'y':
    extract_phase_amp(meg_subj, meg_sess, database, rois, fs, band_dict)

print('%s: Finished' % pu.ctime())