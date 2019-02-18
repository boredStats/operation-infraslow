# -*- coding: utf-8 -*-
"""
Get phase, amplitude data for the different frequency bands
Run CFC with circular correlation

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

#Functionized analyses
def _extract_phase_amp(meg_subj, meg_sess, database, rois, fs, band_dict):
    
    def build_output(ts_data, fs, rois, band):
        #Load in a subject, calculate phase/amplitude for each roi
        ts_len = len(ts_data[rois[0]])
        phase_mat = np.ndarray(shape=[ts_len, len(rois)])
        amp_mat = np.ndarray(shape=[ts_len, len(rois)])
        
        for r, roi in enumerate(rois):
            phase, amp = pac.get_phase_amp_data(ts_data[roi], fs, band, band)
            phase_mat[:, r] = phase
            amp_mat[:, r] = amp
    
        return phase_mat, amp_mat
    
    print('%s: Extracting phase/amp data' % pu.ctime())
    phase_amp_file = pdir + '/data/MEG_phase_amp_data.hdf5' #TO-DO: CHANGE FILENAME
    for sess in meg_sess:
        for subj in meg_subj:
            for b in band_dict:
                band = band_dict[b]
                out_file = h5py.File(phase_amp_file)
                
                print('%s: %s %s %s' % (pu.ctime(), sess, str(subj), b))
                dset = database['/'+ subj +'/MEG/'+ sess +'/timeseries']
                meg_data = pu.read_database(dset, rois)
                phase_mat, amp_mat = build_output(meg_data, fs, rois, band)
                
                grp = out_file.require_group('/' + subj + '/' + sess + '/' + b)
                grp.create_dataset('phase_data', data=phase_mat)
                grp.create_dataset('amplitude_data', data=amp_mat)
                out_file.close()          

def _corr_bands(dataset, roi_index, slow_bands, reg_bands):
    r_mat = np.ndarray(shape=[len(slow_bands), len(reg_bands)])
    p_mat = np.ndarray(shape=[len(slow_bands), len(reg_bands)])
    r2_mat = np.ndarray(shape=[len(slow_bands), len(reg_bands)])
    stderr_mat = np.ndarray(shape=[len(slow_bands), len(reg_bands)])
    
    for slow_index, slow in enumerate(slow_bands):
        slow_group = dataset.get(slow)
        slow_ts = slow_group.get('phase_data')[:, roi_index]
        for reg_index, reg in enumerate(reg_bands):
            reg_group = dataset.get(reg)
            reg_ts = reg_group.get('amplitude_data')[:, roi_index]
            
            r_val, p_val, r2, se = pac.circCorr(slow_ts, reg_ts)
            r_mat[slow_index, reg_index] = r_val
            p_mat[slow_index, reg_index] = p_val
            r2_mat[slow_index, reg_index] = r2
            stderr_mat[slow_index, reg_index] = se
    
    return r_mat, p_mat, r2_mat, stderr_mat

#_extract_phase_amp(meg_subj, meg_sess, database, rois, fs, band_dict)
#print('%s: Finished extracting phase/amplitude data' % pu.ctime())

print('%s: Running phase-amplitdue coupling' % pu.ctime())
data_path = pdir + '/data/MEG_phase_amp_data.hdf5'
slow_bands = ['BOLD', 'Slow 4', 'Slow 3', 'Slow 2', 'Slow 1']
reg_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

for subj in meg_subj:
    for sess in meg_sess:
        phase_amp_file = h5py.File(data_path, 'r')
        subj_data = phase_amp_file.require_group('/' + subj + '/' + sess)
        for r, roi in enumerate(rois):
            print('%s: %s %s %s' % (pu.ctime(), sess, str(subj), roi))
            cfc_file = h5py.File(pdir + '/data/MEG_pac_first_level.hdf5')
            current_group = '/' + subj + '/' + sess + '/' + roi + '/'
            out_group = cfc_file.require_group(current_group)
            
            rs, ps, r2s, ses = _corr_bands(subj_data, r, slow_bands, reg_bands)
            
            out_group.create_dataset('r_vals', data=rs)
            out_group.create_dataset('p_vals', data=ps)
            out_group.create_dataset('r2_vals', data=r2s)
            out_group.create_dataset('std_err_vals', data=ses)
            
            cfc_file.close()
            
        phase_amp_file.close()

print('%s: Finished' % pu.ctime())