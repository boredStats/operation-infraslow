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

#Functionized analyses
def _extract_phase_amp(meg_subj, meg_sess, database, rois, fs, band_dict, phase_amp_file):

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
    phase_amp_file = pdir + '/data/MEG_phase_amp_data.hdf5'
    for sess in meg_sess:
        for subj in meg_subj:
            for b in band_dict:
                band = band_dict[b]
                out_file = h5py.File(phase_amp_file)
                group_path = subj + '/' + sess + '/' + b
                if group_path in out_file:
                    continue

                print('%s: %s %s %s' % (pu.ctime(), sess, str(subj), b))
                dset = database[subj +'/MEG/'+ sess +'/timeseries']
                meg_data = pu.read_database(dset, rois)
                phase_mat, amp_mat = build_output(meg_data, fs, rois, band)

                grp = out_file.require_group(group_path)
                grp.create_dataset('phase_data', data=phase_mat)
                grp.create_dataset('amplitude_data', data=amp_mat)
                out_file.close()
                

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

data_path = pdir + '/data/MEG_BOLD_phase_amp_data.hdf5'
# _extract_phase_amp(meg_subj, meg_sess, database, rois, fs, band_dict, data_path)
print('%s: Finished extracting phase/amplitude data' % pu.ctime())

print('%s: Running phase-amplitdue coupling' % pu.ctime())
coupling_path = pdir + '/data/MEG_BOLD_phase_amp_coupling.hdf5'

slow_bands = ['BOLD bandpass']
reg_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

band1 = slow_bands
band2 = reg_bands

for sess in meg_sess:
    for subj in meg_subj:
        phase_amp_file = h5py.File(data_path, 'r')
#         subj_data = phase_amp_file.require_group(subj + '/' + sess)
        subj_data = phase_amp_file.get(subj + '/' + sess)

        for r, roi in enumerate(rois):
            cfc_file = h5py.File(coupling_path)

            group_path = sess + '/' + subj + '/' + roi
            if group_path in cfc_file:
                continue #check if work has already been done
            print('%s: %s %s %s' % (pu.ctime(), sess, str(subj), roi))
            r_mat = np.ndarray(shape=[len(band1), len(band2)])
            p_mat = np.ndarray(shape=[len(band1), len(band2)])

            for slow_index, slow in enumerate(band1):
                slow_group = subj_data.get(slow)
                slow_ts = slow_group.get('phase_data')[:, r]
                for reg_index, reg in enumerate(band2):
                    reg_group = subj_data.get(reg)
                    reg_ts = reg_group.get('amplitude_data')[:, r]

                    r_val, p_val = pac.circCorr(slow_ts, reg_ts)
                    r_mat[slow_index, reg_index] = r_val
                    p_mat[slow_index, reg_index] = p_val

            out_group = cfc_file.require_group(group_path)
            out_group.create_dataset('r_vals', data=r_mat)
            out_group.create_dataset('p_vals', data=p_mat)
            cfc_file.close()

        phase_amp_file.close()

print('%s: Finished' % pu.ctime())
