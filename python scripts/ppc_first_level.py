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
comp = 'lzf'  # h5py compression param

print('%s: Running phase-phase coupling' % pu.ctime())
coupling_path = pdir + '/data/MEG_phase_phase_coupling.hdf5'
data_path = pdir + '/data/MEG_phase_amp_data.hdf5'

# BOLD bandpass with higher frequency only
# slow_bands = ['BOLD bandpass']
# reg_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# Full combination of phase-amplitude coupling
phase_bands = list(band_dict)
phase_bands_2 = list(band_dict)

for sess in meg_sess:
    for subj in meg_subj:
        data_file = h5py.File(data_path, 'r')
        subj_data = data_file.get(subj + '/' + sess)
        for r, roi in enumerate(rois):
            cfc_file = h5py.File(coupling_path)
            group_path = sess + '/' + subj + '/' + roi
            if group_path in cfc_file:
                continue  # check if work has already been done

            print('%s: PPC for %s %s %s' % (pu.ctime(), sess, str(subj), roi))
            r_mat = np.ndarray(shape=(len(phase_bands), len(phase_bands_2)))
            p_mat = np.ndarray(shape=(len(phase_bands), len(phase_bands_2)))
            for phase_index, phase_band in enumerate(phase_bands):
                p_grp = subj_data.get(phase_band)
                phase_spect = p_grp.get('phase_data')[:, r]
                for phase_index_2, phase_band_2 in enumerate(phase_bands_2):
                    a_grp = subj_data.get(phase_band_2)
                    amp_spect = a_grp.get('phase_data')[:, r]

                    r_val, p_val = pac.circCorr(phase_spect, amp_spect)
                    r_mat[phase_index, phase_index_2] = r_val
                    p_mat[phase_index, phase_index_2] = p_val

            out_group = cfc_file.require_group(group_path)
            out_group.create_dataset(
                'r_vals',
                data=r_mat,
                compression=comp)
            out_group.create_dataset(
                'p_vals',
                data=p_mat,
                compression=comp)
            cfc_file.close()

        data_file.close()

print('%s: Finished' % pu.ctime())
