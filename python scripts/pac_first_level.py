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
comp = 'lzf' #h5py compression param

# data_path = pdir + '/data/MEG_BOLD_phase_amp_data.hdf5'
data_path = pdir + '/data/MEG_phase_amp_data.hdf5'
check = input('Extract phase/amplitude data? y/n ')
if check=='y':
    print('%s: Extracting phase/amp data' % pu.ctime())
    for sess in meg_sess:
        for subj in meg_subj:
            for b in band_dict:
                band = band_dict[b]

                out_file = h5py.File(data_path)
                group_path = subj + '/' + sess + '/' + b
                if group_path in out_file:
                    continue

                print('%s: %s %s %s' % (pu.ctime(), sess, str(subj), b))
                dset = database[subj +'/MEG/'+ sess +'/timeseries']
                meg_data = pu.read_database(dset, rois)
                phase_mat, amp_mat = build_output(meg_data, fs, rois, band)

                grp = out_file.require_group(group_path)
                grp.create_dataset(
                    'phase_data',
                    data=phase_mat,
                    compression=comp)
                grp.create_dataset(
                    'amplitude_data',
                    data=amp_mat,
                    compression=comp)
                out_file.close()

print('%s: Running phase-amplitdue coupling' % pu.ctime())
# coupling_path = pdir + '/data/MEG_BOLD_phase_amp_coupling.hdf5' #BOLD only
coupling_path = pdir + '/data/MEG_phase_amp_coupling.hdf5' #all combinations

# #BOLD bandpass with higher frequency only
# slow_bands = ['BOLD bandpass']
# reg_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

#Full combination of phase-amplitude coupling
phase_bands = list(band_dict)
amp_bands = list(band_dict)

for sess in meg_sess:
    for subj in meg_subj:
        data_file = h5py.File(data_path, 'r')
        subj_data = data_file.get(subj + '/' + sess)
        for r, roi in enumerate(rois):
            cfc_file = h5py.File(coupling_path)
            group_path = sess + '/' + subj + '/' + roi
            if group_path in cfc_file:
                continue #check if work has already been done

            print('%s: CFC for %s %s %s' % (pu.ctime(), sess, str(subj), roi))
            r_mat = np.ndarray(shape=(len(phase_bands), len(amp_bands)))
            p_mat = np.ndarray(shape=(len(phase_bands), len(amp_bands)))
            for phase_index, phase_band in enumerate(phase_bands):
                p_grp = subj_data.get(phase_band)
                phase_spect = p_grp.get('phase_data')[:, r]
                for amp_index, amp_band in enumerate(amp_bands):
                    a_grp = subj_data.get(amp_band)
                    amp_spect = a_grp.get('amplitude_data')[:, r]

                    r_val, p_val = pac.circCorr(phase_spect, amp_spect)
                    r_mat[phase_index, amp_index] = r_val
                    p_mat[phase_index, amp_index] = p_val

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
