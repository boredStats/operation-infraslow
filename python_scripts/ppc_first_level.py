# -*- coding: utf-8 -*-
"""
Get phase, amplitude data for the different frequency bands
Run CFC with circular correlation

Created on Wed Feb  6 09:54:17 2019
"""

import h5py
import numpy as np
import proj_utils as pu
from astropy.stats.circstats import circcorrcoef as circ_corr


def main():
    pd_obj = pu.proj_data()
    pdata = pd_obj.get_data()
    rois = pdata['roiLabels']
    band_dict = pdata['bands']
    meg_subj, meg_sess = pd_obj.get_meg_metadata()

    coupling_path = '../data/MEG_phase_phase_coupling.hdf5'
    data_path = '../data/MEG_phase_amp_data.hdf5'

    # Full combination of phase-phase coupling
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

                r_mat = np.ndarray(shape=(len(phase_bands), len(phase_bands_2)))
                for phase_index, phase_band in enumerate(phase_bands):
                    p_grp = subj_data.get(phase_band)
                    phase_spect = p_grp.get('phase_data')[:, r]
                    for phase_index_2, phase_band_2 in enumerate(phase_bands_2):
                        p2_grp = subj_data.get(phase_band_2)
                        phase_spect_2 = p2_grp.get('phase_data')[:, r]

                        rho = circ_corr(phase_spect, phase_spect_2)
                        r_mat[phase_index, phase_index_2] = rho

                out_group = cfc_file.require_group(group_path)
                out_group.create_dataset('r_vals', data=r_mat, compression='lzf')
                cfc_file.close()

            data_file.close()


main()
