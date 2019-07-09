import h5py
import numpy as np
import proj_utils as pu
import pac_functions as pac
from astropy.stats.circstats import circcorrcoef as circ_corr


def extract_phase_amp_downsamp(meg_sess, meg_subj, rois, downsamp_file):
    # Extract instantaneous phase, amplitude of downsampled data in the BOLD bandpass range
    def _build_output(ts_data, fs, roi_list, phase_band):
        ts_len = len(ts_data[:, 0])
        phase_array = np.ndarray(shape=[ts_len, len(rois)])
        amp_array = np.ndarray(shape=[ts_len, len(rois)])

        for r, roi in enumerate(roi_list):
            phase, amp = pac.get_phase_amp_data(ts_data[:, r], fs, phase_band, phase_band)
            phase_array[:, r] = phase
            amp_array[:, r] = amp

        return phase_array, amp_array

    data_path = '../data/MEG_downsampled_phase_amp_data.hdf5'
    for sess in meg_sess:
        for subj in meg_subj:
            band = (.01, .1)

            out_file = h5py.File(data_path)
            group_path = subj + '/' + sess + '/BOLD bandpass'
            if group_path in out_file:
                continue

            h5 = h5py.File(downsamp_file)
            meg_data = h5[subj + '/MEG/' + sess + '/resampled_truncated'][...]
            phase_mat, amp_mat = _build_output(meg_data, fs=1/.72, roi_list=rois, phase_band=band)

            grp = out_file.require_group(group_path)
            grp.create_dataset('phase_data', data=phase_mat, compression='lzf')
            grp.create_dataset('amplitude_data', data=amp_mat, compression='lzf')
            out_file.close()


def main():
    pd_obj = pu.proj_data()
    pdata = pd_obj.get_data()
    rois = pdata['roiLabels']
    meg_subj, meg_sess = pd_obj.get_meg_metadata()

    downsamp_file = '../data/downsampled_MEG_truncated.hdf5'
    extract_phase_amp_downsamp(meg_sess, meg_subj, rois, downsamp_file)

    output_path = '../data/MEG_phase_phase_coupling.hdf5'
    data_path = '../data/MEG_downsampled_phase_amp_data.hdf5'

    for sess in meg_sess:
        for subj in meg_subj:
            subj_ppc = np.ndarray(shape=(len(rois), len(rois)))
            data_file = h5py.File(data_path, 'r')
            subj_data = data_file[subj + '/' + sess + '/BOLD bandpass/phase_data'][...]

            for r1 in range(len(rois)):
                for r2 in range(len(rois)):
                    phase_1 = subj_data[:, r1]
                    phase_2 = subj_data[:, r2]
                    rho = circ_corr(phase_1, phase_2)
                    subj_ppc[r1, r2] = rho

            ppc_file = h5py.File(output_path)
            prog = sess + '/' + subj
            if prog in ppc_file:
                continue
            out_group = ppc_file.require_group(prog)
            out_group.create_dataset('ppc', data=subj_ppc, compression='lzf')
            ppc_file.close()


main()
