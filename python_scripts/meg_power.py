import h5py
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils as pu


def calc_phase_amp_power(how='location'):
    p_data = pu.proj_data()
    subjects, sessions = p_data.get_meg_metadata()
    rois = p_data.roiLabels
    fs = 500

    if how is 'location':
        df_list = []
        for session in sessions:
            session_df = pd.DataFrame(index=subjects)
            for subject in subjects:
                prog = "%s - %s" % (session, subject)
                print('%s: Calculating infraslow power for %s with location' % (pu.ctime(), prog))

                database = h5py.File('../data/multimodal_HCP.hdf5', 'r+')
                dset = database[subject + '/MEG/' + session + '/timeseries'][...]
                for ROIindex in range(len(rois)):
                    data = dset[:, ROIindex]
                    label = rois[ROIindex]

                    # Get real amplitudes of FFT (only in postive frequencies)
                    # Squared to get power
                    fft_power = np.absolute(np.fft.rfft(data)) ** 2
                    # Get frequencies for amplitudes in Hz

                    fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)
                    infraslow_band = (.01, .1)  # ('BOLD bandpass', (.01, .1))

                    freq_ix = np.where((fft_freq >= infraslow_band[0]) &
                                       (fft_freq <= infraslow_band[1]))[0]
                    colname = '%s %s' % (session, label)
                    if colname not in session_df:
                        session_df[colname] = np.nan

                    avg_power = np.mean(fft_power[freq_ix])
                    session_df.loc[subject][colname] = avg_power

                database.close()

            df_list.append(session_df)

        grand_df = pd.concat(df_list, axis=1)
        with open('../data/MEG_infraslow_power_location_calc.pkl', 'wb') as file:
            pkl.dump(grand_df, file)

    elif how is 'bandpass':
        from scipy.signal import butter, sosfilt

        def _butter_filter(timeseries, fs, cutoffs, btype='band', order=4):
            nyquist = fs / 2
            butter_cut = np.divide(cutoffs, nyquist)  # butterworth param (digital)
            sos = butter(order, butter_cut, output='sos', btype=btype)
            return sosfilt(sos, timeseries)

        df_list = []
        for sess in sessions:
            session_colnames = ['%s %s' % (sess, r) for r in rois]
            session_df = pd.DataFrame(index=subjects, columns=session_colnames)
            for subj in subjects:
                prog = "%s - %s" % (sess, subj)
                print('%s: Calculating infraslow power for %s with bandpass' % (pu.ctime(), prog))

                f = h5py.File('../data/multimodal_HCP.hdf5', 'r')
                data = f[subj + '/MEG/' + sess + '/timeseries'][...]
                f.close()

                data = _butter_filter(data, fs=500, cutoffs=[.01, .1])

                fft_power = np.absolute(np.fft.rfft(data, axis=0)) ** 2
                average_power = np.mean(fft_power, axis=0)

                session_df.loc[subj] = average_power

            df_list.append(session_df)

        grand_df = pd.concat(df_list, axis=1)
        with open('../data/MEG_infraslow_power_bandpass_calc.pkl', 'wb') as file:
            pkl.dump(grand_df, file)

    print('%s: Finished' % pu.ctime())


if __name__ == "__main__":
    calc_phase_amp_power(how='location')
    calc_phase_amp_power(how='bandpass')
