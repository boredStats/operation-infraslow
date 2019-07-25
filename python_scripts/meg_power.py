import h5py
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils as pu


def calc_phase_amp_power():
    pdObj = pu.proj_data()
    subjects, sessions = pdObj.get_meg_metadata()
    labels = pdObj.roiLabels
    bands = pdObj.bands
    fs = 500

    iterables = [sessions, subjects, ['Power'], bands]
    names = ['Session', 'Subject', 'Spectrum', 'Freq. Band']

    dfIdx = pd.MultiIndex.from_product(iterables, names=names)
    df = pd.DataFrame(index=dfIdx, columns=labels)

    df_list = []
    for session in sessions:
        session_df = pd.DataFrame(index=subjects)
        for subject in subjects:
            prog = "%s - %s" % (session, subject)
            print('%s: Calculating phase/amp/power for %s' % (pu.ctime(), prog))

            database = h5py.File('../data/multimodal_HCP.hdf5', 'r+')
            dset = database[subject + '/MEG/' + session + '/timeseries'][...]

            for ROIindex in range(0, 360):
                data = dset[:, ROIindex]
                label = labels[ROIindex]

                # Get real amplitudes of FFT (only in postive frequencies)
                # Squared to get power
                fft_power = np.absolute(np.fft.rfft(data)) ** 2
                # Get frequencies for amplitudes in Hz
                fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)

                # Take the mean of the fft amplitude for each EEG band
                # eeg_band_fft = dict()
                for band in bands:
                    freq_ix = np.where((fft_freq >= bands[band][0]) &
                                       (fft_freq <= bands[band][1]))[0]
                    colname = '%s,%s,%s' % (session, label, band)
                    if colname not in session_df:
                        session_df[colname] = np.nan

                    avg_power = np.mean(fft_power[freq_ix])
                    session_df.loc[subject][colname] = avg_power

                    # eeg_band_fft[band] = dict()
                    # eeg_band_fft[band]['Power'] = np.mean(fft_power[freq_ix])
                    # df.at[(session, subject, 'Power', band), label] = eeg_band_fft[band]['Power']

            database.close()

        df_list.append(session_df)

    grand_df = pd.concat(df_list, axis=1)
    with open('../data/MEG_phase_amp_power.pkl', 'wb') as file:
        pkl.dump(grand_df, file)

    print('%s: Finished' % pu.ctime())


if __name__ == "__main__":
    calc_phase_amp_power()
