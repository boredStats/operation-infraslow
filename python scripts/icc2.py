import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from scipy.stats import zscore

def ICC_rep_anova(Y):
    from numpy import ones, kron, mean, eye, hstack, dot, tile
    from numpy.linalg import pinv
    '''
    the data Y are entered as a 'table' ie subjects are in rows and repeated
    measures in columns
    One Sample Repeated measure ANOVA
    Y = XB + E with X = [FaTor / Subjects]

    The ICC returned corresponds to a two-way random effects, consistency, single rater/measurement
    See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4913118/ Table 3 for more
    '''

    [nb_subjects, nb_conditions] = Y.shape
    dfc = nb_conditions - 1
    dfe = (nb_subjects - 1) * dfc
    dfr = nb_subjects - 1

    # Compute the repeated measure effect
    # ------------------------------------

    # Sum Square Total
    mean_Y = mean(Y)
    SST = ((Y - mean_Y)**2).sum()

    # create the design matrix for the different levels
    x = kron(eye(nb_conditions), ones((nb_subjects, 1)))  # sessions
    x0 = tile(eye(nb_subjects), (nb_conditions, 1))  # subjects
    X = hstack([x, x0])

    # Sum Square Error
    predicted_Y = dot(dot(dot(X, pinv(dot(X.T, X))), X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals**2).sum()

    residuals.shape = Y.shape

    MSE = SSE / dfe

    # Sum square session effect - between colums/sessions
    SSC = ((mean(Y, 0) - mean_Y)**2).sum() * nb_subjects
    MSC = SSC / dfc / nb_subjects

    session_effect_F = MSC / MSE

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    # ICC(3,1) = (mean square subjeT - mean square error) /
    #            (mean square subjeT + (k-1)*-mean square error)
    ICC = (MSR - MSE) / (MSR + dfc * MSE)

    e_var = MSE  # variance of error
    r_var = (MSR - MSE) / nb_conditions  # variance between subjects

    return ICC, r_var, e_var, session_effect_F, dfc, dfe

def butter_filter(timeseries, fs, cutoffs, btype='band', order=4):
    #Scipy v1.2.0
    nyquist = fs/2
    butter_cut = np.divide(cutoffs, nyquist) #butterworth param (digital)
    sos = butter(order, butter_cut, output='sos', btype=btype)
    return sosfilt(sos, timeseries)

def _extract_average_power(hdf5_file, sessions, subjects, rois, image_type, bp=False):
    """
    Extract instantaneous power at each timepoint and average across the signal

    Quick function for calculating power data using our hdf5 hierarchy

    Parameters
    ----------
    hdf5_file : str
    Path to hdf5 file

    sessions : list
    List of session names

    subjects : list
    List of subject codes

    rois : list
    List of ROIs

    image_type : str, "MRI" or "MEG"

    bp : bool, default is False
    Flag for applying a .01 - .1 Hz bandpass filter to the signals

    Returns
    -------
    power_data : dict
    Dictionary of power_data
    """
    power_data = {}
    for sess in sessions:
        session_data = []
        for subj in subjects:
            f = h5py.File(hdf5_file, 'r')
            if 'MEG' in image_type:
                h_path = subj + '/MEG/' + sess + '/resampled_truncated'
                data = f.get(h_path).value
                f.close()

            if 'MRI' in image_type:
                h_path = subj + '/rsfMRI/' + sess + '/timeseries'
                data = f.get(h_path).value
                f.close()

            if bp:
                fs = 1/.72
                cutoffs = [.01, .1]
                data = butter_filter(data, fs, cutoffs)

            fft_power = np.absolute(np.fft.rfft(data, axis=0))**2
            average_power = np.mean(fft_power, axis=0)
            session_data.append(average_power)

        session_df = pd.DataFrame(np.asarray(session_data),
                                  index=subjects,
                                  columns=rois)
        power_data[sess] = session_df

    return power_data

import sys
sys.path.append("..")
import proj_utils as pu

print('%s: Starting...' % pu.ctime())
pdir = pu._get_proj_dir()
pdObj = pu.proj_data()
rois = pdObj.roiLabels
meg_subj, meg_sess = pdObj.get_meg_metadata()
mri_subj, mri_sess = pdObj.get_mri_metadata()
subj_overlap = [s for s in mri_subj if s in meg_subj]

meg_path = pdir + '/data/downsampled_MEG_truncated.hdf5'
mri_path = pdir + '/data/multimodal_HCP.hdf5'
roi_path = pdir + '/data/glasser_atlas/'
fig_path = pdir + '/figures/cron_alpha'

print('%s: BP - Extracting average power in each ROI and subject, MRI' % pu.ctime())
mri_data = _extract_average_power(mri_path, mri_sess, subj_overlap, rois, 'MRI', True)

print('%s: BP - Extracting average power in each ROI and subject, MEG' % pu.ctime())
meg_data = _extract_average_power(meg_path, meg_sess, subj_overlap, rois, 'MEG', True)

grand_dict = {}
for mri_session in list(mri_data):
    new_key = 'MRI_%s' % mri_session
    grand_dict[new_key] = mri_data[mri_session]

for meg_session in list(meg_data):
    new_key = 'MEG_%s' % meg_session
    grand_dict[new_key] = meg_data[meg_session]

icc_res = pd.DataFrame(index=rois, columns=['ICC'])
for r, roi in enumerate(rois):
    icc_input = pd.DataFrame(index=subj_overlap, columns=list(grand_dict))

    for table_key in list(grand_dict):
        table = grand_dict[table_key]
        icc_input[table_key] = table[roi].values
    icc_input.apply(zscore, axis=1, raw=True)

    ICC, _, _, _, _, _ = ICC_rep_anova(icc_input.values)
    # print(ICC)
    icc_res.iloc[r, 0] = ICC

icc_res.to_excel(pdir + '/figures/icc_res.xlsx')
print('%s: Finished' % pu.ctime())