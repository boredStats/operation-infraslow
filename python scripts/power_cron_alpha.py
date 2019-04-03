import h5py
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.stats import zscore
from scipy.signal import butter, sosfilt

def cron_alpha(array):
    k = array.shape[1] #Columns are the groups
    variances_sum = np.sum(np.var(array, axis=0, ddof=1))
    variances_total = np.var(np.sum(array, axis=1), ddof=1)
    
    return (k / (k-1)) * (1 - (variances_sum / variances_total))

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

def _calculate_cron_alpha_per_roi(dataframe_dict, rois):
    """Calculate Cronbach's alpha for each ROI
    
    Procedure:
    For each ROI, get power data for each session -> matrix of size Subjects X Sessions
    Calculate Cronbach's alpha for the matrix
    Save result to ROI table
    
    Parameters
    ----------
    dataframe_dict : dict
    Dictionary of dataframes corresponding to the different sessions
    Each dataframe should be size Subject X ROI
    
    rois : list
    A list of ROIs, for convenience (these can technically be taken from the dataframe)
    
    Returns
    -------
    output : dataframe
    A dataframe of size ROI x 1, filled with c-alpha values for each ROI
    """
    
    sessions = list(dataframe_dict)
    ca_values = np.ndarray(shape=len(rois))
    for r, roi in enumerate(rois):
        session_data = []
        for s, sess in enumerate(sessions):
            df = dataframe_dict[sess]
            data = df[roi]
            session_data.append(data.values)
        
        session_array = np.asarray(session_data).T
        ca_values[r] = cron_alpha(session_array) 
        
    output = pd.DataFrame(ca_values, index=rois, columns=["Cronbach's alpha"], dtype=np.float16)
    
    return output
    
def _conjunction(mri_crons, meg_crons, thresh=.7, out_type='summary'):
    """Calculate the conjunction between two sets of Cronbach's alpha values
    
    Parameters
    ----------
    mri_crons, meg_crons : dataframe
    Dataframe of Cronbach's alpha values for each ROI
    
    thresh : float, default is .7
    Alpha value to run conjunction
    
    out_type : str, "summary" or "full", default is summary
    "summary" returns a dataframe with only conjunctions
    "full" returns a full dataframe with zeroed out rows
    """
    
    mri_cron_values = mri_crons.values
    meg_cron_values = meg_crons.values
    rois = mri_crons.index
    
    conjunction = np.ones(len(mri_cron_values))
    for i in range(len(mri_cron_values)):
        if mri_cron_values[i] < thresh or meg_cron_values[i] < thresh:
            conjunction[i] = 0
    
    if any(conjunction):
        if out_type == "full":
            output = pd.DataFrame(conjunction, index=mri_crons.index, columns=['Conjunctions'])
            return output
        elif out_type == "summary":
            conj, c_rois = [], []
            for r, roi in enumerate(rois):
                if conjunction[r] != 0:
                    conj.append(conjunction[r])
                    c_rois.append(roi)

            output = pd.DataFrame(conj, index=c_rois, columns=['Conjunctions'])
            return output
    
    else:
        print('No conjunctions found.')

if __name__ == "__main__":
    import mPLSC_functions as mf
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
    fig_path = pdir + '/figures/'
    
    print('%s: Extracting average power in each ROI and subject, MRI' % pu.ctime())
    mri_data = _extract_average_power(mri_path, mri_sess, subj_overlap, rois, 'MRI')
    
    print('%s: Extracting average power in each ROI and subject, MEG' % pu.ctime())
    meg_data = _extract_average_power(meg_path, meg_sess, subj_overlap, rois, 'MEG')
    
    print('%s: Calculating Cronbach''s alpha for MRI sessions' % pu.ctime())
    mri_crons = _calculate_cron_alpha_per_roi(mri_data, rois)
    
    print('%s: Calculating Cronbach''s alpha for MEG sessions' % pu.ctime())
    meg_crons = _calculate_cron_alpha_per_roi(meg_data, rois)
    
    print("Cronbach's alpha values for MRI")
    print(mri_crons.head())
    
    print("Cronbach's alpha values for MEG")
    print(meg_crons.head())

    print('%s: Running conjuntion between modes' % pu.ctime())
    conj = _conjunction(mri_crons, meg_crons)
    if conj is not None:
        print('Number of conjunctions found: %d out of %d' % (len(conj.values), len(rois)))
        print(conj)
    
    print('\n')

    #Bandpass version
    print('%s: BP - Extracting average power in each ROI and subject, MRI' % pu.ctime())
    mri_data = _extract_average_power(mri_path, mri_sess, subj_overlap, rois, 'MRI', True)
    
    print('%s: BP - Extracting average power in each ROI and subject, MEG' % pu.ctime())
    meg_data = _extract_average_power(meg_path, meg_sess, subj_overlap, rois, 'MEG', True)
    
    print('%s: BP - Calculating Cronbach''s alpha for MRI sessions' % pu.ctime())
    mri_crons = _calculate_cron_alpha_per_roi(mri_data, rois)
    
    print('%s: BP - Calculating Cronbach''s alpha for MEG sessions' % pu.ctime())
    meg_crons = _calculate_cron_alpha_per_roi(meg_data, rois)
    
    print("Cronbach's alpha values for MRI - bandpass filter .01-.1 Hz")
    print(mri_crons.head())
    
    print("Cronbach's alpha values for MEG - bandpass filter .01-.1 Hz")
    print(meg_crons.head())
    
    print('%s: Running conjuntion between modes' % pu.ctime())
    conj = _conjunction(mri_crons, meg_crons)
    if conj is not None:
        print('Number of conjunctions found: %d out of %d' % (len(conj.values), len(rois)))
        print(conj)
        
        print('%s: Creating conjunction brain' % pu.ctime())
        conj = _conjunction(mri_crons, meg_crons, out_type='full')
        mags = conj.values
        rois_to_stack = conj.index
        custom_roi = mf.create_custom_roi(roi_path, rois_to_stack, mags)
        fname = fig_path + 'cron_alpha_mri_vs_meg.png'
        fig = mf.plot_brain_saliences(custom_roi, minval=None, figpath=fname)
        
    print('%s: Finished' % pu.ctime())