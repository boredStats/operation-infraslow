# -*- coding: utf-8 -*-
"""
Run multi-table PLS-C using MEG power data

Created on Thu Mar 28 12:55:29 2019
"""

import h5py
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.signal import butter, sosfilt

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

def _merge_mri_meg(mri_data, meg_data):
    """Quick function for merging MRI and MEG tables into one list"""
    mri_tables = [mri_data[t] for t in list(mri_data)]
    meg_tables = [meg_data[t] for t in list(meg_data)]
    return mri_tables + meg_tables

def _x_conjunctions(x_saliences, latent_variable_names, rois, return_avg=True):
    """Do conjunctions over multiple latent variables
    Essentially a for loop around the conjunction_analysis function

    This function takes all tables from the x_saliences dict as subtables
    to compare. If you're interested in running conjunctions on specific 
    subtables, recommend creating a custom dict
    
    Parameters
    ----------
    x_saliences : dict of bootstrapped saliences
    latent_variable_names : list of latent variable labels
    rois : list of roi names
    return_avg : bool parameter for conjunction_analysis    
    
    Returns
    Dataframe of conjunctions for each latent variable
    """
    keys = list(x_saliences)
    conjunctions = []
    for name in latent_variable_names:
        brains = []
        for key in keys:
            data = x_saliences[key]
            
            brains.append(data[name].values)
            
        conj_data = pd.DataFrame(np.asarray(brains).T, index=rois)
        res = mf.conjunction_analysis(conj_data, 'sign', return_avg=return_avg)
        conjunctions.append(np.ndarray.flatten(res.values))

    return pd.DataFrame(np.asarray(conjunctions).T,
                        index=res.index,
                        columns=latent_variable_names)

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    import proj_utils as pu
    import mPLSC_functions as mf
    
    from boredStats import pls_tools as pls
    
    pdir = pu._get_proj_dir()
    pdObj = pu.proj_data()
    rois = pdObj.roiLabels
    meg_subj, meg_sess = pdObj.get_meg_metadata()
    mri_subj, mri_sess = pdObj.get_mri_metadata()
    all_subj = [s for s in mri_subj if s in meg_subj]
    
    meg_path = pdir + '/data/downsampled_MEG_truncated.hdf5'    
    mri_path = pdir + '/data/multimodal_HCP.hdf5'
    
    print('%s: Building subtables of power data for MEG' % pu.ctime())
    meg_data = _extract_average_power(meg_path, meg_sess, all_subj, rois, 'MEG', bp=True)
    x_tables = [meg_data[t] for t in list(meg_data)]
    
    print('%s: Building subtables of behavior data' % pu.ctime())
    behavior_metadata = pd.read_csv(pdir + '/data/b_variables_mPLSC.txt',
                                   delimiter='\t', header=None)

    behavior_metadata.rename(dict(zip([0, 1], ['category','name'])),
                                axis='columns', inplace=True)
    
    behavior_raw = pd.read_excel(pdir + '/data/hcp_behavioral.xlsx',
                                  index_col=0, sheet_name='cleaned')
    
    behavior_data = mf.load_behavior_subtables(behavior_raw, behavior_metadata)
    y_tables = [behavior_data[t] for t in list(behavior_data)]
    
    p = pls.MultitablePLSC(n_iters=10000, return_perm=True)
    print('%s: Running permutation testing on latent variables' % pu.ctime())
    res_perm = p.mult_plsc_eigenperm(y_tables, x_tables)

    print('%s: Running bootstrap testing on saliences' % pu.ctime())
    res_boot = p.mult_plsc_bootstrap_saliences(y_tables, x_tables, 4)
    
    num_latent_vars = len(np.where(res_perm['p_values'] < .001)[0])
    latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]
    
    print('%s: Organizing behavior saliences' % pu.ctime())
    y_saliences = mf.create_salience_subtables(
            sals=res_boot['y_saliences'][:, :num_latent_vars],
            dataframes=y_tables,
            subtable_names=list(behavior_data),
            latent_names=latent_names)
    
    print('%s: Averaging saliences within behavior categories' % pu.ctime())
    res_behavior = mf.average_behavior_scores(y_saliences, latent_names)
    
    print('%s: Organizing brain saliences' % pu.ctime())
#     mri_subtable_names = ['mri_%s' % s for s in mri_sess]
    meg_subtable_names = ['meg_%s' % s for s in meg_sess]
    x_table_names = meg_subtable_names #+ mri_subtable_names 
    x_saliences = mf.create_salience_subtables(
            sals=res_boot['x_saliences'][:, :num_latent_vars],
            dataframes=x_tables,
            subtable_names=x_table_names,
            latent_names=latent_names)
    
    print('%s: Running conjunction analysis' % pu.ctime())
    res_conj = _x_conjunctions(x_saliences, latent_names, rois)
    
    print('%s: Saving results' % pu.ctime())
    output = {'permutation_tests':res_perm,
              'bootstrap_tests':res_boot,
              'y_saliences':y_saliences,
              'x_saliences':x_saliences,
              'behaviors':res_behavior,
              'conjunctions':res_conj}
    
    with open(pdir + '/data/mPLSC_power_bandpass_filtered.pkl', 'wb') as file:
        pkl.dump(output, file)
        
    print('%s: Finished' % pu.ctime())