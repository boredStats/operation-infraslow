"""
Run multi-table PLS-C using MEG power data from each session separately
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

def _y_conjunctions_single_session(single_session_res, latent_variable_names, return_avg=True):
    """Run conjunctions on behavior data across the three models"""

    sessions = list(single_session_res)

    y_salience_list = {}
    for sess in sessions:
        output = single_session_res[sess]
        y_salience_list[sess] = output['y_saliences']

    behavior_categories = list(y_salience_list[sessions[0]])

    output = {}
    for cat in behavior_categories:
        category_conjunctions = []
        for name in latent_variable_names:
            behaviors = []
            for sess in sessions: 
                df = y_salience_list[sess][cat]
                behaviors.append(df[name].values ** 2)
                sub_behaviors = df.index

            conj_data = pd.DataFrame(np.asarray(behaviors).T, index=sub_behaviors)
            res = mf.conjunction_analysis(conj_data, 'any', return_avg=return_avg)
            
            res_squared = mf.conjunction_analysis(conj_data**2, 'any', return_avg=return_avg)
            for row in range(res_squared.values.shape[0]):
                for col in range(res_squared.values.shape[1]):
                    if res[row, col] == 0:
                        res_squared[row, col] = 0

            category_conjunctions.append(res_squared.values)
        conj_all_latent_variables = np.squeeze(np.asarray(category_conjunctions).T)

        output[cat] = pd.DataFrame(conj_all_latent_variables,
                                   index=sub_behaviors,
                                   columns=latent_variable_names)

    return output
    
def _x_conjunctions_single_session(single_session_res, latent_variable_names, return_avg=True):
    """Run conjunctions on brain data across the three models"""

    sessions = list(single_session_res)
    print(sessions)
    x_salience_list = {}
    for sess in sessions:
        output = single_session_res[sess]
        print(list(output))
        x_salience_list[sess] = output['x_saliences']

    output = {}
    brain_conjunctions = []
    for name in latent_variable_names: # iterate through latent vars
        brains = []
        for sess in sessions: 
            df = x_salience_list[sess]
            brains.append(df[name].values)
            rois = df.index

        conj_data = pd.DataFrame(np.asarray(brains).T, index=rois)
        res = mf.conjunction_analysis(conj_data, 'sign', return_avg=return_avg)
        
        res_squared = mf.conjunction_analysis(conj_data**2, 'any', return_avg=return_avg)
        for row in range(res_squared.values.shape[0]):
            for col in range(res_squared.values.shape[1]):
                if res[row, col] == 0:
                    res_squared[row, col] = 0
        
        brain_conjunctions.append(res_squared.values)

    conj_all_latent_variables = np.squeeze(np.asarray(brain_conjunctions).T)
    output = pd.DataFrame(conj_all_latent_variables,
                          index=rois,
                          columns=latent_variable_names)

    return output
    
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
    behavior_metadata = pd.read_csv(pdir + '/data/b_variables_mPLSC.txt', delimiter='\t', header=None)

    behavior_metadata.rename(dict(zip([0, 1], ['category','name'])), axis='columns', inplace=True)
    
    behavior_raw = pd.read_excel(pdir + '/data/hcp_behavioral.xlsx',
                                  index_col=0, sheet_name='cleaned')
    
    behavior_data = mf.load_behavior_subtables(behavior_raw, behavior_metadata)
    y_tables = [behavior_data[t] for t in list(behavior_data)]
    
    fig_path = pdir + '/figures/mPLSC_power_per_session'
    single_session_mPLSC = {}
    p = pls.MultitablePLSC(n_iters=10000, return_perm=False)
    alpha = .001
    latent_variable_check = []
    for index, x_table in enumerate(x_tables):
        x_tables_jr = [x_table]
        print('%s: Running permutation testing on latent variables' % pu.ctime())
        res_perm = p.mult_plsc_eigenperm(y_tables, x_tables_jr)
        
        num_latent_vars = len(np.where(res_perm['p_values'] < alpha)[0])
        latent_variable_check.append(num_latent_vars)
        
        print('%s: Plotting scree' % pu.ctime())   
        mf.plotScree(res_perm['true_eigenvalues'],
                     res_perm['p_values'],
                     alpha=alpha,
                     fname=fig_path + '/scree_%s.png' % meg_sess[index])

    best_num_latent_vars = np.min(latent_variable_check)
    latent_names = ['LatentVar%d' % (n+1) for n in range(best_num_latent_vars)]
    print(latent_names)
    
    for index, x_table in enumerate(x_tables):
        x_tables_jr = [x_table]
        print('%s: Running bootstrap testing on saliences' % pu.ctime())
        res_boot = p.mult_plsc_bootstrap_saliences(y_tables, x_tables_jr, 4)
    
        print('%s: Organizing behavior saliences' % pu.ctime())
        y_saliences = res_boot['y_saliences'][:, :best_num_latent_vars]
        y_saliences_tables = mf.create_salience_subtables(y_saliences, y_tables, list(behavior_data), latent_names)
        
        y_salience_z = sals=res_boot['zscores_y_saliences'][:, :best_num_latent_vars]
        y_saliences_ztables = mf.create_salience_subtables(y_salience_z, y_tables, list(behavior_data), latent_names)
        
        print('%s: Averaging saliences within behavior categories' % pu.ctime())
        print(list(y_saliences))
        res_behavior = mf.average_behavior_scores(y_saliences_tables, latent_names)
    
        print('%s: Organizing brain saliences' % pu.ctime())
        meg_subtable_name = 'meg_%s' % meg_sess[index]
        x_table_name = meg_subtable_name 
        x_saliences = res_boot['x_saliences'][:, :best_num_latent_vars]
        x_saliences_tables = pd.DataFrame(x_saliences, index=rois, columns=latent_names)
        
        x_salience_z = sals=res_boot['zscores_x_saliences'][:, :best_num_latent_vars]
        x_saliences_ztables = pd.DataFrame(x_salience_z, index=rois, columns=latent_names)
        
        output = {'permutation_tests':res_perm,
                  'bootstrap_tests':res_boot,
                  'y_saliences':y_saliences_tables,
                  'x_saliences':x_saliences_tables,
                  'y_saliences_zscores':y_saliences_ztables,
                  'x_saliences_zscores':x_saliences_ztables,
                  'behaviors':res_behavior}
        
        single_session_mPLSC[meg_subtable_name] = output
    
#     print('%s: Running behavior conjunctions' % pu.ctime())
#     behavior_conj = _y_conjunctions_single_session(single_session_mPLSC, latent_names, return_avg=True)
    
#     print('%s: Running brain conjunctions' % pu.ctime())
#     brain_conj = _x_conjunctions_single_session(single_session_mPLSC, latent_names, return_avg=True)
    
    print('%s: Saving results' % pu.ctime())
    with open(pdir + '/data/mPLSC/mPLSC_power_per_session.pkl', 'wb') as file:
#         single_session_mPLSC['behavior_conjunction'] = behavior_conj
#         single_session_mPLSC['brain_conjunction'] = brain_conj
        pkl.dump(single_session_mPLSC, file)
        
    print('%s: Finished' % pu.ctime())