# -*- coding: utf-8 -*-
"""
Run multi-table PLS-C using MEG BOLD-passband power data + fMRI data

Created on Thu Mar 28 12:55:29 2019
"""

import h5py
import numpy as np
import pandas as pd
import pickle as pkl

def _extract_metadata(meg_indices):
    sess, subj, band = [], [], []
    for idx in meg_indices:
        sess.append(idx[0])
        subj.append(idx[1])
        band.append(idx[2])
    
    sessions = list(pd.unique(sess))
    subjects = list(pd.unique(subj))
    bands = list(pd.unique(band))
    return sessions, subjects, bands

def _extract_average_power(hdf5_file, sessions, subjects, image_type):
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

            fft_power = np.absolute(np.fft.rfft(data, axis=0))**2
            average_power = np.mean(fft_power, axis=0)
            session_data.append(average_power)
        
        power_data[sess] = np.asarray(session_data)
        
    return power_data

def _create_salience_subtables(sals, dataframes, subtable_names, latent_names):
    salience_subtables = {}
    start = 0
    for t, table in enumerate(dataframes):
        num_variables_in_table = table.values.shape[1]
        end = start + num_variables_in_table
        
        saliences = sals[start:end, :]
        df = pd.DataFrame(saliences, index=list(table), columns=latent_names)
        salience_subtables[subtable_names[t]] = df
        start = end
    return salience_subtables

def _merge_mri_meg(mri_data, meg_data):
    x_meg_tables = [meg_data[t] for t in list(meg_data)]
    x_mri_tables = [mri_data[t] for t in list(mri_data)]
    x_tables = x_meg_tables + x_mri_tables
    
    return x_tables
    
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    import proj_utils as pu
    from mPLSC_functions import _load_behavior_subtables
    
    from boredStats import pls_tools as pls
    
    pdir = pu._get_proj_dir()
    pdObj = pu.proj_data()
    meg_subj, meg_sess = pdObj.get_meg_metadata()
    mri_subj, mri_sess = pdObj.get_mri_metadata()
    subj_overlap = [s for s in mri_subj if s in meg_subj]
    
    meg_path = pdir + '/data/downsampled_MEG_truncated.hdf5'    
    mri_path = pdir + '/data/multimodal_HCP.hdf5'
    
    print('%s: Building subtables of power data for MRI, MEG' % pu.ctime())
    mri_data = _extract_average_power(mri_path, mri_sess, mri_subj, 'MRI')
    meg_data = _extract_average_power(meg_path, meg_sess, meg_subj, 'MEG')
    x_tables = _merge_mri_meg(mri_data, meg_data)
    
    print('%s: Building subtables of behavior data' % pu.ctime())
    behavior_metadata = pd.read_csv(pdir + '/data/b_variables_mPLSC.txt',
                                   delimiter='\t', header=None)

    behavior_metadata.rename(dict(zip([0, 1], ['category','name'])),
                                axis='columns', inplace=True)
    
    behavior_raw = pd.read_excel(pdir + '/data/hcp_behavioral.xlsx',
                                  index_col=0, sheet_name='cleaned')
    
    behavior_data = _load_behavior_subtables(behavior_raw, behavior_metadata)
    y_tables = [behavior_data[t] for t in list(behavior_data)]
    
    p = pls.MultitablePLSC(n_iters=10000)
    print('%s: Running permutation testing on latent variables' % pu.ctime())
    res_perm = p.mult_plsc_eigenperm(y_tables, x_tables)

    print('%s: Running bootstrap testing on saliences' % pu.ctime())
    res_boot = p.mult_plsc_bootstrap_saliences(y_tables, x_tables, 3)
    
    output = {'permutation_tests':res_perm,
              'bootstrap_tests':res_boot,
              'behavior_subtables':behavior_data,
              'mri_data':mri_data,
              'meg_data':meg_data}
    
    with open(pdir + '/data/mPLSC_power.pkl', 'wb') as file:
        pkl.dump(output, file)