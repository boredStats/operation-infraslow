# -*- coding: utf-8 -*-
"""
Run multi-table PLS-C using MEG power data

Created on Thu Mar 28 12:55:29 2019
"""

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

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

def get_highest_squared_brain_salience(res_conj, latent_names):
        vals = []
        for name in latent_names:
            vals.append(res_conj[name].values **2)
        return np.max(vals)

def plot_bar(series):
    x = np.arange(len(series.values))
    fig, ax = plt.subplots()
    plt.bar(x, series.values)

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    import proj_utils as pu
    import mPLSC_functions as mf

    from boredStats import pls_tools as pls

    pdir = pu._get_proj_dir()
    ddir = pdir + '/data'
    pdObj = pu.proj_data()
    rois = pdObj.roiLabels
    meg_subj, meg_sessions = pdObj.get_meg_metadata()
    mri_subj, mri_sess = pdObj.get_mri_metadata()
    subject_overlap = [s for s in mri_subj if s in meg_subj]

    alpha = .001
    z_test = 0
    output_file = ddir + '/mPLSC/mPLSC_power_all_sessions.pkl'
    check = input('Run multitable PLS-C? y/n ')
    if check=='y':
        print('%s: Building subtables of power data for MEG' % pu.ctime())
        meg_data = mf.extract_average_power(
            hdf5_file=ddir+'/downsampled_MEG_truncated.hdf5',
            sessions=meg_sessions,
            subjects=subject_overlap,
            rois=rois,
            image_type='MEG',
            bp=True
            )
        x_tables = [meg_data[session] for session in list(meg_data)]

        print('%s: Building subtables of behavior data' % pu.ctime())
        behavior_metadata = pd.read_csv(
            ddir+'/b_variables_mPLSC.txt',
            delimiter='\t',
            header=None
            )
        behavior_metadata.rename(
            dict(zip([0, 1], ['category','name'])),
            axis='columns',
            inplace=True
            )
        behavior_raw = pd.read_excel(
            ddir+'/hcp_behavioral.xlsx',
            index_col=0,
            sheet_name='cleaned'
            )
        behavior_data = mf.load_behavior_subtables(behavior_raw, behavior_metadata)
        y_tables = [behavior_data[category] for category in list(behavior_data)]

        p = pls.MultitablePLSC(n_iters=10000, return_perm=False)
        print('%s: Running permutation tests on latent variables' % pu.ctime())
        res_perm = p.mult_plsc_eigenperm(y_tables, x_tables)

        print('%s: Running bootstrap testing on saliences' % pu.ctime())
        res_boot = p.mult_plsc_bootstrap_saliences(y_tables, x_tables, z_test)
        num_latent_vars = len(np.where(res_perm['p_values'] < alpha)[0])
        latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]

        print('%s: Organizing saliences' % pu.ctime())
        y_saliences, y_saliences_z = mf.organize_behavior_saliences(
            res_boot=res_boot,
            y_tables=y_tables,
            sub_names=list(behavior_data),
            num_latent_vars=num_latent_vars
            )
        x_saliences, x_saliences_z = mf.organize_brain_saliences(
            res_boot=res_boot,
            x_tables=x_tables,
            sub_names=['MEG_%s' % session for session in meg_sessions],
            num_latent_vars=num_latent_vars,
            )

        print('%s: Saving results' % pu.ctime())
        output = {'permutation_tests':res_perm,
                  'bootstrap_tests':res_boot,
                  'y_saliences':y_saliences,
                  'x_saliences':x_saliences,
                  'y_saliences_zscores':y_saliences_z,
                  'x_saliences_zscores':x_saliences_z
                  }

        with open(output_file, 'wb') as file:
            pkl.dump(output, file)
    else:
        with open(output_file, 'rb') as file:
            output = pkl.load(file)

        res_perm = output['permutation_tests']
        res_boot = output['bootstrap_tests']
        y_saliences = output['y_saliences']
        y_saliences_z = output['y_saliences_zscores']
        x_saliences = output['x_saliences']
        x_saliences_z = output['x_saliences_zscores']

    fig_path = pdir + '/figures/mPLSC_power_all_sessions'
    mf.save_xls(y_saliences, fig_path + '/behavior_saliences.xlsx')
    mf.save_xls(x_saliences, fig_path+'/brain_saliences.xlsx')

    mf.save_xls(y_saliences_z, fig_path+'/behavior_saliences_z.xlsx')
    mf.save_xls(x_saliences_z, fig_path+'/brain_saliences_z.xlsx')

    num_latent_vars = len(np.where(res_perm['p_values'] < alpha)[0])
    latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]

    print('%s: Plotting scree' % pu.ctime())
    mf.plotScree(res_perm['true_eigenvalues'],
                 res_perm['p_values'],
                 alpha=alpha,
                 fname=fig_path+'/scree.png')

    z_thresh = 1
    print('%s: Running conjunction on brain data' % pu.ctime())
    brain_conjunction = mf.single_table_conjunction(x_saliences_z, z_thresh)
    brain_conjunction.to_excel(fig_path+'/brain_conjunction.xlsx')
    # roi_path = ddir + '/glasser_atlas/'
    # print('%s: Creating brain figures' % pu.ctime())
    # maxval = get_highest_squared_brain_salience(res_conj, latent_names)
    # print('Max salience is %.3f' % maxval)
    # for name in latent_names:
    #     mags = res_conj[name].values **2
    #     bin_mags = []
    #     for m in mags:
    #         if m > 0:
    #             bin_mags.append(1)
    #         else:
    #             bin_mags.append(0)
    #
    #     fname = fig_path + '/brain_binarized_%s.png' % name
    #     custom_roi = mf.create_custom_roi(roi_path, rois, bin_mags)
    #     minval = np.min(mags[np.nonzero(mags)])
    #     if len(np.nonzero(mags)) == 1:
    #         minval = None
    #     mf.plot_brain_saliences(custom_roi, minval, maxval, figpath=fname)
    #
    #     fname = fig_path + '/brain_%s.png' % name
    #     custom_roi = mf.create_custom_roi(roi_path, rois, mags)
    #     minval = np.min(mags[np.nonzero(mags)])
    #     if len(np.nonzero(mags)) == 1:
    #         minval = None
    #     mf.plot_brain_saliences(custom_roi, minval, maxval, figpath=fname)

    print('%s: Finished' % pu.ctime())
