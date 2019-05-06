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

def _signed_average_brain_sal(res_conj, sals_per_session):
    sessions = list(sals_per_session)
    rois = res_conj.index
    latent_names = list(res_conj)

    output = pd.DataFrame(index=rois, columns=latent_names)
    for roi in rois:
        for latent_variable in latent_names:
            conj_test = res_conj.loc[roi, latent_variable]
            if conj_test != 0:
                sals = []
                for sess in sessions:
                    df = sals_per_session[sess]
                    val = df.loc[roi, latent_variable]
                    sals.append(val)
                mean_sal = np.mean(np.abs(sals))
                output.loc[roi, latent_variable] = mean_sal
            else:
                output.loc[roi, latent_variable] = 0

    return output

def print_zmeta(brain_conjunction, latent_names, fname=None):
    all_values = []
    for name in latent_names:
        mags = brain_conjunction[name]
        all_values.append(mags.values)
    min_array = np.ma.masked_equal(all_values, 0.0, copy=False)
    min_z = min_array.min()
    max_z = np.max(all_values)
    mu = np.mean(all_values)
    std = np.std(all_values, ddof=1)

    if fname is not None:
        with open(fname, 'w') as file:
            file.write('Min z-score is %.4f\n' % min_z)
            file.write('Max z-score is %.4f\n' % max_z)
            file.write('Mean z-score is: %.4f\n' % mu)
            file.write('Std-dev z-score is: %.4f' % std)
    else:
        print('Mean brain z-score is: %.4d' % mu)
        print('Std-dev brain z-score is: %.4d' % std)

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
    colors = pdObj.colors
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

    #Number of latent variables decided by scree
    num_latent_vars = 5#len(np.where(res_perm['p_values'] < alpha)[0])
    latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]

    print('%s: Plotting scree' % pu.ctime())
    mf.plotScree(res_perm['true_eigenvalues'],
                 res_perm['p_values'],
                 alpha=alpha,
                 fname=fig_path+'/scree.png')

    print('%s: Running conjunction on brain data' % pu.ctime())
    brain_conjunction = mf.single_table_conjunction(x_saliences_z, comp='any', thresh=4)
    brain_conjunction.to_excel(fig_path+'/brain_conjunction.xlsx')
    # brain_conjunction_signed = _signed_average_brain_sal(brain_conjunction, x_saliences_z)
    # brain_conjunction_signed.to_excel(fig_path+'/brain_conjunction.xlsx')

    print('%s: Averaging saliences within behavior categories' % pu.ctime())
    behavior_avg = mf.average_subtable_saliences(y_saliences_z)
    behavior_avg.to_excel(fig_path+'/behavior_average_z.xlsx')

    print('%s: Plotting behavior bar plots' % pu.ctime())
    max_z = 0
    for latent_variable in latent_names:
        current_max = np.max(behavior_avg[latent_variable])
        if current_max > max_z:
            max_z = current_max

    for latent_variable in latent_names:
        series = behavior_avg[latent_variable]
        mf.plot_bar(series, max_z, colors, fig_path+'/behavior_z_%s.svg' % latent_variable)

    y_saliences_zscores_thresh = {}
    for behavior_category in list(y_saliences_z):
        unthresh_df = y_saliences_z[behavior_category]
        unthresh_vals = unthresh_df.values
        thresh_vals = np.ndarray(shape=unthresh_vals.shape)
        for r in range(len(unthresh_df.index)):
            for c in range(len(list(unthresh_df))):
                if np.abs(unthresh_vals[r, c]) < 4:
                    thresh_vals[r, c] = 0
                else:
                    thresh_vals[r, c] = unthresh_vals[r, c]
        thresh_df = pd.DataFrame(thresh_vals, index=unthresh_df.index, columns=list(unthresh_df))
        y_saliences_zscores_thresh[behavior_category] = thresh_df

    print('%s: Creating bar/table combo figure' % pu.ctime())
    def _quick_hist(data, fname=None):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.hist(data, bins=10)
        if fname is not None:
            fig.savefig(fname, bbox_inches='tight')

    print('%s: Creating bar/table combo figures' % pu.ctime())
    mu = []
    for name in latent_names:
        latent_data = []
        for behavior_category in list(y_saliences_zscores_thresh):
            behavior_df = y_saliences_zscores_thresh[behavior_category]
            series = behavior_df[name]
            latent_data.append(np.abs(series.values))
        latent_data_onelist = [v for vals in latent_data for v in vals]
        mu.append(np.mean(latent_data_onelist))
        _quick_hist(latent_data_onelist, fig_path+'/behavior_hist_%s.png' % name)

    for n, name in enumerate(latent_names):
        mean = mu[n]
        print(mean)
        fname = fig_path+'/behavior_fullbar_%s.svg' % name
        mf.bar_all_behaviors(y_saliences_zscores_thresh, name, mean, colors, 80, fname)

    roi_path = ddir + '/glasser_atlas/'
    print('%s: Creating brain figures' % pu.ctime())

    import nibabel as nib
    print_zmeta(brain_conjunction, latent_names, fig_path+'/brain_z_meta.txt')
    check = input('Plot brains? y/n ')
    if check == 'y':
        for name in latent_names:
            mags = brain_conjunction[name]
            mu = np.mean(mags)
            # custom_roi = mf.create_custom_roi(roi_path, rois, mags)

            custom_roi = nib.load(fig_path + '/%s.nii.gz' % name)
            fname = fig_path + '/brain_%s.pdf' % name
            fname = 'brain_%s.pdf' % name
            mf.plot_brain_saliences(
                custom_roi,
                minval=4,
                maxval=40,
                figpath=fname,
                cbar=False,
                cmap='viridis')

        true_brain_means = pd.DataFrame(columns=latent_names)
        for name in latent_names:
            session_data = np.ndarray(shape=(len(rois), len(meg_sessions)))
            for s, sess in enumerate(list(x_saliences_z)):
                session_df = x_saliences_z[sess]
                session_lv = session_df[name]
                lv_vals = np.abs(session_lv.values)
                session_data[:, s] = lv_vals
            # hist_data = np.ndarray.flatten(session_data)
            # _quick_hist(hist_data, fig_path + '/brain_hist_%s.png' % name)
            average_lv = np.mean(session_data, axis=1)
            true_mu = np.mean(average_lv)
            true_brain_means.loc['mu', name] = true_mu
        #
        # for name in latent_names:
        #     mags = brain_conjunction[name]
        #     mu = true_brain_means[name].values
        #     _quick_hist(mags.values, fig_path + '/brain_hist_%s.png' % name)
        #     print(mu)
        #     # custom_roi = mf.create_custom_roi(roi_path, rois, mags)
        #     # nib.save(custom_roi, fig_path + '/%s.nii.gz' % name)
        #     custom_roi = nib.load(fig_path + '/%s.nii.gz' % name)
        #     fname = fig_path + '/brain_%s' % name
        #     mf.plot_brain_saliences_no_subplots(
        #         custom_roi,
        #         minval=4,
        #         maxval=80,#mu,
        #         figpath=fname,
        #         cmap='viridis',
        #         # cbar=True
        #     )
    print('%s: Finished' % pu.ctime())
