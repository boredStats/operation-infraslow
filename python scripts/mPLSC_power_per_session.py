"""
Run multi-table PLS-C using MEG power data from each session separately
"""

import h5py
import numpy as np
import pandas as pd
import pickle as pkl

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
    ddir = pdir + '/data'
    pdObj = pu.proj_data()
    rois = pdObj.roiLabels
    meg_subj, meg_sessions = pdObj.get_meg_metadata()
    mri_subj, mri_sess = pdObj.get_mri_metadata()
    subject_overlap = [s for s in mri_subj if s in meg_subj]

    alpha = .001
    z = 0
    output_file = ddir + '/mPLSC/mPLSC_power_per_session.pkl'
    fig_path = pdir + '/figures/mPLSC_power_per_session'
    roi_path = ddir + '/glasser_atlas/'

    check = input('Run multitable PLS-C? y/n')
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
        x_tables = [meg_data[session] for session in meg_sessions]

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

        single_session_mPLSC = {}
        p = pls.MultitablePLSC(n_iters=10000, return_perm=False)

        print('%s: Running permutation tests' % pu.ctime())
        significant_latent_variables = []
        for index, x_table in enumerate(x_tables):
            res_perm = p.mult_plsc_eigenperm(y_tables, [x_table])

            num_latent_vars = len(np.where(res_perm['p_values'] < alpha)[0])
            significant_latent_variables.append(num_latent_vars)

            print('%s: Plotting scree' % pu.ctime())
            mf.plotScree(
                eigs=res_perm['true_eigenvalues'],
                pvals=res_perm['p_values'],
                alpha=alpha,
                fname=fig_path + '/scree_%s.png' % meg_sessions[index]
                )

        best_num = np.min(significant_latent_variables)
        latent_names = ['LatentVar%d' % (n+1) for n in range(best_num)]
        print('Best number of latent variables is: %d' % best_num)

        print('%s: Running bootstrap tests' % pu.ctime())
        for index, x_table in enumerate(x_tables):
            res_boot = p.mult_plsc_bootstrap_saliences(y_tables, [x_table], z)

            print('%s: Organizing behavior saliences' % pu.ctime())
            y_saliences_tables = mf.create_salience_subtables(
                sals=res_boot['y_saliences'][:, :best_num],
                dataframes=y_tables,
                subtable_names=list(behavior_data),
                latent_names=latent_names
                )
            y_saliences_ztables = mf.create_salience_subtables(
                sals=res_boot['zscores_y_saliences'][:, :best_num],
                dataframes=y_tables,
                subtable_names=list(behavior_data),
                latent_names=latent_names)

            print('%s: Organizing brain saliences' % pu.ctime())
            x_saliences_tables = pd.DataFrame(
                res_boot['x_saliences'][:, :best_num],
                index=rois,
                columns=latent_names
                )
            x_saliences_ztables = pd.DataFrame(
                res_boot['zscores_x_saliences'][:, :best_num],
                index=rois,
                columns=latent_names
                )

            output = {
                'permutation_tests':res_perm,
                'bootstrap_tests':res_boot,
                'y_saliences':y_saliences_tables,
                'x_saliences':x_saliences_tables,
                'y_saliences_zscores':y_saliences_ztables,
                'x_saliences_zscores':x_saliences_ztables,
                }
            meg_subtable_name = meg_sessions[index]
            single_session_mPLSC[meg_subtable_name] = output

        print('%s: Saving results' % pu.ctime())
        with open(output_file, 'wb') as file:
            pkl.dump(single_session_mPLSC, file)

    else:
        with open(output_file, 'rb') as file:
            single_session_mPLSC = pkl.load(file)

    session_behavior_saliences, session_brain_saliences = {}, {}
    session_behavior_z, session_brain_z = {}, {}
    for session in meg_sessions:
        session_dict = single_session_mPLSC[session]
        session_behavior_saliences[session] = session_dict['y_saliences']
        fname = fig_path+'/behavior_saliences_%s.xlsx' % session
        mf.save_xls(session_dict['y_saliences'], fname)
        session_behavior_z[session] = session_dict['y_saliences_zscores']

        session_brain_saliences[session] = session_dict['x_saliences']
        session_brain_z[session] = session_dict['x_saliences_zscores']

    mf.save_xls(session_brain_saliences, fig_path+'/brain_saliences.xlsx')
    mf.save_xls(session_brain_z, fig_path+'/brain_saliences_z.xlsx')

    print('%s: Running conjunction on behavior data' % pu.ctime())
    behavior_conjunction_s = mf.behavior_conjunctions(
        session_behavior_saliences,
        comp='sign'
        )
    mf.save_xls(behavior_conjunction_s, fig_path+'/behavior_conjunction_sals_sign.xlsx')

    behavior_conjunction_z = mf.behavior_conjunctions(
        session_behavior_z,
        thresh=4
    )
    mf.save_xls(behavior_conjunction_z, fig_path+'/behavior_conjunction_z_magnitude.xlsx')

    behavior_conjunction_zs = mf.behavior_conjunctions(
        session_behavior_z,
        thresh=4,
        comp='sign'
    )
    mf.save_xls(behavior_conjunction_zs, fig_path+'/behavior_conjunction_z_sign.xlsx')

    print('%s: Averaging saliences within behavior categories' % pu.ctime())
    behavior_avg = mf.average_subtable_saliences(behavior_conjunction_z)
    behavior_avg.to_excel(fig_path+'/behavior_average_z.xlsx')

    print('%s: Running conjunction on brain data' % pu.ctime()) ###########
    brain_conjunction_s = mf.single_table_conjunction(
        session_brain_saliences,
        comp='sign'
        )
    brain_conjunction_z = mf.single_table_conjunction(
        session_brain_z,
        thresh=4
    )
    brain_conjunctoin_zs = mf.single_table_conjunction(
        session_brain_z,
        thresh=4,
        comp='sign'
    )
    brain_conjunction = {
        'sals_sign':brain_conjunction_s,
        'z_magnitude':brain_conjunction_z,
        'z_sign':brain_conjunctoin_zs
        }
    mf.save_xls(brain_conjunction, fig_path+'/brain_conjunction.xlsx')

    print('%s: Plotting behavior bar plots' % pu.ctime())
    num_latent_vars = len(list(behavior_avg))
    latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]
    for latent_variable in latent_names:
        series = behavior_avg[latent_variable]
        mf.plot_bar(series, fig_path+'/behavior_z_%s.png' % latent_variable)

    print('%s: Plotting brain pictures' % pu.ctime())
    for latent_variable in latent_names:
        mags = brain_conjunction_z[latent_variable]
        fname = fig_path + '/brain_z_%s.png' % latent_variable

        custom_roi = mf.create_custom_roi(roi_path, rois, mags)
        mf.plot_brain_saliences(custom_roi, figpath=fname)

    print('%s: Finished' % pu.ctime())
