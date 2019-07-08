import os
import logging
import pls_functions
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils as pu


logging.basicConfig(level=logging.INFO)


def organize_brain_sals(x_zscores, rois, sessions, latent_vars, comp='any'):
    # Utility function for these analyses

    def _conjunction_analysis(brain_data, compare='any', thresh=0, return_avg=False):
        conjunction = [0 for row in brain_data.index]
        for r in range(brain_data.shape[0]):
            vals = brain_data[r, :]
            if compare == 'any':
                if all(vals) and all(np.abs(vals) > thresh):
                    if return_avg:
                        conjunction[r] = np.mean(vals)
                    else:
                        conjunction[r] = 1

            elif compare == 'sign':
                if all(np.abs(vals) > thresh):
                    if all(np.sign(vals) > 0) or all(np.sign(vals) < 0):
                        if return_avg:
                            conjunction[r] = np.mean(vals)
                        else:
                            conjunction[r] = 1
        return conjunction

    nv = len(latent_vars)
    latent_brain_data = x_zscores[:, :nv]
    session_data = np.array_split(latent_brain_data, 3, axis=0)
    res_dict = {}
    for s, sess in enumerate(sessions):
        res_df = pd.DataFrame(session_data[s], index=rois, columns=latent_vars)
        key = '%s_brain_zscores' % sess
        res_dict[key] = res_df

    conj_df = pd.DataFrame(index=rois, columns=latent_vars)
    for l, lv in enumerate(latent_vars):
        brains = np.ndarray(shape=(len(rois), len(sessions)))
        for s, sess in enumerate(sessions):
            sess_df = res_dict[sess]
            brains[:, s] = sess_df[lv].values
        conj = _conjunction_analysis(brains, thresh=4, return_avg=True)
        conj_df[lv] = conj

    res_dict['brain_conjunction'] = conj_df
    return res_dict


def pls_sleep():
    fig_dir = '../figures/PLS/psqi_components'
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)

    with open('../data/MEG_power_data.pkl', 'rb') as file:
        meg_data = pkl.load(file)
    sessions = list(meg_data)
    rois = list(meg_data[sessions[0]])
    meg_list = [meg_data[sess] for sess in list(meg_data)]
    meg_df = pd.concat(meg_list, axis=1)
    x = meg_df.values

    sleep_variables = ['PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7']
    behavior_raw = pd.read_excel('../data/hcp_behavioral.xlsx', index_col=0, sheet_name='cleaned')

    sleep_df = behavior_raw[sleep_variables]
    y = sleep_df.values

    logging.info('%s: Running PLSC on PSQI components' % pu.ctime())
    p = pls_functions.PLSC(n_iters=10000)

    pres = p.permutation_tests(x, y)
    eigs = pres['true_eigs']
    pvals = pres['p_values']
    alpha = .001
    nv = len(np.where(pvals < alpha)[0])
    latent_vars = ['LV_%d' % (v+1) for v in range(nv)]
    pls_functions.plot_scree(eigs=eigs, pvals=pvals, alpha=alpha, fname=fig_dir + '/scree.png')

    bres = p.bootstrap_tests(x, y)
    print(bres['y_zscores'][0, :])
    print(bres['x_zscores'][:, 0])

    behavior_df = pd.DataFrame(bres['y_zscores'][:nv, :], index=latent_vars, columns=sleep_variables)
    print(behavior_df)

    behavior_df.to_excel(fig_dir+'/behavior_res.xlsx')
    brain_res = organize_brain_sals(bres['x_zscores'], rois, sessions, latent_vars)
    pu.save_xls(brain_res, fig_dir+'/brain_res.xlsx')

    logging.info('%s: Finished' % pu.ctime())


pls_sleep()

