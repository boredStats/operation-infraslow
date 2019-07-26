"""
Analysis script for PLSC
Behavior variables - PSQI components

See pls_functions for code pertaining to PLSC itself
"""

import os
import h5py
import logging
import pls_functions
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils as pu
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy
from nilearn import surface, plotting, datasets
import mPLSC_functions as mf


logging.basicConfig(level=logging.INFO)


def load_psqi_data():
    sleep_variables = ['PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7']
    behavior_raw = pd.read_excel('../data/hcp_behavioral.xlsx', index_col=0, sheet_name='cleaned')
    sleep_df = behavior_raw[sleep_variables].astype(float)

    return sleep_df, sleep_variables


def create_custom_roi(roi_path, rois_to_combine, roi_magnitudes, fname=None):
    def stack_3d_dynamic(template, roi_indices, mag):
        t_copy = deepcopy(template)
        for num_counter in range(len(roi_indices[0])):
            x = roi_indices[0][num_counter]
            y = roi_indices[1][num_counter]
            z = roi_indices[2][num_counter]
            t_copy[x, y, z] = mag
        return t_copy

    rn = '%s.nii.gz' % rois_to_combine[0]
    t_vol = nib.load(os.path.join(roi_path, rn))
    temp = t_vol.get_data()
    temp[temp > 0] = 0
    for r, roi in enumerate(rois_to_combine):
        if roi_magnitudes[r] == 0:
            pass
        rn = '%s.nii.gz' % roi
        volume_data = nib.load(os.path.join(roi_path, rn)).get_data()
        r_idx = np.where(volume_data > 0)
        temp = stack_3d_dynamic(temp, r_idx, roi_magnitudes[r])

    nifti = nib.Nifti1Image(temp, t_vol.affine, t_vol.header)
    if fname is not None:
        nib.save(nifti, fname)
    return nifti


def plot_brain_saliences(custom_roi, minval=0, maxval=None, figpath=None, cbar=False, cmap=None):
    mpl.rcParams.update(mpl.rcParamsDefault)
    if cmap is None:
        cmap = 'coolwarm'

    fsaverage = datasets.fetch_surf_fsaverage()

    orders = [('medial', 'left'), ('medial', 'right'), ('lateral', 'left'), ('lateral', 'right')]

    fig, ax = plt.subplots(2, 2, figsize=(8.0, 6.0), dpi=300, subplot_kw={'projection': '3d'})

    fig.subplots_adjust(hspace=0., wspace=0.00005)
    axes_list = fig.axes

    for index, order in enumerate(orders):
        view = order[0]
        hemi = order[1]

        texture = surface.vol_to_surf(custom_roi, fsaverage['pial_%s' % hemi])
        plotting.plot_surf_roi(
                fsaverage['infl_%s' % hemi],
                texture,
                cmap=cmap,
                hemi=hemi,
                view=view,
                bg_on_data=True,
                axes=axes_list[index],
                bg_map=fsaverage['sulc_%s' % hemi],
                vmin=minval,
                vmax=maxval,
                output_file=figpath,
                symmetric_cbar=False,
                figure=fig,
                darkness=.5,
                colorbar=cbar)
    plt.clf()


def organize_brain_sals(x_zscores, roi_list, sessions, latent_vars, comp='any'):
    def _conjunction_analysis(brain_data, compare='any', thresh=0, return_avg=False):
        conjunction = [0 for row in range(brain_data.shape[0])]
        for r in range(brain_data.shape[0]):
            vals = brain_data[r, :]
            if compare == 'any':
                if all(vals) and all(np.abs(vals) > thresh):
                    if return_avg:
                        conjunction[r] = np.mean(np.abs(vals))
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
        res_df = pd.DataFrame(session_data[s], index=roi_list, columns=latent_vars)
        key = '%s_brain_zscores' % sess
        res_dict[key] = res_df

    conj_df = pd.DataFrame(index=roi_list, columns=latent_vars)
    for l, lv in enumerate(latent_vars):
        brains = np.ndarray(shape=(len(roi_list), len(sessions)))
        for s, sess in enumerate(sessions):
            key = '%s_brain_zscores' % sess
            sess_df = res_dict[key]
            brains[:, s] = sess_df[lv].values
        conj = _conjunction_analysis(brains, compare=comp, thresh=4, return_avg=True)
        conj_df[lv] = conj

    res_dict['brain_conjunction'] = conj_df
    return res_dict


def plot_roi_saliences(rois, conjunction_result, fig_dir, cmap=None, maxv=40, create_rois=False):
    roi_path = '../data/glasser_atlas/'
    for lv in list(conjunction_result):
        if create_rois:
            mags = conjunction_result[lv]
            mags[mags < 40] = 0
            custom_roi = create_custom_roi(roi_path, rois, mags)
            nib.save(custom_roi, fig_dir + '/%s.nii.gz' % lv)
        else:
            custom_roi = nib.load(fig_dir + '/%s.nii.gz' % lv)
        fname = fig_dir + '/brain_%s' % lv
        plot_brain_saliences(custom_roi, minval=4, maxval=maxv, figpath=fname, cmap=cmap, cbar=False)


def pls_psqi_with_power(sessions, rois, fig_dir, run_check=True):
    logging.info('%s: Running PLSC on PSQI components with power' % pu.ctime())
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)

    meg_df = pd.read_excel('../data/MEG_infraslow_power.xlsx', sheet_name='location', index_col=0)
    sleep_df, sleep_variables = load_psqi_data()

    if run_check:
        p = pls_functions.PLSC(n_iters=10000, center_scale='ss1')
        logging.info('%s: Running permutation tests' % pu.ctime())
        pres = p.permutation_tests(meg_df.values, sleep_df.values)
        logging.info('%s: Running bootstrap tests' % pu.ctime())
        bres = p.bootstrap_tests(meg_df.values, sleep_df.values)
        res = {'permutation tests': pres, 'bootstrap_tests': bres}
        with open(fig_dir + '/pls_sleep.pkl', 'wb') as file:
            pkl.dump(res, file)
    else:
        logging.info('%s: Loading raw output' % pu.ctime())
        with open(fig_dir + '/pls_sleep.pkl', 'rb') as file:
            res = pkl.load(file)
        pres = res['permutation tests']
        bres = res['bootstrap_tests']

    alpha = .001
    nv = len(np.where(pres['p_values'] < alpha)[0])
    latent_vars = ['LV_%d' % (v + 1) for v in range(nv)]
    pls_functions.plot_scree(eigs=pres['true_eigs'], pvals=pres['p_values'], alpha=alpha, fname=fig_dir + '/scree.png')

    behavior_df = pd.DataFrame(bres['y_zscores'][:nv, :], index=latent_vars, columns=sleep_variables)
    behavior_df.to_excel(fig_dir+'/behavior_res.xlsx')
    brain_res = organize_brain_sals(bres['x_zscores'], rois, sessions, latent_vars, comp='sign')
    pu.save_xls(brain_res, fig_dir+'/brain_res.xlsx')

    conj = brain_res['brain_conjunction']
    plot_roi_saliences(rois, conj, fig_dir, cmap=None, maxv=80, create_rois=True)

    logging.info('%s: Finished' % pu.ctime())


def pls_psqi_with_bold_alpha_pac(fig_dir, run_check=True):
    logging.info('%s: Running PLSC on PSQI components with phase-amplitude coupling' % pu.ctime())
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)

    # Extracting metadata
    h5_file = h5py.File('../data/MEG_phase_amp_coupling.hdf5')
    sessions = list(h5_file)
    meg_subj = list(h5_file[sessions[0]])
    rois = list(h5_file[sessions[0] + '/' + meg_subj[0]])
    h5_file.close()

    bold_pac_index = 0
    alpha_pac_index = 3
    meg_data = []
    for sess in sessions:
        session_df = pd.DataFrame(index=meg_subj, columns=rois)
        for roi in rois:
            h5_file = h5py.File('../data/MEG_phase_amp_coupling.hdf5')
            for subj in meg_subj:
                key = sess + '/' + subj + '/' + roi + '/r_vals'
                dset = h5_file[key][...]
                session_df.loc[subj][roi] = dset[bold_pac_index, alpha_pac_index]
            h5_file.close()
        meg_data.append(session_df)
    meg_df = pd.concat(meg_data, axis=1)

    sleep_df, sleep_variables = load_psqi_data()

    if run_check:
        p = pls_functions.PLSC(n_iters=10000)
        logging.info('%s: Running permutation tests' % pu.ctime())
        pres = p.permutation_tests(meg_df.to_numpy().astype(float), sleep_df.values)
        logging.info('%s: Running bootstrap tests' % pu.ctime())
        bres = p.bootstrap_tests(meg_df.to_numpy().astype(float), sleep_df.values)
        res = {'permutation tests': pres, 'bootstrap_tests': bres}
        with open(fig_dir + '/pls_sleep.pkl', 'wb') as file:
            pkl.dump(res, file)
    else:
        logging.info('%s: Loading raw output' % pu.ctime())
        with open(fig_dir + '/pls_sleep.pkl', 'rb') as file:
            res = pkl.load(file)
        pres = res['permutation tests']
        bres = res['bootstrap_tests']

    alpha = .001
    nv = len(np.where(pres['p_values'] < alpha)[0])
    latent_vars = ['LV_%d' % (v+1) for v in range(nv)]
    pls_functions.plot_scree(eigs=pres['true_eigs'], pvals=pres['p_values'], alpha=alpha, fname=fig_dir + '/scree.png')

    behavior_df = pd.DataFrame(bres['y_zscores'][:nv, :], index=latent_vars, columns=sleep_variables)
    behavior_df.to_excel(fig_dir+'/behavior_res.xlsx')

    brain_res = organize_brain_sals(np.abs(bres['x_zscores']), rois, sessions, latent_vars, comp='sign')
    pu.save_xls(brain_res, fig_dir+'/brain_res.xlsx')

    conj = brain_res['brain_conjunction']
    plot_roi_saliences(rois, conj, fig_dir, cmap='viridis', maxv=40, create_rois=False)

    logging.info('%s: Finished' % pu.ctime())


def pls_psqi_with_ppc_roi_version(fig_dir):
    logging.info('%s: Running PLSC on PSQI components with phase-phase coupling' % pu.ctime())
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)

    ppc_first_level = pd.read_excel('../data/attention_networks/ppc_first_level.xlsx', index_col=0)
    self_connections = [c for c in list(ppc_first_level) if all(ppc_first_level[c] == 1)]
    meg_df = ppc_first_level.drop(columns=self_connections)
    sessions = pd.unique([t.split(' ')[0] for t in list(meg_df)])
    connections = pd.unique([t.split(' ')[1] for t in list(meg_df)])

    sleep_df, sleep_variables = load_psqi_data()

    p = pls_functions.PLSC(n_iters=10000)
    pres = p.permutation_tests(meg_df.values, sleep_df.values)
    bres = p.bootstrap_tests(meg_df.values, sleep_df.values)

    alpha = .001
    nv = len(np.where(pres['p_values'] < alpha)[0])
    latent_vars = ['LV_%d' % (v+1) for v in range(nv)]
    pls_functions.plot_scree(eigs=pres['true_eigs'], pvals=pres['p_values'], alpha=alpha, fname=fig_dir + '/scree.png')

    behavior_df = pd.DataFrame(bres['y_zscores'][:nv, :], index=latent_vars, columns=sleep_variables)
    behavior_df.to_excel(fig_dir+'/behavior_res.xlsx')

    brain_res = organize_brain_sals(bres['x_zscores'], connections, sessions, latent_vars, comp='sign')
    pu.save_xls(brain_res, fig_dir+'/brain_res.xlsx')

    logging.info('%s: Finished' % pu.ctime())


if __name__ == "__main__":
    meg_subj, meg_sess = pu.proj_data.get_meg_metadata()
    rois = pu.proj_data().roiLabels
    pls_psqi_with_power(meg_sess, rois, fig_dir='../figures/PLS/psqi_components/power_recalculated', run_check=True)
    # pls_psqi_with_bold_alpha_pac(fig_dir='../figures/PLS/psqi_components/pac_bold_alpha', run_check=False)
    # pls_psqi_with_ppc_roi_version(fig_dir='../figures/PLS/psqi_components/ppc_network_rois')
