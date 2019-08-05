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


def plot_brains(custom_roi, minval=0, maxval=None, figpath=None, cbar=False, cmap=None):
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
        plotting.plot_surf_roi(surf_mesh=fsaverage['infl_%s' % hemi], roi_map=texture,
                               bg_map=fsaverage['sulc_%s' % hemi], cmap=cmap,
                               hemi=hemi, view=view, bg_on_data=True,
                               axes=axes_list[index], vmin=-maxval, vmax=maxval,
                               output_file=figpath, symmetric_cbar=True, figure=fig,
                               darkness=.5, colorbar=cbar, threshold=4)
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


def plot_roi_saliences(rois, conj_res, fig_dir, cmap=None, maxv=40, create_rois=False):
    roi_path = '../data/glasser_atlas/'
    for lv in list(conj_res):
        if create_rois:
            mags = conj_res[lv]
            custom_roi = create_custom_roi(roi_path, rois, mags)
            nib.save(custom_roi, fig_dir + '/%s.nii.gz' % lv)
        else:
            custom_roi = nib.load(fig_dir + '/%s.nii.gz' % lv)
        fname = fig_dir + '/brain_%s' % lv
        plot_brains(custom_roi, minval=4, maxval=maxv, figpath=fname, cmap=cmap, cbar=True)


def mirror_strfind(strings):
    # Given all possible combinations of strings in a list, find the mirrored strings
    checkdict = {}  # Creating dict for string tests
    for string_1 in strings:
        for string_2 in strings:
            checkdict['%s-%s' % (string_1, string_2)] = False

    clean, mirror = [], []
    for string_1 in strings:
        for string_2 in strings:
            test = '%s-%s' % (string_1, string_2)
            mir = '%s-%s' % (string_2, string_1)
            if string_1 == string_2:
                checkdict[test] = True
                mirror.append(test)
                continue

            if not checkdict[test] and not checkdict[mir]:
                checkdict[test] = True
                checkdict[mir] = True
                clean.append(test)
            else:
                mirror.append(test)

    return clean, mirror


def run_pls(x, y, output_dir, n_iters=10000, scaling='ss1'):
    p = pls_functions.PLSC(n_iters=n_iters, center_scale=scaling)
    logging.info('%s: Running permutation tests' % pu.ctime())
    pres = p.permutation_tests(x, y)
    logging.info('%s: Running bootstrap tests' % pu.ctime())
    bres = p.bootstrap_tests(x, y)

    res = {'permutation tests': pres,
           'bootstrap_tests': bres}
    with open(output_dir + '/pls_sleep.pkl', 'wb') as file:
        pkl.dump(res, file)

    return pres, bres


def pls_psqi_with_power(sessions, rois, fig_dir, run_check=False):
    logging.info('%s: Running PLSC on PSQI components with power' % pu.ctime())
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)

    meg_df = pd.read_excel('../data/MEG_infraslow_power.xlsx', sheet_name='location', index_col=0)
    sleep_df, sleep_variables = load_psqi_data()

    if run_check:
        pres, bres = run_pls(x=meg_df.values, y=sleep_df.values, output_dir=fig_dir)
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

    # conj = brain_res['brain_conjunction']
    # plot_roi_saliences(rois, conj, fig_dir, maxv=120, create_rois=True)

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
        pres, bres = run_pls(x=meg_df.values, y=sleep_df.values, output_dir=fig_dir)
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
    plot_roi_saliences(rois, conj, fig_dir, maxv=120, create_rois=False)

    logging.info('%s: Finished' % pu.ctime())


def pls_psqi_with_ppc_roi_version(fig_dir, run_check=False):
    import matplotlib.pyplot as plt
    from seaborn import heatmap
    logging.info('%s: Running PLSC on PSQI components with phase-phase coupling' % pu.ctime())
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)

    ppc_first_level = pd.read_excel('../data/attention_networks/ppc_first_level.xlsx', index_col=0)
    colnames = list(ppc_first_level)
    connections = [c.split(' ')[1] for c in colnames]
    rois = pd.unique([c.split('-')[0].replace('\n', '') for c in connections])
    same, mirror = mirror_strfind(rois)
    columns_to_drop = [c for m in mirror for c in colnames if m in c]
    meg_df = ppc_first_level.drop(columns=columns_to_drop)

    sessions = pd.unique([t.split(' ')[0] for t in list(meg_df)])
    connections = pd.unique([t.split(' ')[1] for t in list(meg_df)])

    sleep_df, sleep_variables = load_psqi_data()
    if run_check:
        pres, bres = run_pls(x=meg_df.values, y=sleep_df.values, output_dir=fig_dir)
    else:
        logging.info('%s: Loading raw output' % pu.ctime())
        with open(fig_dir + '/pls_sleep.pkl', 'rb') as file:
            res = pkl.load(file)
        pres = res['permutation tests']
        bres = res['bootstrap_tests']

    print(pres['p_values'])
    alpha = .001
    nv = 1  # len(np.where(pres['p_values'] < alpha)[0])
    latent_vars = ['LV_%d' % (v+1) for v in range(nv)]
    pls_functions.plot_scree(eigs=pres['true_eigs'], pvals=pres['p_values'], alpha=alpha, fname=fig_dir + '/scree.png')

    behavior_df = pd.DataFrame(bres['y_zscores'][:nv, :], index=latent_vars, columns=sleep_variables)
    behavior_df.to_excel(fig_dir+'/behavior_res.xlsx')

    brain_res = organize_brain_sals(bres['x_zscores'], connections, sessions, latent_vars, comp='sign')
    pu.save_xls(brain_res, fig_dir+'/brain_res.xlsx')

    conj_res = brain_res['brain_conjunction']
    heatmap_data = pd.DataFrame(np.full(shape=(len(rois), len(rois)), fill_value=np.nan), index=rois, columns=rois)
    for roi1 in rois:
        for roi2 in rois:
            idx_label = '%s-%s' % (roi1, roi2)
            if idx_label not in conj_res.index:
                continue
            else:
                val = conj_res.loc[idx_label]['LV_1']
                heatmap_data.loc[roi2][roi1] = val

    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap(data=heatmap_data, cmap='coolwarm', center=0.0, annot=True, fmt='.2f', cbar=True, square=True, ax=ax)
    fig.savefig(fig_dir+'/heatmap.svg')
    # plt.show()

    logging.info('%s: Finished' % pu.ctime())


if __name__ == "__main__":
    meg_subj, meg_sess = pu.proj_data.get_meg_metadata()
    rois = pu.proj_data().roiLabels
    # pls_psqi_with_power(meg_sess, rois, fig_dir='../figures/PLS/psqi_components/power', run_check=True)
    pls_psqi_with_bold_alpha_pac(fig_dir='../figures/PLS/psqi_components/pac_bold_alpha', run_check=False)
    # pls_psqi_with_ppc_roi_version(fig_dir='../figures/PLS/psqi_components/ppc_network_rois', run_check=True)
