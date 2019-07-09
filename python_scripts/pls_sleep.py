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


def create_custom_roi(roi_path, rois_to_combine, roi_magnitudes, fname=None):
    """
    Create a custom ROI

    This function takes ROIs from a directory of ROIs, extracts its 3d data,
    and replaces nonzero indices with a given magnitude. It does this for
    each ROI in rois_to_combine, then merges all of the 3d data into a single
    3d image, then transforms it back into a Nifti-compatible object.

    Parameters
    ----------
    roi_path : str
        Path to ROI nifti images (must work with nibabel)

    rois_to_combine : list
        A list of ROIs to combine into one 3d image
        ROIs in this list must exist in the roi_path

    roi_magnitudes : list or numpy array
        A list or vector of magnitudes
        Can be integers (indices) or floats (e.g. stat values)

    fname : path (optional)
    """
    def stack_3d_dynamic(template, roi_indices, mag):
        t_copy = deepcopy(template)
        for num_counter in range(len(roi_indices[0])):
            x = roi_indices[0][num_counter]
            y = roi_indices[1][num_counter]
            z = roi_indices[2][num_counter]
            t_copy[x, y, z] = mag
        return t_copy

    print('Creating custom roi')
    rn = '%s.nii.gz' % rois_to_combine[0]
    t_vol = nib.load(os.path.join(roi_path, rn))
    template = t_vol.get_data()
    template[template > 0] = 0
    for r, roi in enumerate(rois_to_combine):
        print('Stacking %s, %d out of %d' % (roi, r+1, len(rois_to_combine)))
        if roi_magnitudes[r] == 0:
            pass
        rn = '%s.nii.gz' % roi
        volume_data = nib.load(os.path.join(roi_path, rn)).get_data()
        roi_indices = np.where(volume_data > 0)
        template = stack_3d_dynamic(template, roi_indices, roi_magnitudes[r])

    nifti = nib.Nifti1Image(template, t_vol.affine, t_vol.header)
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


def organize_brain_sals(x_zscores, rois, sessions, latent_vars, comp='any'):
    # Utility function for PLS analyses

    def _conjunction_analysis(brain_data, compare='any', thresh=0, return_avg=False):
        conjunction = [0 for row in range(brain_data.shape[0])]
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
            key = '%s_brain_zscores' % sess
            sess_df = res_dict[key]
            brains[:, s] = sess_df[lv].values
        conj = _conjunction_analysis(brains, compare=comp, thresh=4, return_avg=True)
        conj_df[lv] = conj

    res_dict['brain_conjunction'] = conj_df
    return res_dict


def pls_psqi_with_power():
    logging.info('%s: Running PLSC on PSQI components with power' % pu.ctime())
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

    p = pls_functions.PLSC(n_iters=10000)
    logging.info('%s: Running permutation tests' % pu.ctime())
    pres = p.permutation_tests(x, y)
    logging.info('%s: Running bootstrap tests' % pu.ctime())
    bres = p.bootstrap_tests(x, y)
    with open(fig_dir + '/pls_sleep.pkl', 'wb') as file:
        pkl.dump([pres, bres], file)

    logging.info('%s: Loading raw output' % pu.ctime())
    with open(fig_dir + '/pls_sleep.pkl', 'rb') as file:
        pres = pkl.load(file)[0]
        bres = pkl.load(file)[1]

    eigs = pres['true_eigs']
    pvals = pres['p_values']
    alpha = .001
    nv = len(np.where(pvals < alpha)[0])
    latent_vars = ['LV_%d' % (v+1) for v in range(nv)]

    pls_functions.plot_scree(eigs=eigs, pvals=pvals, alpha=alpha, fname=fig_dir + '/scree.png')

    behavior_df = pd.DataFrame(bres['y_zscores'][:nv, :], index=latent_vars, columns=sleep_variables)

    behavior_df.to_excel(fig_dir+'/behavior_res.xlsx')
    brain_res = organize_brain_sals(bres['x_zscores'], rois, sessions, latent_vars, comp='sign')
    pu.save_xls(brain_res, fig_dir+'/brain_res.xlsx')

    logging.info('%s: Finished' % pu.ctime())


def pls_psqi_with_pac():
    logging.info('%s: Running PLSC on PSQI components with PAC' % pu.ctime())
    fig_dir = '../figures/PLS/psqi_components/bold_alpha_PAC_attn_rois'
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)

    bold_pac_index = 0
    alpha_pac_index = 3

    h5_file = h5py.File('../data/MEG_phase_amp_coupling.hdf5')
    sessions = list(h5_file)
    sess_level = h5_file[sessions[0]]
    meg_subj = list(sess_level)
    subj_level = sess_level[meg_subj[0]]
    rois = list(subj_level)
    h5_file.close()

    meg_data = {}
    for sess in sessions:
        session_df = pd.DataFrame(index=meg_subj, columns=rois)
        h5_file = h5py.File('../data/MEG_phase_amp_coupling.hdf5')
        for roi in rois:
            for subj in meg_subj:
                key = sess + '/' + subj + '/' + roi + '/r_vals'
                dset = h5_file[key][...]
                session_df.loc[subj][roi] = dset[bold_pac_index, alpha_pac_index]
        h5_file.close()
        meg_data[sess] = session_df
    meg_list = [meg_data[sess] for sess in list(meg_data)]
    meg_df = pd.concat(meg_list, axis=1)
    x = meg_df.to_numpy()
    print(x[0, 0].dtype)
    x = x.astype(float)

    sleep_variables = ['PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7']
    behavior_raw = pd.read_excel('../data/hcp_behavioral.xlsx', index_col=0, sheet_name='cleaned')
    sleep_df = behavior_raw[sleep_variables]
    y = sleep_df.values

    # p = pls_functions.PLSC(n_iters=10000)
    # logging.info('%s: Running permutation tests' % pu.ctime())
    # pres = p.permutation_tests(x, y)
    # logging.info('%s: Running bootstrap tests' % pu.ctime())
    # bres = p.bootstrap_tests(x, y)
    # res = {'permutation tests': pres, 'bootstrap_tests': bres}
    # with open(fig_dir + '/pls_sleep.pkl', 'wb') as file:
    #     pkl.dump(res, file)

    logging.info('%s: Loading raw output' % pu.ctime())
    with open(fig_dir + '/pls_sleep.pkl', 'rb') as file:
        res = pkl.load(file)
    pres = res['permutation tests']
    bres = res['bootstrap_tests']

    eigs = pres['true_eigs']
    pvals = pres['p_values']
    alpha = .001
    nv = len(np.where(pvals < alpha)[0])
    latent_vars = ['LV_%d' % (v + 1) for v in range(nv)]

    pls_functions.plot_scree(eigs=eigs, pvals=pvals, alpha=alpha, fname=fig_dir + '/scree.png')

    behavior_df = pd.DataFrame(bres['y_zscores'][:nv, :], index=latent_vars, columns=sleep_variables)

    behavior_df.to_excel(fig_dir+'/behavior_res.xlsx')
    brain_res = organize_brain_sals(np.abs(bres['x_zscores']), rois, sessions, latent_vars, comp='any')
    pu.save_xls(brain_res, fig_dir+'/brain_res.xlsx')

    roi_path = '../data/glasser_atlas/'
    conj_res = brain_res['brain_conjunction']
    for lv in list(conj_res):
        mags = conj_res[lv]
        # mags[mags < 40] = 0
        custom_roi = create_custom_roi(roi_path, rois, mags)
        nib.save(custom_roi, fig_dir + '/%s.nii.gz' % lv)
        custom_roi = nib.load(fig_dir + '/%s.nii.gz' % lv)
        fname = fig_dir + '/brain_%s' % lv
        plot_brain_saliences(custom_roi, minval=4, maxval=40, figpath=fname, cmap='viridis', cbar=False)

    logging.info('%s: Finished' % pu.ctime())


# pls_psqi_with_power()
pls_psqi_with_pac()

