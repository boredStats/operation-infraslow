# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:39:26 2019
"""

import os
import h5py
import numpy as np
import pandas as pd
import pickle as pkl
import mPLSC_functions as mf
from boredStats import pls_tools

import sys
sys.path.append("..")
import proj_utils as pu

print('%s: Loading data' % pu.ctime())
pdir = pu._get_proj_dir()
pdObj = pu.proj_data()
rois = pdObj.roiLabels
colors = pdObj.colors
meg_subj, meg_sessions = pdObj.get_meg_metadata()
mri_subj, mri_sess = pdObj.get_mri_metadata()
subjects = [s for s in mri_subj if s in meg_subj]

bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
meg_sess = ['Session1', 'Session2', 'Session3']

ddir = pdir + '/data'
roi_path = ddir + '/glasser_atlas/'
fig_path = pdir + '/figures/mPLSC_cfc/'

# Creating list of x tables
def _load_cfc_subtables(filelist):
    #Support function - load x tables

    # flist = []
    # for f in os.listdir(fpath):
    #     if f.endswith('.xlsx'):
    #         flist.append(os.path.join(fpath, f))

    sessions = ['session1', 'session2', 'session3'] #lowercase notation???
    subtable_data = {}
    for f, file in enumerate(filelist):
        table = pd.read_excel(file, sheet_name=None, index_col=0)
        subtable_data[sessions[f]] = dict(table)

    return subtable_data

def _load_cfc_tables_from_hdf5(hdf5_file, sessions, subjects, rois, bands):
    session_dict = {}
    for sess in sessions:
        band_dict = {}
        for b, band in enumerate(bands):
            print('Loading data from %s - %s' %  (sess, band))
            df = pd.DataFrame(index=subjects, columns=rois)
            for subj in subjects:
                for roi in rois:
                    data_file = h5py.File(hdf5_file, 'r')
                    group_path = sess + '/' + subj + '/' + roi + '/r_vals'
                    data = data_file.get(group_path).value
                    band_val = data[0][b]
                    df.loc[subj, roi] = band_val
            band_dict[band] = df
        session_dict[sess] = band_dict

    return session_dict

outfile = ddir + '/mPLSC_cfc.pkl'
check = input('Run mPLSC? y/n ')
if check == 'y':
    check = input('Load cfc data from hdf5? y/n ')
    if check == 'y':
        hdf5_file = ddir + '/MEG_BOLD_phase_amp_coupling.hdf5'
        cfc_tables = _load_cfc_tables_from_hdf5(hdf5_file, meg_sess, subjects, rois, bands)
        for session in list(cfc_tables):
            session_dict = cfc_tables[session]
            fname = ddir + '/MEG_BOLD_phase_amp_coupling_%s.xlsx' % session
            mf.save_xls(session_dict, fname)

    cfc_fnames = ['MEG_BOLD_phase_amp_coupling_%s.xlsx' % sess for sess in meg_sess]
    cfc_filelist = [os.path.join(ddir, f) for f in cfc_fnames]
    cfc_tables = _load_cfc_subtables(cfc_filelist)
    x_tables = []
    for sess in list(cfc_tables):
        for t in list(cfc_tables[sess]):
            x_tables.append(cfc_tables[sess][t])

    # Creating list of y tables
    behavior_metadata = pd.read_csv(pdir + '/data/b_variables_mPLSC.txt',
                                       delimiter='\t', header=None)

    behavior_metadata.rename(dict(zip([0, 1], ['category','name'])),
                                axis='columns', inplace=True)

    behavior_raw = pd.read_excel(pdir + '/data/hcp_behavioral.xlsx',
                                  index_col=0, sheet_name='cleaned')

    behavior_tables = mf.load_behavior_subtables(behavior_raw, behavior_metadata)
    y_tables = [behavior_tables[t] for t in list(behavior_tables)]

    p = pls_tools.MultitablePLSC(n_iters=10000)
    print('%s: Running permutation testing on latent variables' % pu.ctime())
    res_perm = p.mult_plsc_eigenperm(y_tables, x_tables)
    print('%s: Running bootstrap testing on saliences' % pu.ctime())
    res_boot = p.mult_plsc_bootstrap_saliences(y_tables, x_tables, 0)

    num_latent_vars = len(np.where(res_perm['p_values'] < .001)[0])
    latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]

    print('%s: Organizing saliences' % pu.ctime())
    # y_saliences = mf.create_salience_subtables(
    #         sals=res_boot['y_saliences'][:, :num_latent_vars],
    #         dataframes=y_tables,
    #         subtable_names=list(behavior_tables),
    #         latent_names=latent_names)

    y_saliences, y_saliences_z = mf.organize_behavior_saliences(
        res_boot,
        y_tables,
        list(behavior_tables),
        num_latent_vars
        )

    x_table_names = []
    for sess in list(cfc_tables):
        session_dict = cfc_tables[sess]
        full_table_names = ["%s %s" % (sess, cfc) for cfc in list(session_dict)]
        x_table_names = x_table_names + full_table_names

    # x_saliences = mf.create_salience_subtables(
    #         sals=res_boot['x_saliences'][:, :num_latent_vars],
    #         dataframes=x_tables,
    #         subtable_names=x_table_names,
    #         latent_names=latent_names)

    x_saliences, x_saliences_z = mf.organize_brain_saliences(
        res_boot,
        x_tables,
        x_table_names,
        num_latent_vars
        )

    output = {'permutation_tests':res_perm,
              'bootstrap_tests':res_boot,
              'y_saliences':y_saliences,
              'x_saliences':x_saliences,
              'y_saliences_zscores':y_saliences_z,
              'x_saliences_zscores':x_saliences_z
              }

    with open(outfile, 'wb') as file:
        pkl.dump(output, file)
else:
    with open(outfile, 'rb') as file:
        output = pkl.load(file)

res_perm = output['permutation_tests']
alpha = .001
# print('%s: Plotting scree' % pu.ctime())
# mf.plotScree(res_perm['true_eigenvalues'],
#              res_perm['p_values'],
#              alpha=alpha,
#              fname=fig_path + '/scree.png')
mf.save_scree_data(res_perm, fig_path+'/scree_data.xlsx')

num_latent_vars = 4#len(np.where(res_perm['p_values'] < alpha)[0])
latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]

# mf.save_xls(output['y_saliences'], fig_path + '/behavior_saliences.xlsx')
y_saliences_zscores = output['y_saliences_zscores']
y_saliences_zscores_thresh = {}
for behavior_category in list(y_saliences_zscores):
    unthresh_df = y_saliences_zscores[behavior_category]
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
mf.save_xls(y_saliences_zscores_thresh, fig_path + '/behavior_saliences_z.xlsx')

print('%s: Averaging saliences within behavior categories' % pu.ctime())
behavior_avg = mf.average_subtable_saliences(y_saliences_zscores_thresh)
behavior_avg.to_excel(fig_path+'/behavior_average_z.xlsx')

meg_sessions = ['session1', 'session2', 'session3']
x_saliences_z = output['x_saliences_zscores']

def _signed_average_brain_sal(res_conj, sals_per_session):
    #From a signed conjunction, make values nonsigned then calc average
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

print('%s: Running conjunction on brain data' % pu.ctime())
conjunctions_sign_matters, conjunctions_no_sign = {}, {}
for band in bands:
    sals_per_sess = {}
    for session in meg_sessions:
        for brain_table in list(x_saliences_z):
            if session in brain_table and band in brain_table:
                sals_per_sess[session] = x_saliences_z[brain_table]

    mf.save_xls(sals_per_sess, fig_path+'/brain_saliences_z_%s.xlsx' % band)

    # print('%s: Running signed conjunction on %s' % (pu.ctime(), band))
    # res_conj = mf.single_table_conjunction(
    #     sals_per_sess,
    #     comp='sign',
    #     thresh=4,
    #     return_avg=False)
    #
    # res_conj_sign_corrected = _signed_average_brain_sal(res_conj, sals_per_sess)
    # res_conj_sign_corrected.to_excel(fig_path+'/conjunction_sign_matters_%s.xlsx' % band)
    # conjunctions_sign_matters[band] = res_conj_sign_corrected

    print('%s: Running unsigned conjunction on %s' % (pu.ctime(), band))
    res_conj = mf.single_table_conjunction(
        sals_per_sess,
        comp='any',
        thresh=4)
    res_conj.to_excel(fig_path+'/conjunction_no_sign_%s.xlsx' % band)
    conjunctions_no_sign[band] = res_conj

print('%s: Plotting behavior bar plots' % pu.ctime())
max_z = 0
for latent_variable in latent_names:
    current_max = np.max(behavior_avg[latent_variable])
    if current_max > max_z:
        max_z = current_max

max_z = max_z + 1
for latent_variable in latent_names:
    series = behavior_avg[latent_variable]
    mf.plot_bar(series, max_z, colors, fig_path+'/behavior_z_%s.svg' % latent_variable)

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

def _stacked_bar_with_tables(z_scores_series, fname=None):
    original_z = np.abs(series_to_plot.values)
    # columns = list(series_to_plot)
    rows = series_to_plot.index
    n_rows = len(rows)
    n_cols = len(list(series_to_plot))
    columns = ['Latent Variable %d' % (x+1) for x in range(n_cols)]

    # Transforming data to plot contribution to the mean
    data = series_to_plot.values
    data = np.abs(data)

    col_means = np.mean(data, axis=0)
    mean_array = np.tile(col_means, (n_rows, 1))
    col_sums = np.sum(data, axis=0)
    sum_array = np.tile(col_sums, (n_rows, 1))

    data = data / sum_array
    data = data * mean_array

    # Get some pastel shades for the colors
    colors = plt.cm.Reds(np.linspace(0, 0.5, len(rows)))

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    fig, ax = plt.subplots(figsize=(12, 8))
    cell_text = []
    for row in range(n_rows):
        data_to_plot = data[row] #/ np.sum(data[row])
        plt.bar(index, data_to_plot, bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data_to_plot
        # cell_text.append(['%1.1f' % (x / np.sum(data[row])) for x in y_offset])
        cell_text.append(['%1.1f' % x for x in original_z[row]])

    the_table = plt.table(
        cellText=cell_text,
        rowLabels=rows,
        rowColours=colors,
        colLabels=columns,
        loc='bottom'
        )
    # plt.subplots_adjust(left=0.2, bottom=0.3)
    ax.set_ylabel('Contribution to average z-score')
    ax.set_xticks([])
    ax.set_ylim(0, max_z)
    # plt.show()
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', pad_inches=.5)
    fig.clf()

print('%s: Creating stacked bar plots with tables' % pu.ctime())
for behavior_category in list(y_saliences_zscores_thresh):
    behavior_df = y_saliences_zscores_thresh[behavior_category]
    series_to_plot = behavior_df[latent_names]
    fname = fig_path+'/behavior_stacked_bar_%s.png' % behavior_category
    _stacked_bar_with_tables(series_to_plot, fname)

def _bar_table_combo(z_scores_series, color, fname=None):
    original_z = np.abs(series_to_plot.to_numpy())
    # columns = list(series_to_plot)
    rows = series_to_plot.index
    n_rows = len(rows)
    n_cols = len(list(series_to_plot))
    columns = ['Latent Variable %d' % (x+1) for x in range(n_cols)]

    # Transforming data to plot contribution to the mean
    data = series_to_plot.values
    data = np.abs(data)

    # Get some pastel shades for the colors
    colors = plt.cm.Reds(np.linspace(0, 0.5, len(rows)))

    index = np.arange(n_rows)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.barh(index, original_z, align='center', height=1, color=color)
    ax.set_yticks(index)
    ax.set_yticklabels(z_scores_series.index)
    ax.set_xlim(0, 30)
    # plt.show()
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight')
    fig.clf()

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
    mf.bar_all_behaviors(y_saliences_zscores_thresh, name, mean, colors, 40, fname)

# for name in latent_names:
#     color_list = []
#     rowname_extract = []
#     series_extract = []
#
#     for b, behavior_category in enumerate(list(y_saliences_zscores_thresh)):
#         # color = list(np.divide(colors[b], 255))
#         # color.append(1)
#         # print(color)
#         # for name in latent_names:
#         #     behavior_df = y_saliences_zscores_thresh[behavior_category]
#         #     series_to_plot = behavior_df[name]
#         #     fname = fig_path+'/behavior_bar_table_%s.png' % behavior_category
#         #     _bar_table_combo(series_to_plot, color, fname)
#         #     break
#
#         color = list(np.divide(colors[b], 255))
#         color.append(1)
#         color = tuple(color)
#
#         behavior_df = y_saliences_zscores_thresh[behavior_category]
#         series_to_plot = behavior_df[name]
#
#         color_list.append([color] * len(series_to_plot.index))
#         rowname_extract.append(series_to_plot.index)
#         series_extract.append(np.abs(series_to_plot.values))
#
#     series_combined = _dumb_extraction(series_extract)
#     rownames = _dumb_extraction(rowname_extract)
#     plot_colors = [c for cols in color_list for c in cols]
#     for c in plot_colors:
#         print(c)
#
#     # Reverse order of bars, row names, colors for plot
#     series_combined.reverse()
#     rownames.reverse()
#     color_list.reverse()
#
#     fname = fig_path+'/behavior_fullbar_%s.svg' % name
#
#     index = np.arange(len(rownames))
#     fig, ax = plt.subplots(figsize=(12, 30))
#     ax.barh(index, series_combined, align='center', color=plot_colors, height=1)
#     ax.set_yticks(index)
#     ax.set_yticklabels([])
#     ax.set_xlim(0, 25)
#     # plt.show()
#     fig.savefig(fname, bbox_inches='tight')
#     fig.clf()

print('%s: Extracting brain z-score metadata' % pu.ctime())
def print_zmeta(brain_conjunction, bands, latent_names, fname):
    mins, maxs, mus, stds = {}, {}, {}, {}
    for band in bands:
        conjunction_data = brain_conjunction[band]
        band_values = []
        for name in latent_names:
            mags = conjunction_data[name].values
            # min_z = np.min(mags[np.nonzero(mags)])
            # print(min_z)
            band_values.append(mags)
        # min_array = np.ma.masked_equal(band_values, 0.0, copy=False)
        # mins[band] = np.min(min_array)
        maxs[band] = np.max(band_values)
        mus[band] = np.mean(band_values)
        stds[band] = np.std(band_values, ddof=1)

    if fname is not None:
        with open(fname, 'w') as file:
            # for band in bands:
                # file.write('Min z-score for %s is %.4f\n' % (band, mins[band]))
            for band in bands:
                file.write('Max z-score for %s is %.4f\n' % (band, maxs[band]))
            for band in bands:
                file.write('Mean z-score for %s is %.4f\n' % (band, mus[band]))
            for band in bands:
                file.write('Std-dev z-score for %s is %.4f\n' % (band, stds[band]))

print_zmeta(conjunctions_no_sign, bands, latent_names, fig_path+'/brain_zmeta.txt')

check = input('Make brain figures? y/n ')
if check == 'y':
    print('%s: Creating brain figures' % pu.ctime())
    for band in bands:
        brain_conjunction = conjunctions_no_sign[band]#conjunctions_sign_matters[band]
        for name in latent_names:
            mags = brain_conjunction[name]
            fname = fig_path + '/brain_%s.svg' % name
            custom_roi = mf.create_custom_roi(roi_path, rois, mags)
            mf.plot_brain_saliences(custom_roi, minval=4, maxval=20, figpath=fname, cbar=False, cmap='PiYG_r')
print('%s: Finished' % pu.ctime())
