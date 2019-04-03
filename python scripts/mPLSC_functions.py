# -*- coding: utf-8 -*-
"""
Functions for mPLSC

Created on Mon Mar 25 12:43:05 2019
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy
from boredStats import pls_tools
from nilearn import surface, plotting, datasets

def permutation_pls(y_tables, x_tables, scree_path=None):
    p = pls_tools.MultitablePLSC(n_iters=1000, return_perm=False)
    res_perm = p.mult_plsc_eigenperm(y_tables, x_tables)
    
    if scree_path is not None:
        plotScree(res_perm['true_eigenvalues'],
                  res_perm['p_values'],
                  fname=scree_path + 'scree.png')
    return res_perm

def boostrap_sal(y_tables, x_tables, z=2):
    p = pls_tools.MultitablePLSC(n_iters=1000, return_perm=False)
    res_boot = p.mult_plsc_bootstrap_saliences(y_tables, x_tables, z)
    return res_boot

def load_behavior_subtables(behavior_raw, variable_metadata):
    #Support function - load y tables
    names = list(variable_metadata['name'].values)
    
    overlap = [b for b in names if b in list(behavior_raw)]
    to_drop = [b for b, n in enumerate(names) if n not in list(behavior_raw)]
    
    
    variable_metadata.drop(to_drop, inplace=True)
    categories = list(variable_metadata['category'].values)
    
    behavior_data = behavior_raw.loc[:, overlap]
    
    subtable_data = {}
    for c in list(pd.unique(categories)):
        blist = [beh for b, beh in enumerate(overlap) if categories[b] == c]
        subtable_data[c] = behavior_data.loc[:, blist]
        
    return subtable_data

def create_salience_subtables(sals, dataframes, subtable_names, latent_names):
    salience_subtables = {}
    start = 0
    for t, table in enumerate(dataframes):
        if isinstance(table, pd.DataFrame):
            num_variables_in_table = table.values.shape[1]
        else:
            num_variables_in_table = table.shape[1]
        end = start + num_variables_in_table
        
        saliences = sals[start:end, :]
        df = pd.DataFrame(saliences, index=list(table), columns=latent_names)
        salience_subtables[subtable_names[t]] = df
        start = end
        
    return salience_subtables

def conjunction_analysis(brains, compare='any', return_avg=False):
    """
    Run a sort of conjunction analysis between brain tables to
    find common loadings between subtables for a given latent variable
    
    Parameters
    ----------
    brain_tables : pandas Dataframe
        An N x M DataFrame where 
            N corresponds to the number of ROIs
            M corresponds to the subtable which the ROIs belong to
        The rows of brain_tables will be compared.
        
    compare : str, default is 'any'
        How comparisons should be made. 
        'any' if loadings must exist
        'sign' if loadings must have the same sign
        The 'any' parameter is fine if squared loadings are being used
    
    return_avg : bool , default is False
        If the average loading across subtables should be returned
        If true, returns a pandas df of length N with average loadings
        Otherwise, returns a pandas df of length N with bools
    """
    
    cpr, avg = [], []
    for r, row in enumerate(brains.index):
        vals = brains.values[r, :]
        if compare == 'any':
            if all(vals):
                cpr.append(True)
                avg.append(np.mean(vals))
            else:
                cpr.append(False)
                avg.append(0)
        elif compare == 'sign':
            signs = np.sign(vals)
            if all(signs > 0) or all(signs < 0):
                cpr.append(True)
                avg.append(np.mean(vals))
            else:
                cpr.append(False)
                avg.append(0)
    
    if not return_avg:
        return pd.DataFrame(cpr, index=brains.index)
    else:
        return pd.DataFrame(avg, index=brains.index)

def average_behavior_scores(y_saliences, latent_variable_names):
    """Calculate the average salience of a behavioral category"""
    
    keys = list(y_saliences)
    avg_sals = np.ndarray(shape=(len(keys), len(latent_variable_names)))
    for l, latent_name in enumerate(latent_variable_names):
        for k, key in enumerate(keys):
            behavior_saliences = y_saliences[key]
            avg_sals[k, l] = np.mean(behavior_saliences[latent_name])
        
    return pd.DataFrame(avg_sals, index=keys, columns=latent_variable_names)

def plotScree(eigs, pvals=None, alpha=.05, percent=True, kaiser=False, fname=None):
    """
    Create a scree plot for factor analysis using matplotlib
    
    Parameters
    ----------
    eigs : numpy array
        A vector of eigenvalues
        
    Optional
    --------
    pvals : numpy array
        A vector of p-values corresponding to a permutation test
    
    alpha : float
        Significance level to threshold eigenvalues (Default = .05)
        
    percent : bool
        Plot percentage of variance explained
    
    kaiser : bool
        Plot the Kaiser criterion on the scree
        
    fname : filepath
        filepath for saving the image
    Returns
    -------
    fig, ax1, ax2 : matplotlib figure handles
    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    percentVar = (np.multiply(100, eigs)) / np.sum(eigs)
    cumulativeVar = np.zeros(shape=[len(percentVar)])
    c = 0
    for i,p in enumerate(percentVar):
        c = c+p
        cumulativeVar[i] = c
    
    fig,ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Scree plot", fontsize='xx-large')
    ax.plot(np.arange(1,len(percentVar)+1), eigs, '-k')
    ax.set_ylim([0,(max(eigs)*1.2)])
    ax.set_ylabel('Eigenvalues', fontsize='xx-large')
    ax.set_xlabel('Factors', fontsize='xx-large')
#    ax.set_xticklabels(fontsize='xx-large') #TO-DO: make tick labels bigger
    if percent:
        ax2 = ax.twinx()
        ax2.plot(np.arange(1,len(percentVar)+1), percentVar,'ok')
        ax2.set_ylim(0,max(percentVar)*1.2)
        ax2.set_ylabel('Percentage of variance explained', fontsize='xx-large')

    if pvals is not None and len(pvals) == len(eigs):
        #TO-DO: add p<.05 legend?
        p_check = [i for i,t in enumerate(pvals) if t < alpha]
        eigenCheck = [e for i,e in enumerate(eigs) for j in p_check if i==j]
        ax.plot(np.add(p_check,1), eigenCheck, 'ob', markersize=10)
    
    if kaiser:
        ax.axhline(1, color='k', linestyle=':', linewidth=2)
    
    if fname:
        fig.savefig(fname, bbox_inches='tight')
    return fig, ax, ax2

def plot_radar2(saliences_series, choose=True, separate_neg=True, fname=None):
    """Plot behavior saliences in a radar plot
    """
    def choose_saliences(series, num_to_plot=10):
        series_to_sort = np.abs(series)
        series_sorted = series_to_sort.sort_values(ascending=False)
        return series[series_sorted.index[:num_to_plot]]
    
    if choose:
        sals = choose_saliences(saliences_series)
    else:
        sals = saliences_series
    values = list(sals.values)
    values.append(values[0])
    N = len(sals.index)
    
    theta = [n / float(N) * 2 * np.pi for n in range(N)]
    theta.append(theta[0])
    
    ax = plt.subplot(111, projection='polar')
    ax.set_rmax(2)
    ax.set_rticks([])
    ticks = np.linspace(0, 360, N+1)[:-1] 
    
    pos_values = np.asarray(deepcopy(values))
    pos_values[pos_values < 0] = 0
    ax.plot(theta, pos_values, 'r')
    ax.fill(theta, pos_values, 'r', alpha=0.1)
    
    if separate_neg:
        neg_values = np.asarray(deepcopy(values))
        neg_values[neg_values > 0] = 0
        neg_values = np.abs(neg_values)
    
        ax.plot(theta, neg_values, 'b')
        ax.fill(theta, neg_values, 'b', alpha=0.1)
    
    max_val = .3
    ax.set_ylim(-.01, max_val)
    ax.set_xticks(np.deg2rad(ticks))
    ticklabels = list(sals.index)
    ax.set_xticklabels(ticklabels, fontsize=10)
    ax.set_yticks(np.arange(.05, max_val, .05))
    
    plt.gcf().canvas.draw()
    angles = np.linspace(0,2*np.pi,len(ax.get_xticklabels())+1)
    angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
    angles = np.rad2deg(angles)
    labels = []
    for label, angle in zip(ax.get_xticklabels(), angles):
        x,y = label.get_position()
        lab = ax.text(x,y-.3, label.get_text(), transform=label.get_transform(),
                      ha=label.get_ha(), va=label.get_va())
        lab.set_rotation(angle)
        labels.append(lab)
    ax.set_xticklabels([])
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=600)
    plt.show()

def create_custom_roi(roi_path, rois_to_combine, roi_magnitudes):
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
        
    roi_magnidues : list or numpy array
        A list or vector of magnitudes
        Can be integers (indices) or floats (e.g. stat values)
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
    return nifti

def plot_brain_saliences(custom_roi, minval, figpath=None):
    fsaverage = datasets.fetch_surf_fsaverage()
    orders = [('medial', 'left'), ('medial', 'right'),
             ('lateral', 'left'), ('lateral', 'right')]
    
    fig, ax = plt.subplots(nrows=2,
                           ncols=2,
                           figsize=(8.0, 6.0),
                           dpi=300,
                           frameon=False,
                           sharex=True,
                           sharey=True,
                           subplot_kw={'projection':'3d'},)
    fig.subplots_adjust(hspace=0., wspace=0.00005)
    axes_list = fig.axes
    for index, order in enumerate(orders):
        view = order[0]
        hemi = order[1]
        
        texture = surface.vol_to_surf(custom_roi, fsaverage['pial_%s' % hemi])
        plotting.plot_surf_stat_map(
                fsaverage['infl_%s' % hemi],
                texture,
                cmap='seismic',
                hemi=hemi,
                view=view,
                bg_on_data=True,
                axes=axes_list[index],
                bg_map=fsaverage['sulc_%s' % hemi],
                threshold=minval,
                output_file=figpath,
                figure=fig,
                colorbar=False)
    return fig

def _avg_behavior_saliences_squared(y_salience_dict, num_latent_vars):
    #Function for squaring and averaging behavior saliences, legacy function
    keys = list(y_salience_dict)
    avg_squared_saliences = np.ndarray(shape=(len(keys), num_latent_vars))
    for k, key in enumerate(keys):
        input_df = y_salience_dict[key]
        
        for l, latent_var in enumerate(list(input_df)):
            saliences = input_df[latent_var].values
            sq_salience = np.square(saliences)
            avg_squared_saliences[k, l] = np.mean(sq_salience)
            
    return pd.DataFrame(
            avg_squared_saliences,
            index=keys,
            columns=['LatentVar%d' % (n+1) for n in range(num_latent_vars)])