# -*- coding: utf-8 -*-
"""
Functions for mPLSC

Created on Mon Mar 25 12:43:05 2019
"""

import os
import numpy as np
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

def boostrap_sal(y_tables, x_tables, z=3):
    p = pls_tools.MultitablePLSC(n_iters=1000, return_perm=False)
    res_boot = p.mult_plsc_bootstrap_saliences(y_tables, x_tables, z)
    return res_boot

def plotScree(eigs, pvals=None, percent=True, kaiser=False, fname=None):
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
        p_check = [i for i,t in enumerate(pvals) if t<.05]
        eigenCheck = [e for i,e in enumerate(eigs) for j in p_check if i==j]
        ax.plot(np.add(p_check,1), eigenCheck, 'ob', markersize=10)
    
    if kaiser:
        ax.axhline(1, color='k', linestyle=':', linewidth=2)
    
    if fname:
        fig.savefig(fname, bbox_inches='tight')
    return fig, ax, ax2

def plot_radar2(saliences_series, fname=None):
    
    def choose_saliences(series, num_to_plot=10):
        series_to_sort = np.abs(series)
        series_sorted = series_to_sort.sort_values(ascending=False)
        return series[series_sorted.index[:num_to_plot]]
    
    sals = choose_saliences(saliences_series)
#    sals = saliences_series
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
    
    neg_values = np.asarray(deepcopy(values))
    neg_values[neg_values > 0] = 0
    neg_values = np.abs(neg_values)
    
    ax.plot(theta, pos_values, 'r')
    ax.plot(theta, neg_values, 'b')
    
    ax.set_ylim(-.02, .5)
    ax.set_xticks(np.deg2rad(ticks))
    ticklabels = list(sals.index)
    ax.set_xticklabels(ticklabels, fontsize=10)
    ax.set_yticks([0, .25, .5])
    ax.fill(theta, pos_values, 'r', alpha=0.1)
    ax.fill(theta, neg_values, 'b', alpha=0.1)
    
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
#        plt.tight_layout()
        plt.savefig(fname, bbox_inches='tight', dpi=300)
#    plt.subplots_adjust(top=0.68,bottom=0.32,left=0.05,right=0.95)
    plt.show()
    return pos_values, neg_values

def create_custom_roi(roi_path, rois_to_combine, roi_magnitudes):
    from copy import deepcopy
    
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

def plot_brain_saliences(custom_roi, minval, figpath):
    fsaverage = datasets.fetch_surf_fsaverage()
    hemispheres = ['left', 'right']
    views = ['medial', 'lateral']
    for hemi in hemispheres:
        texture = surface.vol_to_surf(custom_roi, fsaverage['pial_%s' % hemi])
        for view in views:
            outfile = "%s_%s_%s.png" % (figpath, hemi, view)
            plotting.plot_surf_stat_map(fsaverage['infl_%s' % hemi],
                                        texture,
                                        cmap='seismic',
                                        hemi=hemi,
                                        view=view,
                                        bg_on_data=True,
                                        bg_map=fsaverage['sulc_%s' % hemi],
                                        threshold=minval,
                                        output_file=outfile,
                                        colorbar=False,)