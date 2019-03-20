# -*- coding: utf-8 -*-
"""
Multitable PLS-C attempt

Created on Mon Mar 18 13:56:17 2019
"""

import os
import nilearn
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn import surface
from nilearn import datasets
from boredStats import pls_tools

import sys
sys.path.append("..")
import proj_utils as pu
nilearn.EXPAND_PATH_WILDCARDS = False

##### Load data ######
print('%s: Getting metadata, parameters' % pu.ctime())
pdir = pu._get_proj_dir()
fig_path = pdir + '/figures/mPLSC/'

data_path = pdir + '/data/mPLSC/'
cfc_path = data_path + 'phase_amplitude_excelsheets'
cfc_files = sorted([os.path.join(cfc_path, f) for f in os.listdir(cfc_path)])
cfc_tables = pd.read_excel(cfc_files[0], sheet_name=None, index_col=0) 
cfc_arrays = [cfc_tables[t].values for t in cfc_tables]

with open(pdir + '/data/cog_emotion_variables.txt', 'r') as boi:
    behavior_of_interest = [b.replace('\n', '') for b in boi]

behavior_path = pdir + '/data/hcp_behavioral.xlsx'
behavior_raw = pd.read_excel(behavior_path, index_col=0, sheet_name='cleaned')
behavior_df = behavior_raw.loc[:, behavior_of_interest]

##### Run analyses ######
def perm_eig(behavior_df, cfc_arrays):
    p = pls_tools.MultitablePLSC(n_iters=1000, return_perm=False)
    print('%s: Running eigenvalue permutation' % pu.ctime())
    res_perm = p.mult_plsc_eigenperm(behavior_df.values, cfc_arrays)
    
    pu.plotScree(res_perm['true_eigenvalues'],
                    res_perm['p_values'],
                    fname=fig_path + 'scree.png')
    return res_perm

def boostrap_sal(behavior_df, cfc_arrays):
    p = pls_tools.MultitablePLSC(n_iters=1000, return_perm=False)
    print('%s: Running boostrap tests for saliences' % pu.ctime())
    return p.mult_plsc_bootstrap_saliences(behavior_df.values, cfc_arrays, 3)

outfile = data_path + 'mPLSC_MEG_session1.pkl'
#res_perm = perm_eig(behavior_df, cfc_arrays) 
#res_boot = boostrap_sal(behavior_df, cfc_arrays)
#with open(outfile, 'wb') as file:
#    pkl.dump(res_perm, file)
#    pkl.dump(res_boot, file)

with open(outfile, 'rb') as file:
    res_perm = pkl.load(file)
    res_boot = pkl.load(file)

num_latent_vars = len(np.where(res_perm['p_values'] < .05)[0])
latent_colnames = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]
with open(data_path + 'renamed_variables_short.txt', 'r') as f:
    renamed_behavior = []
    for b in f:
        renamed_behavior.append(b)
bootstrap_y = pd.DataFrame(res_boot['y_saliences'][:, :num_latent_vars],
                           index=renamed_behavior,
                           columns=latent_colnames)
bootstrap_y.to_excel(data_path + 'mPLSC_MEG_session1_behavior_saliences.xlsx')

bootstrap_x = {}
num_vars_in_x = [cfc_tables[x].values.shape[1] for x in cfc_tables]
start_index = 0
for x_index, x_table in enumerate(cfc_tables):
    end_index = start_index + num_vars_in_x[x_index]
    df = pd.DataFrame(res_boot['x_saliences'][start_index:end_index, :6],
                      index=list(cfc_tables[x_table]),
                      columns=latent_colnames)
    bootstrap_x[x_table] = df
    start_index = start_index + num_vars_in_x[x_index]

with pd.ExcelWriter(data_path + 'mPLSC_MEG_session1_pac_saliences.xlsx') as wr:
    for table_name in bootstrap_x:
        bootstrap_x[table_name].to_excel(wr, table_name)
    wr.save()
    
###### Plot behavior results ######
def plot_radar2(saliences_series, fname=None):
    
    def choose_saliences(series, num_to_plot=10):
        series_to_sort = np.abs(series)
        series_sorted = series_to_sort.sort_values(ascending=False)
        return series[series_sorted.index[:num_to_plot]]
    
    sals = choose_saliences(saliences_series)
    values = list(sals.values)
    values.append(values[0])
    N = len(sals.index)
    
    theta = [n / float(N) * 2 * np.pi for n in range(N)]
    theta.append(theta[0])
    
    ax = plt.subplot(111, projection='polar')
    ax.set_rmax(2)
    ax.set_rticks([])
    ticks = np.linspace(0,360,N+1)[:-1] 
    ax.plot(theta, values)
    ax.set_ylim(-1, 1)
    ax.set_xticks(np.deg2rad(ticks))
    ticklabels = list(sals.index)
    ax.set_xticklabels(ticklabels, fontsize=10)
    ax.set_yticks([-1, -.5, 0, .5, 1])
    ax.fill(theta, values, 'b', alpha=0.1)
    
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
        plt.savefig(fname)
#    plt.subplots_adjust(top=0.68,bottom=0.32,left=0.05,right=0.95)
    plt.show() 

test = bootstrap_y['LatentVar1']
test_ = plot_radar2(test)

for l_var in latent_colnames:
    fname = fig_path + '/' + '%s_behavior_top10' % l_var
    series = bootstrap_y[l_var]
    plot_radar2(series, fname)
    
###### Plot brain results ######
print('%s: Plotting results' % pu.ctime())
roi_path = pdir + '/data/glasser_atlas/'

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

def plot_brain_saliences(custom_roi, minval, figname):
    fsaverage = datasets.fetch_surf_fsaverage()
    hemispheres = ['left', 'right']
    views = ['medial', 'lateral']
    for hemi in hemispheres:
    
        texture = surface.vol_to_surf(custom_roi, fsaverage['pial_%s' % hemi])
        for view in views:
            outfile = "%s_%s_%s.png" % (figname, hemi, view)
            plotting.plot_surf_stat_map(fsaverage['infl_%s' % hemi], texture,
                                    cmap='seismic',
                                    hemi=hemi,
                                    view=view,
                                    bg_on_data=True,
                                    bg_map=fsaverage['sulc_%s' % hemi],
                                    threshold=minval,
                                    output_file=outfile,
                                    colorbar=False,)
  
#for l_var in latent_colnames:
#    for k, key in enumerate(keys):
#        test_df = bootstrap_x[key]
#        sal = test_df[l_var]
#        mags = sal.values
#        minval = np.min(mags[np.nonzero(mags)])
#        custom_roi = create_custom_roi(roi_path, list(test_df.index), mags)
#        
#        figname = fig_path + '/' + '%s_%s' % (l_var, key)
#        plot_brain_saliences(custom_roi, minval, figname)