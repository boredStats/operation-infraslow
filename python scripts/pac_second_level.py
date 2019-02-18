# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:01:21 2019
"""

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, sem

import sys
sys.path.append("..")
import proj_utils as pu

print('%s: Starting' % pu.ctime())   

print('%s: Getting metadata, parameters' % pu.ctime())
pdir = pu._get_proj_dir()
pdObj = pu.proj_data()
pData = pdObj.get_data()
rois = pData['roiLabels']
slow_bands = ['BOLD', 'Slow 4', 'Slow 3', 'Slow 2', 'Slow 1'] #rows
reg_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'] #cols
meg_subj, meg_sess = pdObj.get_meg_metadata()
first_level_file = pdir + '/data/MEG_pac_first_level.hdf5'
sess = meg_sess[0] #Doing session 1

#Testing: getting data from one subject
subj = meg_subj[0]
single_subj_data = []
for roi in rois:
    cfc_file = h5py.File(first_level_file, 'r')
    roi_path = '/' + subj + '/' + sess + '/' + roi + '/'
    rval_path = roi_path + '/' + 'r_vals'
    dset = cfc_file.get(rval_path).value
    single_subj_data.append(dset)
    cfc_file.close()
single_subj_array = np.asarray(single_subj_data)
single_subj_fisherz = np.arctanh(single_subj_array)
single_subj_avg_cfc = np.mean(single_subj_fisherz, axis=0)

#Getting correlation between BOLD and high frequences from all subjects, sess1
between_subj_data = []
for subj in meg_subj:
    print('%s: %s Getting correlation data for session 1' % (pu.ctime(), subj))
    within_subj_data = []
    for roi in rois:
        cfc_file = h5py.File(first_level_file, 'r')
        roi_path = '/' + subj + '/' + sess + '/' + roi + '/'
        rval_path = roi_path + '/' + 'r_vals'
        dset = cfc_file.get(rval_path).value
        within_subj_data.append(dset[:, :])
        cfc_file.close()
    within_subj_array = np.arctanh(np.asarray(within_subj_data))
    between_subj_data.append(within_subj_array)
between_subj_array = np.asarray(between_subj_data)

##Doing t-tests
#t_mat = np.ndarray(shape=[len(rois), len(reg_bands)])
#p_mat = np.ndarray(shape=[len(rois), len(reg_bands)])
#for roi_index in range(len(rois)):
#    for fast_band_index in range(len(reg_bands)):
#        vector_to_test = between_subj_array[:, roi_index, fast_band_index]
#        t, p_2sided = ttest_1samp(vector_to_test, 0)
#        p_1sided = p_2sided/2
#        p_bonferroni = p_1sided * (360*5*5)
#        t_mat[roi_index, fast_band_index] = t
#        p_mat[roi_index, fast_band_index] = p_bonferroni

mean_roi_data = np.mean(between_subj_array, axis=0)

band_data = np.ndarray(shape=[360*5, 5])
reg_band_ref = []
roi_rep = []
for b, band in enumerate(reg_bands):
    band_to_plot = mean_roi_data[:, :, b]
    band_data[360 * b: 360* (b+1), :] = band_to_plot
    
    #creatings lists to be repeated later
    reg_band_ref = reg_band_ref + [band]*len(rois)
    roi_rep = roi_rep + rois

band_df = pd.DataFrame(band_data, index=roi_rep, columns=slow_bands)

vector_list = []
slow_band_ref = []
reg_band_full_ref = []
roi_full_rep = []
for slow_band in list(band_df):
    data_to_grab = band_df[slow_band].values
    vector_list = vector_list + list(data_to_grab)
    
    #creating more lists for final dataframe 
    slow_band_ref = slow_band_ref + [slow_band]*band_df.values.shape[0]
    reg_band_full_ref = reg_band_full_ref + reg_band_ref
    roi_full_rep = roi_full_rep + roi_rep
vector_data = np.asarray(vector_list)

final_df = pd.DataFrame(vector_data, columns=['Cross-Frequency Coupling'], index=roi_full_rep) 
final_df['Phase bands'] = slow_band_ref
final_df['Amplitude bands'] = reg_band_full_ref

sns.set_context('talk')
fig = sns.catplot(x='Phase bands', y='Cross-Frequency Coupling', 
                   height=15, aspect=1.78, data=final_df,
                   hue='Amplitude bands', kind='violin')

#---- 3D distribution figure ----# for posterity
#ax.set_xticklabels([])
#ax.tick_params(axis='x', length=0)

##import matplotlib.pyplot as plt
##plt.hist(data_array, density=True)
#
#fisherz_array = np.arctanh(data_array)
#
#pop_mean = np.repeat(0, fisherz_array.shape[0])
#
#t, p = ttest_1samp(fisherz_array, pop_mean, axis=1)
#
#from mpl_toolkits.mplot3d import axes3d
#import matplotlib.pyplot as plt
#
#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(111, projection='3d')
#
#nbins = 10
#test_data = fisherz_array[:, 0]
#hist_range = (0, np.max(fisherz_array))
#hist, bin_edges = np.histogram(test_data, bins=nbins, range=hist_range)
#bin_edges_trunc = bin_edges[1::]
#
#hist_list = []
#for z in np.arange(1, fisherz_array.shape[1]):
#    data_vect = fisherz_array[:, z]
#    hist, _ = np.histogram(data_vect, bins=nbins, range=hist_range)
#    hist_list.append(hist)
#    xs = hist
#    ys = bin_edges_trunc
#    ax.bar(xs, ys, zs=z, zdir='y')
#    
#ax.ticklabel_format(style='sci', scilimits=(.0001, .2,), useMathText=True)
#ax.set_xlabel('Circular Correlation Magnitude', fontsize=16)
#ax.set_ylabel('Regions of Interest', fontsize=16)
#ax.set_zlabel('Count', fontsize=16)
#ax.set_xticklabels(bin_edges)
#ax.set_zticklabels(np.arange(0, np.max(hist_list)))
#plt.show()
#
##fig = plt.figure(figsize=(10, 10))
##ax = fig.add_subplot(111, projection='3d')
##
##for z in np.arange(1, fisherz_array.shape[1]):
##    data_vect = fisherz_array[:, z]
##    
##    xs = np.arange(1, fisherz_array.shape[0]+1)
##    ys = data_vect
##    ax.bar(xs, ys, zs=z, zdir='y',)
##
##ax.set_xlabel('Subjects', fontsize=16)
##ax.set_ylabel('Regions of Interest', fontsize=16)
##ax.set_zlabel('Circular Correlation', fontsize=16)
##plt.show()
##fig.savefig('r_distributions_session1.png', bbox_inches='tight')