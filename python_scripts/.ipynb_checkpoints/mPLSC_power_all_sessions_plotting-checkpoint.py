"""
Plot multitable PLSC results

"""

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

def _unpickle(pickle_path):
    """Get mPLSC results"""
    
    with open(pickle_path, 'rb') as file:
        output = pkl.load(file)
        
    res_perm = output['permutation_tests']
    res_boot = output['bootstrap_tests']
    y_saliences = output['y_saliences']
    x_saliences = output['x_saliences']
    res_behavior = output['behaviors']
    res_conj = output['conjunctions']
    
    return res_perm, res_boot, y_saliences, x_saliences, res_behavior, res_conj 

def _get_highest_squared_brain_salience(res_conj, latent_names):
        vals = []
        for name in latent_names:
            vals.append(res_conj[name].values **2)
            
        return np.max(vals)

def plot_bar(series):
    x = np.arange(len(series.values))
    fig, ax = plt.subplots()
    plt.bar(x, series.values)
    
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    import proj_utils as pu
    import mPLSC_functions as mf
    
    pdir = pu._get_proj_dir()
    pdObj = pu.proj_data()
    rois = pdObj.roiLabels
    roi_path = pdir + '/data/glasser_atlas/'
    
    fig_path = pdir + '/figures/mPLSC_power_all_sessions'
    pickle_path = pdir + '/data/mPLSC_power_bandpass_filtered.pkl'
    
    res_perm, res_boot, y_saliences, x_saliences, res_behavior, res_conj = _unpickle(pickle_path)
    
    alpha = .001
    num_latent_vars = len(np.where(res_perm['p_values'] < alpha)[0])
    latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]
    
    mf.save_xls(y_saliences, fig_path + '/behavior_saliences.xlsx')
    behavior_categories = list(y_saliences)
    y_squared_saliences = {}
    for cat in behavior_categories:
        df = y_saliences[cat]
        y_squared_saliences[cat] = df**2
    mf.save_xls(y_squared_saliences, fig_path + '/behavior_saliences_squared.xlsx')

    mf.save_xls(x_saliences, fig_path + '/brain_saliences.xlsx')
    sessions = list(x_saliences)
    x_squared_saliences = {}
    for sess in sessions:
        df = x_saliences[sess]
        x_squared_saliences[sess] = df**2
    mf.save_xls(x_squared_saliences, fig_path + '/brain_saliences_squared.xlsx')
    
    print('%s: Plotting scree' % pu.ctime())   
    mf.plotScree(res_perm['true_eigenvalues'],
                 res_perm['p_values'],
                 alpha=alpha,
                 fname=fig_path + '/scree.png')

    y_summed_squared_saliences = pd.DataFrame(index=behavior_categories, columns=latent_names)
    max_vals = []
    for c, cat in enumerate(behavior_categories):
        df = y_squared_saliences[cat]
        sums = np.sum(df.values, axis=0)
        y_summed_squared_saliences.iloc[c] = sums
        max_vals.append(np.max(sums))
    max_val = np.max(max_vals)
    
    print('%s: Creating radar plots' % pu.ctime())
    for name in latent_names:
        fname = fig_path + '/behavior_summed_%s.png' % name 
        series = y_summed_squared_saliences[name]
        mf.plot_radar2(series, max_val=.5, choose=False, separate_neg=False, fname=fname)
    
    print('%s: Creating brain figures' % pu.ctime())
    maxval = _get_highest_squared_brain_salience(res_conj, latent_names)
    print('Max salience is %.3f' % maxval)
    for name in latent_names:
        mags = res_conj[name].values **2
        bin_mags = []
        for m in mags:
            if m > 0:
                bin_mags.append(1)
            else:
                bin_mags.append(0)
        
        fname = fig_path + '/brain_binarized_%s.png' % name
        custom_roi = mf.create_custom_roi(roi_path, rois, bin_mags)   
        minval = np.min(mags[np.nonzero(mags)])
        if len(np.nonzero(mags)) == 1:
            minval = None
        mf.plot_brain_saliences(custom_roi, minval, maxval, figpath=fname)
        
        fname = fig_path + '/brain_%s.png' % name
        custom_roi = mf.create_custom_roi(roi_path, rois, mags)
        minval = np.min(mags[np.nonzero(mags)])
        if len(np.nonzero(mags)) == 1:
            minval = None
        mf.plot_brain_saliences(custom_roi, minval, maxval, figpath=fname)
    
    print('%s: Finished' % pu.ctime())