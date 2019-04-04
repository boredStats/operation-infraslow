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
    
def save_xls(dict_df, path):
    """
    Save a dictionary of dataframes to an excel file, with each dataframe as a seperate page
    """

    writer = pd.ExcelWriter(path)
    for key in dict_df:
        dict_df[key].to_excel(writer, '%s' % key)

    writer.save()

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
    
    fig_path = pdir + '/figures/mPLSC_power_bandpass_filtered'
    pickle_path = pdir + '/data/mPLSC_power_bandpass_filtered.pkl'
    
    res_perm, res_boot, y_saliences, x_saliences, res_behavior, res_conj = _unpickle(pickle_path)
    
    alpha = .001
    num_latent_vars = len(np.where(res_perm['p_values'] < alpha)[0])
    latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]
    
    save_xls(y_saliences, pdir + '/data/mPLSC_power_behavior_saliences.xlsx')
    
    y_squared_saliences = {}
    keys = list(y_saliences)
    for key in keys:
        df = y_saliences[key]
        y_squared_saliences[key] = df**2
        
    save_xls(y_squared_saliences, pdir + '/data/mPLSC_power_behavior_saliences_squared.xlsx')
#     print('%s: Plotting scree' % pu.ctime())   
#     mf.plotScree(res_perm['true_eigenvalues'],
#                  res_perm['p_values'],
#                  alpha=alpha,
#                  fname=fig_path + '/scree.png')

#     print('%s: Creating bar plots' % pu.ctime())
#     name = latent_names[0]
#     fname = fig_path + '/behavior_bar_%s.png' % name
    
#     behavior_categories = list(y_saliences)
    
#     series = y_saliences[behavior_categories[0]][name]
#     print(series)
#     plot_bar(series)
    
#     print('%s: Creating radar plots' % pu.ctime())
#     for name in latent_names:
#         fname = fig_path + '/behavior_%s.png' % name 
#         series = res_behavior[name] ** 2
#         mf.plot_radar2(series, choose=False, separate_neg=False, fname=fname)
        
#     print('%s: Creating brain figures' % pu.ctime())
#     maxval = _get_highest_squared_brain_salience(res_conj, latent_names)
#     print('Max salience is %.3f' % maxval)
#     for name in latent_names:
#         mags = res_conj[name].values **2
#         bin_mags = []
#         for m in mags:
#             if m > 0:
#                 bin_mags.append(1)
#             else:
#                 bin_mags.append(0)

#         custom_roi = mf.create_custom_roi(roi_path, rois, bin_mags)        
#         fname = fig_path + '/brain_binarized_%s.png' % name
# #         fname = fig_path + '/brain_%s.png' % name
#         minval = np.min(mags[np.nonzero(mags)])
#         if len(np.nonzero(mags)) == 1:
#             minval = None
#         mf.plot_brain_saliences(custom_roi, minval, maxval, figpath=fname)
    
    print('%s: Finished' % pu.ctime())