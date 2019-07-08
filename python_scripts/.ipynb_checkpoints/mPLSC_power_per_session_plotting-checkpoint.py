"""
Plot multitable PLSC power-per-session analysis results
"""

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

def _unpickle(pickle_path):
    with open(pickle_path, 'rb') as file:
        output = pkl.load(file)
    
    session_data = {}
    conjunction_data = {}
    for key in list(output):
        if "meg" in key:
            session_data[key] = output[key]
        else:
            conjunction_data[key] = output[key]
            
    return session_data, conjunction_data

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    import proj_utils as pu
    import mPLSC_functions as mf
    
    pdir = pu._get_proj_dir()
    pdObj = pu.proj_data()
    rois = pdObj.roiLabels
    roi_path = pdir + '/data/glasser_atlas/'
    
    fig_path = pdir + '/figures/mPLSC_power_per_session'
    pickle_path = pdir + '/data/mPLSC_power_per_session.pkl'
    
    session_data, conjunction_data = _unpickle(pickle_path)
    conjunction_data['brain_conjunction'].to_excel(fig_path + '/brain_conjunction.xlsx')
    mf.save_xls(conjunction_data['behavior_conjunction'], fig_path + '/behavior_conjunction.xlsx')
    
    latent_names = list(conjunction_data['brain_conjunction'])
    
    behavior_categories = list(conjunction_data['behavior_conjunction'])
    y_summed_squared_saliences = pd.DataFrame(index=behavior_categories, columns=latent_names)
    max_vals = []
    for c, cat in enumerate(behavior_categories):
        df = conjunction_data['behavior_conjunction'][cat]
        sums = np.sum(df.values, axis=0)
        y_summed_squared_saliences.iloc[c] = sums
        max_vals.append(np.max(sums))
    max_val = np.max(max_vals)
    
    print('%s: Creating radar plots' % pu.ctime())
    for name in latent_names:
        fname = fig_path + '/behavior_summed_%s.png' % name 
        series = y_summed_squared_saliences[name]
        mf.plot_radar2(series, max_val=.5, choose=False, separate_neg=False, fname=fname)
        
    print('%s: Finished' % pu.ctime())