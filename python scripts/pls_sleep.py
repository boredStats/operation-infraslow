# Running PLS-correlation using sklearn's PLSSVD function

import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils as pu
import mPLSC_functions as mf
from sklearn.cross_decomposition import PLSSVD

def main():
    pdir = pu._get_proj_dir()
    pdata = pu.proj_data()
    rois = pdata.roiLabels
    colors = pdata.colors
    meg_subj, meg_sessions = pdata.get_meg_metadata()
    print(meg_sessions)
    print('%s: Building tables of power data for MEG' % pu.ctime())
    meg_data = mf.extract_average_power(hdf5_file=pdir + '/data/downsampled_MEG_truncated.hdf5',
                                        sessions=[meg_sessions[0]],
                                        subjects=meg_subj,
                                        rois=rois,
                                        image_type='MEG',
                                        bp=True)
    X = pd.concat(meg_data, axis=1)

    sleep_variables = ['PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7']
    behavior_raw = pd.read_excel(pdir + '/data/hcp_behavioral.xlsx', index_col=0, sheet_name='cleaned')

    sleep_df = behavior_raw[sleep_variables]
    Y = sleep_df.values

    plsc = PLSSVD(n_components=10, scale=True)
    plsc.fit(X, Y)
    X_c, Y_c = plsc.transform(X, Y)
    # print(X_c.shape, Y_c.shape)
    print(plsc.x_weights_.shape)
    print(plsc.y_weights_.shape)


main()

