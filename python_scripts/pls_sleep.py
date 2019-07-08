import proj_utils as pu
import mPLSC_functions
from sklearn.cross_decomposition import PLSSVD
# import sys
# sys.path.append("..")
import numpy as np
import pandas as pd
import pickle as pkl
# import proj_utils as pu


def main():
    pdata = pu.proj_data()
    rois = pdata.roiLabels
    # colors = pdata.colors
    meg_subj, meg_sessions = pdata.get_meg_metadata()

    with open('../data/MEG_power_data.pkl', 'rb') as file:
        meg_data = pkl.load(file)
    meg_list = [meg_data[sess] for sess in list(meg_data)]
    x = pd.concat(meg_list, axis=1)

    sleep_variables = ['PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7']
    behavior_raw = pd.read_excel('../data/hcp_behavioral.xlsx', index_col=0, sheet_name='cleaned')

    sleep_df = behavior_raw[sleep_variables]
    y = sleep_df.values

    plsc = PLSSVD(n_components=10, scale=True)
    plsc.fit(x, y)
    X_c, Y_c = plsc.transform(x, y)
    print(plsc.x_weights_.shape)
    print(plsc.y_weights_.shape)
    # https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/cross_decomposition/pls_.py#L753


main()
