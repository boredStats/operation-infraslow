import h5py
import logging
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import proj_utils as pu
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.stats import ttest_1samp as ttest
from astropy.stats import circcorrcoef as circ_corr


logging.basicConfig(level=logging.INFO)


def pac(bold_data, alpha_data, attn_rois):
    import scipy as sp

    def _circ_corr(ang, line):
        # Correlate periodic data with linear data
        n = len(ang)
        rxs = sp.stats.pearsonr(line, np.sin(ang))
        rxs = rxs[0]
        rxc = sp.stats.pearsonr(line, np.cos(ang))
        rxc = rxc[0]
        rcs = sp.stats.pearsonr(np.sin(ang), np.cos(ang))
        rcs = rcs[0]
        rho = np.sqrt((rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / (1 - rcs ** 2))  # r
        # r_2 = rho**2 #r squared
        pval = 1 - sp.stats.chi2.cdf(n * (rho ** 2), 1)
        # standard_error = np.sqrt((1-r_2)/(n-2))

        return rho, pval  # , r_2,standard_error

    rmat = np.ndarray(shape=(bold_data.shape[1]**2))
    pmat = deepcopy(rmat)
    count = 0
    cols = []
    for b in range(bold_data.shape[1]):
        bold_str = '%s_BOLD'
        for a in range(alpha_data.shape[1]):
            alpha_str = '%s_Alpha'
            cols.append('%s %s' % (bold_str, alpha_str))
            r, p = _circ_corr(bold_data[:, b], alpha_data[:, a])
            rmat[count] = r
            pmat[count] = p
            count += 1

    res_df = pd.DataFrame(index=['pac', 'p_vals'], columns=cols)
    res_df.loc['pac'] = rmat
    res_df.loc['p_vals'] = pmat

    return res_df

def phase_phase_coupling(ts_data):
    rho_vect = np.ndarray(shape=(ts_data.shape[1]**2))
    count = 0
    for i in range(ts_data.shape[1]):
        for j in range(ts_data.shape[1]):
            if i == j:
                rho_vect[count] = 1
                count += 1
                continue
            rho_vect[count] = circ_corr(ts_data[:, i], ts_data[:, j])
            count += 1

    return rho_vect


def first_level(phase_amp_file, meg_subj, meg_sess, rois, attn_rois):
    attn_indices = []
    for a, aroi in enumerate(attn_rois):
        for roi in rois:
            if aroi == roi:
                attn_indices.append(a)

    # f = h5py.File(phase_amp_file)
    # subj_level = f[meg_subj[0]]
    # sess_level = subj_level[meg_sess[0]]
    # band_level = f[meg_subj[0] + '/' + meg_sess[0] + '/BOLD bandpass']  # sess_level['BOLD bandpass']
    # d_level = band_level['phase_data']
    # # print(list(band_level))
    # print(d_level[...].shape)
    # f.close()

    conns = ['%s_%s' % (r1, r2) for r2 in attn_rois for r1 in attn_rois]
    res = []
    for sess in meg_sess:
        sess_conns = ['%s %s' % (sess, c) for c in conns]
        sess_res = pd.DataFrame(index=meg_subj, columns=sess_conns)
        for subj in meg_subj:
            logging.info(' %s: Phase-amplitude coupling for %s %s' % (pu.ctime(), sess, subj))
            f = h5py.File(phase_amp_file)
            bold_dset = f[subj + '/' + sess + '/BOLD bandpass/phase_data'][...]
            bold_data = bold_dset[:, attn_indices]

            alpha_dset = f[subj + '/' + sess + '/Alpha/amplitude_data'][...]
            alpha_data = alpha_dset[:, attn_indices]
            f.close()

            res_df = pac(bold_data, alpha_data, attn_rois)
            sess_res.loc[subj] = res_df.loc['pac'].values

        res.append(sess_res)

    return pd.concat(res, axis=1)


def second_level(first_level_tests):
    mu = np.zeros(shape=(len(list(first_level_tests))))
    t, p = ttest(first_level_tests.values, popmean=mu, axis=0)
    res_df = pd.DataFrame(index=['t', 'p'], columns=list(first_level_tests))
    res_df.loc['t'] = t
    res_df.loc['p'] = p

    return res_df


def cron_alpha_test(first_level_tests, attn_rois, meg_sess):
    conns = ['%s_%s' % (r1, r2) for r2 in attn_rois for r1 in attn_rois]
    res_df = pd.DataFrame(index=['cron_alpha'], columns=conns)
    for conn in conns:
        ca_data = []
        for colname in list(first_level_tests):
            for sess in meg_sess:
                if conn in colname and sess in colname:
                    ca_data.append(first_level_tests[colname].values)
        res_df.loc['cron_alpha'][conn] = pu.cron_alpha(np.asarray(ca_data).T)

    return res_df


def mirror_strfind(strings):
    # Given all possible combinations of strings in a list, find the mirrored strings
    checkdict = {}  # Creating dict for string tests
    for string_1 in strings:
        for string_2 in strings:
            checkdict['%s_%s' % (string_1, string_2)] = False

    yuki, yuno = [], []
    for string_1 in strings:
        for string_2 in strings:
            test = '%s_%s' % (string_1, string_2)
            mir = '%s_%s' % (string_2, string_1)
            if string_1 == string_2:
                checkdict[test] = True
                yuno.append(test)
                continue

            if not checkdict[test] and not checkdict[mir]:
                checkdict[test] = True
                checkdict[mir] = True
                yuki.append(test)
            else:
                yuno.append(test)

    return yuki, yuno


def plot_grouped_boxplot(first_level_tests, attn_rois):
    colnames = list(first_level_tests)
    sessions = pd.unique([c.split()[0] for c in colnames])
    connections, mirrors = mirror_strfind(attn_rois)

    melty = first_level_tests.melt(var_name='Old Columns', value_name='Phase-phase coupling')
    filo, raph = [], []
    for old_col in melty['Old Columns'].values:
        raph.append(old_col.split()[0])
        filo.append(old_col.split()[1])
    melty['Connection'] = filo
    melty['Session'] = raph

    for mir in mirrors:
        idx = melty[melty['Connection'] == mir].index
        melty.drop(idx, inplace=True)

    print(melty)

    sns.set(style='darkgrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    sns.boxenplot(x='Connection', y='Phase-phase coupling', hue='Session', data=melty)
    plt.show()


def main():
    pdir = pu.get_proj_dir()
    pdata = pu.proj_data()
    rois = pdata.roiLabels
    meg_subj, meg_sess = pdata.get_meg_metadata()
    phase_amp_file = pdir + '/data/MEG_phase_amp_data.hdf5'
    attn_rois = ['IPS1_R' , 'FEF_R', 'TPOJ1_R', 'AVI_R']

    first_level_tests = first_level(phase_amp_file, meg_subj, meg_sess, rois, attn_rois)
    first_level_tests.to_excel(pdir+'/figures/attention_networks/phase_amp_first_level.xlsx')
    # first_level_tests = pd.read_excel(pdir+'/figures/attention_networks/phase_phase_first_level.xlsx', index_col=0)
    # second_level_res = second_level(first_level_tests)
    # cron_alpha_res = cron_alpha_test(first_level_tests, attn_rois, meg_sess)
    #
    # res = {'first_level_tests': first_level_tests,
    #        'second_level_tests': second_level_res,
    #        'cron_alpha_tests': cron_alpha_res}
    # pu.save_xls(res, pdir+'/figures/attention_networks/phase_phase_cross_correlations.xlsx')
    # plot_grouped_boxplot(first_level_tests, attn_rois)


main()
