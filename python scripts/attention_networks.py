import h5py
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import proj_utils as pu
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.stats import pearsonr, ttest_1samp, chi2
from astropy.stats.circstats import circcorrcoef as circ_corr


logging.basicConfig(level=logging.INFO)


def ppc(phase_data):
    rho_vect = np.ndarray(shape=(phase_data.shape[1]**2))
    count = 0
    for i in range(phase_data.shape[1]):
        for j in range(phase_data.shape[1]):
            rho = circ_corr(phase_data[:, i], phase_data[:, j])
            rho_vect[count] = rho
            count += 1

    # rmat = circ_corr(phase_data, phase_data, axis=1)
    # print(rmat)
    # rho_vect = np.ndarray.flatten(rmat)
    return rho_vect


def pac(bold_data, alpha_data, attn_rois):
    def _circ_line_corr(ang, line):
        # Correlate periodic data with linear data
        # https://github.com/voytekresearch/tutorials
        n = len(ang)
        rxs = pearsonr(line, np.sin(ang))
        rxs = rxs[0]
        rxc = pearsonr(line, np.cos(ang))
        rxc = rxc[0]
        rcs = pearsonr(np.sin(ang), np.cos(ang))
        rcs = rcs[0]
        rho = np.sqrt((rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / (1 - rcs ** 2))  # r
        # r_2 = rho**2 #r squared
        pval = 1 - chi2.cdf(n * (rho ** 2), 1)
        # standard_error = np.sqrt((1-r_2)/(n-2))

        return rho, pval  # , r_2,standard_error

    rmat = np.ndarray(shape=(bold_data.shape[1]**2))
    pmat = deepcopy(rmat)
    count = 0
    cols = []
    for b in range(bold_data.shape[1]):
        bold_str = '%s_BOLD' % attn_rois[b]
        for a in range(alpha_data.shape[1]):
            alpha_str = '%s_Alpha' % attn_rois[a]
            cols.append('%s %s' % (bold_str, alpha_str))
            r, p = _circ_line_corr(bold_data[:, b], alpha_data[:, a])
            rmat[count] = r
            pmat[count] = p
            count += 1

    res_df = pd.DataFrame(index=['pac', 'p_vals'], columns=cols)
    res_df.loc['pac'] = rmat
    res_df.loc['p_vals'] = pmat

    return res_df


def first_level_ppc(phase_amp_file, meg_subj, meg_sess, rois, attn_rois):
    attn_indices = []
    for aroi in attn_rois:
        for r, roi in enumerate(rois):
            if aroi == roi:
                attn_indices.append(r)

    conns = ['%s-%s' % (r1, r2) for r2 in attn_rois for r1 in attn_rois]
    res = []
    for sess in meg_sess:
        sess_conns = ['%s %s' % (sess, c) for c in conns]
        sess_res = pd.DataFrame(index=meg_subj, columns=sess_conns)
        count = 0
        for subj in meg_subj:
            logging.info(' %s: Phase-phase coupling for %s %s' % (pu.ctime(), sess, subj))
            f = h5py.File(phase_amp_file)
            dset = f[subj + '/' + sess + '/BOLD bandpass/phase_data'][...]
            attn_data = dset[:, attn_indices]
            f.close()
            sess_res.loc[subj] = ppc(attn_data)
            count += 1
        res.append(sess_res)
    return pd.concat(res, axis=1)


def first_level_pac(phase_amp_file, meg_subj, meg_sess, rois, attn_rois):
    """


    :param phase_amp_file:
    :param meg_subj:
    :param meg_sess:
    :param rois:
    :param attn_rois:
    :return:
    """
    attn_indices = []
    for aroi in attn_rois:
        for r, roi in enumerate(rois):
            if aroi == roi:
                attn_indices.append(r)

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
    t, p = ttest_1samp(first_level_tests.values, popmean=mu, axis=0)
    res_df = pd.DataFrame(index=['t', 'p'], columns=list(first_level_tests))
    res_df.loc['t'] = t
    res_df.loc['p'] = p

    return res_df


def cron_alpha_test(first_level_tests, attn_rois, meg_sess):
    conns = ['%s-%s' % (r1, r2) for r2 in attn_rois for r1 in attn_rois]
    res_df = pd.DataFrame(index=['cron_alpha'], columns=conns)
    colnames = list(first_level_tests)
    for conn in conns:
        ca_data = []
        for colname in colnames:
            for sess in meg_sess:
                if sess in colname:
                    if conn in colname:
                        ca_data.append(first_level_tests[colname].values)
        if not ca_data:
            print('CA not run')
            continue
        res_df.loc['cron_alpha'][conn] = pu.cron_alpha(np.asarray(ca_data).T)

    return res_df


def mirror_strfind(strings):
    # Given all possible combinations of strings in a list, find the mirrored strings
    checkdict = {}  # Creating dict for string tests
    for string_1 in strings:
        for string_2 in strings:
            checkdict['%s-%s' % (string_1, string_2)] = False

    yuki, yuno = [], []
    for string_1 in strings:
        for string_2 in strings:
            test = '%s-%s' % (string_1, string_2)
            mir = '%s-%s' % (string_2, string_1)
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


def plot_grouped_boxplot(first_level_tests, attn_rois, cron_alpha_df=None, fname=None):
    connections, mirrors = mirror_strfind(attn_rois)
    vn = "Functional connectivity (Circular R)"
    melty = first_level_tests.melt(var_name='Old Columns', value_name=vn)
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

    sns.boxplot(x='Connection', y=vn, hue='Session', data=melty)
    if cron_alpha_df is not None:
        new_xlabels = []
        for conn in connections:
            for colname in list(cron_alpha_df):
                if conn in colname:
                    val = cron_alpha_df.loc['cron_alpha'][conn]
                    xlabel = '%s\n'r'$\alpha$ = %.3f' % (conn, val)
                    new_xlabels.append(xlabel)
        ax.set_xticklabels(new_xlabels)
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight')
    else:
        plt.show()


def main():
    pdir = pu.get_proj_dir()
    pdata = pu.proj_data()
    rois = pdata.roiLabels
    meg_subj, meg_sess = pdata.get_meg_metadata()
    phase_amp_file = pdir + '/data/MEG_phase_amp_data.hdf5'
    attn_rois =  ['IPS1_R', 'FEF_R', 'TPOJ1_R', 'AVI_R']

    # first_level_tests_ppc = first_level_ppc(phase_amp_file, meg_subj, meg_sess, rois, attn_rois)
    # first_level_tests_ppc.to_excel(pdir+'/figures/attention_networks/ppc_first_level.xlsx')
    first_level_tests_ppc = pd.read_excel(pdir+'/figures/attention_networks/ppc_first_level.xlsx', index_col=0)
    second_level_res = second_level(first_level_tests_ppc)
    cron_alpha_res = cron_alpha_test(first_level_tests_ppc, attn_rois, meg_sess)

    res = {'first_level_tests': first_level_tests_ppc,
           'second_level_tests': second_level_res,
           'cron_alpha_tests': cron_alpha_res}
    pu.save_xls(res, pdir+'/figures/attention_networks/ppc_second_level.xlsx')

    plot_grouped_boxplot(first_level_tests_ppc, attn_rois,
                         cron_alpha_df=cron_alpha_res,
                         fname=pdir+'/figures/attention_networks/ppc_boxplot.pdf')


main()
