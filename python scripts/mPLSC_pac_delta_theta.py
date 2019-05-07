"""
mPLSC using phase-amplitude coupling between delta phase/theta amplitude and
theta phase/gamma amplitude tables
"""

import h5py
import numpy as np
import pandas as pd
import pickle as pkl
import mPLSC_functions as mf

def load_delta_theta_cfc(hdf5_path, rois, subjects, sessions, outfile=None):
    bands_to_extract = [('BOLD bandpass', 'Delta'), ('Delta', 'Theta'), ('Theta', 'Gamma')]
    bands = ['BOLD bandpass', 'Delta', 'Theta', 'Alpha', 'Gamma']

    #Getting metadata from hdf5_path
    # file = h5py.File(hdf5_path)
    # sessions = list(file)
    # layer = file.get(sessions[0] + '/' + subjects[0] + '/' + rois[0] + '/r_vals')
    # data = layer5[...]
    # file.close()

    tables = {}
    for band_pair in bands_to_extract:
        phase_band = band_pair[0]
        amp_band = band_pair[1]
        for b, band in enumerate(bands):
            #Getting indcies
            if phase_band in band:
                p = b
            if amp_band in band:
                a = b
        for session in sessions:
            table_name = "%s_%s_%s" % (phase_band, amp_band, session)
            print('Creating table %s' % table_name)
            cfc_output = np.ndarray(shape=(len(subjects), len(rois)))
            for s, subject in enumerate(subjects):
                for r, roi in enumerate(rois):
                    cfc_key = session + '/' + subject + '/' + roi + '/r_vals'
                    file = h5py.File(hdf5_path)
                    cfc_data = file.get(cfc_key)[...]
                    cfc_value = cfc_data[p, a]
                    cfc_output[s, r] = cfc_value
                    file.close()

            cfc_table = pd.DataFrame(cfc_output, index=subjects, columns=rois)
            tables[table_name] = cfc_table

    if outfile is not None:
        mf.save_xls(tables, outfile)

    return tables

if __name__ == '__main__':
    from boredStats import pls_tools

    import sys
    sys.path.append("..")
    import proj_utils as pu

    print('%s: Loading data' % pu.ctime())
    pdir = pu._get_proj_dir()
    ddir = pdir + '/data/'
    roi_path = ddir + '/glasser_atlas/'
    fig_path = pdir + '/figures/mPLSC_delta_theta/'

    pdObj = pu.proj_data()
    rois = pdObj.roiLabels
    colors = pdObj.colors
    meg_subj, meg_sessions = pdObj.get_meg_metadata()
    mri_subj, mri_sess = pdObj.get_mri_metadata()
    subj = [s for s in mri_subj if s in meg_subj]
    meg_sess = ['Session1', 'Session2', 'Session3']

    pls_path = ddir + 'mPLSC_delta_theta_cfc.pkl'
    check_0 = input('Run mPLSC? y/n ')
    if check_0 == 'y':
        xls_path = ddir + 'MEG_delta_theta_pac.xlsx'
        check_1 = input('Create excel sheets? y/n ')
        if check_1 == 'y':
            hdf5_path = ddir + 'MEG_phase_amp_coupling.hdf5'
            cfc_tables = load_delta_theta_cfc(hdf5_path, rois, subj, meg_sess, xls_path)
        else:
            xls = pd.ExcelFile(xls_path)
            sheets = xls.sheet_names
            cfc_tables = {}
            for sheet in sheets:
                cfc_tables[sheet] = xls.parse(sheet)
        x_tables = [cfc_tables[key] for key in list(cfc_tables)]

        print('%s: Building subtables of behavior data' % pu.ctime())
        behavior_metadata = pd.read_csv(
            ddir + '/b_variables_mPLSC.txt',
            delimiter='\t',
            header=None)
        behavior_metadata.rename(
            dict(zip([0, 1], ['category','name'])),
            axis='columns',
            inplace=True)
        behavior_raw = pd.read_excel(
            ddir + '/hcp_behavioral.xlsx',
            index_col=0,
            sheet_name='cleaned')
        behavior_tables = mf.load_behavior_subtables(
            behavior_raw,
            behavior_metadata)
        y_tables = [behavior_tables[category] for category in list(behavior_tables)]

        p = pls_tools.MultitablePLSC(n_iters=10000)
        print('%s: Running permutation testing on latent variables' % pu.ctime())
        res_perm = p.mult_plsc_eigenperm(y_tables, x_tables)
        print('%s: Running bootstrap testing on saliences' % pu.ctime())
        res_boot = p.mult_plsc_bootstrap_saliences(y_tables, x_tables, 0)

        num_latent_vars = len(np.where(res_perm['p_values'] < .001)[0])
        latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]

        print('%s: Organizing saliences' % pu.ctime())
        y_saliences, y_saliences_z = mf.organize_behavior_saliences(
            res_boot,
            y_tables,
            list(behavior_tables),
            num_latent_vars)
        x_saliences, x_saliences_z = mf.organize_brain_saliences(
            res_boot,
            x_tables,
            list(cfc_tables),
            num_latent_vars)

        output = {'permutation_tests':res_perm,
                  'bootstrap_tests':res_boot,
                  'y_saliences':y_saliences,
                  'x_saliences':x_saliences,
                  'y_saliences_zscores':y_saliences_z,
                  'x_saliences_zscores':x_saliences_z}
        with open(pls_path, 'wb') as file:
            pkl.dump(output, file)
    else:
        with open(pls_path, 'rb') as file:
            output = pkl.load(file)

    print('%s: Plotting scree' % pu.ctime())
    res_perm = output['permutation_tests']
    mf.plotScree(res_perm['true_eigenvalues'],
                 res_perm['p_values'],
                 alpha=.001,
                 fname=fig_path + '/scree.png')

    num_latent_vars = 3#len(np.where(res_perm['p_values'] < alpha)[0])
    latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]

    print('%s: Saving behvaior salience z-scores' % pu.ctime())
    y_saliences_zscores = output['y_saliences_zscores']
    y_saliences_zscores_thresh = {}
    for behavior_category in list(y_saliences_zscores):
        unthresh_df = y_saliences_zscores[behavior_category]
        unthresh_vals = unthresh_df.values
        thresh_vals = np.ndarray(shape=unthresh_vals.shape)
        for r in range(len(unthresh_df.index)):
            for c in range(len(list(unthresh_df))):
                if np.abs(unthresh_vals[r, c]) < 4:
                    thresh_vals[r, c] = 0
                else:
                    thresh_vals[r, c] = unthresh_vals[r, c]
        thresh_df = pd.DataFrame(thresh_vals, index=unthresh_df.index, columns=list(unthresh_df))
        y_saliences_zscores_thresh[behavior_category] = thresh_df
    mf.save_xls(y_saliences_zscores_thresh, fig_path + '/behavior_saliences_z.xlsx')

    print('%s: Averaging saliences within behavior categories' % pu.ctime())
    behavior_avg = mf.average_subtable_saliences(y_saliences_zscores_thresh)
    behavior_avg.to_excel(fig_path+'/behavior_average_z.xlsx')

    print('%s: Creating quick histograms, getting means' % pu.ctime())
    def _quick_hist(data, fname=None):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.hist(data, bins=20)
        if fname is not None:
            fig.savefig(fname, bbox_inches='tight')

    mu = []
    for name in latent_names:
        latent_data = []
        for behavior_category in list(y_saliences_zscores_thresh):
            behavior_df = y_saliences_zscores_thresh[behavior_category]
            series = behavior_df[name]
            latent_data.append(np.abs(series.values))
        histogram_data = [v for vals in latent_data for v in vals]
        mu.append(np.mean(histogram_data))
        _quick_hist(histogram_data, fig_path+'/behavior_hist_%s.png' % name)

    print('%s: Creating bar/table combo figures' % pu.ctime())
    for n, name in enumerate(latent_names):
        print(mu[n])
        mf.bar_all_behaviors(
            behavior_dict=y_saliences_zscores_thresh,
            latent_variable=name,
            mu=mu[n],
            colors=colors,
            xlim=40,
            fname=fig_path + '/behavior_fullbar_%s.svg' % name)


    print('%s: Organizing brain saliences' % pu.ctime())
    xls_path = ddir + 'MEG_delta_theta_pac.xlsx'
    xls_data = pd.read_excel(xls_path)

    bands_delta_theta_analysis = ['BOLD bandpass_Delta', 'Delta_Theta', 'Theta_Gamma']
    sessions = ['Session1', 'Session2', 'Session3']
    print('%s: Running conjunction on brain data' % pu.ctime())
    conjunctions_no_sign = {}
    x_saliences_z = output['x_saliences_zscores']
    raw_hist_data, conj_hist_data = [], []
    for band_combo in bands_delta_theta_analysis:
        sals_per_sess = {}
        for sess in sessions:
            for brain_table in list(x_saliences_z):
                if band_combo in brain_table and sess in brain_table:
                    sals_per_sess[sess] = x_saliences_z[brain_table]

        mf.save_xls(sals_per_sess, fig_path + '/brain_saliences_z_%s.xlsx' % band_combo)

        for sess in sals_per_sess:
            print(sals_per_sess[sess].values[:, :3].shape)
            raw_hist_data.append(np.abs(np.ndarray.flatten(sals_per_sess[sess].values[:, :3])))

        res_conj = mf.single_table_conjunction(
            sals_per_sess,
            comp='any',
            thresh=4)
        res_conj.to_excel(fig_path + '/conjunction_no_sign_%s.xlsx' % band_combo)
        conjunctions_no_sign[band_combo] = res_conj
        conj_hist_data.append(np.abs(np.ndarray.flatten(res_conj.values[:, :3])))

    res_conj_band_combo = mf.single_table_conjunction(conjunctions_no_sign)
    res_conj_band_combo.to_excel(fig_path + '/conjunction_over_bands.xlsx')

    _quick_hist(raw_hist_data, fig_path + '/brain_histogram.png')
    _quick_hist(conj_hist_data, fig_path + '/brain_histogram_conj.png')
    import nibabel as nib
    check = input('Make brain figures? y/n ')
    if check == 'y':
        print('%s: Creating brain figures' % pu.ctime())
        for band_combo in bands_delta_theta_analysis:
            if bands_delta_theta_analysis[0] in band_combo:
                cmap = 'autumn'
            elif bands_delta_theta_analysis[1] in band_combo:
                cmap = 'summer'
            else:
                cmap = 'winter'
            brain_conjunction = conjunctions_no_sign[band_combo]#conjunctions_sign_matters[band_combo]
            for name in latent_names:
                mags = brain_conjunction[name]
                fname = fig_path + '/brain_%s_%s.pdf' % (band_combo, name)
                # custom_roi = mf.create_custom_roi(roi_path, rois, mags)
                custom_roi = nib.load(fig_path + '/brain_%s_%s.nii.gz' % (band_combo, name))
                # nib.save(custom_roi, fig_path + '/brain_%s_%s.nii.gz' % (band_combo, name))
                mf.plot_brain_saliences(
                    custom_roi,
                    minval=4,
                    maxval=30,
                    figpath=fname,
                    cbar=True,
                    cmap=cmap)

                if band_combo == bands_delta_theta_analysis[0]:
                    mags_full = res_conj_band_combo[name]
                    fname = fig_path + '/brain_full_%s.pdf' % name
                    # custom_roi = mf.create_custom_roi(roi_path, rois, mags_full)
                    # nib.save(custom_roi, fig_path + '/brain_full_%s.nii.gz' % name)
                    custom_roi = nib.load(fig_path + '/brain_full_%s.nii.gz' % name)
                    mf.plot_brain_saliences(
                        custom_roi,
                        minval=4,
                        maxval=30,
                        figpath = fname,
                        cbar=True,
                        cmap='viridis')
    print('%s: Finished' % pu.ctime())
