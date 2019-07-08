"""
Sandbox for plotting brains
"""

import nilearn
import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn import surface, plotting, datasets

nilearn.EXPAND_PATH_WILDCARDS = False

def plot_brain_saliences(custom_roi, minval=0, maxval=None, figpath=None, cbar=False, cmap=None):
    mpl.rcParams.update(mpl.rcParamsDefault)
    if cmap is None:
        cmap = 'coolwarm'

    fsaverage = datasets.fetch_surf_fsaverage()

    orders = [
        ('medial', 'left'),
        ('medial', 'right'),
        ('lateral', 'left'),
        ('lateral', 'right')]

    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(8.0, 6.0),
        dpi=300,
        frameon=False,
        sharex=True,
        sharey=True,
        subplot_kw={'projection':'3d'})

    fig.subplots_adjust(hspace=0., wspace=0.00005)
    axes_list = fig.axes

    for index, order in enumerate(orders):
        view = order[0]
        hemi = order[1]

        texture = surface.vol_to_surf(custom_roi, fsaverage['pial_%s' % hemi])
        plotting.plot_surf_roi(
                fsaverage['infl_%s' % hemi],
                texture,
                cmap=cmap,
                hemi=hemi,
                view=view,
                bg_on_data=True,
                axes=axes_list[index],
                bg_map=fsaverage['sulc_%s' % hemi],
                threshold=minval,
                vmax=maxval,
                output_file=figpath,
                symmetric_cbar=False,
                figure=fig,
                darkness=.5,
                colorbar=cbar)
    plt.clf()

def plot_brain_heatmap(plot_matrix, color_list, fname=None):
    mpl.rcParams['image.interpolation'] = 'nearest'

    cm = LinearSegmentedColormap.from_list(
        'my_cmap',
        full_color_list,
        N=len(full_color_list))

    fig, ax = plt.subplots(figsize=(4, 48))
    im = ax.matshow(
        plot_matrix,
        # interpolation='none',
        origin='lower',
        aspect='auto',
        vmax=7,
        cmap=cm)

    # ylines = np.arange(1, len(left_rois), 1)
    # for y in ylines:
    #     ax.axhline(y, linewidth=.2)

    # ax.set_yticklabels(left_rois[::-1])
    # fig.colorbar(im)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', length=0, width=0, labelsize=0)

    fig.savefig(fname, bbox_inches='tight')

if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    import sys
    sys.path.append("..")
    import proj_utils as pu

    pdir = pu._get_proj_dir()
    fig_path = pdir + '/figures/mPLSC_power_all_sessions/'
    pdObj = pu.proj_data()
    rois = pdObj.roiLabels
    colors = pdObj.colors
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    alpha_colors = []
    for c in colors:
        color = list(np.divide(c, 255))
        # color.append(1)
        alpha_colors.append(tuple(color))

    ###Plotting saliences on the brain
    # cmap = 'autumn'
    # plot_brain_saliences(
    #     custom_roi=fig_path + 'LatentVar1.nii.gz',
    #     minval=4,
    #     maxval=40,
    #     figpath=fig_path + 'test_brain_%s.png' %cmap,
    #     cbar=True,
    #     cmap=cmap)

    ###Plotting saliences on a heatmap
    cfc_path = fig_path + 'mPLSC_cfc/'
    dir_list = os.listdir(cfc_path)
    conjunction_list_unsort = [f for f in dir_list if 'conjunction' in f]
    raw_saliences_list_unsort = [f for f in dir_list if 'brain_saliences' in f]

    #Sorting files into proper band order
    conjunction_list, raw_saliences_list = [], []
    for band in bands:
        for file in conjunction_list_unsort:
            if band in file:
                conjunction_list.append(file)
        for file in raw_saliences_list_unsort:
            if band in file:
                raw_saliences_list.append(file)

    #Creating list of latent variable names
    num_latent_vars = 4
    latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]

    #Getting salience data from excel files
    conjunction_sals = {}
    for conj in conjunction_list:
        table_name = conj.replace('.xlsx', '')
        raw_df = pd.read_excel(os.path.join(cfc_path, conj))
        conjunction_sals[table_name] = raw_df[latent_names]

    #Getting average saliences for each band, latent variable
    average_sals_over_sessions = {}
    for r, raw in enumerate(raw_saliences_list):
        average_sals = pd.DataFrame(index=rois, columns=latent_names)
        xls = pd.ExcelFile(os.path.join(cfc_path, raw))

        session_data = {}
        for sheet in xls.sheet_names:
            df = xls.parse(sheet, index_col=0)
            session_data[sheet] = df[latent_names]

        for name in latent_names:
            for roi in rois:
                z_vals = []
                for sess in list(session_data):
                    df = session_data[sess]
                    val = df.loc[roi, name]
                    z_vals.append(np.abs(val))
                mu = np.mean(z_vals)
                average_sals.loc[roi, name] = mu
        average_sals_over_sessions[bands[r]] = average_sals

    #Creating custom colormap
    white = np.array([1, 1, 1])
    gray = np.multiply([1, 1, 1], .5)
    colors_for_plot = alpha_colors[:len(conjunction_sals)]

    full_color_list = []
    full_color_list.append(white)
    full_color_list.append(gray)
    for plot_color in colors_for_plot:
        full_color_list.append(plot_color)

    left_rois = [r for r in rois if '_L' in r]
    right_rois = [r for r in rois if '_R' in r]

    #Creating plot data as integer list
    for name in latent_names:
        plot_matrix = []
        for s, sal in enumerate(list(conjunction_sals)):
            #Calulating average z-score for the latent variable
            band = bands[s]
            average_sals = average_sals_over_sessions[band]
            average_sals_in_lv = average_sals[name]
            mu = np.mean(average_sals_in_lv.values)

            conj_df = conjunction_sals[sal]
            conj_data = conj_df[name].values
            #Dummy-coding conjunction z-scores
            plot_data = []
            for val in conj_data:
                if val < mu:
                    if val < 4:
                        plot_data.append(0)
                    else:
                        plot_data.append(1)
                else:
                    plot_data.append(s+2)
            plot_matrix.append(plot_data)

        #Turning nested lists into array
        plot_matrix = np.asarray(plot_matrix).T
        plot_df = pd.DataFrame(plot_matrix, index=rois, columns=bands)

        lr_plot_data = []
        for band in bands:
            series = plot_df[band]
            left_series = series.loc[left_rois]
            right_series = series.loc[right_rois]
            lr_plot_data.append(left_series.values)
            lr_plot_data.append(right_series.values)
        lr_plot_matrix = np.asarray(lr_plot_data).T
        lr_plot_matrix = np.flipud(lr_plot_matrix)
        print(lr_plot_matrix.shape)

        fname = cfc_path + 'brain_heatmap_%s.svg' % name
        plot_brain_heatmap(lr_plot_matrix, full_color_list, fname)

    #Creating 3-stacked list of ROIs
    cleaned_roi_list = [r.replace('_L', '') for r in left_rois]
    df = pd.DataFrame(columns=['col1', 'col2', 'col3'])
    df['col1'] = cleaned_roi_list[0::3]
    df['col2'] = cleaned_roi_list[1::3]
    df['col3'] = cleaned_roi_list[2::3]
    df.to_excel(fig_path + 'roi_list_3column.xlsx')

    print('Finished')
