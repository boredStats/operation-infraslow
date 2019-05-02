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
    fig_path = pdir + '/figures/'
    pdObj = pu.proj_data()
    rois = pdObj.roiLabels
    colors = pdObj.colors
    alpha_colors = []
    for c in colors:
        color = list(np.divide(c, 255))
        # color.append(1)
        alpha_colors.append(tuple(color))

    ###Plotting saliences on the brain###
    cmap = 'viridis'
    # plot_brain_saliences(
    #     custom_roi=fig_path + 'brain_LatentVar1.nii',
    #     minval=4,
    #     maxval=5.242388956,
    #     figpath=fig_path + 'test_brain.png',
    #     cbar=False,
    #     cmap=cmap)

    ###Plotting saliences on a grid###
    cfc_path = fig_path + 'mPLSC_cfc/'
    dir_list = os.listdir(cfc_path)
    sal_list = [f for f in dir_list if 'conjunction' in f]

    #Creating list of latent variable names
    num_latent_vars = 4
    latent_names = ['LatentVar%d' % (n+1) for n in range(num_latent_vars)]

    #Getting salience data from excel files
    sals = {}
    for sal in sal_list:
        table_name = sal.replace('.xlsx', '')
        raw_df = pd.read_excel(os.path.join(cfc_path, sal))
        df = raw_df[latent_names]
        sals[table_name] = df

    #Creating custom colormap
    white = np.array([1, 1, 1])
    gray = np.multiply([1, 1, 1], .2)
    colors_for_plot = alpha_colors[:len(sals)]

    full_color_list = []
    full_color_list.append(white)
    full_color_list.append(gray)
    for plot_color in colors_for_plot:
        full_color_list.append(plot_color)

    cm = LinearSegmentedColormap.from_list(
        'my_cmap',
        full_color_list,
        N=len(full_color_list))

    #Creating plot data as MxNxRGBA array
    # pixel_data = np.ndarray(shape=(360, len(sals), 4))
    # for s, sal in enumerate(list(sals)):#name in latent_names:
    #     color = colors_for_plot[s]
    #     df = sals[sal]
    #     test_data = df[latent_names[0]].values
    #     mu = np.mean(test_data)
    #     for v, val in enumerate(test_data):
    #         pixel_data[v, :, :] = v
    #         pixel_data[:, s, :] = s
    #         if val < mu:######
    #             if val < 4:
    #                 pixel_data[v, s, :] = white
    #             else:
    #                 pixel_data[v, s, :] = gray
    #         else:
    #             pixel_data[v, s, :] = color
    # print(pixel_data[0, :, :])
    # # x = pixel_data[:, 0, 0, 0, 0, 0]
    # # y = pixel_data[0, :, 0, 0, 0, 0]
    # # extent = np.min(x), np.max(x), np.min(y), np.max(y)
    # fig, ax = plt.subplots(figsize=(8, 6))
    # im = ax.imshow(
    #     pixel_data,
    #     interpolation='none',
    #     origin='lower',
    #     aspect='auto',
    #     # extent=extent,
    #     # vmax=5,
    #     # cmap=cm
    #     )
    #
    # # fig.colorbar(im)
    # fig.savefig('test_pixel.png', bbox_inches='tight')

    left_rois = [r for r in rois if '_L' in r]
    right_rois = [r for r in rois if '_R' in r]

    #Creating plot data as integer list
    for name in latent_names:
        plot_matrix = []
        for s, sal in enumerate(list(sals)):#name in latent_names:
            df = sals[sal]
            test_data = df[name].values
            mu = np.mean(test_data)
            print(mu)
            plot_data = []
            for val in test_data:
                if val < mu:
                    if val < 4:
                        plot_data.append(0)
                    else:
                        plot_data.append(1)
                else:
                    plot_data.append(s+2)
            plot_matrix.append(plot_data)
        plot_matrix = np.asarray(plot_matrix).T
        print(plot_matrix.shape)
        plot_df = pd.DataFrame(plot_matrix, index=rois, columns=list(sals))

        lr_plot_data = []
        for sal in list(sals):
            series = plot_df[sal]
            left_series = series.loc[left_rois]
            right_series = series.loc[right_rois]
            lr_plot_data.append(left_series.values)
            lr_plot_data.append(right_series.values)
        lr_plot_matrix = np.asarray(lr_plot_data).T
        lr_plot_matrix = np.flipud(lr_plot_matrix)
        print(lr_plot_matrix.shape)

        mpl.rcParams['image.interpolation'] = 'nearest'
        fig, ax = plt.subplots(figsize=(4, 48))
        im = ax.matshow(
            lr_plot_matrix,
            # interpolation='none',
            origin='lower',
            aspect='auto',
            vmax=7,
            cmap=cm)
        ylines = np.arange(1, len(left_rois), 1)
        for y in ylines:
            ax.axhline(y, linewidth=.2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', length=0, width=0, labelsize=0)
        # ax.set_yticklabels(left_rois[::-1])
        # fig.colorbar(im)
        # fname = cfc_path + 'brain_heatmap_%s.svg' % name
        # fig.savefig(fname, bbox_inches='tight')
        # plt.show()

    # cleaned_roi_list = [r.replace('_L', '') for r in left_rois]
    # print(cleaned_roi_list)
    #
    # col_1 = cleaned_roi_list[0::3]
    # col_2 = cleaned_roi_list[1::3]
    # col_3 = cleaned_roi_list[2::3]
    #
    # df = pd.DataFrame(columns=['col1', 'col2', 'col3'])
    # df['col1'] = col_1
    # df['col2'] = col_2
    # df['col3'] = col_3
    # df.to_excel(fig_path + 'roi_list_3column.xlsx')
    # print('Finished')
