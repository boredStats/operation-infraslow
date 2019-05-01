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
        plotting.plot_surf_stat_map(
                fsaverage['infl_%s' % hemi],
                texture,
                cmap=cmap,
                hemi=hemi,
                view=view,
                bg_on_data=True,
                axes=axes_list[index],
                bg_map=fsaverage['sulc_%s' % hemi],
                threshold=minval,
                # vmax=maxval,
                output_file=figpath,
                symmetric_cbar=False,
                figure=fig,
                colorbar=cbar)
    plt.clf()

if __name__ == '__main__':
    import sys
    sys.path.append("..")
    import proj_utils as pu

    pdir = pu._get_proj_dir()
    fig_path = pdir + '/figures/'

    cmap = 'PiYG_r'
    plot_brain_saliences(
        custom_roi=fig_path + 'brain_LatentVar1.nii',
        minval=4,
        maxval=20,
        figpath=fig_path + 'test_brain.png',
        cbar=False,
        cmap=cmap)

    print('Finished')
