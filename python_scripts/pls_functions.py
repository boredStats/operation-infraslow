import numpy as np
from scipy.stats import zscore
from sklearn.utils import resample


def center_matrix(a):
    # Remove the means from each column in a matrix
    return a - np.mean(a, axis=0)


def cross_corr(x, y):
    # Calculate Pearson's R the columns of two matrices

    s = x.shape[0]
    if s != y.shape[0]:
        raise ValueError("x and y must have the same number of subjects")

    std_x = x.std(0, ddof=s - 1)
    std_y = y.std(0, ddof=s - 1)

    cov = np.dot(center_matrix(x).T, center_matrix(y))

    return cov / np.dot(std_x[:, np.newaxis], std_y[np.newaxis, :])


def _center_scale_xy(X, Y, scale=True):
    """ Center X, Y and scale if the scale parameter==True

    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    # center
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = Y.mean(axis=0)
    Y -= y_mean
    # scale
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])
    return X, Y, x_mean, y_mean, x_std, y_std


def resample_matrix(matrix):
    # Create a new matrix by resampling the rows of an old matrix
    n_rows = matrix.shape[0]
    resample_indices = np.random.randint(low=0, high=n_rows, size=n_rows)
    new_matrix = np.ndarray(shape=matrix.shape)
    for i, r in enumerate(resample_indices):
        new_matrix[i, :] = matrix[r, :]
    return new_matrix


def permutation_p(observed, perm_array):
    # see Phipson & Smyth 2010 for more information
    n_iters = len(perm_array)
    n_hits = np.where(np.abs(perm_array) >= np.abs(observed))
    return (len(n_hits[0]) + 1) / (n_iters + 1)


class PLSC:
    def __init__(self, center_scale='scale', n_iters=1000):
        self.center_scale = center_scale  # Options are 'center', 'scale', or None
        self.n_iters = n_iters

    @staticmethod
    def _procrustes(true_svd, perm_svd):
        """
        Apply a Procrustes rotation to resampled SVD results
        This rotation is to correct for:
            - axis rotation(change in order of components)
            - axis reflection(change in sign of loadings)

        See McIntosh & Lobaugh, 2004 - 'Assessment of significance'
        """
        s = true_svd[1]
        v = true_svd[2]
        s_sq = np.diagflat(s)

        u_ = perm_svd[0]
        v_ = perm_svd[2]
        s_sq = perm_svd[1]   # <<<<<< continue from here !!!!!

        n, _, p = np.linalg.svd(np.dot(v.T, v_))
        q = np.dot(n, p.T)

        u_r = np.linalg.multi_dot((u_, s_sq, q))
        v_r = np.linalg.multi_dot((v_, s_sq, q))
        print(v_r)
        try:
            sum_of_squares = np.sum(u_r[:, :] ** 2, 0)
            s_r = np.sqrt(sum_of_squares)
        except RuntimeWarning as err:
            if 'overflow' in err:
                raise OverflowError  # catch overflow to force re-permutation of data

        return u_r, s_r, v_r

    @staticmethod
    def _p_from_perm_mat(obs_vect, perm_array):
        """Calculate p-values columnwise

        Parameters:
        -----------
        obs_vect : numpy array
            Vector of true observations

        perm_array : numpy array
            N x M array of observations obtained through permutation
                N is the number of permutations used
                M is the number of variables

        Returns:
        --------
        p_values : numpy array
            Vector of p-values corresponding to obs_vect
        """
        p_values = np.ndarray(shape=obs_vect.shape)
        for t, true in enumerate(obs_vect):
            perm_data = perm_array[:, t]
            p_values[t] = permutation_p(true, perm_data)
        return p_values

    @staticmethod
    def _bootstrap_z(true_observations, permutation_cube):
        # Calculate "z-scores" from a cube of randomized data
        standard_dev = np.std(permutation_cube, axis=-1)
        standard_err = standard_dev / np.sqrt(permutation_cube.shape[2])
        bootz = np.divide(true_observations, standard_err)

        return bootz

    @staticmethod
    def _mplsc(x, y, center_scale='scale'):
        if center_scale is None:
            clean_x = x
            clean_y = y
        elif center_scale is 'center':
            clean_x, clean_y, _, _, _, _ = _center_scale_xy(x, y, scale=False)
        elif center_scale is 'scale':
            clean_x, clean_y, _, _, _, _ = _center_scale_xy(x, y, scale=True)

        corr_xy = np.dot(clean_x.T, clean_y)
        u, s, v = np.linalg.svd(corr_xy, full_matrices=False)

        return u, s, v.T

    def permutation_tests(self, x, y):
        true_svd = self._mplsc(x, y, center_scale=self.center_scale)
        true_eigs = true_svd[1]

        perm_eigs = np.ndarray(shape=(self.n_iters, len(true_eigs)))
        n = 0
        while n != self.n_iters:
            try:
                perm_y = resample(y, replace=False)  # perm_matrix(y)
                perm_x = resample(x, replace=False)  # perm_matrix(x)
                perm_svd = self._mplsc(perm_x, perm_y, center_scale=self.center_scale)
            except np.linalg.LinAlgError:
                continue  # Re-permute data if SVD doesn't converge

            try:
                _, rotated_eigs, _ = self._procrustes(true_svd, perm_svd)
            except OverflowError:
                continue
            perm_eigs[n, :] = rotated_eigs
            n += 1

        p_values = self._p_from_perm_mat(true_eigs, perm_eigs)
        res = {'original_svd': true_svd,
               'true_eigs': true_eigs,
               'p_values': p_values}

        return res

    def bootstrap_tests(self, x, y):
        true_svd = self._mplsc(x, y, center_scale=self.center_scale)
        true_y_saliences = true_svd[2]
        true_x_saliences = true_svd[0]

        perm_x_cube = np.ndarray(shape=(true_x_saliences.shape[0], true_x_saliences.shape[1], self.n_iters))
        perm_y_cube = np.ndarray(shape=(true_y_saliences.shape[0], true_y_saliences.shape[1], self.n_iters))
        n = 0
        while n != self.n_iters:
            try:
                resamp_y = resample(y, replace=True)
                resamp_x = resample(x, replace=True)
                resamp_svd = self._mplsc(resamp_x, resamp_y, center_scale=self.center_scale)
            except np.linalg.LinAlgError:
                continue  # Re-resample data if SVD doesn't converge

            # TO-DO: Procrustes Rotation
            try:
                rotated_ysals = self._procrustes(true_svd, resamp_svd)[2]
                rotated_xsals = self._procrustes(true_svd, resamp_svd)[0]
            except OverflowError:
                continue

            perm_y_cube[:, :, n] = rotated_ysals  # resamp_svd[0]
            perm_x_cube[:, :, n] = rotated_xsals  # resamp_svd[2]
            n += 1

        x_zscores = self._bootstrap_z(true_x_saliences, perm_x_cube)
        y_zscores = self._bootstrap_z(true_y_saliences, perm_y_cube)

        res = {'y_saliences': true_y_saliences,
               'y_zscores': y_zscores,
               'x_saliences': true_x_saliences,
               'x_zscores': x_zscores}

        return res


def plot_scree(eigs, pvals=None, alpha=.05, percent=True, kaiser=False, fname=None):
    """
    Create a scree plot for factor analysis using matplotlib

    Parameters
    ----------
    eigs : numpy array
        A vector of eigenvalues

    Optional
    --------
    pvals : numpy array
        A vector of p-values corresponding to a permutation test

    alpha : float
        Significance level to threshold eigenvalues (Default = .05)

    percent : bool
        Plot percentage of variance explained

    kaiser : bool
        Plot the Kaiser criterion on the scree

    fname : filepath
        filepath for saving the image
    Returns
    -------
    fig, ax1, ax2 : matplotlib figure handles
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update(mpl.rcParamsDefault)

    percent_var = (np.multiply(100, eigs)) / np.sum(eigs)
    cumulative_var = np.zeros(shape=[len(percent_var)])
    c = 0
    for i, p in enumerate(percent_var):
        c = c+p
        cumulative_var[i] = c

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Scree plot", fontsize='xx-large')
    ax.plot(np.arange(1, len(percent_var)+1), eigs, '-k')
    ax.set_ylim([0, (max(eigs)*1.2)])
    ax.set_ylabel('Eigenvalues', fontsize='xx-large')
    ax.set_xlabel('Factors', fontsize='xx-large')
    if percent:
        ax2 = ax.twinx()
        ax2.plot(np.arange(1, len(percent_var)+1), percent_var, 'ok')
        ax2.set_ylim(0, max(percent_var)*1.2)
        ax2.set_ylabel('Percentage of variance explained', fontsize='xx-large')

    if pvals is not None and len(pvals) == len(eigs):
        p_check = [i for i,t in enumerate(pvals) if t < alpha]
        eigen_check = [e for i, e in enumerate(eigs) for j in p_check if i==j]
        ax.plot(np.add(p_check, 1), eigen_check, 'ob', markersize=10)

    if kaiser:
        ax.axhline(1, color='k', linestyle=':', linewidth=2)

    if fname:
        fig.savefig(fname, bbox_inches='tight')
    else:
        plt.show()

    return fig, ax, ax2


if __name__ == "__main__":
    # Performance testing
    import logging
    import pandas as pd
    import pickle as pkl
    import proj_utils as pu

    logging.basicConfig(level=logging.INFO)

    with open('../data/MEG_power_data.pkl', 'rb') as file:
        meg_data = pkl.load(file)
    meg_list = [meg_data[sess] for sess in list(meg_data)]
    meg_df = pd.concat(meg_list, axis=1)
    x = meg_df.values

    sleep_variables = ['PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7']
    behavior_raw = pd.read_excel('../data/hcp_behavioral.xlsx', index_col=0, sheet_name='cleaned')

    sleep_df = behavior_raw[sleep_variables]
    y = sleep_df.values.astype(float)

    logging.info('%s: Running PLSC' % pu.ctime())
    p = PLSC(n_iters=2)
    pres = p.permutation_tests(x, y)

    eigs = pres['true_eigs']
    pvals = pres['p_values']
    # plot_scree(eigs=eigs, pvals=pvals)

    bres = p.bootstrap_tests(x, y)
    print(bres['y_saliences'])
    print(bres['x_zscores'].shape)

    logging.info('%s: Finished' % pu.ctime())
