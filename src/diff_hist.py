import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from statsmodels.stats.weightstats import DescrStatsW


def _gauss(x: np.ndarray, a: float, mu: float, sigma: float) -> float:
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def get_gauss_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    :param x: time
    :param y: voltage
    :return: mean, std
    """
    # regular statistics
    weighted_stats = DescrStatsW(x, weights=y, ddof=0)
    mean_stat = weighted_stats.mean
    std_stat = weighted_stats.std

    # fitted gaussian statistics
    popt, _ = curve_fit(_gauss, x, y, p0=[1, mean_stat, std_stat]) # second parameter: Cov 
    gauss_mean = popt[1]
    gauss_std = abs(popt[2])

    return gauss_mean, gauss_std, mean_stat, std_stat


def _diff_hist_stats(timestamps_diff: np.ndarray, show: bool, n_bins: int, hist_range: tuple[float, float],
                     hist_alpha: float, hist_label: str, plot_gauss: bool) -> tuple[float, float]:
    hist_data = plt.hist(timestamps_diff, bins=n_bins, range=hist_range, alpha=hist_alpha, label=hist_label)

    # retrieve bins
    bins_x, bins_y = hist_data[1][:-1], hist_data[0]
    x_step = (bins_x[1] - bins_x[0]) / 2
    bins_x += x_step

    mean, std, mean_stat, std_stat = get_gauss_stats(bins_x, bins_y)

    if plot_gauss:
        gauss_y = norm.pdf(bins_x, mean, std)
        # TODO: gauss_y *= popt[0]
        gauss_y *= np.max(bins_y) / np.max(gauss_y) # use other normalisation (popt[0])
        plt.plot(bins_x, gauss_y, 'r--', linewidth=2)

    if show:
        plt.show()

    return mean, std, mean_stat, std_stat


def plot_diff_hist_stats(y_true: np.ndarray, y_pred: np.ndarray, show: bool = True, n_bins: int = 100,
                         hist_range: tuple[float, float] = (-0.5, 0.5), hist_alpha: float = 1., hist_label: str = None,
                         plot_gauss: bool = True, xlabel: str = 'time [ns]'):
    """
    Find the mean and std of a histogram of differences between y_true and y_pred timestamps
    :param y_true: Ground-truth timestamps
    :param y_pred: Predicted timestamps
    :param show: If True: the histogram is shown (plt.show())
    :param n_bins: Number of the histogram bins
    :param hist_range: Range of the histogram
    :param hist_alpha: Alpha of the plotted histogram
    :param hist_label: Label of the histogram
    :param plot_gauss: If True: a fitted Gaussian is plotted with the histogram
    :param xlabel: plot x label
    :return: tuple: (mean, std, mean_stat, std_stat) of the histogram
    """

    # histogram
    timestamps_diff = y_pred - y_true

    plt.xlabel(xlabel)
    return _diff_hist_stats(timestamps_diff, show, n_bins, hist_range, hist_alpha, hist_label, plot_gauss)
