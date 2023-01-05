import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from statsmodels.stats.weightstats import DescrStatsW


def _gauss(x: np.ndarray, a: float, mu: float, sigma: float) -> float:
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def get_gauss_stats(x: np.ndarray, y: np.ndarray, a_0: float = None, mean_0: float = None, std_0: float = None) -> \
        tuple[float, float, float, float, float, float]:
    """
    :param x: e.g. time
    :param y: e.g. voltage
    :param a_0: base a value for the curve fit
    :param mean_0: base mean value for the curve fit
    :param std_0: base std value for the curve fit
    :return: gauss_a, mean, std, mean_stat, std_stat, pcov
    """
    # regular statistics
    weighted_stats = DescrStatsW(x, weights=y, ddof=0)
    mean_stat = weighted_stats.mean
    std_stat = weighted_stats.std

    if a_0 is None:
        a_0 = np.max(y)
    if mean_0 is None:
        mean_0 = mean_stat
    if std_0 is None:
        std_0 = std_stat

    # fitted gaussian statistics
    popt, pcov = curve_fit(_gauss, x, y, p0=[a_0, mean_0, std_0])
    a = popt[0]
    gauss_mean = popt[1]
    gauss_std = abs(popt[2])

    return a, gauss_mean, gauss_std, mean_stat, std_stat, pcov


def plot_gauss_hist(x: np.ndarray, show: bool = True, n_bins: int = 100, hist_range: tuple[float, float] = (-0.5, 0.5),
                    hist_alpha: float = 1., hist_label: str = None, plot_gauss: bool = True,
                    xlabel: str = 'time [ns]') -> tuple[float, float, float, float, float]:
    """
    Find the mean and std of a histogram based on x. Optionally plot a Gaussian fitted to the histogram.
    :param x: histogram data
    :param show: If True: the histogram is shown (plt.show())
    :param n_bins:
    :param hist_range:
    :param hist_alpha:
    :param hist_label:
    :param plot_gauss: If True: the fitted Gaussian is plotted with the histogram
    :param xlabel:
    :return: tuple of 5 numbers: Gaussian mean, Gaussian std, stat mean, stat std, covariance matrix of the Gaussian fit
    """
    hist_data = plt.hist(x, bins=n_bins, range=hist_range, alpha=hist_alpha, label=hist_label)

    # retrieve bins
    bins_x, bins_y = hist_data[1][:-1], hist_data[0]
    x_step = (bins_x[1] - bins_x[0]) / 2
    bins_x += x_step

    a, mean, std, mean_stat, std_stat, pcov = get_gauss_stats(bins_x, bins_y)

    if plot_gauss:
        gauss_y = norm.pdf(bins_x, mean, std)
        gauss_y *= a / np.max(gauss_y)
        plt.plot(bins_x, gauss_y, 'r--', linewidth=2)

    plt.xlabel(xlabel)
    if show:
        plt.show()

    return mean, std, mean_stat, std_stat, pcov


def plot_diff_hist_stats(y_true: np.ndarray, y_pred: np.ndarray, show: bool = True, n_bins: int = 100,
                         hist_range: tuple[float, float] = (-0.5, 0.5), hist_alpha: float = 1., hist_label: str = None,
                         plot_gauss: bool = True, xlabel: str = 'time [ns]') -> \
        tuple[float, float, float, float, float]:
    """
    Find the mean and std of a histogram of differences between y_true and y_pred timestamps. Optionally plot a
    Gaussian fitted to the histogram.
    :param y_true: Ground-truth timestamps
    :param y_pred: Predicted timestamps
    :param show: If True: the histogram is shown (plt.show())
    :param n_bins:
    :param hist_range:
    :param hist_alpha:
    :param hist_label:
    :param plot_gauss: If True: a fitted Gaussian is plotted with the histogram
    :param xlabel: plot x label
    :return: tuple of 5 numbers: Gaussian mean, Gaussian std, stat mean, stat std, covariance matrix of the Gaussian fit
    """
    timestamps_diff = y_pred - y_true
    return plot_gauss_hist(timestamps_diff, show, n_bins, hist_range, hist_alpha, hist_label, plot_gauss, xlabel)
