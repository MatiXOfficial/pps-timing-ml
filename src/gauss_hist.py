import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.optimize import curve_fit
from scipy.stats import norm
from statsmodels.stats.weightstats import DescrStatsW


def _gauss(x: np.ndarray, a: float, mu: float, sigma: float) -> float:
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def get_gauss_stats(x: np.ndarray, y: np.ndarray, a_0: float = None, mean_0: float = None, std_0: float = None) -> \
        tuple[float, float, float, float]:
    """
    :param x: e.g. time
    :param y: e.g. voltage
    :param a_0: base a value for the curve fit
    :param mean_0: base mean value for the curve fit
    :param std_0: base std value for the curve fit
    :return: gauss_a, mean, std, pcov
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

    return a, gauss_mean, gauss_std, pcov


def compute_fwhm(x: np.ndarray, y: np.ndarray, hist_range, n_interp):
    """
    Use a spline interpolation of degree 3 to find full width at half maximum
    :param x:
    :param y:
    :param hist_range: (range_min, range_max)
    :param n_interp: number of interpolation points
    :return: fwhm, first_cross, last_cross, half_max
    """
    half_max = np.max(y) / 2.

    s = splrep(x, y, k=3)

    x2 = np.linspace(hist_range[0], hist_range[1], n_interp)
    y2 = splev(x2, s)

    y_cross = y2 > half_max
    first_cross = x2[np.argmax(y_cross)]
    last_cross = x2[len(x2) - np.argmax(y_cross[::-1]) - 1]
    fwhm = last_cross - first_cross
    return fwhm, first_cross, last_cross, half_max


def plot_gauss_hist(x: np.ndarray, show: bool = True, n_bins: int = 100, hist_range: tuple[float, float] = (-0.5, 0.5),
                    hist_alpha: float = 1., hist_label: str = None, plot_gauss: bool = True, plot_fwhm: bool = True,
                    xlabel: str = 'time [ns]') -> tuple[float, float, float, float]:
    """
    Find the mean and std of a histogram based on x. Optionally plot a Gaussian fitted to the histogram.
    :param x: histogram data
    :param show: If True: the histogram is shown (plt.show())
    :param n_bins:
    :param hist_range:
    :param hist_alpha:
    :param hist_label:
    :param plot_gauss: If True: the fitted Gaussian is plotted with the histogram
    :param plot_fwhm: If True: full width at half maximum is marked
    :param xlabel:
    :return: Gaussian mean, Gaussian std, covariance matrix of the Gaussian fit, fwhm
    """
    hist_data = plt.hist(x, bins=n_bins, range=hist_range, alpha=hist_alpha, label=hist_label)

    # retrieve bins
    bins_x, bins_y = hist_data[1][:-1], hist_data[0]
    x_step = (bins_x[1] - bins_x[0]) / 2
    bins_x += x_step

    a, mean_gauss, std_gauss, pcov = get_gauss_stats(bins_x, bins_y)
    fwhm, first_cross, last_cross, half_max = compute_fwhm(bins_x, bins_y, hist_range, n_bins * 100)

    if plot_gauss:
        gauss_y = norm.pdf(bins_x, mean_gauss, std_gauss)
        gauss_y *= a / np.max(gauss_y)
        plt.plot(bins_x, gauss_y, 'r--', linewidth=2)

    # if plot_fwhm:
    #     plt.plot([first_cross, last_cross], [half_max, half_max], c='purple', linewidth=2)

    plt.xlabel(xlabel)
    if show:
        plt.show()

    return mean_gauss, std_gauss, pcov, fwhm


def plot_diff_hist_stats(y_true: np.ndarray, y_pred: np.ndarray, show: bool = True, n_bins: int = 100,
                         hist_range: tuple[float, float] = (-0.5, 0.5), hist_alpha: float = 1., hist_label: str = None,
                         plot_gauss: bool = True, plot_fwhm: bool = True, xlabel: str = 'time [ns]') -> \
        tuple[float, float, float, float]:
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
    :param plot_fwhm: If True: full width at half maximum is marked
    :param xlabel: plot x label
    :return: Gaussian mean, Gaussian std, covariance matrix of the Gaussian fit, fwhm
    """
    timestamps_diff = y_pred - y_true
    return plot_gauss_hist(timestamps_diff, show, n_bins, hist_range, hist_alpha, hist_label, plot_gauss, plot_fwhm,
                           xlabel)
