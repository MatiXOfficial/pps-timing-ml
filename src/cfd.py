import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy import signal
from scipy.ndimage import filters

from src.network_utils import plot_difference_hist


class CFD:
    """
    Constant fraction discriminator implemented using the normalized threshold algorithm
    """

    def __init__(self, n_baseline: int = 20, threshold: float = 0.3, smooth: bool = False, kernel_width: float = 10,
                 kernel_sigma: float = 5, log: bool = False) -> None:
        """
        :param n_baseline: number of first samples taken into account when calculating the baseline
        :param threshold: threshold to cross by the signal
        :param smooth: If True: apply a gaussian kernel to smooth the series
        :param kernel_width: Kernel width for gaussian smoothing
        :param kernel_sigma: Kernel sigma for gaussian smoothing
        :param log: if True, log messages are printed
        """
        self.n_baseline = n_baseline
        self.threshold = threshold
        self.smooth = smooth
        self.kernel_width = kernel_width
        self.kernel_sigma = kernel_sigma
        self.log = log

    def predict(self, X: np.ndarray, Y: np.ndarray, baseline_threshold: float = None) -> None | float:
        """
        Find the timestamp
        :param X: x axis data (time)
        :param Y: y axis data (ampl)
        :param baseline_threshold: max - min threshold. if None: np.std(Y[:self.n_baseline]) * 6
        :return: timestamp
        """
        if X is None:
            X = np.arange(len(Y))
        if baseline_threshold is None:
            baseline_threshold = np.std(Y[:self.n_baseline]) * 6

        samples = Y.astype(float)

        # if max - min < baseline_threshold there is no peak for sure
        if np.max(samples) - np.min(samples) < baseline_threshold:
            if self.log:
                print('max - min < threshold')
            return None

        # gaussian smoothing
        if self.smooth:
            kernel = signal.gaussian(self.kernel_width, self.kernel_sigma)
            samples = filters.convolve(samples, kernel)

        # work only with positive and normalized signals
        samples -= np.mean(samples[:self.n_baseline])
        if abs(np.max(samples)) < abs(np.min(samples)):
            samples /= np.min(samples)
        else:
            samples /= np.max(samples)

        # Find the moment of crossing the threshold
        i = 0
        while i < len(samples) and samples[i] <= self.threshold:
            i += 1
        if i == len(samples):
            if self.log:
                print('Signal is not crossing the threshold')
            return None

        # Apply fit between the two points closest to the crossing and return the timestamp
        x1 = X[i]
        x2 = X[i - 1]
        v1 = samples[i]
        v2 = samples[i - 1]
        return x1 + (x2 - x1) * (self.threshold - v1) / (v2 - v1)


def find_optimal_cfd_threshold(thresholds, n_baseline, X, y_true, x_time, n_jobs=8, plot=True, log=True) -> float:
    def compute_cfd_resolution(threshold):
        print(f'Processing threshold={threshold:0.2f}...')
        cfd = CFD(n_baseline=n_baseline, threshold=threshold)

        y_pred = []
        for x in X:
            y_pred.append(cfd.predict(x_time, x))

        y_pred = np.array(y_pred)
        std_cfd, _, _ = plot_difference_hist(y_true, y_pred, show=False)
        std_stat_cfd = np.std(y_pred - y_true)
        max_diff = max(abs(y_pred - y_true))

        return threshold, std_cfd, std_stat_cfd, max_diff

    cfd_all_stds = Parallel(n_jobs=n_jobs)(
        delayed(compute_cfd_resolution)(threshold) for threshold in thresholds)

    cfd_stds = {key: v for key, v, _, _ in cfd_all_stds}
    cfd_stat_stds = {key: v for key, _, v, _ in cfd_all_stds}
    cfd_max_diffs = {key: v for key, _, _, v in cfd_all_stds}

    # optimal_cfd_threshold = list(cfd_stds.keys())[np.argmin(list(cfd_stds.values()))]
    optimal_cfd_threshold = list(cfd_stat_stds.keys())[np.argmin(list(cfd_stat_stds.values()))]
    # optimal_cfd_threshold = list(cfd_max_diffs.keys())[np.argmin(list(cfd_max_diffs.values()))]

    if plot:
        plt.figure(figsize=(11, 7))

        plt.subplot(2, 2, 1)
        plt.plot(cfd_stds.keys(), cfd_stds.values(), marker='.')
        plt.title('Gauss - CFD resolution (train dataset)')
        plt.xlabel('CFD threshold')
        plt.ylabel('CFD resolution')
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(cfd_stat_stds.keys(), cfd_stat_stds.values(), marker='.')
        plt.title('Stat - CFD resolution (train dataset)')
        plt.xlabel('CFD threshold')
        plt.ylabel('CFD resolution')
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(cfd_max_diffs.keys(), cfd_max_diffs.values(), marker='.')
        plt.title('Max CFD differences')
        plt.xlabel('CFD threshold')
        plt.ylabel('Max difference')
        plt.grid()

        plt.tight_layout()
        plt.show()

    if log:
        print(f'Optimal CFD threshold: {optimal_cfd_threshold:0.3f}')

    return optimal_cfd_threshold
