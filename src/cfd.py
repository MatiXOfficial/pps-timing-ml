import numpy as np
from scipy import signal
from scipy.ndimage import filters


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
