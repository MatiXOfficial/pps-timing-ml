import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.dataset import PlaneChannel

PLANES = [1, 2, 3]
PLANE_0 = 1
N_PLANES = 3


def save_plt(path: Path | str, **kwargs) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parents[0].mkdir(parents=True, exist_ok=True)
    plt.savefig(path, **kwargs)


def _sorted_pair(a, b):
    return (a, b) if b > a else (b, a)


def deconvolve_precision(p: int, prec_dict: dict[PlaneChannel, float]) -> float:
    """
    sigma(1)^2 = sigma(1, 2)^2 + sigma(1, 3)^2 - sigma(2, 3)^2
    """
    p1 = (p - PLANE_0 + 1) % N_PLANES + PLANE_0
    p2 = (p - PLANE_0 + 2) % N_PLANES + PLANE_0

    pos_pair_1 = prec_dict[_sorted_pair(p, p1)]
    pos_pair_2 = prec_dict[_sorted_pair(p, p2)]
    neg_pair = prec_dict[_sorted_pair(p1, p2)]

    return math.sqrt((pos_pair_1 ** 2 + pos_pair_2 ** 2 - neg_pair ** 2) / 2)


def deconvolve_precisions(prec_dict: dict[tuple[PlaneChannel, PlaneChannel], float]) -> dict[PlaneChannel, float]:
    channel_mutual_precisions: dict[int, dict[tuple[int, int], float]] = defaultdict(dict)
    for ((x_p, x_ch), (y_p, y_ch)), precision in prec_dict.items():
        assert x_ch == y_ch
        channel_mutual_precisions[x_ch][(x_p, y_p)] = precision

    deconvolved_precisions = {}
    for ch, prec_dict in channel_mutual_precisions.items():
        for p in PLANES:
            deconvolved_precisions[(p, ch)] = deconvolve_precision(p, prec_dict)

    return deconvolved_precisions


def print_pairwise_precisions(precisions: dict[tuple[PlaneChannel, PlaneChannel], float]) -> None:
    channel_mutual_precisions: dict[int, dict[PlaneChannel, float]] = defaultdict(dict)
    for ((x_p, x_ch), (y_p, y_ch)), precision in precisions.items():
        assert x_ch == y_ch
        channel_mutual_precisions[x_ch][(x_p, y_p)] = precision

    for ch, data in channel_mutual_precisions.items():
        for (p_1, p_2), prec in data.items():
            print(f'ch {ch:2}: (p{p_1} vs p{p_2}): {prec:0.2f} ps')


def scatter_random(data, classes, class_limit=None, size=1, seed=42):
    if isinstance(data, list):
        data = np.array(data)
    if isinstance(classes, list):
        classes = np.array(classes)

    np.random.seed(seed)
    plot_idx = np.random.permutation(data.shape[0])
    data = data[plot_idx]
    classes = classes[plot_idx]

    if class_limit is not None:
        data_to_use = []
        classes_to_use = []
        for class_ in np.unique(classes):
            data_to_use.extend(data[classes == class_][:class_limit])
            classes_to_use.extend([class_] * len(data[classes == class_][:class_limit]))

        data = np.array(data_to_use)
        classes = np.array(classes_to_use)

        plot_idx = np.random.permutation(data.shape[0])
        data = data[plot_idx]
        classes = classes[plot_idx]

    colors, labels = pd.factorize(classes)

    fig, ax = plt.subplots()
    sc = ax.scatter(data[:, 0], data[:, 1], c=colors, cmap='tab10', s=size, vmax=9)

    h = lambda c: plt.Line2D([], [], color=c, ls="", marker="o")
    plt.legend(handles=[h(sc.cmap(sc.norm(i))) for i in range(len(labels))], labels=list(labels))

    plt.tight_layout()
