import math
from pathlib import Path

from matplotlib import pyplot as plt

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


def deconvolve_precision(p: int, prec_dict: dict[tuple[int, int], float]) -> float:
    """
    sigma(1)^2 = sigma(1, 2)^2 + sigma(1, 3)^2 - sigma(2, 3)^2
    """
    p1 = (p - PLANE_0 + 1) % N_PLANES + PLANE_0
    p2 = (p - PLANE_0 + 2) % N_PLANES + PLANE_0

    pos_pair_1 = prec_dict[_sorted_pair(p, p1)]
    pos_pair_2 = prec_dict[_sorted_pair(p, p2)]
    neg_pair = prec_dict[_sorted_pair(p1, p2)]

    return math.sqrt(pos_pair_1 ** 2 + pos_pair_2 ** 2 - neg_pair ** 2)
