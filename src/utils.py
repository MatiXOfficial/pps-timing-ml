from matplotlib import pyplot as plt
from pathlib import Path

def save_plt(plot: plt, path: Path | str, **kwargs) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parents[0].mkdir(parents=True, exist_ok=True)
    plt.savefig(path, **kwargs)
