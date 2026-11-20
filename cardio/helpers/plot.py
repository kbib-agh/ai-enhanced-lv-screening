import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def create_fig(*args, width=2000, height=1000, flatten=True, dpi=100) -> tuple[Figure, Axes | tuple[Axes]]:
    fig, axs = plt.subplots(*args, figsize=(width / dpi, height / dpi), dpi=dpi)
    if flatten and isinstance(axs, np.ndarray):
        axs = axs.flatten()
    return fig, axs
