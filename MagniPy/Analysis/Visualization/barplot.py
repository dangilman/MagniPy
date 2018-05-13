import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample

plt.rcParams['axes.linewidth'] = 2.5

plt.rcParams['xtick.major.width'] = 2.5
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.minor.size'] = 2

plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.minor.size'] = 2

def bar_plot(bar_heights, coords, color='0', alpha='1', rebin=None, ax=None):

    if ax is None:
        ax = plt.subplot(111)

    if rebin is not None:
        new = []
        if len(bar_heights) % rebin == 0:
            fac = int(len(bar_heights) / rebin)
            for i in range(0, len(bar_heights), fac):
                new.append(np.mean(bar_heights[i:(i + fac)]))

            bar_heights = new
        else:
            bar_heights = resample(bar_heights, rebin)

    bar_width = np.absolute(coords[-1] - coords[0])*len(bar_heights)**-1
    bar_centers = []
    for i in range(0, len(bar_heights)):
        bar_centers.append(coords[0] + bar_width*(0.5 + i))

    for i, h in enumerate(bar_heights):
        x1, x2, y = bar_centers[i] - bar_width * .5, bar_centers[i] + bar_width * .5, h
        ax.plot([x1, x2], [y, y], color=color)
        ax.fill_between([x1, x2], y, color=color, alpha=alpha)
        ax.plot([x1, x1], [0, y], color=color)
        ax.plot([x2, x2], [0, y], color=color)

    ax.set_xlim(coords[0], coords[-1])
    ax.set_ylim(0, max(bar_heights) * 1.05)
    return ax