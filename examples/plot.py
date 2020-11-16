import matplotlib.pyplot as plt
import numpy as np


LOGLAMMIN, LOGLAMMAX = 3.5836, 3.9559
EPS = 0.0005
N_FEATURES = 512
WAVE = np.power(10, np.linspace(LOGLAMMIN + EPS, LOGLAMMAX - EPS, N_FEATURES))

LYALPHA = 1216
CIV = 1549
CIII = 1909
MGII = 2796
HBETA = 4862
HALPHA = 6563
LINES = [LYALPHA, CIV, CIII, MGII, HBETA, HALPHA]


def spectrum(ax, flux, wave=WAVE, label=None):
    ax.plot(wave, flux, label=label)

def redshift(ax, z, z_std=None, color="k"):
    ax.set_xlim(10 ** LOGLAMMIN, 10 ** LOGLAMMAX)
    for line in LINES:
        ax.axvline((1 + z) * line, color=color, linestyle="--")
        if z_std is not None:
            lower = (1 + (z - 1.96 * z_std)) * line
            upper = (1 + (z + 1.96 * z_std)) * line
            ax.fill_betweenx([-1, 1], lower, upper, color=color, alpha=0.2)
