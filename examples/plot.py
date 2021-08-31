import matplotlib.pyplot as plt
import numpy as np


LOGLAMMIN, LOGLAMMAX = 3.5832, 3.9583
N_FEATURES = 3752
WAVE = np.power(10, np.linspace(LOGLAMMIN, LOGLAMMAX, N_FEATURES))

LYALPHA = 1216
CIV = 1549
CIII = 1909
MGII = 2796
HBETA = 4862
HALPHA = 6563
LINES = [LYALPHA, CIV, CIII, MGII, HBETA, HALPHA]


def spectrum(ax, flux, wave=WAVE, label=None):
    ax.plot(wave, flux, label=label)

def redshift(ax, z, color="k"):
    ax.set_xlim(10 ** LOGLAMMIN, 10 ** LOGLAMMAX)
    for line in LINES:
        ax.axvline((1 + z) * line, color=color, linestyle="--")
