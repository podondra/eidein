import numpy as np


LOGLAMMIN, LOGLAMMAX = 3.5832, 3.9583
N_FEATURES = 3752
LAM = np.power(10, np.linspace(LOGLAMMIN, LOGLAMMAX, N_FEATURES))
LINES = [
    (1033.82, "O VI"),
    (1215.24, "Lyα"),
    (1549.48, "C IV"),
    (1908.734, "C III"),
    (2326.0, "C II"),
    (2799.117, "Mg II"),
    (4102.89, "HΔ"),
    (4341.68, "Hγ"),
    (4862.68, "Hβ"),
    (6564.61, "Hα")]
ARROWPROPS = {"arrowstyle": "-|>", "facecolor": "black"}


def z2lam_emit(z, lam_obsv):
    return lam_obsv / (1 + z)

def plot_function(ax, filename, flux, y, label, lam=LAM):
    label_str = "{}\n$\hat{{z}} = {:.2f}$\n$z = {:.2f}$ (shown)".format(
        filename, y, label)
    
    lam_emit = [z2lam_emit(label, l) for l in lam]
    ax.plot(lam_emit, flux, label=label_str)
    ax.legend()
    ax.set_xlabel("Rest Frame Wavelength [Å]")
    ax.set_ylabel("Flux [10$^{-17}$ erg cm$^{-2}$ s$^{-1}$ Å$^{-1}$]")
    
    # plot spectral lines
    for line, name in LINES:
        ax.annotate(
            name,
            xy=(line, 0), xytext=(line, -2),
            arrowprops=ARROWPROPS,
            horizontalalignment='center')
