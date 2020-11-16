import h5py
import numpy as np
from sklearn.preprocessing import minmax_scale
from spectres import spectres


N_WAVES = 3724
LOGLAMMIN, LOGLAMMAX = 3.5836, 3.9559
N_FEATURES = 512
EPS = 0.0005
N_VAL, N_TEST = 50000, 50000


with h5py.File("data/dataset.hdf5", "r+") as datafile:
    ids = datafile["id"][:]
    fluxes = datafile["flux"][:]
    z = datafile["z"][:]

    # resample
    loglam = np.linspace(LOGLAMMIN, LOGLAMMAX, N_WAVES)
    # EPS else will get nans in output
    new_loglam = np.linspace(LOGLAMMIN + EPS, LOGLAMMAX - EPS, N_FEATURES)
    X = spectres(new_loglam, loglam, fluxes, verbose=True).astype(np.float32, copy=False)
    X = minmax_scale(X, feature_range=(-1, 1), axis=1, copy=False)
    X_dset = datafile.create_dataset("X", data=X)

    # split into training, validation and test set (sizes almost according to ILSVRC)
    # seed from random.org
    rng = np.random.default_rng(seed=83)
    n = ids.shape[0]
    rnd_idx = rng.permutation(n)
    n_tr = n - N_VAL - N_TEST
    idx_tr, idx_va, idx_te = rnd_idx[:n_tr], rnd_idx[n_tr:n_tr + N_VAL], rnd_idx[n_tr + N_VAL:]

    for name, idx in [("tr", idx_tr), ("va", idx_va), ("te", idx_te)]:
        datafile.create_dataset("id_" + name, data=ids[idx])
        datafile.create_dataset("X_" + name, data=X[idx])
        datafile.create_dataset("z_" + name, data=z[idx])
