from math import ceil

from astropy.io import fits
import h5py
from mpi4py import MPI
import numpy as np


N_WAVES = 3724
LOGLAMMIN, LOGLAMMAX = 3.5836, 3.9559

FILE_TEMPLATE = "data/DR16Q_Superset_v3/{:04d}/spec-{:04d}-{}-{:04d}.fits"


comm = MPI.COMM_WORLD    # communicator which links all our processes together
rank = comm.Get_rank()    # number which identifies this process
size = comm.Get_size()    # number of processes in a communicator

with h5py.File("data/dataset.hdf5", "r+", driver="mpio", comm=comm) as datafile:
    ids = datafile["id"][:]

    # divide data between processes
    n = ids.shape[0]
    chunk = ceil(n / size)
    start = rank * chunk
    end = start + chunk if start + chunk <= n else n

    fluxes = datafile.create_dataset("flux", shape=(n, N_WAVES), dtype=np.float32)
    for i in range(start, end):
        plate, mjd, fiber = ids[i]
        filepath = FILE_TEMPLATE.format(plate, plate, mjd, fiber)
        with fits.open(filepath) as hdulist:
            data = hdulist[1].data
            loglam = data["loglam"]
            fluxes[i] = data["flux"][(LOGLAMMIN <= loglam) & (loglam <= LOGLAMMAX)]
