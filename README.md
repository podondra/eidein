# spectraviz

interactive tool for dimensionality reduction of spectra and their visualisation

## Installation

Need to install Node.js.

Need to install JupyterLab widget manager (see https://github.com/matplotlib/ipympl):

    $ jupyter labextension install @jupyter-widgets/jupyterlab-manager
    $ jupyter lab build    # maybe

Need to install Parallel HDF5:

    $ pip install mpi4py
    $ CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/path/to/parallel-hdf5 pip install --no-binary=h5py h5py
