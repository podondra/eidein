# spectraviz

interactive tool for dimensionality reduction of spectra and their visualisation

## installation

Need to install Node.js.

Need to install JupyterLab widget manager (see https://github.com/matplotlib/ipympl):

    $ jupyter labextension install @jupyter-widgets/jupyterlab-manager
    $ jupyter lab build    # maybe

Need to install Parallel HDF5:

    $ pip install mpi4py
    $ CC="mpicc" HDF5_MPI="ON" pip install --no-binary=h5py h5py

## data preparation

1. Run `preparation.ipynb` notebook to create `dataset.hdf5` with identifiers and redshift.
2. Run `data_extr.py`.
3. Run `data_prep.py`.
