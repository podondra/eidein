# eidein

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/podondra/eidein/master?labpath=examples%2Fexample.ipynb)

interactive tool for dimensionality reduction of data and their visualisation

## installation

Need to install Node.js.

Need to install JupyterLab widget manager (see https://github.com/matplotlib/ipympl):

    $ jupyter labextension install @jupyter-widgets/jupyterlab-manager
    $ jupyter lab build    # maybe

## upload to PyPI

    $ rm dist/*
    $ python setup.py sdist bdist_wheel
    $ twine upload dist/*
