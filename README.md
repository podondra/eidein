# eidein

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
