
Data analysis algorithm for unmodeled gravitational-wave transient signals
**************************************************************************

Installation instructions with pip::

  python3 -m venv pyburst_dev
  source pyburst_dev/bin/activate
  (venv) pip install ipykernel
  (venv) pip install < pyburst/requirements.txt
  (venv) ipython kernel install --user --name=pyburst
  (venv) cd pyburst/
  (venv) pip install -e .
  
Installation instructions with conda::

  conda create --name pyburst_dev python=3.7
  conda activate pyburst_dev
  (pyburst_dev) conda config --append channels conda-forge
  (pyburst_dev) conda config --append channels ltfatpy
  (pyburst_dev) conda install --file requirements.txt

Note: developers should execute
  (pyburst_dev) conda-develop pyburst
