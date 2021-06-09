import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.0'
PACKAGE_NAME = 'Top2Phase'
URL = 'https://github.com/moradza/Top2Phase/'
AUTHOR = 'Alireza Moradzadeh'
AUTHOR_EMAIL = 'moradza2@illinois.edu'

LICENSE = 'Apache License 2.0'
DESCRIPTION = 'Prediction of Phase from pairwise distance (topology) using edge-conditioned convolutional graph neural network'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy',
      'tensorflow>=2.1',
      'spektral>=1.0.4',
      'pyboo',
      'pymatgen',
      'tqdm']

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      package_dir={'': 'src'},
      packages=find_packages(where='src'),
      entry_points ={'console_scripts': [ 'top2phase = Top2Phase.main:main','orderparms= Top2Phase.compute_orderprameters:main', 'pwdist=Top2Phase.compute_pairwisedistance:main'  ]},
      python_requires='>3.5',
      )
