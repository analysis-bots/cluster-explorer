import os

from setuptools import setup, find_packages

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version():
    for line in read('src/cluster_explorer/__init__.py').splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

def get_long_description():
    with open('README.md', 'r') as fh:
        return fh.read()

setup(
    name='cluster_explorer',
    version='1.0.2',#get_version(),
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    long_description_content_type="text/markdown",
    long_description=get_long_description(),  # Long description read from the readme file
    project_urls={
        'Git': 'https://github.com/analysis-bots/cluster-explorer',
    },
    install_requires=[
        'wheel',
        'pandas>=2.2.3',
        'numpy>=2.1.3',
        'scikit-learn>=1.6.0',
        'optbinning>=0.20.1'
    ]
)
