#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

__version__ = '0.0.1'
__author__ = "Jordie Shier"
__contact__ = "jordieshier@gmail.com"
__url__ = ""
__license__ = "MIT"


with open("README.md", "r", encoding='utf-8') as f:
    readme = f.read()

setup(
    name='uvic-music-extractor',
    version=__version__,
    author=__author__,
    author_email=__contact__,
    description='Audio feature extractors for research on musical audio conducted at the University of Victoria',
    long_description=readme,
    long_description_content_type='text/markdown',
    url=__url__,
    licence=__license__,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={
        '': [],
    },
    scripts=[
        'scripts/uvic_music_extractor',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'six',
        'tqdm'
    ],
    extras_require={
        'dev': [
        ],
    }
)
