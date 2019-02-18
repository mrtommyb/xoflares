#!/usr/bin/env python

import os
import sys
from setuptools import setup

dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dirname, "xoflares"))
from version import __version__ # NOQA

setup(
    name="xoflares",
    version=__version__,
    author="Tom Barclay",
    author_email="tom@tombarclay.com",
    url="https://github.com/mrtommyb/xoflares",
    license="MIT",
    packages=[
        "xoflares",
    ],
    description="Fast & scalable MCMC for all your exoplanet needs",
    # long_description=readme,
    # install_requires=install_requires,
    # package_data={"": ["README.rst", "LICENSE"]},
    # include_package_data=True,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)