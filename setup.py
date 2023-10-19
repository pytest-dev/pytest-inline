#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os

from setuptools import setup
from setuptools import find_packages


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding="utf-8").read()


setup(
    name="pytest-inline",
    version="1.0.5",
    author="Yu Liu",
    author_email="yuki.liu@utexas.edu",
    maintainer="Alan Han, Yu Liu, Pengyu Nie, Zachary William Thurston",
    maintainer_email="yuki.liu@utexas.edu",
    license="MIT",
    url="https://github.com/pytest-dev/pytest-inline",
    description="A pytest plugin for writing inline tests.",
    long_description_content_type="text/markdown",
    long_description=read("README.md"),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=["pytest>=7.0.0"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: Pytest",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    entry_points={
        "pytest11": [
            "inline = inline.plugin",
        ],
    },
    extras_require={
        "dev": ["flake8", "black", "tox"],
    },
)
