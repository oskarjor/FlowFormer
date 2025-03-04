#!/usr/bin/env python

import os
from setuptools import find_packages, setup

install_requires = [
    "torch>=1.11.0",
    "torchvision>=0.11.0",
    "lightning-bolts",
    "matplotlib",
    "numpy<2.0.0",  # Due to pandas incompatibility
    "scipy",
    "scikit-learn",
    "scprep",
    "scanpy",
    "torchdyn",
    "pot",
    "torchdiffeq",
    "absl-py",
    "clean-fid",
    "Pillow",  # for VAR image loading
]

version_py = os.path.join(os.path.dirname(__file__), "torchcfm", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()
readme = open("README.md", encoding="utf8").read()

setup(
    name="local-packages",  # or choose another name
    version=version,
    description="Local packages including Conditional Flow Matching and VAR",
    author="Oskar Jørgensen",
    author_email="oskarjorgensen@gmail.com",
    url="https://github.com/oskarjor/FlowFormer",
    install_requires=install_requires,
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    extras_require={"forest-flow": ["xgboost", "scikit-learn", "ForestDiffusion"]},
)
