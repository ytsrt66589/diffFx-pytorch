#!/usr/bin/env python3
# Inspired from https://github.com/kennethreitz/setup.py

from pathlib import Path
from setuptools import setup, find_packages

NAME = "diffFx-pytorch"
DESCRIPTION = "Differentiable audio effect processors in PyTorch"
URL = "https://github.com/ytsrt66589/diffFx-pytorch"
EMAIL = "ytsrt66589@gmail.com"
AUTHOR = "Yen-Tung (Arthur) Yeh"
REQUIRES_PYTHON = ">=3.10.0"
VERSION = "0.0.1"

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=["diffFx_pytorch"],
    install_requires=["torch", "numpy", "torchlpc>=0.4", "torchcomp", "torchaudio",  "scipy"],
    extras_require={"extra": ["matplotlib",  "librosa",]},
    include_package_data=True,
    license="Apache License 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
    ],
)