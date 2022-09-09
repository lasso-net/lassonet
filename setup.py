from setuptools import setup
from pathlib import Path


def read(fname):
    return (Path(__file__).parent / fname).open().read()


setup(
    name="lassonet",
    version="0.0.11",
    author="Louis Abraham, Ismael Lemhadri",
    author_email="louis.abraham@yahoo.fr, lemhadri@stanford.edu",
    license="MIT",
    description="Reference implementation of LassoNet",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/lasso-net/lassonet",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    packages=["lassonet"],
    install_requires=[
        "torch >= 1.11",
        "scikit-learn",
        "matplotlib",
        "sortedcontainers",
        "tqdm",
    ],
    tests_require=["pytest"],
    python_requires=">=3.8",
)
