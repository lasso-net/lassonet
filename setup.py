from setuptools import setup
from pathlib import Path


def read(fname):
    return (Path(__file__).parent / fname).open().read()


setup(
    name="lassonet",
    version="0.0.3",
    author="Louis Abraham, Ismael Lemhadri",
    author_email="louis.abraham@yahoo.fr, lemhadri@stanford.edu",
    license="MIT",
    description="Reference implementation of LassoNet",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/ilemhadri/lassonet",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    packages=["lassonet"],
    install_requires=["torch", "scikit-learn"],
    tests_require=["pytest"],
    python_requires=">=3.6.5",
)
