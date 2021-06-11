from setuptools import setup, find_packages
import os

VERSION = '0.0.1'
DESCRIPTION = 'Genetic Algorithms for model selection'
# Setting up
setup(
    name="galearn",
    version=VERSION,
    author="Oliver Holl",
    author_email="<oholl@ethz.ch>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy', 'scikit-learn'],
    keywords=['machine learning', 'hyperparameter tuning', 'hyperparameter', 'scikit-learn'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)