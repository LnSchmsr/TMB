#Train Measures Bridge (TMB)

This is an implementation of the paper "Deep learning based indirect monitoring to identify bridge resonant frequencies using sensors on a passing train" usingl Python 3, Pytorch and Ray. The model predicts the frequency of an acceleration signal using regression.

The repository contains:
- Source code of the model created with Pytorch 3
- Training code for the simulations and real data on the Schmutter bridge.
- The experiments used, which are discussed in the chapters.

The code is documented and designed to make the paper more comprehensible and easily extendable. If these repros are used in your research, please consider citing this repository.
# Installation
- Clone this repository
- Install dependencies

`pip install -r requirements.txt`

- Run setup from the repository root directory

`python3 setup.py install`

# Code Overview
The code is organised as follows:
- **data:** Contains the training data used for the simulation (*dataset_0_30_100_200_1.npz*), as well as for the Schmutter. Furthermore, a smaller dataset (*debug_dataset.npz*) is included for simple debugging.
- **experiments:** Contains the experiments used:
- **models:**Contains the models used.
- **tmb:** Contains the code of the package.
