"""
The build/compilations setup
>> pip install -r requirements.txt
>> python setup.py install
"""
import pip
import logging
import pkg_resources
import os
import yaml
from pathlib import Path

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def _parse_requirements(file_path):
    pip_ver = pkg_resources.get_distribution('pip').version
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path,
                                         session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]


# parse_requirements() returns generator of pip.req.InstallRequirement objects
try:
    install_reqs = _parse_requirements("requirements.txt")
except Exception:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = []

setup(
    name='tmb',
    version='1.0',
    url='https://github.com/matterport/Mask_RCNN',
    author='Matterport',
    author_email='waleed.abdulla@gmail.com',
    license='MIT',
    description='Implementation and code for the paper "Deep learning based indirect monitoring to identify bridge resonant frequencies using sensors on a passing train"',
    packages=["tmb"],
    install_requires=install_reqs,
    include_package_data=True,
    python_requires='>=3.8',
    long_description="""This is the implementation and code for Paper  
    "Deep learning based indirect monitoring to identify bridge resonant frequencies using sensors on a passing train" for IABMAS 2022. 
    The implementation was done with Python 3, Pytorch and Ray.
    """,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: SHM",
        "Topic :: Scientific/Engineering :: Structural Health Monitoring",
        'Programming Language :: Python :: 3.8',
    ],
    keywords="train measures bridge tmb IABMAS2022",
)

pd = os.getcwd()

#Create dir
path = f'{pd}/experiments/conf'
if not Path(path).exists():
    os.mkdir(f'{pd}/experiments/conf')

#Create yaml
yaml_input={
        'paths':
        {
            'dataset_100_200_1': 'dataset_0_30_100_200_1.npz',
            'dataset_100_250_5': 'dataset_100_250_5.npz',
            'debug_dataset': 'debug_dataset.npz',
            'dataset_schmutter': 'dataset_schmutter.npz',
            'ssi_decay': 'SSI_decay.txt',
            'ssi_loaded': 'SSI_loaded.txt'
        },

        'dirs':
        {
            'data_dir': '${hydra:runtime.cwd}/data',
            'schmutter_acc': '${hydra:runtime.cwd}/data/schmutter_acc',
            'schmutter_freq': '${hydra:runtime.cwd}/data/schmutter_freq',
            'output_dir': './output',
            'log_dir': './run',
            'model_dir': './models'
        },
        'setup': 
        {
            'project_dir':pd
        },
        'hydra':
        {
            'run':
                {'dir': '.'},
            'output_subdir': 'null',
            'sweep':
                {'dir': '.'}
        }
}
with open('./experiments/conf/config.yaml', 'w') as outfile:
    yaml.dump(yaml_input, outfile, default_flow_style=False)