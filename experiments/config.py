from dataclasses import dataclass


@dataclass
class Paths:
    dataset_100_200_1: str
    dataset_100_250_5: str
    dataset_schmutter: str
    debug_dataset: str
    ssi_decay: str
    ssi_loaded: str


@dataclass
class Dirs:
    data_dir: str
    schmutter_acc: str
    schmutter_freq: str
    output_dir: str
    log_dir: str
    model_dir: str


@dataclass
class Setup:
    project_dir: str


@dataclass
class tmbConfig:
    paths: Paths
    dirs: Dirs
    setup: Setup
