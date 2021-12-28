"""
Example hypersearch script.
"""

# Imports
import hydra
import torch
import logging
import os

from sklearn.preprocessing import MaxAbsScaler
from config import tmbConfig
from hydra.core.config_store import ConfigStore
from tmb.hyper import ray_search
from tmb.data import AccelerationDataSetFFT
from ray import tune
from pathlib import Path


# Setup Logger
logging.basicConfig(level=logging.INFO)

# Create ConfigStore
cs = ConfigStore.instance()
cs.store(name="tmb_config", node=tmbConfig)


@hydra.main(config_path="conf", config_name="config")
def hyper(cfg: tmbConfig) -> None:
    hydra.output_subdir = "null"
    print(os.getcwd())
    logging.info("Loading the dataset...")
    accPath = Path(f"{cfg.dirs.data_dir}/{cfg.paths.dataset_100_200_1}")
    dataset = AccelerationDataSetFFT(accPath, snr=0, x_scaler=MaxAbsScaler())
    logging.info("Transformed dataset")

    # This config defines the hyperparameter to be optimised.
    # The names of the parameters must stay the same. The values can be changed.
    # If you need more parameters for the hypersearch. You need to add them in the same
    # way is lr and batch_size. Also in the given methods.

    config = {
        "lr": tune.loguniform(0.0005, 0.001),
        "batch_size": tune.choice([512, 1024, 2048]),
    }

    # perform hypersearch
    ray_search(
        dataset=dataset,
        hypersearch_config=config,
        hydra_config=cfg,
        best_model_name="Test",
        criterion=torch.nn.modules.loss.MSELoss(),
        local_dir=r"D:\ray",
        resources_per_trial={"cpu": 4.0},
        num_samples=1,
        num_epochs=3,
    )


if __name__ == "__main__":
    hyper()
