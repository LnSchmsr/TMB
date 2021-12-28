import torch
import ray
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.progress_reporter import CLIReporter
from tmb.train import hypersearch
from tmb.model import FC_FFT
from typing import Dict
import os
from pathlib import Path


def ray_search(
    dataset: np.array,
    criterion,
    hypersearch_config: Dict,
    hydra_config,
    best_model_name: str,
    local_dir: str = None,
    resources_per_trial: Dict[str, float] = None,
    num_samples: int = 10,
    num_epochs: int = 200,
):
    """
    Performs a hypersearch with the hypersearch method in train.py
    Args:
        dataset (np.array): Training dataset
        criterion (Any): Lossfunction for training
        logdir (str): Logpath for tensorboard
        modeldir (str): Path for saving best model
        num_samples (int, optional): Number of models to train. Defaults to 10.
        num_epochs (int, optional): Number of epochs for training. Defaults to 200.
        resources_per_trial (Dict[str, float], optional): Resources used for each trial. Defaults to None.
        local_dir (str, optional): Defines save path for ray files. Defaults to None.
    """

    # Define variables
    log_dir = hydra_config.dirs.log_dir
    model_dir = hydra_config.dirs.model_dir
    output_dir = hydra_config.dirs.output_dir
    project_dir = hydra_config.setup.project_dir
    ray.init()
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["valid_loss", "train_loss", "training_iteration"]
    )
    result = tune.run(
        tune.with_parameters(
            hypersearch,
            dataset=dataset,
            num_epochs=num_epochs,
            criterion=criterion,
            log_dir=log_dir,
            model_name=best_model_name,
            model_dir=model_dir,
            output_dir=output_dir,
            project_dir=project_dir,
        ),
        config=hypersearch_config,
        mode="min",
        num_samples=num_samples,
        progress_reporter=reporter,
        scheduler=scheduler,
        resources_per_trial=resources_per_trial,
        local_dir=local_dir,
    )

    best_trial = result.get_best_trial("valid_loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print(
        "Best trial final validation loss: {}".format(
            best_trial.last_result["valid_loss"]
        )
    )
    print(
        "Best trial final training loss: {}".format(
            best_trial.last_result["train_loss"]
        )
    )

    model = FC_FFT(
        log_path=Path(f"{project_dir}/{log_dir}/{best_model_name}").resolve(),
        input_size=1000,
    )

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(
        os.path.join(best_checkpoint_dir, "checkpoint")
    )
    model.load_state_dict(model_state)
    model_name = (
        best_model_name
        + "_"
        + str(best_trial.config["lr"])
        + "_bs_"
        + str(best_trial.config["batch_size"])
    )
    torch.save(model.state_dict(), model_dir + model_name + ".pt")
