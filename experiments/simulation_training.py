"""
Example Training and Test Script.
"""

import csv
import logging as log
import torch
import hydra
import logging

from tmb.model import FC_FFT
from tmb.data import AccelerationDataSetFFT, get_train_valid_test_loader
from tmb.train import train_valid
from torch.optim import Adam
from torch.nn.modules.loss import MSELoss
from tqdm import tqdm
from sklearn.preprocessing import MaxAbsScaler
from hydra.core.config_store import ConfigStore
from config import tmbConfig
from pathlib import Path

# Setup Logger
logging.basicConfig(level=logging.INFO)

# Create ConfigStore
cs = ConfigStore.instance()
cs.store(name="tmb_config", node=tmbConfig)


@hydra.main(config_path="conf", config_name="config")
def train(cfg: tmbConfig) -> None:

    # Names
    model_name = r"test_12"

    # Path
    log_path = Path(
        f"{cfg.setup.project_dir}/{cfg.dirs.log_dir}/{model_name}"
    ).resolve()
    model_save_path = Path(
        f"{cfg.setup.project_dir}/{cfg.dirs.model_dir}/{model_name}.pt"
    ).resolve()
    accPath = Path(f"{cfg.dirs.data_dir}/{cfg.paths.debug_dataset}")

    log.info("Loading the dataset...")
    dataset = AccelerationDataSetFFT(accPath, snr=0, x_scaler=MaxAbsScaler())
    log.info("Dataset loaded. Prepeare Dataloader...")

    # Model
    model = FC_FFT(log_path=log_path, input_size=1000)
    train_loader, valid_loader, test_loader, test_indices = get_train_valid_test_loader(
        dataset=dataset, valid_size=0.2, test_size=0.1, batch_size=32, num_workers=0
    )
    log.info("Dataloader prepared. Start training...")

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    valid_loss, train_loss, _ = train_valid(
        model=model,
        model_save_path=model_save_path,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=3,
        learning_rate=0.001,
        criterion=MSELoss(),
        optimizer=optimizer,
    )  # CrossEntropyLoss angepasst

    output_path = Path(
        f"{cfg.setup.project_dir}/{cfg.dirs.output_dir}/transfer/{model_name}_predict.csv"
    )

    with output_path.open("w") as csvfile:

        writer = csv.writer(csvfile, delimiter=",")
        model.to("cuda") if torch.cuda.is_available() else model.to("cpu")
        model.eval()
        index = 0
        for acc, freq in tqdm(test_loader, desc=f"Testing on Test Dataset"):
            acc = (
                acc.to("cuda") if torch.cuda.is_available() else acc.to("cpu")
            )  # cpu instead of cuda if no gpu is available
            freq = (
                freq.to("cuda") if torch.cuda.is_available() else freq.to("cpu")
            )  # cpu instead of cuda if no gpu is available
            # forward pass: compute predicted outputs by passing inputs to the model
            pred_freq = model(acc)

            for freq_curr, pred_freq_curr in zip(freq, pred_freq):
                log.info(
                    f"Freq: {freq_curr.item():.3f} , Pred. Freq: {pred_freq_curr.item():.3f} , Indice: {test_indices[index]}, diff: {abs(freq_curr.item()-pred_freq_curr.item()):.3f}"
                )
                # update running validation loss

                writer.writerow(
                    [
                        freq_curr.item(),
                        pred_freq_curr.item(),
                        test_indices[index],
                        abs(freq_curr.item() - pred_freq_curr.item()),
                    ]
                )
                index += 1


if __name__ == "__main__":
    train()
