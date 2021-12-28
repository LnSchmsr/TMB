from tmb.train import train_valid
from tmb.data import AcclerationDataSetSchmutter, get_train_valid_test_loader
from torch.nn.modules.loss import MSELoss
import torch
from tmb.model import FC_FFT
import os
import csv
from tqdm import tqdm
from sklearn.preprocessing import MaxAbsScaler
import logging
import hydra
from hydra.core.config_store import ConfigStore
from config import tmbConfig
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Create ConfigStore
cs = ConfigStore.instance()
cs.store(name="tmb_config", node=tmbConfig)


@hydra.main(config_path="conf", config_name="config")
def transfer_train(cfg: tmbConfig):

    # Define modelname and dir for loading
    model_name = "simulation_0.0005827668754538285bs_2048"
    model_dir = Path(
        f"{cfg.setup.project_dir}/{cfg.dirs.model_dir}/{model_name}.pt"
    ).resolve()

    # Define model name for saving
    model_save_name = "transfer_fft_15_hz_loaded_final_test"

    # Define outputnames
    output_name = "loaded_loss.csv"

    # Data paths
    accPath = Path(f"{cfg.dirs.schmutter_acc}").resolve()
    freqPath = Path(f"{cfg.dirs.schmutter_freq}/{cfg.paths.ssi_decay}").resolve()
    log_dir = cfg.dirs.log_dir
    loss_path = Path(f"{cfg.setup.project_dir}/{cfg.dirs.output_dir}/{output_name}")
    predict_path = Path(
        f"{cfg.setup.project_dir}/{cfg.dirs.output_dir}/transfer/{model_name}.csv"
    ).resolve()

    logging.info("Loading the dataset...")
    dataset = AcclerationDataSetSchmutter(
        accPath,
        sampling_rate=1200,
        freq_path=freqPath,
        repeats_f=8,
        load_from_txt=True,
        x_scaler=MaxAbsScaler(),
    )

    logging.info("Dataset loaded. Prepare Dataloader...")
    config = {"lr": 0.001, "batch_size": 32}

    # Model
    model = FC_FFT(log_path=os.path.join(log_dir, "dummy"), input_size=1000)
    map_location = (
        torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
    )
    model.load_state_dict(torch.load(model_dir, map_location=map_location))
    # Enable transfer learning
    model.transfer_learning()
    model.to("cuda") if torch.cuda.is_available() else model.to("cpu")

    # Create Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    train_loader, valid_loader, test_loader, test_indices = get_train_valid_test_loader(
        dataset, 0.20, 0.10, config["batch_size"], num_workers=0
    )
    logging.info("Dataloader prepared. Start training...")
    # Zum Speichern und Laden des Models, Name des NEUEN MODELLS
    model_name = model_save_name
    model_save_path = Path(
        f"{cfg.setup.project_dir}/{cfg.dirs.model_dir}/{model_save_name}.pt"
    ).resolve()
    train_loss, valid_loss, _ = train_valid(
        model=model,
        model_save_path=model_save_path,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=10,
        learning_rate=config["lr"],
        criterion=MSELoss(),
        optimizer=optimizer,
    )

    with loss_path.open("w") as csvfile:
        writer = csv.writer(csvfile)
        for loss in [train_loss, valid_loss]:
            writer.writerow(loss)

    predict(
        test_loader=test_loader,
        output_path=predict_path,
        model_dir=model_dir,
        log_dir=log_dir,
        test_indices=test_indices,
    )


def predict(
    test_loader: torch.utils.data.DataLoader,
    output_path: str,
    model_dir: str,
    log_dir: str,
    test_indices,
):
    with open(output_path, "w") as csvfile:

        writer = csv.writer(csvfile, delimiter=",")

        model = FC_FFT(log_path=os.path.join(log_dir, "dummy"), input_size=1000)

        map_location = (
            torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
        )
        model.load_state_dict(torch.load(model_dir, map_location=map_location))
        model.to("cuda") if torch.cuda.is_available() else model.to("cpu")
        model.eval()

        for acc, freq in tqdm(test_loader, desc=f"Testing on Test Dataset"):
            acc = (
                acc.to("cuda") if torch.cuda.is_available() else acc.to("cpu")
            )  # cpu instead of cuda if no gpu is available
            freq = (
                freq.to("cuda") if torch.cuda.is_available() else freq.to("cpu")
            )  # cpu instead of cuda if no gpu is available
            # forward pass: compute predicted outputs by passing inputs to the models
            pred_freq = model(acc)

            for i, (freq_curr, pred_freq_curr) in enumerate(zip(freq, pred_freq)):
                logging.info(
                    f"Freq: {freq_curr.item():.3f} , Pred. Freq: {pred_freq_curr.item():.3f} , Dif.: {abs(freq_curr.item()-pred_freq_curr.item()):.3f}, Indice: {test_indices[i]}"
                )
                # update running validation loss
                writer.writerow(
                    [
                        freq_curr.item(),
                        pred_freq_curr.item(),
                        abs(freq_curr - pred_freq_curr).item(),
                        test_indices[i],
                    ]
                )


if __name__ == "__main__":
    transfer_train()
