"""
Example prediction file.
"""


import os
import torch
import csv
import logging
import hydra

from tmb.data import AcclerationDataSetSchmutter
from tmb.model import FC_FFT
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
def predict_schmutter(cfg: tmbConfig) -> None:

    model_name = "transfer_fft_15_hz.pt"
    output_name = "schmutter_15Hz_loaded"

    log_dir = cfg.dirs.log_dir
    model_path = Path(f"{cfg.setup.project_dir}/{cfg.dirs.model_dir}/{model_name}")
    output_path = Path(
        f"{cfg.setup.project_dir}/{cfg.dirs.output_dir}/{output_name}_predict.csv"
    )

    with open(output_path.resolve(), "w") as csvfile:

        writer = csv.writer(csvfile, delimiter=",")

        model = FC_FFT(log_path=os.path.join(log_dir, model_name), input_size=1000)

        map_location = (
            torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
        )
        model.load_state_dict(
            torch.load(model_path.resolve(), map_location=map_location)
        )
        model.to("cuda") if torch.cuda.is_available() else model.to("cpu")
        model.eval()
        model.transfer_learning()

        accPath = Path(f"{cfg.dirs.schmutter_acc}").resolve()
        freqPath = Path(f"{cfg.dirs.schmutter_freq}/{cfg.paths.ssi_decay}").resolve()
        data_schmutter = AcclerationDataSetSchmutter(
            accPath,
            sampling_rate=1200,
            freq_path=freqPath,
            repeats_f=8,
            load_from_txt=True,
            x_scaler=MaxAbsScaler(),
        )
        for i, (_, _) in enumerate(zip(data_schmutter.x.T, data_schmutter.y)):
            x, y = data_schmutter.__getitem__(i)
            with torch.no_grad():
                x = torch.tensor(x.reshape(shape=(1, len(x))))
                x = x.to("cuda") if torch.cuda.is_available() else x.to("cpu")
                predict_f = model(x)
                f = predict_f.item()
                logging.info(
                    f"Frequenz: {y.item():.3f}, Prediction: {f:.3f}, Diff: {abs(y.item() - f):.3f}"
                )
                writer.writerow([y.item(), f, abs(y.item() - f)])


if __name__ == "__main__":
    predict_schmutter()
