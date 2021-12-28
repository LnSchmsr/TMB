import torch
import numpy as np
import torch.nn as nn
import os
import csv

from tmb.model import FC_FFT, init_weights
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from ray import tune
from tmb.data import AccelerationDataSet, get_train_valid_test_loader
from tqdm import tqdm
from typing import Dict, List, Tuple




def hypersearch(
    config: Dict,
    dataset: AccelerationDataSet = None,
    log_dir: str = None,
    criterion: torch.optim = None,
    num_epochs: int = None,
    checkpoint_dir: str = None,
    model_name: str = None,
    output_dir: str = None,
    model_dir: str = None,
    project_dir: str = None,
) -> None:
    """Performs a hypersearch with the given config and the given dataset

    Args:
        config (Dict): Hypersearch config
        dataset (AccelerationDataSet, optional): Dataset for training. Defaults to None.
        log_dir (str, optional): Directory for logging to tensorboard. Defaults to None.
        criterion (torch.optim, optional): Loss function. Defaults to None.
        num_epochs (int, optional): Number of epoch to train. Defaults to None.
        checkpoint_dir (str, optional): Directory for checkpoints for ray tune. Defaults to None.
        model_name (str, optional): Name of the Model. Will be saved as model_name+batch_size+learning_rate.. Defaults to None.
        output_dir (str, optional): Directory for outputs (Loss, Prediction). Defaults to None.
        model_dir (str, optional): Directory for saving the model. Defaults to None.
        project_dir (str, optional): Project directory. Defaults to None.
    """

    # Model
    model_name = (
        model_name + "_" + str(config["lr"]) + "_bs_" + str(config["batch_size"])
    )
    model = FC_FFT(
        log_path=Path(f"{project_dir}/{log_dir}/{model_name}").resolve(),
        input_size=1000,
    )

    # Create Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    train_loader, valid_loader, test_loader, test_index = get_train_valid_test_loader(
        dataset, 0.2, 0.1, config["batch_size"], num_workers=0
    )

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )

        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Define Hypterparamters
    num_epochs = num_epochs  # epochs

    criterion = criterion  # loss - function
    optimizer = optimizer
    model.apply(init_weights)
    model.to("cuda") if torch.cuda.is_available() else model.to("cpu")
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf  # set initial "min" to infinity
    # Initalize train, valid losses
    train_loss_epoch = torch.zeros(num_epochs)
    valid_loss_epoch = torch.zeros(num_epochs)

    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        # monitor losses
        train_loss = 0
        valid_loss = 0

        ###################
        # train the model #
        ###################

        model.train()  # prep model for training
        for acc, freq in tqdm(train_loader, desc=f"Training for Epoch {epoch}"):
            acc = (
                acc.to("cuda") if torch.cuda.is_available() else acc.to("cpu")
            )  # cpu instead of cuda if no gpu is available
            freq = (
                freq.to("cuda") if torch.cuda.is_available() else freq.to("cpu")
            )  # cpu instead of cuda if no gpu is available
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            #####################

            pred_freq = model(acc)
            loss = criterion(pred_freq, torch.squeeze(freq))
            loss.backward()

            # update running training loss
            train_loss += loss.item() * acc.size(0)
            optimizer.step()

            model.log_weights_and_bias(epoch)
            #####################

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        valid_steps = 0
        for acc, freq in tqdm(valid_loader, desc=f"Valid for Epoch {epoch}"):
            acc = (
                acc.to("cuda") if torch.cuda.is_available() else acc.to("cpu")
            )  # cpu instead of cuda if no gpu is available
            freq = (
                freq.to("cuda") if torch.cuda.is_available() else freq.to("cpu")
            )  # cpu instead of cuda if no gpu is available
            # forward pass: compute predicted outputs by passing inputs to the model
            pred_freq = model(acc)

            # calculate the loss
            loss = criterion(pred_freq, torch.squeeze(freq))
            # update running validation loss
            valid_loss += loss.item() * acc.size(0)
            valid_steps += 1
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)

        model.logger.write_scalar("Train loss", train_loss, epoch)
        model.logger.write_scalar("Test loss", valid_loss, epoch)

        train_loss_epoch[epoch] = train_loss
        valid_loss_epoch[epoch] = valid_loss

        if valid_loss <= valid_loss_min:
            save_path = Path(f"{project_dir}/{model_dir}/{model_name}.pt")
            torch.save(model.state_dict(), str(save_path.resolve()))

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(valid_loss=(valid_loss), train_loss=(train_loss))

        #####################
    model.logger.close()

    output_path_loss = Path(f"{project_dir}/{output_dir}/{model_name}_loss.csv")
    output_path_predict = Path(f"{project_dir}/{output_dir}/{model_name}_predict.csv")

    with output_path_loss.open("w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        for loss in [train_loss_epoch, valid_loss_epoch]:
            loss = [x.item() for x in loss]
            writer.writerow(loss)

    model.eval()
    with output_path_predict.open("w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        index = 0
        for acc, freq in tqdm(test_loader, desc="Testing on Test Dataset"):
            acc = (
                acc.to("cuda") if torch.cuda.is_available() else acc.to("cpu")
            )  # cpu instead of cuda if no gpu is available
            freq = (
                freq.to("cuda") if torch.cuda.is_available() else freq.to("cpu")
            )  # cpu instead of cuda if no gpu is available
            # forward pass: compute predicted outputs by passing inputs to the model
            pred_freq = model(acc)

            for freq_curr, pred_freq_curr in zip(freq, pred_freq):
                writer.writerow(
                    [
                        freq_curr.item(),
                        pred_freq_curr.item(),
                        test_index[index],
                        abs(freq_curr.item() - pred_freq_curr.item()),
                    ]
                )
                index += 1


def train_valid(
    model: nn.Module,
    model_save_path: str,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    criterion,
    optimizer,
) -> Tuple(List[float], List[float], List[int]):
    """Performs a training and validation on a given model.

    Args:
        model (nn.Module, model_save_path): Model for training and validation.
        model_save_path (str): Path for saving the trained model.
        train_loader(DataLoader): Train Loader for training.
        valid_loader(DataLoader): Validaiotn Loader for training.
        num_epoochs (int): Number of epochs for training.
        learning_rate(float): Learning rate for training.
        criterion (torch.nn.modules): Lossfunction for optimisation.
        optimizier (torch.optim): Optimiser for training.

    Returns:
        Tuple (List[float], List[float], List[int]): Returns the list of training loss, validation loss, and test indices.
    """

    # Define Hypterparamters
    num_epochs = num_epochs  # epochs
    learning_rate = learning_rate  # lr
    criterion = criterion  # loss - function
    optimizer = optimizer
    model.apply(init_weights)
    model.to("cuda") if torch.cuda.is_available() else model.to("cpu")
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf  # set initial "min" to infinity
    # Initalize train, valid losses
    train_loss_epoch = torch.zeros(num_epochs)
    valid_loss_epoch = torch.zeros(num_epochs)
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        # monitor losses
        train_loss = 0
        valid_loss = 0

        ###################
        # train the model #
        ###################

        model.train()  # prep model for training
        for acc, freq in tqdm(train_loader, desc=f"Training for Epoch {epoch}"):
            acc = (
                acc.to("cuda") if torch.cuda.is_available() else acc.to("cpu")
            )  # cpu instead of cuda if no gpu is available
            freq = (
                freq.to("cuda") if torch.cuda.is_available() else freq.to("cpu")
            )  # cpu instead of cuda if no gpu is available
            optimizer.zero_grad()
            #####################

            pred_freq = model(acc)

            # Make freq to dist

            loss = criterion(pred_freq, freq.float())
            loss.backward()

            # update running training loss
            train_loss += loss.item() * acc.size(0)
            optimizer.step()

            model.log_weights_and_bias(epoch)
            #####################

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for acc, freq in tqdm(valid_loader, desc=f"Valid for Epoch {epoch}"):
            acc = (
                acc.to("cuda") if torch.cuda.is_available() else acc.to("cpu")
            )  # cpu instead of cuda if no gpu is available
            freq = (
                freq.to("cuda") if torch.cuda.is_available() else freq.to("cpu")
            )  # cpu instead of cuda if no gpu is available
            # forward pass: compute predicted outputs by passing inputs to the model
            pred_freq = model(acc)

            # calculate the loss
            # Make freq to dist

            loss = criterion(pred_freq, freq.float())

            # update running validation loss
            valid_loss += loss.item() * acc.size(0)
            # f1 += f1_score(freq.cpu(), pred_freq.cpu() > 0.5, average="samples")
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        # f1 = f1 / len(valid_loader)
        model.logger.write_scalar("Train loss", train_loss, epoch)
        model.logger.write_scalar("Test loss", valid_loss, epoch)
        # model.logger.write_scalar('F1', f1, epoch) # f1 score eingefügt

        train_loss_epoch[epoch] = train_loss
        valid_loss_epoch[epoch] = valid_loss
        # F1_SCORE[epoch] = f1
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch + 1,
                train_loss,
                valid_loss,
            )
        )

        #####################
        # save model if validation loss has decreased (early stopping)
        if valid_loss <= valid_loss_min:
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    valid_loss_min, valid_loss
                )
            )
            torch.save(model.state_dict(), model_save_path)
            valid_loss_min = valid_loss

        #####################
    model.logger.close()
    return train_loss_epoch, valid_loss_epoch, model
