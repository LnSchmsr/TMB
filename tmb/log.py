"""
Define the logging module used with tensorboard
"""
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np


class Logger:
    def __init__(self, path: str):
        self.writer = SummaryWriter(log_dir=path)

    def close(self):
        """
        Closes the writer
        """
        self.writer.close()

    def write_scalar(self, label: str, scalar: float, total_steps: int):
        """Writes a scalar in the log (e.g. Loss or Accuracy)

        Args:
            label (str): name of the scalar
            scalar (float): value to log
            total_steps (int): actual step
        """
        self.writer.add_scalar(label, scalar, total_steps)

    def visualize_weights_and_bias(self, label: str, input: np.ndarray, epoch: int):
        """
        Visualiszes the weight and bias of a layer

        Args:
            label (str): name of the layer
            input (np.ndarray): weight or bias
            epoch (int): epoch
        """
        self.writer.add_histogram(label, input, epoch)

    def visualize_model_architecture(self, model: nn.Module, batch: np.ndarray):
        """
        Shows the model architecture in tensorboard

        Args:
            model (nn.Module): Torch model
            batch (np.ndarray): Batch of input data
        """
        self.writer.add_graph(model, batch)
