import torch.nn as nn
import torch
from tmb.log import Logger


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.LSTM:
        for name, parameter in m.named_parameters():
            if "weight" in name and parameter.requires_grad:
                torch.nn.init.xavier_uniform_(
                    parameter.data, gain=nn.init.calculate_gain("tanh")
                )
    elif type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.Conv1d:
        for name, parameter in m.named_parameters():
            if parameter.requires_grad:
                torch.nn.init.xavier_uniform_(m.weight)


class FC_FFT(nn.Module):
    def __init__(self, log_path: str, output_size=1, input_size=800):
        super(FC_FFT, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(50),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(25),
            nn.Linear(25, 25),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(25),
            nn.Linear(25, 25),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(25),
            nn.Linear(25, output_size),
        )

        self.logger = Logger(log_path)

    def forward(self, x):
        return torch.squeeze(self.fc(x))

    def transfer_learning(self):
        for name, parameter in self.named_parameters():
            if len(parameter.flatten()) > 25:
                parameter.requires_grad = False
        for layer in self.modules():
            if (
                type(layer) == torch.nn.modules.batchnorm.BatchNorm1d
                or type(layer) == torch.nn.modules.dropout.Dropout
            ):
                layer.requires_grad_(True)
            elif type(layer) == torch.nn.modules.activation.ReLU:
                layer.requires_grad_(True)

    def log_weights_and_bias(self, epoch: int):
        for name, parameter in self.fc.named_parameters():
            if parameter.requires_grad:
                self.logger.visualize_weights_and_bias(name, parameter.data, epoch)
                self.logger.write_scalar(
                    name + "_mean", torch.mean(parameter.data.flatten()), epoch
                )
