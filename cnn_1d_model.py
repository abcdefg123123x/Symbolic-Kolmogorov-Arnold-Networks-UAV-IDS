import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DTabular(nn.Module):
    """1D-CNN built specifically for 2D tabular feature inputs."""

    def __init__(self, input_size, num_classes, cnn_channels, kernel_size,
                 dropout):
        super(CNN1DTabular, self).__init__()

        layers = []
        in_ch = 1  # Treats input features as a single-channel sequence of length input_size

        for out_ch in cnn_channels:
            layers.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.conv_stack = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(in_ch, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Expects strictly 2D tabular input: (Batch, Features) -> add channel dim -> (Batch, 1, Features)
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = self.conv_stack(x)
        x = self.pool(x).squeeze(-1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_cnn_model(input_size, num_classes, params):
    num_conv_layers = params["num_conv_layers"]
    cnn_channels = [params[f"layer_{i}_channels"] for i in
                    range(1, num_conv_layers + 1)]

    return CNN1DTabular(
        input_size=input_size,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        kernel_size=params["kernel_size"],
        dropout=params["dropout"]
    )