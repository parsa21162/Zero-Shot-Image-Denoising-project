"""
Denoising Network: Simple 8-layer Fully Convolutional Network
All layers use 3x3 kernels and 48 channels except the last one.
"""
import torch
import torch.nn as nn


class DenoisingNetwork(nn.Module):
    def __init__(self, in_channels=3, num_layers=8, num_features=48):
        super(DenoisingNetwork, self).__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True))
                layers.append(nn.ReLU(inplace=True))
            elif i == num_layers - 1:
                layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=True))
            else:
                layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True))
                layers.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)
