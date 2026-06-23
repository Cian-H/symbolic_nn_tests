import torch
from torch import nn


class QMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(784, 10),
        )

    def forward(self, x):
        return self.net(x)


class PubChemModel(nn.Module):
    def __init__(self, return_module_y=False):
        super().__init__()

        self.return_module_y = return_module_y

        self.x0_encoder = nn.TransformerEncoderLayer(7, 7, dim_feedforward=512)
        self.x1_encoder = nn.TransformerEncoderLayer(10, 10, dim_feedforward=1024)
        self.encode_x0 = self.create_xval_encoding_fn(self.x0_encoder)
        self.encode_x1 = self.create_xval_encoding_fn(self.x1_encoder)
        self.ff = nn.Sequential(
            nn.Linear(17, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    @staticmethod
    def create_xval_encoding_fn(layer):
        def encoding_fn(xbatch):
            return torch.stack([layer(x)[-1] for x in xbatch])

        return encoding_fn

    def forward(self, x):
        x0, x1 = x
        y0 = self.encode_x0(x0)
        y1 = self.encode_x1(x1)
        y = torch.cat([y0, y1], dim=1)
        y = self.ff(y)
        if self.return_module_y:
            return x, (y, y0, y1)
        return x, y
