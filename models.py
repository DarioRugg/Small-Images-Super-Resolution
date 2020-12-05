import torch
from torch import nn

from blocks import Classifier, Scaler, RRDB, AddNoise


class Model1(nn.Module):
    def __init__(self, device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"

        super(Model1, self).__init__()

        self.layers = nn.Sequential(
            Classifier()
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        X_downsampled = Scaler(56)(X)
        X_upsampled = X
        y_pred = self.layers(X)
        return X_downsampled, X_upsampled, y_pred


class Model2(nn.Module):
    def __init__(self, rrdb_pretrained_weights_path: str, device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"
        # checks that the weights are correctly given
        assert isinstance(rrdb_pretrained_weights_path, str)

        super(Model2, self).__init__()

        self.layers = nn.Sequential(
            Scaler(56),
            # AddNoise(),
            RRDB(pretrained_weights_path=rrdb_pretrained_weights_path),
            Classifier()
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        X_downsampled = self.layers[:2](X)
        X_upsampled = self.layers[2:-1](X_downsampled)
        y_pred = self.layers[-1](X_upsampled)
        return X_downsampled, X_upsampled, y_pred


class Model3(nn.Module):
    def __init__(self, device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"

        super(Model3, self).__init__()

        self.layers = nn.Sequential(
            Scaler(56),
            # AddNoise(),
            Scaler(224),
            Classifier()
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        X_downsampled = self.layers[:2](X)
        X_upsampled = self.layers[2:-1](X_downsampled)
        y_pred = self.layers[-1](X_upsampled)
        return X_downsampled, X_upsampled, y_pred
