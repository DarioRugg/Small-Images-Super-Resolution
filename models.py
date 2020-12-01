import torch
from torch import nn

from blocks import Classifier, Scaler, RRDB


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
        out = self.layers(X)
        return out


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
            RRDB(pretrained_weights_path=rrdb_pretrained_weights_path),
            Classifier()
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        out = self.layers(X)
        return out

class Model3(nn.Module):
    def __init__(self, device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"

        super(Model3, self).__init__()

        self.layers = nn.Sequential(
            Scaler(56),
            Scaler(224),
            Classifier()
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        out = self.layers(X)
        return out
