from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

import RRDBNet_arch as arch


class Classifier(nn.Module):
    def __init__(self, device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"

        super(Classifier, self).__init__()

        self.resnet18 = models.resnet18(pretrained=True)

        self.layers = nn.Sequential(
            self.resnet18
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        out = self.layers(X)
        return out


class Scaler(nn.Module):
    def __init__(self, size: Union[int, tuple], mode: str = "bicubic"):
        # checks that the mode is correctly inserted
        assert isinstance(mode, str)
        assert mode in {"nearest", "linear", "bilinear", "bicubic", "trilinear", "area"}
        super(Scaler, self).__init__()

        self.size = size

    def forward(self, X: torch.Tensor):
        out = F.interpolate(X, size=self.size)
        return out


class RRDB(nn.Module):
    def __init__(self, pretrained_weights_path: str, device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"
        # checks that the weights are correctly given
        assert isinstance(pretrained_weights_path, str)

        super(RRDB, self).__init__()

        self.rrdb = arch.RRDBNet(3, 3, 64, 23, gc=32)
        self.rrdb.load_state_dict(torch.load(pretrained_weights_path), strict=True)

        self.layers = nn.Sequential(
            self.rrdb
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        out = self.layers(X).data
        return out
