from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms

import RRDBNet_arch as arch


class Classifier(nn.Module):
    def __init__(self, device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"

        super(Classifier, self).__init__()

        resnet18 = models.resnet18(pretrained=True)
        resnet18.eval()

        self.layers = nn.Sequential(
            resnet18
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        X_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])(X)
        out = self.layers(X_normalized)
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

        rrdb = arch.RRDBNet(3, 3, 64, 23, gc=32)
        rrdb.load_state_dict(torch.load(pretrained_weights_path), strict=True)
        rrdb.eval()

        self.layers = nn.Sequential(
            rrdb
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        out = self.layers(X).data
        return out


class AddNoise(nn.Module):
    def __init__(self, std: float = 0.025, mean: float = 0, device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"
        assert isinstance(float(std), float) and std >= 0
        assert isinstance(float(mean), float) and mean >= 0
        self.std, self.mean = std, mean

        super(AddNoise, self).__init__()

        # moves the block to the chosen device
        self.to(self.device)

    def forward(self, X):
        X_noisy = X + torch.randn(X.size()).to(self.device) * self.std + self.mean
        return X_noisy
