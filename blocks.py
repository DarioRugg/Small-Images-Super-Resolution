from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL.Image import BICUBIC, NEAREST, BILINEAR
import RRDBNet_arch as arch


class Classifier(nn.Module):
    def __init__(self, device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"
        super(Classifier, self).__init__()

        resnet50 = models.resnet50(pretrained=True)
        for parameter in resnet50.parameters():
            parameter.requires_grad = False
        resnet50.eval()

        self.layers = nn.Sequential(
            resnet50
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        X_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])(X)
        out = self.layers(X_normalized)
        return out


class Scaler(nn.Module):
    def __init__(self, size: Union[int, tuple]):


        super(Scaler, self).__init__()

        self.size = size

    def forward(self, X: torch.Tensor):
        if int(self.size) < X.shape[-1]:
            trans = transforms.Compose([
                transforms.GaussianBlur(kernel_size=3, sigma=0.2),
                transforms.Resize(self.size)
            ])
            out = trans(X)
        else:
            out = transforms.Resize(self.size)(X)
        return out


class RRDB(nn.Module):
    def __init__(self, pretrained_weights_path: str = '0', trainable: bool = False, device: str = "auto",
                 nb: int = 23):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"
        # checks that the weights are correctly given
        assert isinstance(pretrained_weights_path, str)

        super(RRDB, self).__init__()

        rrdb = arch.RRDBNet(3, 3, 64, nb, gc=32)
        if pretrained_weights_path != '0':
            rrdb.load_state_dict(torch.load(pretrained_weights_path), strict=True)
        if trainable:
            rrdb.train()
        else:
            for parameter in rrdb.parameters():
                parameter.requires_grad = False
            rrdb.eval()

        self.layers = nn.Sequential(
            rrdb
        )

        # moves the entire model to the chosen device
        self.to(self.device)


    def forward(self, X: torch.Tensor):
        out = self.layers(X)
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
