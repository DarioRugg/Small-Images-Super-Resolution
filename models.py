import time
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.utils import save_image

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
        out = self.layers(X)
        return out


def test_model(model: nn.Module, data: DataLoader, verbose: bool = True):
    assert isinstance(model, nn.Module)
    assert isinstance(data, DataLoader)
    assert isinstance(verbose, bool)

    loss_function = nn.CrossEntropyLoss()
    losses, times, starting_time = np.zeros(shape=len(data)), \
                                   np.zeros(shape=len(data)), \
                                   time.time()
    with torch.no_grad():
        for i_batch, batch in enumerate(data):
            # plot a sample image if it's the first time
            if i_batch == 0 and verbose:
                show_img(batch[0][0])
            batch_starting_time = time.time()
            # make a prediction
            X, y = batch[0].to(model.device), batch[1].to(model.device)
            y_pred = model(X)
            losses[i_batch], times[i_batch] = loss_function(y_pred, y), \
                                              time.time() - batch_starting_time
            # prints some stats
            if i_batch != 0 and i_batch % (len(data) / 20) == 0 and verbose:
                print(pd.DataFrame(index=[f"batch {i_batch} of {len(data)}"], data={
                    "avg loss": [np.mean(losses[:i_batch])],
                    "avg time per batch (s)": [np.mean(times[:i_batch])],
                    "total elapsed time (s)": [time.time() - starting_time]
                }))

    return np.mean(losses)


def show_img(*imgs: torch.Tensor, save_to_folder: str = None):
    assert not save_to_folder or isinstance(save_to_folder, str)
    imgs = list(imgs)
    for i_img, img in enumerate(imgs):
        assert isinstance(img, torch.Tensor)
        assert len(img.shape) == 3
        if save_to_folder:
            save_image(img, f"{save_to_folder}/img{i_img}_{int(time.time())}.png")
        imgs[i_img] = img.permute(1, 2, 0).to("cpu").numpy()
    fig, axs = plt.subplots(1, len(imgs), squeeze=False)
    for i_ax, ax in enumerate(axs.flat):
        ax.imshow(imgs[i_ax])
    plt.show()
