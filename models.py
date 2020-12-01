import time

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from utils import show_img


class Classifier(nn.Module):
    def __init__(self, device="auto"):
        super(Classifier, self).__init__()
        assert device in {"cpu", "cuda", "auto"}

        self.resnet18 = models.resnet18(pretrained=True)

        self.layers = nn.Sequential(
            self.resnet18
        )

        # passes the model to the selected device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        out = self.layers(X)
        return out


def test_model(model: nn.Module, dataloader: DataLoader, verbose: bool = True):
    assert isinstance(model, nn.Module)
    assert isinstance(dataloader, DataLoader)
    assert isinstance(verbose, bool)

    loss_function = nn.CrossEntropyLoss()
    losses, times, starting_time = np.zeros(shape=len(dataloader)), \
                                   np.zeros(shape=len(dataloader)), \
                                   time.time()
    for i_batch, batch in enumerate(dataloader):
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
        if i_batch != 0 and i_batch % (len(dataloader) / 20) == 0 and verbose:
            print(pd.DataFrame(index=[i_batch], data={
                "avg loss": [np.mean(losses[:i_batch])],
                "avg time per batch (s)": [np.mean(times[:i_batch])],
                "total elapsed time (s)": [time.time() - starting_time]
            }))

    return np.mean(losses)
