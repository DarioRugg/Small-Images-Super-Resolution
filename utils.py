import os
from os.path import join

import copy
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from blocks import Scaler, Classifier

pd.set_option('display.max_columns', None)


def read_json(filepath: str):
    with open(filepath, "r") as fp:
        return json.load(fp)


def show_img(*imgs: torch.Tensor, filename: str = None, save_to_folder: str = None):
    assert not save_to_folder or isinstance(save_to_folder, str)
    imgs = list(imgs)
    for i_img, img in enumerate(imgs):
        assert isinstance(img, torch.Tensor)
        assert len(img.shape) == 3
        if save_to_folder:
            filename = filename if filename else f"img_{i_img}"
            save_image(img, join(save_to_folder, f"{filename}.png"))
        imgs[i_img] = img.permute(1, 2, 0).to("cpu").numpy()
    fig, axs = plt.subplots(1, len(imgs), squeeze=False)
    for i_ax, ax in enumerate(axs.flat):
        ax.imshow(imgs[i_ax])
    plt.show()


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return np.inf
    return 20 * torch.log10(1 / torch.sqrt(mse))


def test_model(model: nn.Module, data: DataLoader,
               batches_per_epoch: int = None,
               verbose: bool = True, logs_path: str = None):
    # checks about model's parameter
    assert isinstance(model, nn.Module)
    assert isinstance(data, DataLoader)
    if batches_per_epoch:
        assert batches_per_epoch > 1
    # checks on other parameters
    assert isinstance(verbose, bool)
    assert not logs_path or isinstance(logs_path, str)
    assert not batches_per_epoch or isinstance(batches_per_epoch, int)

    loss_function = nn.CrossEntropyLoss()
    losses, psnrs, corrects = np.zeros(shape=len(data)), \
                              np.zeros(shape=len(data)), \
                              np.zeros(shape=len(data))
    starting_time = time.time()
    with torch.no_grad():
        for i_batch, batch in enumerate(data):
            # checks wheter to stop
            if batches_per_epoch and i_batch == batches_per_epoch:
                break

            # make a prediction
            X, y = batch[0].to(model.device), \
                   batch[1].to(model.device)

            X_downsampled, X_upsampled, y_pred = model(X)
            y_pred_as_labels = torch.argmax(F.softmax(y_pred, dim=1), dim=-1)
            losses[i_batch], psnrs[i_batch], corrects[i_batch] = loss_function(y_pred, y), \
                                                                 psnr(X, X_upsampled), \
                                                                 (y_pred_as_labels == y).sum()
            # plot a sample image if it's the first time
            if i_batch == 0 and verbose:
                show_img(X_upsampled[0], filename=model.name.lower().strip(), save_to_folder=logs_path)

            # prints some stats

            if i_batch != 0 and i_batch % int((batches_per_epoch if batches_per_epoch else len(data)) / 20) == 0 and verbose:
                print(pd.DataFrame(index=[f"batch {i_batch} of {(batches_per_epoch if batches_per_epoch else len(data))}"], data={
                    "avg loss": [np.mean(losses[:i_batch])],
                    "total elapsed time (s)": [time.time() - starting_time]
                }))

    return {
        "loss": losses[:i_batch],
        "psnr": psnrs[:i_batch],
        "corrects": corrects[:i_batch],
        "total_time": time.time() - starting_time
    }


def train_darionet(model: nn.Module, data_train: DataLoader, data_val: DataLoader,
                   lr: float = 1e-4, epochs=25, batches_per_epoch: int = None,
                   filepath: str = None, verbose: bool = True):
    # checks about model's parameters
    assert isinstance(model, nn.Module)
    assert isinstance(data_train, DataLoader)
    assert isinstance(data_val, DataLoader)
    assert not filepath or isinstance(filepath, str)
    # checks on other parameters
    assert isinstance(verbose, bool)
    assert isinstance(lr, float) and lr > 0
    assert isinstance(epochs, int) and epochs >= 1

    since = time.time()
    best_epoch_loss, best_model_weights = np.inf, \
                                          copy.deepcopy(model.state_dict())

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    cross_entropy, l1 = nn.CrossEntropyLoss(), nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            data = data_train if phase == "train" else data_val
            if phase == 'train':
                #
                # for parameter in model.parameters():
                #     parameter.requires_grad = False
                # # for parameter in model.layers[0].conv_last.parameters():
                # #     parameter.requires_grad = True
                # # for parameter in model.layers[0].HRconv.parameters():
                # #     parameter.requires_grad = True
                model.train()
            else:
                with torch.no_grad():
                    model.eval()

            batches_to_do = min(batches_per_epoch if batches_per_epoch else len(data), len(data))

            epoch_losses, epoch_psnrs = np.zeros(shape=batches_to_do), \
                                        np.zeros(shape=batches_to_do)
            epoch_MAE, epoch_CrossEntropy = np.zeros(shape=batches_to_do), \
                                            np.zeros(shape=batches_to_do)

            for i_batch, batch in enumerate(data):
                # eventually early stops the training
                if batches_per_epoch and i_batch >= batches_to_do:
                    break

                # gets input data
                X, y = batch[0].to(model.device), \
                       batch[1].to(model.device)
                X_downsampled = Scaler(X.shape[-1] // 4)(X)

                optimizer.zero_grad()

                # forward pass
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(phase == 'train'):
                        X_supersampled = model(X_downsampled)

                    y_pred = Classifier()(X_supersampled)

                    loss = l1(X_supersampled, X)*cross_entropy(y_pred, y)

                    # backward pass
                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                epoch_losses[i_batch], epoch_psnrs[i_batch] = loss, \
                                                              psnr(X, X_supersampled)

                epoch_MAE[i_batch], epoch_CrossEntropy[i_batch] = l1(X_supersampled, X), \
                                                                  cross_entropy(y_pred, y)

                # statistics
                if verbose and i_batch in np.linspace(start=1, stop=batches_to_do, num=20, dtype=np.int):
                    time_elapsed = time.time() - since
                    print(pd.DataFrame(
                        index=[
                            f"batch {i_batch + 1} of {batches_to_do}"],
                        data={
                            "epoch": epoch,
                            "phase": phase,
                            "avg loss": np.mean(epoch_losses[:i_batch]),
                            "avg PSNR": np.mean(epoch_psnrs[:i_batch]),
                            "avg MAE": np.mean(epoch_MAE[:i_batch]),
                            "avg CE": np.mean(epoch_CrossEntropy[:i_batch]),
                            "time": "{:.0f}:{:.0f}".format(time_elapsed // 60, time_elapsed % 60)
                        }))

            # deep copy the model
            avg_epoch_loss = np.mean(epoch_losses)
            if phase == 'val' and avg_epoch_loss < best_epoch_loss:
                print(f"Found best model with loss {avg_epoch_loss}")
                best_epoch_loss, best_model_weights = avg_epoch_loss, \
                                                      copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_weights)
    # saves to a file
    if filepath:
        torch.save(model, filepath)
        print(f"Model saved to {filepath}")
    return model
