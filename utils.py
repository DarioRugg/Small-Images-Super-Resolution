import os
from os.path import join
import re
import copy
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from positional_encodings.positional_encodings import PositionalEncodingPermute2D
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from blocks import Scaler, Classifier
from torchvision import transforms

pd.set_option('display.max_columns', None)


def read_json(filepath: str):
    with open(filepath, "r") as fp:
        return json.load(fp)


def save_json(d: dict, filepath: str):
    with open(filepath, "w") as fp:
        return json.dump(d, fp, indent=4)


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

    batches_to_do = min(batches_per_epoch if batches_per_epoch else len(data), len(data))
    loss_function = nn.CrossEntropyLoss()
    losses, psnrs, corrects = np.zeros(shape=len(data)), \
                              np.zeros(shape=len(data)), \
                              np.zeros(shape=len(data))
    y_true_final, y_pred_final = [], []
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
            X_upsampled = torch.clip(X_upsampled, 0, 1)
            y_pred_as_labels = torch.argmax(F.softmax(y_pred, dim=1), dim=-1)
            losses[i_batch], psnrs[i_batch], corrects[i_batch] = loss_function(y_pred, y), \
                                                                 psnr(X, X_upsampled) \
                                                                     if X_upsampled is not None else None, \
                                                                 (y_pred_as_labels == y).sum()
            y_true_final += y.tolist()
            y_pred_final += y_pred_as_labels.tolist()
            # plot a sample image if it's the first time
            if i_batch == 0 and verbose:
                if X_upsampled is not None:
                    show_img(X_upsampled[0], filename=model.name.lower().strip(), save_to_folder=logs_path)
                elif X_downsampled is not None:
                    show_img(X_downsampled[0], filename=model.name.lower().strip(), save_to_folder=logs_path)

            # prints some stats

            if verbose and i_batch in np.linspace(start=1, stop=batches_to_do, num=20, dtype=np.int):
                print(pd.DataFrame(
                    index=[f"batch {i_batch} of {(batches_per_epoch if batches_per_epoch else len(data))}"], data={
                        "avg loss": [np.mean(losses[:i_batch])],
                        "total elapsed time (s)": [time.time() - starting_time]
                    }))

    return {
        "loss": losses[:i_batch],
        "psnr": psnrs[:i_batch],
        "corrects": corrects[:i_batch],
        "total_time": time.time() - starting_time,
        "y": y_true_final,
        "y_pred": y_pred_final
    }


def train_darionet(model: nn.Module, data_train: DataLoader, data_val: DataLoader,
                   lr: float = 3e-5, epochs=25, batches_per_epoch: int = None,
                   filepath: str = None, verbose: bool = True,
                   scale: float = 0.25, train_crop_size: int = 256,
                   val_crop_size: int = 256, save: bool = True,
                   checkpoints: str = None):
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


    # Since checkpoints may be either a new directory or the path to the checkpoint model, checking if checkpoint
    # is a directory gives the econdition to understand if training is resuming or starting from zero
    if os.path.isdir(checkpoints):
        starting_epoch = 0

    else:
        starting_epoch = int(re.search(r"\d+", os.path.basename(checkpoints)).group(0)) + 1
        print(f"... Resuming training from epoch {starting_epoch}")
        model = torch.load(checkpoints)



    # optimizer_MSE = optim.Adam(params=model.parameters(), lr=lr)
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    cross_entropy, l1 = nn.CrossEntropyLoss(), nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(starting_epoch, epochs):
        # if epoch % 2 == 0:
        #     print("Minimizing Cross-entropy")
        # else:
        #     print("Minimizing L1")

        for phase in ['train', 'val']:
            data = data_train if phase == "train" else data_val


            batches_to_do = min(batches_per_epoch if batches_per_epoch else len(data), len(data))

            epoch_losses, epoch_psnrs = np.zeros(shape=batches_to_do), \
                                        np.zeros(shape=batches_to_do)
            epoch_pred_loss, epoch_CrossEntropy = np.zeros(shape=batches_to_do), \
                                            np.zeros(shape=batches_to_do)

            for i_batch, batch in enumerate(data):
                # eventually early stops the training
                if batches_per_epoch and i_batch >= batches_to_do:
                    break

                # gets input data
                X, y = batch[0].to(model.device), \
                       batch[1].to(model.device)


                X_downsampled = Scaler(int(X.shape[-1] * scale))(X)



                # forward pass
                with torch.cuda.amp.autocast():
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            for parameter in model.parameters():
                                parameter.requires_grad = True
                            model.train()
                            X_supersampled = torch.clip(transforms.Resize(train_crop_size)(model(X_downsampled)), 0,1)
                        else:
                            with torch.no_grad():
                                model.eval()
                            X_supersampled = torch.clip(transforms.Resize(val_crop_size)(model(X_downsampled)), 0,1)

                    resnet = Classifier()
                    for par in resnet.parameters():
                        par.requires_grad=False

                    outputs = []

                    def hook(module, input, output):
                        outputs.append(output)
                    for i in range(3):
                        resnet.layers[0].layer1[i].conv1.register_forward_hook(hook)
                    for i in range(4):
                        resnet.layers[0].layer2[i].conv1.register_forward_hook(hook)

                    gt_pred = resnet(X)

                    bl_ce = cross_entropy(gt_pred, y)
                    y_pred = resnet(X_supersampled)

                    posenc_loss = 0
                    for i in range(len(outputs)//2):
                        pos_enc = PositionalEncodingPermute2D(512)(outputs[i])
                        posenc_loss += nn.L1Loss()(pos_enc + outputs[i], pos_enc + outputs[i+7])
                    CE = cross_entropy(y_pred, y)
                    pred_loss = nn.L1Loss()(y_pred, gt_pred)

                    loss = pred_loss + posenc_loss


                # backward pass
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                epoch_losses[i_batch] = loss
                epoch_psnrs[i_batch] = psnr(X, X_supersampled)

                epoch_pred_loss[i_batch] = pred_loss
                epoch_CrossEntropy[i_batch] = CE-bl_ce


                # statistics
                if verbose and i_batch in np.linspace(start=1, stop=batches_to_do, num=20, dtype=np.int):
                    time_elapsed = time.time() - since
                    print(pd.DataFrame(
                        index=[
                            f"batch {i_batch + 1} of {batches_to_do}"],
                        data={
                            "epoch": epoch,
                            "phase": phase,
                            f"avg loss": np.mean(epoch_losses[:i_batch]),
                            "avg PSNR": np.mean(epoch_psnrs[:i_batch]),
                            "avg prediction loss": np.mean(epoch_pred_loss[:i_batch]),
                            "avg CE": np.mean(epoch_CrossEntropy[:i_batch]),
                            "time": "{:.0f}:{:.0f}".format(time_elapsed // 60, time_elapsed % 60)
                        }))

            # deep copy the model
            avg_epoch_loss = np.mean(epoch_losses)
            if phase == 'val' and avg_epoch_loss < best_epoch_loss:
                print(f"Found best model with loss {avg_epoch_loss}")
                best_epoch_loss, best_model_weights = avg_epoch_loss, \
                                                      copy.deepcopy(model.state_dict())
        if checkpoints and os.path.isdir(checkpoints):
            torch.save(model, join(checkpoints, f'darionet_epoch_{epoch}.pt'))
        elif checkpoints:
            torch.save(model, join(os.path.dirname(checkpoints), f'darionet_epoch_{epoch}.pt'))

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_weights)
    # saves to a file
    if save:
        torch.save(model, filepath)
        print(f"Model saved to {filepath}")
    return model
