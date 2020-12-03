import os
import time
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def prepare_div2k_datasets(datasets_folder, high_resolution=True):
    # eventually creates the base directory
    if not os.path.basename(datasets_folder) in os.listdir():
        os.mkdir(datasets_folder)

    # retrieves info about the online dataset
    datasets_website_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K"
    datasets_urls_high_resolution = [f"{datasets_website_url}/DIV2K_train_HR.zip",
                                     f"{datasets_website_url}/DIV2K_valid_HR.zip"]
    datasets_urls_low_resolution = [f"{datasets_website_url}/DIV2K_train_LR_x8.zip",
                                    f"{datasets_website_url}/DIV2K_valid_LR_x8.zip"]
    datasets_urls = datasets_urls_high_resolution if high_resolution else datasets_urls_low_resolution

    # eventually downloads each missing dataset
    for dataset_url in datasets_urls:
        dataset_zip_name = os.path.basename(dataset_url)
        dataset_name = os.path.splitext(dataset_zip_name)[0]
        if dataset_zip_name not in os.listdir(datasets_folder) and dataset_name not in os.listdir(datasets_folder):
            # downloads the .zip containing the dataset
            print(f"Downloading {dataset_url}")
            urllib.request.urlretrieve(dataset_url, f"./{dataset_zip_name}")
            # extracts the content into the folder
            with zipfile.ZipFile(dataset_zip_name, 'r') as fp:
                fp.extractall(datasets_folder)
            # deletes the original .zip file
            os.remove(dataset_zip_name)


def show_img(*imgs: torch.Tensor, filename: str = None, save_to_folder: str = None):
    assert not save_to_folder or isinstance(save_to_folder, str)
    imgs = list(imgs)
    for i_img, img in enumerate(imgs):
        assert isinstance(img, torch.Tensor)
        assert len(img.shape) == 3
        if save_to_folder:
            if not filename:
                filename = f"img{int(time.time())}_{i_img}"
            save_image(img, f"{save_to_folder}/{filename}.png")
        imgs[i_img] = img.permute(1, 2, 0).to("cpu").numpy()
    fig, axs = plt.subplots(1, len(imgs), squeeze=False)
    for i_ax, ax in enumerate(axs.flat):
        ax.imshow(imgs[i_ax])
    plt.show()


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1 / torch.sqrt(mse))


def test_model(model: nn.Module, data: DataLoader, early_stop: int = None, verbose: bool = True):
    assert isinstance(model, nn.Module)
    assert isinstance(data, DataLoader)
    assert isinstance(verbose, bool)
    assert not early_stop or isinstance(early_stop, int)
    if early_stop:
        assert early_stop > 1

    loss_function = nn.CrossEntropyLoss()
    losses, psnrs, corrects = np.zeros(shape=len(data)), \
                              np.zeros(shape=len(data)), \
                              np.zeros(shape=len(data))
    starting_time = time.time()
    with torch.no_grad():
        for i_batch, batch in enumerate(data):
            # checks wheter to stop
            if early_stop and i_batch == early_stop:
                break
            # plot a sample image if it's the first time
            if i_batch == 0 and verbose:
                show_img(batch[0][0])
            # make a prediction
            X, y = batch[0].to(model.device), batch[1].to(model.device)
            X_downsampled, X_upsampled, y_pred = model(X)
            y_pred_as_labels = torch.argmax(F.softmax(y_pred, dim=1), dim=-1)
            losses[i_batch], psnrs[i_batch], corrects[i_batch] = loss_function(y_pred, y), \
                                                                 psnr(X, X_upsampled), \
                                                                 (y_pred_as_labels == y).sum()

            # prints some stats
            if i_batch != 0 and i_batch % (len(data) / 20) == 0 and verbose:
                print(pd.DataFrame(index=[f"batch {i_batch} of {len(data)}"], data={
                    "avg loss": [np.mean(losses[:i_batch])],
                    "total elapsed time (s)": [time.time() - starting_time]
                }))

    return {
        "loss": losses[:i_batch],
        "psnr": psnrs[:i_batch],
        "corrects": corrects[:i_batch],
        "total_time": time.time() - starting_time
    }
