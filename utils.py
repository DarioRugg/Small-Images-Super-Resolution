import os
import zipfile
import urllib.request

import torch
import matplotlib.pyplot as plt


def show_img(img: torch.Tensor):
    # reshapes the tensor to make it viewable
    img = img.permute(1, 2, 0)
    # prints the image
    plt.imshow(img.numpy())
    plt.show()


def prepare_datasets(datasets_folder, high_resolution=False):
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
