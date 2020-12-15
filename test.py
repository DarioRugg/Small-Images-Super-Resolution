import time
from os import mkdir
from os.path import join

import numpy as np
import pandas as pd

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import Model1, Model3, Model2, Model4
from utils import test_model, read_json

# parameters object
parameters_path = join(".", "parameters.json")
parameters = read_json(parameters_path)

torch.manual_seed(parameters["test"]["seed"])
np.random.seed(parameters["test"]["seed"])

assets_path = join(".", "assets")
logs_path = join(assets_path, "logs")
imagenet2012_path = join(assets_path, "ImageNet2012_val")
models_path = join(assets_path, "models")
rrdb_pretrained_weights_path, darionet_pretrained_model_path = join(models_path, "RRDB_PSNR_x4.pth"), \
                                                               join(models_path, "DarioNet1.pt")

transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=parameters["transformations"]["random_horizontal_flip_probability"]),
    transforms.RandomVerticalFlip(p=parameters["transformations"]["random_vertical_flip_probability"]),
    transforms.Resize(parameters["transformations"]["resize_size"]),
    transforms.RandomCrop(parameters["transformations"]["random_crop_size"]),
    transforms.ToTensor()
])

imagenet2012_val_dataset = datasets.ImageFolder(root=imagenet2012_path, transform=transforms)
if __name__ == '__main__':
    imagenet2012_val_loader = DataLoader(imagenet2012_val_dataset, num_workers=4,
                                         batch_size=parameters["test"]["batch_size"],
                                         shuffle=False, pin_memory=True)

    # computes tests on the different models
    models = [
        Model1(input_image_size=parameters["transformations"]["random_crop_size"]),
        Model2(input_image_size=parameters["transformations"]["random_crop_size"]),
        Model3(input_image_size=parameters["transformations"]["random_crop_size"],
               rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        Model4(input_image_size=parameters["transformations"]["random_crop_size"],
               darionet_pretrained_path=darionet_pretrained_model_path)
    ]

    losses, psnrs, corrects, = np.zeros(shape=len(models)), \
                               np.zeros(shape=len(models)), \
                               np.zeros(shape=len(models))
    total_times = np.zeros(shape=len(models))

    # creates a folder for the logs of this test
    test_logs_path = join(logs_path, f"test_{str(int(time.time()))}")
    if parameters["test"]["verbose"]:
        mkdir(test_logs_path)
    for i_model, model in enumerate(models):
        # tests the model
        test_results = test_model(model=model, data=imagenet2012_val_loader,
                                  batches_per_epoch=parameters["test"]["batches_per_epoch"],
                                  verbose=parameters["test"]["verbose"], logs_path=test_logs_path)
        losses[i_model], psnrs[i_model], corrects[i_model] = np.mean(test_results["loss"]), \
                                                             np.mean(test_results["psnr"]), \
                                                             np.mean(test_results["corrects"])
        total_times[i_model] = test_results["total_time"]

    print(pd.DataFrame(
        index=[model.name for model in models],
        data={
            "Avg loss": losses,
            "Avg PSNR (dB)": psnrs,
            "Accuracy": corrects / parameters["test"]["batch_size"],
            "Total time (s)": total_times
        }
    ))
