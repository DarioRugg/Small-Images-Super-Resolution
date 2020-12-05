from os.path import join

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models import Model1, Model2, Model3, Model4
from utils import train_darionet, test_model, show_img

seed = 69420
torch.manual_seed(seed)
np.random.seed(seed)

assets_path = join(".", "assets")
logs_path = join(assets_path, "logs")
imagenet2012_path = join(assets_path, "ImageNet2012")
models_path = join(assets_path, "models")
rrdb_pretrained_weights_path, darionet_pretrained_model_path = join(models_path, "RRDB_PSNR_x4.pth"), \
                                                               join(models_path, "DarioNet.pt")
batch_size, epochs, early_stop_batches = 2, 1, 20

transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.01),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()
])

imagenet2012_val_dataset = datasets.ImageFolder(root=imagenet2012_path, transform=transforms)

imagenet2012_val_loader = DataLoader(imagenet2012_val_dataset, num_workers=4,
                                     batch_size=batch_size, shuffle=False)

# computes tests on the different models
models = [
    Model1(),
    Model2(rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
    Model3(),
    Model4(darionet_pretrained_path=darionet_pretrained_model_path)
]

losses, psnrs, corrects, = np.zeros(shape=len(models)), \
                           np.zeros(shape=len(models)), \
                           np.zeros(shape=len(models))
total_times = np.zeros(shape=len(models))
for i_model, model in enumerate(models):
    test_results = test_model(model=model, data=imagenet2012_val_loader,
                              early_stop=early_stop_batches, verbose=False)
    losses[i_model], psnrs[i_model], corrects[i_model] = np.mean(test_results["loss"]), \
                                                         np.mean(test_results["psnr"]), \
                                                         np.mean(test_results["corrects"])
    total_times[i_model] = test_results["total_time"]

print(pd.DataFrame(
    index=[f"Model {i + 1}" for i in range(len(models))],
    data={
        "Avg loss": losses,
        "Avg PSNR (dB)": psnrs,
        "Accuracy": corrects / batch_size,
        "Total time (s)": total_times
    }
))

