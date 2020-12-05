import sys

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models import Model1, Model2, Model3
from blocks import RRDB
from utils import train_model, test_model, show_img

seed = 69420
torch.manual_seed(seed)
np.random.seed(seed)

assets_path = f"./assets"
logs_path = f"{assets_path}/logs"
imagenet2012_path = f"{assets_path}/ImageNet2012"
rrdb_pretrained_weights_path = f"{assets_path}/models/RRDB_PSNR_x4.pth"
batch_size, early_stop_batches = 3, 1000

transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.01),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()
])

imagenet2012_val_dataset = datasets.ImageFolder(root=imagenet2012_path, transform=transforms)

imagenet2012_val_loader = DataLoader(imagenet2012_val_dataset, num_workers=4,
                                     batch_size=batch_size, shuffle=False)
'''
# computes tests on the different models
models = [
    Model1(),
    Model2(rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
    Model3()
]
# print(Model2(rrdb_pretrained_weights_path=rrdb_pretrained_weights_path).layers[2])
# sys.exit()
losses, psnrs, corrects, = np.zeros(shape=len(models)), \
                           np.zeros(shape=len(models)), \
                           np.zeros(shape=len(models))
total_times = np.zeros(shape=len(models))
for i_model, model in enumerate(models):
    test_results = test_model(model=model, data=imagenet2012_test_loader, early_stop=early_stop_batches, verbose=False)
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
'''

DarioNet = RRDB(pretrained_weights_path=rrdb_pretrained_weights_path, trainable=True)
train_model(model=DarioNet,
            data_train=imagenet2012_val_loader, data_val=imagenet2012_val_loader,
            epochs=2, batches_per_epoch=early_stop_batches)
