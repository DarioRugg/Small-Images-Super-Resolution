from os import listdir
from os.path import join

import cv2

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
dataset_description_path = join(assets_path, "dataset_description.csv")
batch_size, epochs, early_stop_batches = 2, 1, 20

labels, widths, heights = [], [], []
for i_class, class_name in enumerate(listdir(imagenet2012_path)):
    for img_name in listdir(join(imagenet2012_path, class_name)):
        img_path = join(imagenet2012_path, class_name, img_name)
        img = cv2.imread(img_path)

        labels += [class_name]
        widths += [img.shape[0]]
        heights += [img.shape[1]]
    print(f"Retrieved class {i_class + 1} of {len(listdir(imagenet2012_path))}")

sizes = pd.DataFrame(data={
    "label": labels,
    "width": widths,
    "heights": heights
}, index=list(range(len(labels))))
sizes.to_csv(path_or_buf=dataset_description_path)
print(sizes)
