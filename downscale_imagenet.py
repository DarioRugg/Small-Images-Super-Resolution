from os.path import join
from os import listdir
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

from blocks import Scaler
from blocks import RRDB
from utils import train_darionet, read_json


assets_path = join(".", "assets")
logs_path = join(assets_path, "logs")
imagenet2012_train_path, imagenet2012_val_path = join(assets_path, "ImageNet2012_train"), \
                                                 join(assets_path, "ImageNet2012_val")


# lr_dir = join(assets_path, "ImageNet2012_train_LoRes")
# if not os.path.exists(lr_dir): os.makedirs(lr_dir)
# for class_folder in listdir(imagenet2012_train_path):
#         dest = join(lr_dir, class_folder)
#         if not os.path.exists(dest): os.makedirs(dest)
#         class_path = join(imagenet2012_train_path, class_folder)
#         for img_name in listdir(class_path):
#             img = Image.open(join(class_path, img_name))
#             X = transforms.ToTensor()(img)
#             low_res = Scaler((X.shape[1] // 4, X.shape[-1] // 4))(X)
#             transforms.ToPILImage()(low_res).save(join(dest, img_name))


lr_dir = join(assets_path, "ImageNet2012_val_LoRes")
if not os.path.exists(lr_dir): os.makedirs(lr_dir)

for class_folder in listdir(imagenet2012_val_path):
    dest = join(lr_dir, class_folder)
    if not os.path.exists(dest): os.makedirs(dest)
    class_path = join(imagenet2012_val_path, class_folder)
    for img_name in listdir(class_path):
        img = Image.open(join(class_path, img_name))
        X = transforms.ToTensor()(img)
        low_res = Scaler(X.shape[-1] // 4)(X)
        transforms.ToPILImage()(low_res).save(join(dest, img_name))