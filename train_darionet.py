from os.path import join

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from blocks import RRDB
from utils import train_darionet, read_json

# parameters object
parameters_path = join(".", "parameters.json")
parameters = read_json(parameters_path)

torch.manual_seed(parameters["training"]["seed"])
np.random.seed(parameters["training"]["seed"])

assets_path = join(".", "assets")
logs_path = join(assets_path, "logs")
imagenet2012_train_path, imagenet2012_val_path = join(assets_path, "ImageNet2012_train"), \
                                                 join(assets_path, "ImageNet2012_val")
models_path = join(assets_path, "models")
rrdb_pretrained_weights_path, DarioNet_pretrained_model_path = join(models_path, "RRDB_PSNR_x4.pth"), \
                                                               join(models_path, "DarioNet.pt")
darionet_out_path = join(models_path, "DarioNet1.pt")

transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=parameters["transformations"]["random_horizontal_flip_probability"]),
    transforms.RandomVerticalFlip(p=parameters["transformations"]["random_vertical_flip_probability"]),
    transforms.Resize(parameters["transformations"]["resize_size"]),
    transforms.RandomCrop(parameters["transformations"]["random_crop_size"]),
    transforms.ToTensor()
])

imagenet2012_train_dataset, imagenet2012_val_dataset = datasets.ImageFolder(root=imagenet2012_train_path,
                                                                            transform=transforms), \
                                                       datasets.ImageFolder(root=imagenet2012_val_path,
                                                                            transform=transforms)
if __name__ == '__main__':
    imagenet2012_train_loader, imagenet2012_val_loader = DataLoader(imagenet2012_train_dataset, num_workers=4,
                                                                    batch_size=parameters["training"]["batch_size"],
                                                                    shuffle=parameters["training"]["shuffle"], pin_memory=True), \
                                                         DataLoader(imagenet2012_val_dataset, num_workers=4,
                                                                    batch_size=parameters["training"]["batch_size"],
                                                                    shuffle=parameters["training"]["shuffle"], pin_memory=True)

    darionet = torch.load(DarioNet_pretrained_model_path)
    train_darionet(model=darionet, filepath=darionet_out_path,
                   data_train=imagenet2012_train_loader, data_val=imagenet2012_val_loader,
                   epochs=parameters["training"]["epochs"],
                   batches_per_epoch=parameters["training"]["batches_per_epoch"])
