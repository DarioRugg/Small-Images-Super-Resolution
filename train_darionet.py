from os.path import join

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from blocks import RRDB
from utils import train_darionet

seed = 69420
torch.manual_seed(seed)
np.random.seed(seed)

assets_path = join(".", "assets")
logs_path = join(assets_path, "logs")
imagenet2012_path = join(assets_path, "ImageNet2012")
models_path = join(assets_path, "models")
rrdb_pretrained_weights_path, DarioNet_pretrained_model_path = join(models_path, "RRDB_PSNR_x4.pth"), \
                                                               join(models_path, "DarioNet.pt")
batch_size, epochs, early_stop_batches = 2, 1, 20

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

darionet = RRDB(pretrained_weights_path=rrdb_pretrained_weights_path, trainable=True)
train_darionet(model=darionet, filepath=DarioNet_pretrained_model_path,
               data_train=imagenet2012_val_loader, data_val=imagenet2012_val_loader,
               epochs=epochs, batches_per_epoch=early_stop_batches)
