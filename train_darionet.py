from os.path import join
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL.Image import BICUBIC
from blocks import RRDB
from utils import train_darionet, read_json
import pytorch_lightning as pl
import time

# parameters object
parameters_path = join(".", "parameters.json")
parameters = read_json(parameters_path)

torch.manual_seed(parameters["training"]["seed"])
np.random.seed(parameters["training"]["seed"])

assets_path = join(".", "assets")
logs_path = join(assets_path, "logs")
imagenet2012_train_path, imagenet2012_val_path = join(assets_path, "train_15_percent"), \
                                                 join(assets_path, "ImageNet2012_val")
models_path = join(assets_path, "models")

# to resume training from a given checkpoint, insert path of the checkpoint here
resume_from = None#join(assets_path, 'checkpoints', 'train_1611668888', 'darionet_epoch_10.pt')
if not resume_from:
    check_path = join(assets_path, 'checkpoints', f"train_{str(int(time.time()))}")
    if not os.path.exists(check_path):
        os.mkdir(check_path)
else:
    check_path = resume_from
rrdb_pretrained_weights_path, DarioNet_pretrained_model_path = join(models_path, "RRDB_PSNR_x4.pth"), \
                                                               join(models_path, "DarioNet_fromzero.pt")

transform_train = transforms.Compose([
    transforms.Resize(parameters["transformations"]["resize_size"]),
    transforms.CenterCrop(parameters["transformations"]["train_crop_size"]),
    transforms.ToTensor()
])

transform_val = transforms.Compose([
    transforms.Resize(parameters["transformations"]["resize_size"]),
    transforms.CenterCrop(parameters["transformations"]["val_crop_size"]),
    transforms.ToTensor()
])
imagenet2012_train_dataset, imagenet2012_val_dataset = datasets.ImageFolder(root=imagenet2012_train_path,
                                                                            transform=transform_train), \
                                                       datasets.ImageFolder(root=imagenet2012_val_path,
                                                                            transform=transform_val)
if __name__ == '__main__':
    imagenet2012_train_loader, imagenet2012_val_loader = DataLoader(imagenet2012_train_dataset, num_workers=4,
                                                                    batch_size=parameters["training"]["batch_size"],
                                                                    shuffle=parameters["training"]["shuffle"], pin_memory=True), \
                                                         DataLoader(imagenet2012_val_dataset, num_workers=4,
                                                                    batch_size=parameters["test"]["batch_size"],
                                                                    shuffle=False, pin_memory=True)

    darionet = RRDB(nb=3, trainable=True)
    # train_darionet(model=darionet, data_train=imagenet2012_train_loader, data_val=imagenet2012_val_loader,
    #                epochs=parameters["training"]["epochs"], lr=parameters['training']['learning_rate'],
    #                batches_per_epoch=parameters["training"]["batches_per_epoch"], filepath=DarioNet_pretrained_model_path,
    #                scale=parameters['transformations']['scale'], train_crop_size=parameters['transformations']['train_crop_size'],
    #                val_crop_size=parameters['transformations']['val_crop_size'], save=parameters['training']['save'],
    #                checkpoints=check_path)
    trainer = pl.Trainer(gpus=1, precision=16, num_sanity_val_steps=4, max_epochs=parameters['training']['epochs'], benchmark=True)
    trainer.tune(darionet, imagenet2012_train_loader, imagenet2012_val_loader)
    trainer.fit(darionet, imagenet2012_train_loader, imagenet2012_val_loader)