from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL.Image import BICUBIC, NEAREST, BILINEAR
import RRDBNet_arch as arch
import pytorch_lightning as pl
from positional_encodings import PositionalEncodingPermute2D
import numpy as np
import pandas as pd

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return np.inf
    return 20 * torch.log10(1 / torch.sqrt(mse))

class Classifier(nn.Module):
    def __init__(self, device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"
        super(Classifier, self).__init__()

        resnet50 = models.resnet50(pretrained=True)
        for parameter in resnet50.parameters():
            parameter.requires_grad = False
        resnet50.eval()

        self.layers = nn.Sequential(
            resnet50
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        X_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])(X)
        out = self.layers(X_normalized)
        return out


class Scaler(nn.Module):
    def __init__(self, size: Union[int, tuple]):


        super(Scaler, self).__init__()

        self.size = size

    def forward(self, X: torch.Tensor):
        if int(self.size) < X.shape[-1]:
            trans = transforms.Compose([
                transforms.GaussianBlur(kernel_size=3, sigma=0.2),
                transforms.Resize(self.size)
            ])
            out = trans(X)
        else:
            out = transforms.Resize(self.size)(X)
        return out


class RRDB(pl.LightningModule):
    def __init__(self, pretrained_weights_path: str = '0', trainable: bool = False, device: str = "auto",
                 nb: int = 23):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device_str = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"
        # checks that the weights are correctly given
        assert isinstance(pretrained_weights_path, str)

        super(RRDB, self).__init__()

        rrdb = arch.RRDBNet(3, 3, 64, nb, gc=32)
        if pretrained_weights_path != '0':
            rrdb.load_state_dict(torch.load(pretrained_weights_path), strict=True)
        if trainable:
            rrdb.train()
        else:
            for parameter in rrdb.parameters():
                parameter.requires_grad = False
            rrdb.eval()

        self.layers = nn.Sequential(
            rrdb
        )
        # moves the entire model to the chosen device
        self.epoch = 0
        self.batch_size = None
        self.lr = 1e-4
        self.to(self.device_str)
        self._ce = nn.CrossEntropyLoss
        self._l1 = nn.L1Loss
        self._resnet = Classifier()
        for par in self._resnet.parameters():
            par.requires_grad = False
        self._epoch_losses, self._epoch_psnrs , self._epoch_pred_loss, self._epoch_CrossEntropy = 0,0,0,0
        self._epoch_val_losses, self._epoch_val_psnrs, self._epoch_val_pred_loss, self._epoch_val_CrossEntropy = 0,0,0,0
        self._counter = 0
        self._counter_val = 0
        self.outputs = []
        self.stats = pd.DataFrame(columns=["epoch", "phase", "avg loss", "avg PSNR", "avg prediction loss", "avg CE"])
    def hook(self, module, input, output):
        self.outputs.append(output.detach())


    def configure_optimizers(
            self,
    ):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):

        X, y = train_batch
        if not self.batch_size:
            self.batch_size = X.shape[0]
        X_downsampled = Scaler(int(X.shape[-1] * 0.32))(X)
        self.train()
        X_supersampled = torch.clip(transforms.Resize(256)(self(X_downsampled)), 0, 1)

        self.outputs = []
        self.handles = []
        for i in range(3):
            self.handles.append(self._resnet.layers[0].layer1[i].conv1.register_forward_hook(self.hook))
            self.handles.append(self._resnet.layers[0].layer1[i].conv2.register_forward_hook(self.hook))
        for i in range(4):
            self.handles.append(self._resnet.layers[0].layer2[i].conv1.register_forward_hook(self.hook))
            self.handles.append(self._resnet.layers[0].layer2[i].conv2.register_forward_hook(self.hook))

        gt_pred = self._resnet(X)

        bl_ce = self._ce()(gt_pred, y)
        y_pred = self._resnet(X_supersampled)

        posenc_loss = 0
        for i in range(len(self.outputs) // 2):
            pos_enc = PositionalEncodingPermute2D(int(self.outputs[i].shape[1]))(self.outputs[i])
            posenc_loss += self._l1()(pos_enc + self.outputs[i], pos_enc + self.outputs[i + len(self.outputs) // 2])
        for handle in self.handles:
            handle.remove()
        CE = self._ce()(y_pred, y)
        pred_loss = self._l1()(y_pred, gt_pred)

        loss = pred_loss + posenc_loss
        self._epoch_losses += loss.item()
        self._epoch_psnrs += psnr(X, X_supersampled).item()

        self._epoch_pred_loss += pred_loss.item()
        self._epoch_CrossEntropy += (CE - bl_ce).item()
        self._counter += 1

        return loss

    def validation_step(self, val_batch, batch_idx):

        X, y = val_batch
        X_downsampled = Scaler(int(X.shape[-1] * 0.32))(X)
        self.eval()
        X_supersampled = torch.clip(transforms.Resize(256)(self(X_downsampled)), 0, 1)

        self.outputs = []
        self.handles = []
        for i in range(3):
            self.handles.append(self._resnet.layers[0].layer1[i].conv1.register_forward_hook(self.hook))
            self.handles.append(self._resnet.layers[0].layer1[i].conv2.register_forward_hook(self.hook))
        for i in range(4):
            self.handles.append(self._resnet.layers[0].layer2[i].conv1.register_forward_hook(self.hook))
            self.handles.append(self._resnet.layers[0].layer2[i].conv2.register_forward_hook(self.hook))

        gt_pred = self._resnet(X)

        bl_ce = self._ce()(gt_pred, y)
        y_pred = self._resnet(X_supersampled)

        posenc_loss = 0
        for i in range(len(self.outputs) // 2):
            pos_enc = PositionalEncodingPermute2D(512)(self.outputs[i])
            posenc_loss += self._l1()(pos_enc + self.outputs[i], pos_enc + self.outputs[i + len(self.outputs) // 2])
        for handle in self.handles:
            handle.remove()

        CE = self._ce()(y_pred, y)
        pred_loss = self._l1()(y_pred, gt_pred)

        loss = pred_loss + posenc_loss
        self._epoch_val_losses+= loss.item()
        self._epoch_val_psnrs+= psnr(X, X_supersampled).item()

        self._epoch_val_pred_loss+= pred_loss.item()
        self._epoch_val_CrossEntropy += (CE - bl_ce).item()
        self._counter_val += 1

    def on_epoch_end(self):

        if self._counter > 0:
            print()
            self.stats = self.stats.append(pd.DataFrame(index=[self.epoch, self.epoch], data={"epoch":[self.epoch, self.epoch], "phase":["train", "val"],
                                            "avg loss":[np.round(self._epoch_losses/(self._counter), 3),np.round(self._epoch_val_losses/(self._counter_val), 3)],
                                           "avg PSNR":[np.round(self._epoch_psnrs/(self._counter), 3),np.round(self._epoch_val_psnrs/(self._counter_val), 3)],
                                           "avg prediction loss":[np.round(self._epoch_pred_loss/(self._counter), 3),np.round(self._epoch_val_pred_loss/(self._counter_val), 3)],
                                           "avg CE":[np.round(self._epoch_CrossEntropy/(self._counter), 3),np.round(self._epoch_val_CrossEntropy/(self._counter_val), 3)]}))
            print(self.stats.to_string(index=False))
            print()
        self._epoch_losses, self._epoch_psnrs, self._epoch_pred_loss, self._epoch_CrossEntropy = 0, 0, 0, 0
        self._epoch_val_losses, self._epoch_val_psnrs, self._epoch_val_pred_loss, self._epoch_val_CrossEntropy = 0, 0, 0, 0
        self._counter = 0
        self._counter_val = 0
        self.epoch += 1

    def forward(self, X: torch.Tensor):
        out = self.layers(X)
        return out




class AddNoise(nn.Module):
    def __init__(self, std: float = 0.025, mean: float = 0, device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"
        assert isinstance(float(std), float) and std >= 0
        assert isinstance(float(mean), float) and mean >= 0
        self.std, self.mean = std, mean

        super(AddNoise, self).__init__()

        # moves the block to the chosen device
        self.to(self.device)

    def forward(self, X):
        X_noisy = X + torch.randn(X.size()).to(self.device) * self.std + self.mean
        return X_noisy
