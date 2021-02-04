from typing import Union
from os.path import join
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.utils import save_image
from PIL import Image
import RRDBNet_arch as arch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import math

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return np.inf
    return 20 * torch.log10(1 / torch.sqrt(mse))

def show_img(*imgs: torch.Tensor, filename: str = None, save_to_folder: str = None):
    assert not save_to_folder or isinstance(save_to_folder, str)
    imgs = list(imgs)[0]
    for i_img, img in enumerate(imgs):
        assert isinstance(img, torch.Tensor)
        assert len(img.shape) == 3
        if save_to_folder:
            filename = filename if filename else f"img_{i_img}"
            save_image(img, join(save_to_folder, f"{filename}.png"))
        imgs[i_img] = img.permute(1, 2, 0).to("cpu").numpy()
    fig, axs = plt.subplots(1, len(imgs), squeeze=False)
    for i_ax, ax in enumerate(axs.flat):
        ax.imshow(imgs[i_ax])
    plt.show()

def positionalencoding2d(d_model, tensor):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    height, width = tensor.shape[2:]
    pe = torch.zeros(d_model, height, width, device=tensor.device)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2, device=tensor.device) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width, device=tensor.device).unsqueeze(1)
    pos_h = torch.arange(0., height, device=tensor.device).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe

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
        X = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])(X)
        out = self.layers(X)
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
                 nb: int = 23, lr: float = 1e-4):
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
        self.lr = lr
        self.to(self.device_str)
        self._ce = nn.CrossEntropyLoss
        self._mse = nn.MSELoss
        self._resnet = Classifier()
        for par in self._resnet.parameters():
            par.requires_grad = False
        self._epoch_losses, self._epoch_psnrs , self._epoch_pred_loss, self._epoch_CrossEntropy = 0,0,0,0
        self._epoch_val_losses, self._epoch_val_psnrs, self._epoch_val_pred_loss, self._epoch_val_CrossEntropy = 0,0,0,0
        self._best_loss = np.inf
        self._counter = 0
        self._counter_val = 0
        self.outputs = []
        self.handles = []
        self.stats = pd.DataFrame(columns=["epoch", "phase", "avg loss", "avg PSNR", "avg prediction loss", "avg CE"])
    def hook(self, module, input, output):
        self.outputs.append(output)


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
        with torch.cuda.amp.autocast():
            self.train()
            X_supersampled = transforms.Resize(256)(self(X_downsampled))

            for i in range(3):

                self.handles.append(self._resnet.layers[0].layer1[i].conv3.register_forward_hook(self.hook))
            for i in range(4):

                self.handles.append(self._resnet.layers[0].layer2[i].conv3.register_forward_hook(self.hook))

            self.handles.append(self._resnet.layers[0].layer3[0].conv2.register_forward_hook(self.hook))
            self.handles.append(self._resnet.layers[0].layer4[0].conv2.register_forward_hook(self.hook))

            gt_pred = self._resnet(X)

            bl_ce = self._ce()(gt_pred, y)
            y_pred = self._resnet(X_supersampled)

            percep_loss = 0
            for i in range(len(self.outputs) // 2):
                percep_loss += self._mse()(
                    self.outputs[i], self.outputs[i + len(self.outputs) // 2])
            percep_loss /= len(self.outputs) // 2
            for handle in self.handles:
                handle.remove()
            self.outputs = []
            self.handles = []

            CE = self._ce()(y_pred, y)
            pred_loss = self._mse()(y_pred, gt_pred)

            loss = pred_loss + percep_loss + self._mse()(X_supersampled, X)
            self._epoch_losses += loss.item()
            self._epoch_psnrs += psnr(X, X_supersampled).item()

            self._epoch_pred_loss += pred_loss.item()
            self._epoch_CrossEntropy += (CE - bl_ce).item()
            self._counter += 1

        return loss

    def validation_step(self, val_batch, batch_idx):

        X, y = val_batch
        X_downsampled = Scaler(int(X.shape[-1] * 0.32))(X)

        with torch.cuda.amp.autocast():
            self.eval()
            X_supersampled = transforms.Resize(256)(self(X_downsampled))

            for i in range(3):
                self.handles.append(self._resnet.layers[0].layer1[i].conv1.register_forward_hook(self.hook))
                self.handles.append(self._resnet.layers[0].layer1[i].conv2.register_forward_hook(self.hook))
                self.handles.append(self._resnet.layers[0].layer1[i].conv3.register_forward_hook(self.hook))
            for i in range(4):
                self.handles.append(self._resnet.layers[0].layer2[i].conv1.register_forward_hook(self.hook))
                self.handles.append(self._resnet.layers[0].layer2[i].conv2.register_forward_hook(self.hook))
                self.handles.append(self._resnet.layers[0].layer2[i].conv3.register_forward_hook(self.hook))

            self.handles.append(self._resnet.layers[0].layer3[0].conv2.register_forward_hook(self.hook))
            self.handles.append(self._resnet.layers[0].layer4[0].conv2.register_forward_hook(self.hook))

            gt_pred = self._resnet(X)

            bl_ce = self._ce()(gt_pred, y)
            y_pred = self._resnet(X_supersampled)

            percep_loss = 0
            for i in range(len(self.outputs) // 2):
                percep_loss += self._mse()(
                     self.outputs[i], self.outputs[i + len(self.outputs) // 2])
            percep_loss /= len(self.outputs)//2
            for handle in self.handles:
                handle.remove()
            self.outputs = []
            self.handles = []

            CE = self._ce()(y_pred, y)
            pred_loss = self._mse()(y_pred, gt_pred)

            loss = pred_loss + percep_loss + self._mse()(X_supersampled, X)
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
        val_loss = self._epoch_val_losses/self._counter_val
        if  val_loss < self._best_loss:
            print(f"Found Best Model With Loss: {val_loss}")
            self.cpu()
            torch.save(self.layers.state_dict(), join('assets', 'models', 'Darionet_percep.pth'))
            self.to(self.device_str)
            self._best_loss = val_loss

        self._epoch_losses, self._epoch_psnrs, self._epoch_pred_loss, self._epoch_CrossEntropy = 0, 0, 0, 0
        self._epoch_val_losses, self._epoch_val_psnrs, self._epoch_val_pred_loss, self._epoch_val_CrossEntropy = 0, 0, 0, 0
        self._counter = 0
        self._counter_val = 0
        self.epoch += 1
        goldfish = join('assets', 'sample_images', 'pesce_rosso.jpg')
        colosseum = join('assets', 'sample_images', 'colosseo.png')
        imgs = [Image.open(goldfish), Image.open(colosseum)]
        imgs = [transforms.ToTensor()(transforms.CenterCrop(256)(transforms.Resize(256)(X))).to('cuda') for X in imgs]
        X_downsampled = [Scaler(int(X.shape[-1] * 0.32))(X) for X in imgs]
        X_upsampled = [self(X.unsqueeze(dim=0))[0].detach() for X in X_downsampled]
        show_img(X_downsampled)
        show_img(X_upsampled)

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
