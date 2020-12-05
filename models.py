import torch
from torch import nn

from blocks import Classifier, Scaler, RRDB, AddNoise


class Model1(nn.Module):
    def __init__(self, input_image_size: int,
                 name: str = "No downsampling", device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"
        # checks that the input image size is correctly given
        assert isinstance(input_image_size, int) and input_image_size >= 224
        self.input_image_size = input_image_size
        # checks that the entered name is correct
        assert isinstance(name, str)
        self.name = name

        super(Model1, self).__init__()

        self.layers = nn.Sequential(
            Classifier()
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        X_downsampled = Scaler(self.input_image_size // 4)(X)
        X_upsampled = X
        y_pred = self.layers(X)
        return X_downsampled, X_upsampled, y_pred


class Model2(nn.Module):
    def __init__(self, input_image_size: int,
                 name: str = "Bicubic downsampling", device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"
        # checks that the input image size is correctly given
        assert isinstance(input_image_size, int) and input_image_size >= 224
        self.input_image_size = input_image_size
        # checks that the entered name is correct
        assert isinstance(name, str)
        self.name = name

        super(Model2, self).__init__()

        self.layers = nn.Sequential(
            Scaler(input_image_size // 4),
            Scaler(input_image_size),
            Classifier()
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        X_downsampled = self.layers[:2](X)
        X_upsampled = self.layers[2:-1](X_downsampled)
        y_pred = self.layers[-1](X_upsampled)
        return X_downsampled, X_upsampled, y_pred


class Model3(nn.Module):
    def __init__(self, input_image_size: int,
                 rrdb_pretrained_weights_path: str,
                 name: str = "RRDB", device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"
        # checks that the weights are correctly given
        assert isinstance(rrdb_pretrained_weights_path, str)
        # checks that the input image size is correctly given
        assert isinstance(input_image_size, int) and input_image_size >= 224
        self.input_image_size = input_image_size
        # checks that the entered name is correct
        assert isinstance(name, str)
        self.name = name

        super(Model3, self).__init__()

        self.layers = nn.Sequential(
            Scaler(input_image_size // 4),
            RRDB(pretrained_weights_path=rrdb_pretrained_weights_path),
            Classifier()
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        X_downsampled = self.layers[:2](X)
        X_upsampled = self.layers[2:-1](X_downsampled)
        y_pred = self.layers[-1](X_upsampled)
        return X_downsampled, X_upsampled, y_pred


class Model4(nn.Module):
    def __init__(self, input_image_size: int,
                 darionet_pretrained_path: str,
                 name: str = "DarioNet", device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else \
            "cuda" if torch.cuda.is_available() else "cpu"
        # checks that the weights are correctly given
        assert isinstance(darionet_pretrained_path, str)
        # checks that the input image size is correctly given
        assert isinstance(input_image_size, int) and input_image_size >= 224
        self.input_image_size = input_image_size
        # checks that the entered name is correct
        assert isinstance(name, str)
        self.name = name

        super(Model4, self).__init__()

        self.layers = nn.Sequential(
            Scaler(input_image_size // 4),
            torch.load(darionet_pretrained_path),
            Classifier()
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        X_downsampled = self.layers[:2](X)
        X_upsampled = self.layers[2:-1](X_downsampled)
        y_pred = self.layers[-1](X_upsampled)
        return X_downsampled, X_upsampled, y_pred
