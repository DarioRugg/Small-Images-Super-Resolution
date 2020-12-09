import argparse

parser = argparse.ArgumentParser(description='Try DarioNet to apply 4x Super Resolution to an image.')
parser.add_argument('img_path', type=str,
                    help='image on which apply 4x Super Resolution')

args = parser.parse_args()

# loads the image
from PIL import Image
import torch
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

img = Image.open(args.img_path).convert('RGB')
img = transforms.Compose([
    transforms.Resize(min(*img.size)),
    transforms.ToTensor()
])(img).to(device)

# loads DarioNet
from os.path import join

assets_path = join(".", "assets")
models_path = join(assets_path, "models")
darionet_pretrained_model_path = join(models_path, "DarioNet.pt")

darionet = torch.load(darionet_pretrained_model_path).to(device)
with torch.no_grad():
    img_super_resolution = darionet(img.unsqueeze(dim=0))
