import argparse

parser = argparse.ArgumentParser(description='Try DarioNet to apply 4x Super Resolution to an image.')
parser.add_argument('img_path', type=str,
                    help='image on which apply 4x Super Resolution')

args = parser.parse_args()

# loads the image
from os.path import join, splitext, basename

from PIL import Image
import torch
from torchvision import transforms

from utils import show_img

device = "cuda" if torch.cuda.is_available() else "cpu"

img = Image.open(args.img_path).convert('RGB')
img = transforms.Compose([
    transforms.Resize(min(*img.size) // 4),
    transforms.ToTensor()
])(img).to(device)
show_img(img, filename=f"{splitext(basename(args.img_path))[0]}_LR", save_to_folder=".")

# loads DarioNet
assets_path = join(".", "assets")
models_path = join(assets_path, "models")
darionet_pretrained_model_path = join(models_path, "DarioNet.pt")

# applies the super resolution
darionet = torch.load(darionet_pretrained_model_path).to(device)
with torch.no_grad():
    img_super_resolution = darionet(img.unsqueeze(dim=0))[0]
show_img(img_super_resolution, filename=f"{splitext(basename(args.img_path))[0]}_SR", save_to_folder=".")


