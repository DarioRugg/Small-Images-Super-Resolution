import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import Classifier, RRDB, test_model, show_img

assets_path = f"./assets"
logs_path = f"{assets_path}/logs"
imagenet2012_path = f"{assets_path}/ImageNet2012"
rrdb_pretrained_weights_path = f"{assets_path}/models/RRDB_PSNR_x4.pth"

imagenet2012_dataset = datasets.ImageFolder(root=imagenet2012_path,
                                            transform=transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.RandomCrop(224),
                                                transforms.ToTensor(),
                                                #transforms.Normalize(
                                                 #   mean=[0.485, 0.456, 0.406],
                                                  #  std=[0.229, 0.224, 0.225])
                                            ]))
imagenet2012_loader = DataLoader(imagenet2012_dataset,
                                 batch_size=2, shuffle=True, num_workers=4)

#classifier = Classifier()
#test_model(model=classifier, data=imagenet2012_loader)

rrdb = RRDB(pretrained_weights_path=rrdb_pretrained_weights_path)
with torch.no_grad():
    for batch in imagenet2012_loader:
        X = batch[0].to("cuda")
        output = rrdb(X).data
        show_img(X[0], output[0], save_to_folder=logs_path)
        break
