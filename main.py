from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import Classifier, RRDB, test_model

assets_path = "./assets"
imagenet2012_path = f"{assets_path}/ImageNet2012"
rrdb_pretrained_weights_path = f"{assets_path}/models/RRDB_PSNR_x4.pth"

imagenet2012_dataset = datasets.ImageFolder(root=imagenet2012_path,
                                            transform=transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.RandomCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                            ]))
imagenet2012_loader = DataLoader(imagenet2012_dataset,
                                 batch_size=1, shuffle=True, num_workers=4)

#classifier = Classifier()
#test_model(model=classifier, dataloader=imagenet2012_loader)

rrdb = RRDB(pretrained_weights_path=rrdb_pretrained_weights_path)
for batch in imagenet2012_loader:
    X = batch[0].to("cuda")
    print(X.shape)
    output = rrdb(X).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    print(output)
