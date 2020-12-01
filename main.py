from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import Classifier, test_model

assets_path = "./assets"
imagenet2012_path = f"{assets_path}/ImageNet2012"

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
                                 batch_size=50, shuffle=True, num_workers=4)

classifier = Classifier()
test_model(model=classifier, dataloader=imagenet2012_loader)
