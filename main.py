import pandas as pd

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils import show_img, prepare_datasets

DATASETS_FOLDER = f"./datasets"
prepare_datasets(DATASETS_FOLDER)

if __name__ == '__main__':
    # retrieves the datasets from the folder
    images = ImageFolder(root=DATASETS_FOLDER,
                         transform=transforms.Compose([
                             transforms.Resize(1440),
                             transforms.RandomCrop(1080),
                             transforms.ToTensor()]))
    # separates train and validation set
    train_labels, val_labels = {encoded_label for label, encoded_label in images.class_to_idx.items()
                                if "train" in label}, \
                               {encoded_label for label, encoded_label in images.class_to_idx.items()
                                if "valid" in label}
    images_train, images_val = Subset(images, [i for i, label in enumerate(images.targets) if label in train_labels]), \
                               Subset(images, [i for i, label in enumerate(images.targets) if label in val_labels])
    # builds the dataloaders out of the datasets
    images_train_loader = DataLoader(images_train, batch_size=50, shuffle=True, num_workers=0)
    images_val_loader = DataLoader(images_val, batch_size=50, shuffle=True, num_workers=0)
    print(pd.DataFrame(data={
        "Total images": [len(images_train), len(images_val)],
        "Batches in dataloader": [len(images_train_loader), len(images_val_loader)],
        "Batch size": [images_train_loader.batch_size, images_val_loader.batch_size]
    }, index=["Training set", "Validation set"]))
    for images_batch in images_train_loader:
        # we'll discard the labels
        images_batch = images_batch[0]
        show_img(images_batch[0])
        break
