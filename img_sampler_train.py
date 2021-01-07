import os
from math import ceil
from random import sample, seed
from shutil import copyfile
from tqdm import tqdm
seed(2020)
# percentage of sampled images (between 0 and 1)
perc = 0.15

# path where the dataset is at the moment (from the local folder)
data_path = os.path.join(".", "assets", "ImageNet2012_train_original")
output_path = os.path.join(".", "assets", "ImageNet2012_train")


if not os.path.exists(output_path):
    os.makedirs(output_path)

for class_folder in tqdm(os.listdir(data_path)):
    class_folder_path = os.path.join(data_path, class_folder)

    images = os.listdir(class_folder_path)

    sampled_images = sample(images, ceil(len(images) * perc))

    if sampled_images:

        destination_class_folder = os.path.join(output_path, class_folder)

        if not os.path.exists(destination_class_folder):
            os.makedirs(destination_class_folder)

        for image in sampled_images:
            copyfile(os.path.join(class_folder_path, image),
                     os.path.join(destination_class_folder, image))

    else:
        print("Floder " + class_folder_path + " has too few observations maybe. ",
              "\n - Number of observations sampled: " + str(len(sampled_images)),
              "\n - Total images in that folder: " + str(len(images)))
