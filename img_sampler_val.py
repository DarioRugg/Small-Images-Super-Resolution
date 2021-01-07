import os
from random import shuffle, seed
from shutil import copyfile
from tqdm import tqdm
seed(2020)

# path where the dataset is at the moment (from the local folder)
data_path = os.path.join(".", "assets", "ImageNet2012_val_original")
output_path = os.path.join(".", "assets")


output_val_path = os.path.join(output_path, "ImageNet2012_val")
output_test_path = os.path.join(output_path, "ImageNet2012_test")

if not os.path.exists(output_val_path):
    os.makedirs(output_val_path)
if not os.path.exists(output_test_path):
    os.makedirs(output_test_path)

for class_folder in tqdm(os.listdir(data_path)):
    class_folder_path = os.path.join(data_path, class_folder)

    images = os.listdir(class_folder_path)

    shuffle(images)

    sampled_val = images[:len(images)//2]
    sampled_test = images[len(images) // 2:]

    if sampled_val:

        destination_class_folder = os.path.join(output_val_path, class_folder)

        if not os.path.exists(destination_class_folder):
            os.makedirs(destination_class_folder)

        for image in sampled_val:
            copyfile(os.path.join(class_folder_path, image),
                     os.path.join(destination_class_folder, image))

    else:
        print("Floder " + class_folder_path + " has too few observations maybe. ",
              "\n - Number of observations sampled: " + str(len(sampled_val)),
              "\n - Total images in that folder: " + str(len(images)))

    if sampled_test:

        destination_class_folder = os.path.join(output_test_path, class_folder)

        if not os.path.exists(destination_class_folder):
            os.makedirs(destination_class_folder)

        for image in sampled_test:
            copyfile(os.path.join(class_folder_path, image),
                     os.path.join(destination_class_folder, image))

    else:
        print("Floder " + class_folder_path + " has too few observations maybe. ",
              "\n - Number of observations sampled: " + str(len(sampled_test)),
              "\n - Total images in that folder: " + str(len(images)))
