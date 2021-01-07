# Advanced Machine Learning course 2020/21
# Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SjIv-DGM3X2QDy1SY_hVz-hn75dqsYB_?usp=sharing)

## Who are we?
| Name | Matricola |
| --- | --- |
| Riccardo Ceccaroni | 1884368 |
| Simone Ercolino | 1587229 |
| Romeo Lanzino | 1753403 |
| Dario Ruggeri | 1741637 |

## How to check our results?
### The quick way: Google Colab
You can run an [interactive demo](https://colab.research.google.com/drive/1SjIv-DGM3X2QDy1SY_hVz-hn75dqsYB_?usp=sharing) in Google Colab just by clicking on the banner beneath the title of this README.

### The DIY way: run our code on your machine

#### Test the model
Just run `try_darionet.py <img_path>` replacing the argument with the path of an image that you want to upscale and classify.

Beware that ImageNet2012 has just 1000 labels that you can check into `assets/labels.json`.

This script will plot the image downscaled at different sizes and then reconstructed by DarioNet, with a final classification plot.

#### Train the model

Firstly be sure to have the correct assets downloaded in the right places! 

Download the **ImageNet2012 training and validation datasets** from [ImageNet](www.image-net.org) and extract it into the `assets` folder, in subfolders called `ImageNet2012_train`, and split the validation set in 2 halves, called `ImageNet2012_val` and `ImageNet2012_test` (taking the first 50% of images from each folder of the validation set).
Be aware that **this dataset is not publicly available**, so you'll have to register to the website with your academic credentials to download it.

After this procedure, you should have a similar folder structure:

```
aml_project
    assets
        ImageNet2012_train
            n01440764
            n01443537
            n01484850
            n01491361
            ...
        ImageNet2012_val
            n01414164
            n01412039
            n01483123
            n01491131
            ...
        ImageNet2012_test
            n01414164
            n01412039
            n01483123
            n01491131
            ...
        models
            DarioNet.pt
            RRDB_PSNR_x4.pth
            RRDBNet_arch.py
    train_darionet.py
    try_darionet.py
    ...
```

Then just run `train_darionet.py` and follow the instructions in the script.


## Resources
- [Brief report](https://github.com/rom42pla/aml_project/blob/main/report.pdf) of project's development and performances
- [Final presentation](https://github.com/rom42pla/aml_project/blob/main/presentation.pdf) of the project, publicly explained on December 7th 2021
- [Interactive demo](https://colab.research.google.com/drive/1SjIv-DGM3X2QDy1SY_hVz-hn75dqsYB_?usp=sharing) on Google Colab
