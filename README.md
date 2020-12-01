# Advanced Machine Learning course 2020/21
# Project

## Who are we?
| Name | Matricola |
| --- | --- |
| Riccardo Ceccaroni | 1884368 |
| Simone Ercolino | 1587229 |
| Romeo Lanzino | 1753403 |
| Dario Ruggeri | 1741637 |

## How to check our results?
### Assets used in this project

Firstly be sure to have the correct assets downloaded in the right places! 

Download the **ImageNet2012 validation dataset** from [ImageNet](www.image-net.org) and extract it into the assets folder, in a folder called `ImageNet2012`. 
Be aware that **this dataset is not publicly available**, so you'll have to register to the website with your academic credentials to download it.

You'll have to also download a **[pretrained model for the super resolution task](https://drive.google.com/file/d/1pJ_T-V1dpb1ewoEra1TGSWl5e6H7M4NN/view?usp=sharing)** and the [relative classes](https://raw.githubusercontent.com/xinntao/ESRGAN/master/RRDBNet_arch.py), placing them into `model` folder.

After this procedure, you should have a similar folder structure:

```
aml_project
    assets
        ImageNet2012
            n01440764
            n01443537
            n01484850
            n01491361
            ...
        models
            RRDB_PSNR_x4.pth
            RRDBNet_arch.py
    ...
```

## Resources
TBD
