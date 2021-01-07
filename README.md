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
You can run an ![interactive demo](https://colab.research.google.com/drive/1SjIv-DGM3X2QDy1SY_hVz-hn75dqsYB_?usp=sharing) in Google Colab just by clicking on the banner beneath the title of this README.

### Assets used in this project

Firstly be sure to have the correct assets downloaded in the right places! 

Download the **ImageNet2012 training and validation datasets** from [ImageNet](www.image-net.org) and extract it into the `assets` folder, in subfolders called `ImageNet2012_train` and `ImageNet2012_val`. 
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
        models
            DarioNet.pt
            RRDB_PSNR_x4.pth
            RRDBNet_arch.py
    train_darionet.py
    ...
```



## Resources
- ![Brief report](https://drive.google.com/file/d/1zC_8VIzTyizQplvMEEK7Jqbuh557_V79/view?usp=sharing) of project's development and performances
- ![Final presentation](https://docs.google.com/presentation/d/1aNI4n8BEENaebQcXuPHGfZCY1ghCCCj-523Y_etqXE4/edit?usp=sharing) of the project, publicly explained on December 7th 2021
- ![Interactive demo](https://colab.research.google.com/drive/1SjIv-DGM3X2QDy1SY_hVz-hn75dqsYB_?usp=sharing) on Google Colab
