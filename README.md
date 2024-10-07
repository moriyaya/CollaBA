# [PR 2024] Collaborative brightening and amplification of low-light imagery via bi-level adversarial learning [[Paper]](https://www.sciencedirect.com/science/article/pii/S0031320324003091)
By Jiaxin Gao, Yaohua Liu, Ziyu Yue, Risheng Liu, and Xin Fan

## Pipeline
![](./Figures/pipeline.jpg)

## Dependencies
```
pip install -r requirements.txt
````

## Download the raw training and evaluation datasets
### Paired datasets 
RELLISUR dataset: Andreas Aakerberg, Kamal Nasrollahi, Thomas Moeslund. "RELLISUR: A Real Low-Light Image Super-Resolution Dataset". NeurIPS Datasets and Benchmarks 2021. [UIEB](https://ieeexplore.ieee.org/document/8917818)

DarkFace dataset: Jiang Hai, Zhu Xuan, Ren Yang, Yutong Hao, Fengzhu Zou, Fang Lin, and Songchen Han. "R2RNet: Low-light Image Enhancement via Real-low to Real-normal Network". Journal of Visual Communication and Image Representation, 2023. [[Baiduyun (extracted code: wmrr)]](https://pan.baidu.com/s/1XHWQAS0ZNrnCyZ-bq7MKvA)

### Unpaired datasets 
Please refer to [[Project Page of RetinexNet]](https://daooshee.github.io/BMVC2018website/).

## Pre-trained Models 
You can download our pre-trained model from [[Google Drive]](https://drive.google.com/drive/folders/1m3t15rWw76IDDWJ0exLOe5P0uEnjk3zl?usp=drive_link) and [[Baidu Yun (extracted code:cjzk)]](https://pan.baidu.com/s/1fPLVgnZbdY1n75Flq54bMQ)

## How to train?
You need to modify ```datasets/dataset.py``` slightly for your environment, and then
```
python train.py  
```

## How to test?
```
python evaluate.py
```

## Visual comparison
![](./Figures/visual.jpg)

## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@InProceedings{Jiang_2024_ECCV,
    author    = {Jiang, Hai and Luo, Ao and Liu, Xiaohong and Han, Songchen and Liu, Shuaicheng},
    title     = {LightenDiffusion: Unsupervised Low-Light Image Enhancement with Latent-Retinex Diffusion Models},
    booktitle = {European Conference on Computer Vision},
    year      = {2024},
    pages     = {}
}
```

## Acknowledgement
Part of the code is adapted from previous works: [WeatherDiff](https://github.com/IGITUGraz/WeatherDiffusion) and [MIMO-UNet](https://github.com/chosj95/MIMO-UNet). We thank all the authors for their contributions.

