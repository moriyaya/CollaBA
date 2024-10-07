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
RELLISUR dataset: Andreas Aakerberg, Kamal Nasrollahi, Thomas Moeslund. "RELLISUR: A Real Low-Light Image Super-Resolution Dataset". NeurIPS Datasets and Benchmarks 2021. [RELLISUR](https://vap.aau.dk/rellisur/)

### Unpaired datasets 
Please refer to DARK FACE dataset: Yang, Wenhan and Yuan, Ye and Ren, Wenqi and Liu, Jiaying and Scheirer, Walter J. and Wang, Zhangyang and Zhang, and et al. "DARK FACE: Face Detection in Low Light Condition". IEEE Transactions on Image Processing, 2020. [DARK FACE](https://flyywh.github.io/CVPRW2019LowLight/)

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
![](./visual.pdf)

## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@article{gao2024collaborative,
  title={Collaborative brightening and amplification of low-light imagery via bi-level adversarial learning},
  author={Gao, Jiaxin and Liu, Yaohua and Yue, Ziyu and Fan, Xin and Liu, Risheng},
  journal={Pattern Recognition},
  volume={154},
  pages={110558},
  year={2024},
  publisher={Elsevier}
}
```

## Acknowledgement
Part of the code is adapted from previous works: [SwinIR](https://github.com/JingyunLiang/SwinIR) and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (code structure). We thank all the authors for their contributions.

Please contact me if you have any questions at: jiaxinn.gao@outlook.com

