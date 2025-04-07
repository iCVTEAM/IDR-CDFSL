<h1 align="left"> Free Lunch to Meet the Gap: Intermediate Domain Reconstruction for Cross-Domain Few-Shot Learning </h1>


## Introduction


This repository contains code for the paper named [Free Lunch to Meet the Gap: Intermediate Domain Reconstruction for Cross-Domain Few-Shot Learning](https://arxiv.org/abs/). 

##  Quick Started

### 1. Environment Set Up
Clone this repository and install packages.
```bash
Python 3.8
PyTorch 1.7.1 or higher
torchvision 0.8.2 or higher
numpy
```

### 2. Download Pretrained Weights

Please download our checkpoints from [Google Drive](https://drive.google.com/file/d/1-q90yVBg7-D2nOQgjM6lSgGMkJtCPpZx/view?usp=sharing) and put it in `./results/`.


### 3. Evaluate on your own dataset.

```bash
python meta_test.py --config configs/test.yaml
```

## Train Your Own Data
### 1. Prepare your own data as the metadata\mini-ImageNet.

### 2. Start training!
```bash
python train.py --config configs/train.yaml
python meta_train.py --config configs/meta_train.yaml
```

## Contacts

If you have any questions about our work, please contact us by email.

Tong Zhang: [tongzhang@buaa.edu.cn](tongzhang@buaa.edu.cn)

## Acknowledgments

Our code is build upon [Hawkeye](https://github.com/Hawkeye-FineGrained/Hawkeye) and [FRN](https://github.com/Tsingularity/FRN), thanks to all the contributors!


## Citation

```bibtex
@article{tong2025free,
      title={Free Lunch to Meet the Gap: Intermediate Domain Reconstruction for Cross-Domain Few-Shot Learning},
      author={Tong Zhang and Yifan Zhao and Liangyu Wang sand Jia Li},
      journal={arXiv preprint arXiv:},
      year={2025}
}