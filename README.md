# MGPNet -- PyTorch Implementation

This repository contains the PyTorch Implementation of the following paper:

> **Image Smoothing via Multiscale Global Perception**



## datasets

### trainsets

SPS: [https://drive.google.com/drive/folders/1inuxV8ghABOv60KVc6zY97Ccj0yyJ9uv?usp=sharing](https://drive.google.com/drive/folders/1inuxV8ghABOv60KVc6zY97Ccj0yyJ9uv?usp=sharing)

The approach of generating the trainsets is showed in [https://github.com/YidFeng/Easy2Hard](https://github.com/YidFeng/Easy2Hard)

### testsets

- [SPS](https://drive.google.com/drive/folders/1EDqgjFZt5KndZlHjtD2EjpcXvfzT878s?usp=sharing)
- [NKS](https://drive.google.com/drive/folders/1rsWLc7kpyM2VfGwY_Gu94TH2bnt3ywfV?usp=sharing)



## Usage

The code is tested with **Python 3.7**, **PyTorch 1.9.0** and **CUDA 11.1**, and is saved in `codes` folder.

```shell
cd codes
```

**Training**

First set a config file `train.yml` in `options/train/`, then run as following:

```shell
python train.py -opt options/train/train.yml
```

**Test**

First set a config file `test_smoothing.yml` in `options/test/`, then run as following:

	python test.py -opt options/test/test_smoothing.yml

The test result will be saved in `../results` folder.

**Pretrained model**

Pretrained model is released on `../experiments/MGPNet/models/best_G.pth`.



## The Contents of `codes` folder

**Config**: [`options/`](./options) Configure the options for data loader, network structure, model, training strategies and etc.

**Data**: [`data/`](./data) A data loader to provide data for training, validation and testing.

**Model**: [`models/`](./models) Construct models for training and testing, [`models/MGPNet.py`](./models/MGPNet.py) construct network architectures.



## Citation

```

```
