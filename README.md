# MoQ
This repository is the official Pytorch implementation of **_Be Conservative or Be Aggressive? An Adaptive Trade-Off MoE For Time Series Forecasting_**. 

## Code
There are 6 different python files:
- utils.py
- metrics.py
- MoQ.py
- LSTNet.py
- ex_Train_expert.py
- ex_Train_MoQ.py

`utils.py` and `metrics.py` include some functions which are used in `ex_Train_expert.py` and `ex_Train_MoQ.py`; LSTNet is employed as the expert in our paper, and the file `LSTNet.py` is the official Pytorch implementation of LSTNet, please check the official repository for more details https://github.com/fbadine/LSTNet. `MoQ.py` is the MoE framework used to fuse the predictions of experts, and the details of this model can be found in our paper. `ex_Train_expert.py` and `ex_Train_MoQ.py` are the scripts used to train experts and MoQ; notice that MoQ can only be trained after finishing the training of experts.


## Data
The data required to train experts and MoQ can be downloaded at https://drive.google.com/file/d/15XTKEIX-1VPGregtQHQVHVBIa9GaqZtu/view?usp=sharing. Five files can be found by decompressing the zip file: 
- params_expert.pckl
- params_MoQ.pckl
- PM25_train.pckl
- PM25_valid.pckl
- PM25_test.pckl

**params_expert.pckl** and **params_MoQ.pckl** are the hyperparameters of the experts and MoQ, and the other files are the train set, validation set and test set of PM2.5 dataset used in the paper. For confidentiality reasons, we cannot share the mobile traffic dataset. Notice that the decompessed folder should be put in the same directory as the codes.


## Usage
In order to reproduce the results of MoQ, first you need to run `ex_Train_expert.py` to train experts, and then run `ex_Train_MoQ.py` to train MoQ.

```shell
python ex_Train_expert.py
```

```shell
python ex_Train_MoQ.py
```

In default, two versions of MoQ will be trained in `ex_Train_MoQ.py`:

```shell
penalization = 'mask'  # 'mask' or 'Gaussian'
train(penalization)

penalization = 'Gaussian' 
train(penalization)
```

If you can comment the lines if only one version is needed.


## Environment
The codes were tested on a system with the following versions:
- Python 3.8.8
- Pytorch 1.7.0
- Numpy 1.18.1




