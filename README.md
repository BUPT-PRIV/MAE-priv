## MAE for Self-supervised ViT

## Introduction
This is an unofficial PyTorch implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) for self-supervised ViT.

This repo is mainly based on [moco-v3](https://github.com/facebookresearch/moco-v3), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit)

## TODO
- [ ] visualization of reconstruction image
- [x] linear probing
- [ ] more results
- [ ] transfer learning
- [ ] multi-nodes training
- [ ] ...

## Main Result
### ViT-Base

| pretrain epoch | accuracy | config                                                       |
| -------------- | -------- | ------------------------------------------------------------ |
| 400            | 83.08%   | [pretrain](cfgs/pretrain/ViT-B_400E_Norm_GAP.yaml) [finetune](cfgs/finetune/ViT-B_100E_GAP.yaml) |

## Usage

### Usage: Preparation

The code has been tested with CUDA 11.4, PyTorch 1.8.2.

#### Notes:
1. The batch size specified by `-b` is **batch-size per card**.
1. The learning rate specified by `--lr` is the *base* lr (corresponding to 256 batch-size), and is adjusted by the [linear lr scaling rule](https://arxiv.org/abs/1706.02677).
1. In this repo, only *multi-gpu*, *DistributedDataParallel* training is supported; single-gpu or DataParallel training is not supported.
1. **Only pretraining and finetuning have been tested.**
1. We support cls-token (token) and global averaging pooling (GAP) for classification. **Please verify the correspondence of pretraining and finetuning/linear probing**. For cls-token mode during pretraining, cls-token is trained in encoder.

### Usage: Self-supervised Pre-Training
Below is examples for MAE pre-training.

#### ViT-Small with 1-node (8-GPU, NVIDIA GeForce RTX 3090) training, batch-size 4096, GAP
```
sh run_pretrain.sh --config cfgs/pretrain/Vit-S_100E_GAP.yaml --data_path /path/to/train/data
```

### Usage: End-to-End Fine-tuning ViT
#### ViT-Small with 1-node (8-GPU, NVIDIA GeForce RTX 3090) training, 50epochs, batch-size 4096, GAP
```
sh run_finetune.sh --config cfgs/fintune/ViT-S_50E_GAP.yaml --data_path /path/to/data --finetune /path/to/pretrain/model
```

### Usage: Linear Classification
According to paper, we have two training modes: SGD + 4096 batch-size and LARS + 16384 batch-size. 

#### ViT-Small with 1-node (8-GPU, NVIDIA GeForce RTX 3090) training, 50epochs, SGD + batch-size 4096, GAP
```
sh run_finetune.sh --config cfgs/lincls/ViT-S_SGD_50E_GAP.yaml --data_path /path/to/data --finetune /path/to/pretrain/model
```

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### Citation

If you use the code of this repo, please cite the original paper and this repo:

```
@Article{he2021mae,
  author  = {Kaiming He* and Xinlei Chen* and Saining Xie and Yanghao Li and Piotr Dolla ́r and Ross Girshick},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  journal = {arXiv preprint arXiv:2111.06377},
  year    = {2021},
}
```

```
@misc{yang2021maepriv,
  author       = {Lu Yang* and Pu Cao* and Yang Nie and Qing Song},
  title        = {MAE-priv},
  howpublished = {\url{https://github.com/BUPT-PRIV/MAE-priv}},
  year         = {2021},
}
```
