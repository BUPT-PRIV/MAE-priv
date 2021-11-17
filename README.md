## MAE for Self-supervised ViT

### Introduction
This is an unofficial PyTorch implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) for self-supervised ViT.

This repo is mainly based on [moco-v3](https://github.com/facebookresearch/moco-v3), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit)

### Main Results

The following results are based on ImageNet-1k self-supervised pre-training, followed by ImageNet-1k supervised training for linear evaluation or end-to-end fine-tuning. 

#### Vit-Base
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">pretrain<br/>epochs</th>
<th valign="center">with<br/>pixel-norm</th>
<th valign="center">linear<br/>acc</th>
<th valign="center">fine-tuning<br/>acc</th>
<!-- TABLE BODY -->
<tr>
<td align="right">100</td>
<td align="center">False</td>
<td align="center">--</td>
<td align="center">75.58 [1]</td>
</tr>
<tr>
<td align="right">100</td>
<td align="center">True</td>
<td align="center">--</td>
<td align="center">77.19</td>
</tr>
</tbody></table>

On 8 NVIDIA GeForce 3090 GPUs, pretrain for 100 epochs needs about xx hours, 4096 batch size needs about xx GB GPU memory.

[1]. fine-tuning for 50 epochs;


#### Vit-Large
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">pretrain<br/>epochs</th>
<th valign="center">with<br/>pixel-norm</th>
<th valign="center">linear<br/>acc</th>
<th valign="center">fine-tuning<br/>acc</th>
<!-- TABLE BODY -->
<tr>
<td align="right">100</td>
<td align="center">False</td>
<td align="center">--</td>
<td align="center">--</td>
</tr>
<tr>
<td align="right">100</td>
<td align="center">True</td>
<td align="center">--</td>
<td align="center">--</td>
</tr>
</tbody></table>

On 8 NVIDIA GeForce 3090 GPUs, pretrain for 100 epochs needs about xx hours, 4096 batch size needs about xx GB GPU memory.


### Usage: Preparation

The code has been tested with CUDA 11.0, PyTorch 1.9.0.

### Usage: Self-supervised Pre-Training

Below is examples for MAE pre-training. 

#### Vit-Base with 1-node (8-GPU, NVIDIA GeForce 3090) training, batch 4096

```
python main_mae.py \
  -a vit_base -b 4096 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

#### Vit-Large with 1-node (8-GPU, NVIDIA GeForce 3090) pre-training, batch 2048

```
python main_mae.py \
  -a vit_large -b 2048 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

#### Notes:
1. The batch size specified by `-b` is the total batch size across all GPUs.
1. The learning rate specified by `--lr` is the *base* lr, and is adjusted by the [linear lr scaling rule](https://arxiv.org/abs/1706.02677) in [this line](https://github.com/facebookresearch/moco-v3/blob/main/main_moco.py#L213).
1. In this repo, only *multi-gpu*, *DistributedDataParallel* training is supported; single-gpu or DataParallel training is not supported. This code is improved to better suit the *multi-node* setting, and by default uses automatic *mixed-precision* for pre-training.


### Usage: End-to-End Fine-tuning ViT


Below is examples for MAE fine-tuning.

#### Vit-Base with 1-node (8-GPU, NVIDIA GeForce 3090) training, batch 4096

```
python main_mae.py \
  -a vit_base -b 4096 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

This gives us 83.2% accuracy for ViT-Base with 100-epoch fine-tuning.

**Note**:
1. We use `--resume` rather than `--finetune` in the DeiT repo, as its `--finetune` option trains under eval mode. When loading the pre-trained model, revise `model_without_ddp.load_state_dict(checkpoint['model'])` with `strict=False`.
1. Our ViT-Small is with `heads=12` in the Transformer block, while by default in DeiT it is `heads=6`. Please modify the DeiT code accordingly when fine-tuning our ViT-Small model. 


### Usage: Linear Classification

By default, we use momentum-SGD and a batch size of 1024 for linear classification on frozen features/weights. This can be done with a single 8-GPU node.

```
python main_lincls.py \
  -a [architecture] --lr [learning rate] \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path]/[your checkpoint file].pth.tar \
  [your imagenet-folder with train and val folders]
```


### Model Configs

See the commands listed in [CONFIG.md](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md) for specific model configs, including our recommended hyper-parameters and pre-trained reference models.

### Transfer Learning

See the instructions in the [transfer](https://github.com/facebookresearch/moco-v3/tree/main/transfer) dir.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### Citation

If you use the code of this repo, please cite the original papre and this repo:

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
