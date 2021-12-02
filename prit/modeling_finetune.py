from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple

from utils.layers import trunc_normal_
from utils.registry import register_model

from .layers import (PatchEmbed, PatchDownsample, PatchUpsample,
                     Block, LocalBlock, SRBlock, 
                     Output, LocalOutput, SROutput)
from .utils import build_2d_sincos_position_embedding, _cfg


class PriT(nn.Module):
    """
    PyramidReconstructionImageTransformer
    """

    def __init__(self,
                 # args for ViT (timm)
                 # w/o `distilled`, `detph`, `representation_size` and `weight_init`.
                 # default value of `patch_size` and `embed_dim` changed.
                 img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 # more args for ViT (BeiT)
                 qk_scale=None, init_values=0., init_scale=0.,
                 # args for PriT
                 strides=(1, 2, 2, 2), depths=(2, 2, 6, 2), dims=(48, 96, 192, 384),
                 blocks_type=('normal', 'normal', 'normal', 'normal'),
                 use_mean_pooling=True, pyramid_reconstruction=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer (nn.Module): normalization layer

            qk_scale (float): scale of qk in Attention
            init_values (float): init values of gamma_{1|2} in Block

            strides (tuple): stride for echo stage
            depths (tuple): depth of transformer for echo stage
            dims (tuple): dimension for echo stage
            init_scale (float): init scale of head
            use_mean_pooling (bool): enable mean pool
            pyramid_reconstruction (bool): return pyramid features from stages
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_features = dims[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.img_size = img_size = to_2tuple(img_size)

        # save args for build blocks
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.act_layer = act_layer

        assert self.embed_dim == dims[0]
        self.num_layers = len(depths)
        self.patch_size = patch_size
        self.strides = strides
        self.dims = dims
        self.pyramid_reconstruction = pyramid_reconstruction
        self.use_mean_pooling = use_mean_pooling
        self.use_cls_token = use_cls_token = not use_mean_pooling

        self.stride = stride = patch_size * reduce(mul, strides)  # 32 = 4 * 2 * 2 * 2 * 1
        self.out_size = out_size = (img_size[0] // stride, img_size[1] // stride)  # 7 = 224 / 32
        self.num_patches = out_size[0] * out_size[1]  # 49 = 7 * 7

        grid_size = stride // patch_size  # 8 = 32 / 4
        self.split_shape = (out_size[0], grid_size, out_size[1], grid_size)  # (7, 8, 7, 8)

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        grid_h, grid_w = self.patch_embed.patch_shape
        assert out_size[0] * grid_size == grid_h and out_size[1] * grid_size == grid_w

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = build_2d_sincos_position_embedding(
            grid_h, grid_w, embed_dim, use_cls_token=use_cls_token)
        self.pos_drop = nn.Dropout(p=drop_rate)

        _blocks ={
            "normal": Block,
            "local": partial(LocalBlock, self.num_patches),
            "spacial_reduction": partial(SRBlock, self.num_patches),
        }
        blocks = tuple(_blocks[b] for b in blocks_type)

        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, sum(depths))]  # stochastic depth decay rule
        for i in range(self.num_layers):
            downsample = i > 0 and (strides[i] == 2 or dims[i - 1] != dims[i])
            self.add_module(f'stage{i + 1}', nn.Sequential(
                PatchDownsample(dims[i - 1], dims[i], self.num_patches, stride=strides[i],
                    norm_layer=norm_layer, with_cls_token=use_cls_token) if downsample else nn.Identity(),
                self._build_blocks(dims[i], num_heads, depths[i],
                    dpr=[dpr.pop() for _ in range(depths[i])],
                    init_values=init_values, block=blocks[i]),
            ))
        self.norm = norm_layer(self.num_features) if use_cls_token else nn.Identity()

        self.fc_norm = norm_layer(self.num_features) if use_mean_pooling else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # weight initialization
        self.apply(self._init_weights)
        if use_cls_token:
            trunc_normal_(self.cls_token, std=.02)

        # init scale
        trunc_normal_(self.head.weight, std=.02)
        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _build_blocks(self, dim, num_heads, depth, dpr=None, init_values=0., block=Block):
        dpr = dpr or ([0.] * depth)
        blocks = [block(
            dim=dim, num_heads=num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale, drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i],
            norm_layer=self.norm_layer, act_layer=self.act_layer, init_values=init_values)
            for i in range(depth)]
        return nn.Sequential(*blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return self.num_layers

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def grid_patches(self, x, split_shape):
        B, N, L = x.shape
        x = x.reshape(B, *split_shape, L)               # Bx  (7x8x7x8)  xL
        x = x.permute([0, 1, 3, 2, 4, 5])               # Bx   7x7x8x8   xL
        x = x.reshape(B, x.size(1) * x.size(2), -1, L)  # Bx (7x7)x(8x8) xL
        return x

    def blocks(self, x):
        out = []
        for i in range(self.num_layers):
            x = getattr(self, f'stage{i + 1}')(x)
            if i == self.num_layers - 1:  # last block
                x = self.norm(x)
            out.append(x)
        return out if self.pyramid_reconstruction else [x]

    def forward(self, x):
        x = self.patch_embed(x)  # Bx(56x56)xL
        if self.use_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # get grid-like patches
        x_wo_cls_token = x[:, 1:] if self.use_cls_token else x
        x_wo_cls_token = self.grid_patches(x_wo_cls_token, self.split_shape)  # Bx (7x7)x(8x8) xL
        all_tokens = torch.cat((x[:, [0]], x_wo_cls_token), dim=1) if self.use_cls_token else x_wo_cls_token
        all_tokens = all_tokens.flatten(1, 2) # Bx (7x7x8x8) xL

        # [Bx(12*8*8)xL, Bx(12*4*4)xL, Bx(12*2*2)xL, Bx(12*1*1)xL]
        encoded_all_tokens = self.blocks(all_tokens)

        assert not self.pyramid_reconstruction
        encoded_all_tokens = encoded_all_tokens[-1]

        encoded_all_tokens = encoded_all_tokens[:, 0] if self.use_cls_token else encoded_all_tokens.mean(1)
        return self.head(self.fc_norm(encoded_all_tokens))


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model = PriT(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        strides=[1],
        depths=[12],
        dims=[384],
        blocks_type=['normal'],
        num_heads=6,
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def prit_local_small_GGGG_patch16_224(pretrained=False, **kwargs):
    model = PriT(
        img_size=224,
        patch_size=4,
        embed_dim=96,
        strides=(1, 2, 2, 2),
        depths=(2, 2, 7, 1),
        dims=(96, 192, 384, 768),
        blocks_type=('normal', 'normal', 'normal', 'normal'),
        num_heads=6,
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def prit_local_small_LGGG_patch16_224(pretrained=False, **kwargs):
    model = PriT(
        img_size=224,
        patch_size=4,
        embed_dim=96,
        strides=(1, 2, 2, 2),
        depths=(2, 2, 7, 1),
        dims=(96, 192, 384, 768),
        blocks_type=('local', 'normal', 'normal', 'normal'),
        num_heads=6,
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def prit_local_small_LLGG_patch16_224(pretrained=False, **kwargs):
    model = PriT(
        img_size=224,
        patch_size=4,
        embed_dim=96,
        strides=(1, 2, 2, 2),
        depths=(2, 2, 7, 1),
        dims=(96, 192, 384, 768),
        blocks_type=('local', 'local', 'normal', 'normal'),
        num_heads=6,
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def prit_local_small_LLLG_patch16_224(pretrained=False, **kwargs):
    model = PriT(
        img_size=224,
        patch_size=4,
        embed_dim=96,
        strides=(1, 2, 2, 2),
        depths=(2, 2, 7, 1),
        dims=(96, 192, 384, 768),
        blocks_type=('local', 'local', 'local', 'normal'),
        num_heads=6,
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def prit_local_small_LLLL_patch16_224(pretrained=False, **kwargs):
    model = PriT(
        img_size=224,
        patch_size=4,
        embed_dim=96,
        strides=(1, 2, 2, 2),
        depths=(2, 2, 7, 1),
        dims=(96, 192, 384, 768),
        blocks_type=('local', 'local', 'local', 'local'),
        num_heads=6,
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def prit_local_small_SrGGG_patch16_224(pretrained=False, **kwargs):
    model = PriT(
        img_size=224,
        patch_size=4,
        embed_dim=96,
        strides=(1, 2, 2, 2),
        depths=(2, 2, 7, 1),
        dims=(96, 192, 384, 768),
        blocks_type=('spacial_reduction', 'normal', 'normal', 'normal'),
        num_heads=6,
        **kwargs)
    model.default_cfg = _cfg()
    return model
