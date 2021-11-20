import math
from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn

from timm.models import (Block, PatchEmbed, VisionTransformer,
                         named_apply, trunc_normal_, _init_vit_weights)


class PatchDownsample(nn.Module):
    def __init__(self, dim_in, dim_out, visible_num, stride=2, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.visible_num = visible_num
        self.stride = stride

        if stride == 1:
            self.pool = None
        if stride == 2:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(stride)

        self.reduction = nn.Linear(dim_in, dim_out, bias=False)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, N, L = x.shape  # N = BxLx(visible_num x grid_h x grid_w)
        grid_h = grid_w = int((N // self.visible_num) ** 0.5)

        if self.pool is not None:
            # get grid-like patches
            x = x.permute([0, 2, 1])  # BxLxN
            x = x.reshape(B, L, self.visible_num, grid_h, grid_w)  # BxLx12 x grid_h x grid_w
            x = x.reshape(-1, grid_h, grid_w)  # (BxLx12) x grid_h x grid_w

            x = self.pool(x)  # (BxLx12) x grid_h/2 x grid_w/2

            # reshape
            x = x.view(B, L, -1)  # BxLx(N/4)
            x = x.permute([0, 2, 1])  # Bx(N/4)xL

        x = self.reduction(x)
        x = self.norm(x)
        return x


class PriTEncoder(VisionTransformer):
    """
    PyramidReconstructionImageTransformer Encoder
    """

    def __init__(self,
                 # args for ViT, w/o `distilled`, `detph` and `representation_size`.
                 # default value of `patch_size`, `num_classes` and `embed_dim` changed.
                 img_size=224, patch_size=4, in_chans=3, num_classes=0, embed_dim=96, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, weight_init='',
                 # args for PriT
                 strides=(2, 2, 2, 1), depths=(2, 3, 5, 2), dims=(96, 192, 384, 768),
                 mask_ratio=0.75, use_mean_pooling=True):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_features = dims[-1]  # update
        self.num_tokens = 0 if use_mean_pooling else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # save args for build blocks
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.act_layer = act_layer

        assert self.embed_dim == dims[0]
        self.mask_ratio = mask_ratio
        self.use_mean_pooling = use_mean_pooling

        self.stride = stride = patch_size * reduce(mul, strides)  # 32 = 4 * 2 * 2 * 2 * 1
        self.out_size = out_size = img_size // stride  # 7 = 224 / 32
        self.num_patches = out_size ** 2  # 49 = 7 * 7
        self.visible_num = int(self.num_patches * (1 - self.mask_ratio))  # 12 = ⎣49 * (1 - 0.75)⎦

        self.grid_size = grid_size = stride // patch_size  # 8 = 32 / 4
        grid_h, grid_w = self.patch_embed.grid_size  # 56 = img_size / patch_size = 224 / 4
        self.split_shape = (grid_h // grid_size, grid_size, grid_w // grid_size, grid_size)  # (7, 8, 7, 8)

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        if not use_mean_pooling:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = self.build_2d_sincos_position_embedding(self.embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        for i, (depth, stride) in enumerate(zip(depths, strides)):
            self.add_module(f'stage{i + 1}', nn.Sequential(
                self._build_blocks(dims[i], depth, drop_path=dpr.pop()),
                PatchDownsample(dims[i], dims[i + 1], self.visible_num, stride=stride, norm_layer=norm_layer),
            ))

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # weight initialization, ViT (timm)
        self.init_weights(weight_init)

        # weight initialization, MOCO v3
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

    def _build_blocks(self, dim, depth, drop_path=0.):
        blocks = [Block(
            dim=dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop_rate,
            attn_drop=self.attn_drop_rate, drop_path=drop_path, norm_layer=self.norm_layer, act_layer=self.act_layer)
            for _ in range(depth)]
        return nn.Sequential(*blocks)

    def build_2d_sincos_position_embedding(self, embed_dim=768, temperature=10000., decode=False):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        if self.use_mean_pooling or decode:
            pos_embed = nn.Parameter(pos_emb)
        else:
            assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
            pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
            pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        pos_embed.requires_grad = False
        return pos_embed

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        # trunc_normal_(self.pos_embed, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            if self.cls_token is not None:
                trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def blocks(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x4

    def grid_patches(self, x):
        B, N, L = x.shape
        x = x.reshape(B, self.split_shape, L)         # Bx  (7x8x7x8)  xL
        x = x.permute([0, 1, 3, 2, 4, 5])             # Bx   7x7x8x8   xL
        x = x.reshape(B, -1, self.grid_size ** 2, L)  # Bx (7x7)x(8x8) xL
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)  # Bx(56x56)xL
        if not self.use_mean_pooling:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # get grid-like patches
        x_wo_cls_token = x if self.use_mean_pooling else x[:, 1:]
        x_wo_cls_token = self.grid_patches(x_wo_cls_token)  # Bx (7x7)x(8x8) xL

        # get visible tokens by random shuffle
        if self.mask_ratio == 0.:
            visible_tokens = x_wo_cls_token
        else:
            visible_num = self.visible_num  # 12 = ⎣49 * (1 - 0.75)⎦
            shuffle = torch.randperm(self.num_patches)  # 49 = 7 * 7
            visible_tokens = x_wo_cls_token[:, shuffle[:visible_num]]    # Bx12x(8x8)xL
        visible_tokens = visible_tokens.flatten(1, 3)  # Bx(12x8x8)xL

        visible_tokens = x_wo_cls_token if self.use_mean_pooling else torch.cat((x[: 0], visible_tokens), dim=1)

        visible_tokens = self.blocks(visible_tokens)
        encoded_visible_patches = self.norm(visible_tokens)
        if not self.use_mean_pooling:
            encoded_visible_patches = encoded_visible_patches[:, 1:]  # w/o cls token

        return encoded_visible_patches, shuffle

    def forward(self, x):
        encoded_visible_patches, shuffle = self.forward_features(x)
        return self.head(encoded_visible_patches), shuffle
