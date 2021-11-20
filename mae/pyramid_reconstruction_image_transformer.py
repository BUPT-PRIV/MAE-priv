import math
from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn

from timm.models.vision_transformer import (Block, PatchEmbed,
    VisionTransformer, named_apply, trunc_normal_, _init_vit_weights)
from timm.models.layers import to_2tuple


class PatchDownsample(nn.Module):
    def __init__(self, dim_in, dim_out, num_visible, stride=2,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cls_token=False):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_visible = num_visible
        self.stride = stride
        self.with_cls_token = with_cls_token

        if stride == 1:
            self.pool = None
        if stride == 2:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(stride)

        self.reduction = nn.Linear(dim_in, dim_out, bias=False)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        if self.pool is not None:
            x_wo_cls_token = x[:, 1:] if self.with_cls_token else x

            B, N, L = x_wo_cls_token.shape  # N = BxLx(num_visible x grid_h x grid_w)
            grid_h = grid_w = int((N // self.num_visible) ** 0.5)

            # get grid-like patches
            x_wo_cls_token = x_wo_cls_token.permute([0, 2, 1])  # BxLxN
            x_wo_cls_token = x_wo_cls_token.reshape(B, L, self.num_visible, grid_h, grid_w)  # BxLx12 x grid_h x grid_w
            x_wo_cls_token = x_wo_cls_token.reshape(-1, grid_h, grid_w)  # (BxLx12) x grid_h x grid_w

            x_wo_cls_token = self.pool(x_wo_cls_token)  # (BxLx12) x grid_h/2 x grid_w/2

            # reshape
            x_wo_cls_token = x_wo_cls_token.view(B, L, -1)  # BxLx(N/4)
            x_wo_cls_token = x_wo_cls_token.permute([0, 2, 1])  # Bx(N/4)xL

            x = torch.cat((x[:, [0]], x_wo_cls_token), dim=1) if self.with_cls_token else x_wo_cls_token

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
                 mask_ratio=0.75, use_mean_pooling=True, pyramid_reconstruction=False):
        # super(PriTEncoder, self).__init__()
        super(VisionTransformer, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_features = dims[-1]  # update
        self.num_tokens = 0 if use_mean_pooling else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        img_size = to_2tuple(img_size)

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
        self.num_layers = len(depths)
        self.mask_ratio = mask_ratio
        self.pyramid_reconstruction = pyramid_reconstruction
        self.use_mean_pooling = use_mean_pooling
        self.use_cls_token = use_cls_token = not use_mean_pooling

        self.stride = stride = patch_size * reduce(mul, strides)  # 32 = 4 * 2 * 2 * 2 * 1
        self.out_size = out_size = (img_size[0] // stride, img_size[1] // stride)  # 7 = 224 / 32
        self.num_patches = out_size[0] * out_size[1]  # 49 = 7 * 7
        self.num_visible = int(self.num_patches * (1 - self.mask_ratio))  # 12 = ⎣49 * (1 - 0.75)⎦

        self.grid_size = grid_size = stride // patch_size  # 8 = 32 / 4
        self.split_shape = (out_size[0], grid_size, out_size[1], grid_size)  # (7, 8, 7, 8)

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        grid_h, grid_w = self.patch_embed.grid_size
        assert out_size[0] * grid_size == grid_h and out_size[1] * grid_size == grid_w

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        self.pos_embed = self.build_2d_sincos_position_embedding(grid_h, grid_w, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        for i in range(self.num_layers):
            downsample = i < len(depths) - 1 and (strides[i] == 2 or dims[i] != dims[i + 1])
            self.add_module(f'stage{i + 1}', nn.Sequential(
                self._build_blocks(dims[i], num_heads, depths[i], dpr=[dpr.pop() for _ in range(depths[i])]),
                PatchDownsample(dims[i], dims[i + 1], self.num_visible, stride=strides[i],
                    norm_layer=norm_layer, with_cls_token=use_cls_token) if downsample else nn.Identity(),
            ))
        self.norm = norm_layer(self.num_features)

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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

    def _build_blocks(self, dim, num_heads, depth, dpr=None):
        dpr = dpr or ([0.] * depth)
        blocks = [Block(
            dim=dim, num_heads=num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop_rate,
            attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer, act_layer=self.act_layer)
            for i in range(depth)]
        return nn.Sequential(*blocks)

    def build_2d_sincos_position_embedding(self, h, w, embed_dim, temperature=10000., decode=False):
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

        if not self.use_cls_token or decode:
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

    def grid_patches(self, x):
        B, N, L = x.shape
        x = x.reshape(B, *self.split_shape, L)        # Bx  (7x8x7x8)  xL
        x = x.permute([0, 1, 3, 2, 4, 5])             # Bx   7x7x8x8   xL
        x = x.reshape(B, -1, self.grid_size ** 2, L)  # Bx (7x7)x(8x8) xL
        return x

    def blocks(self, x):
        out = []
        for i in range(self.num_layers):
            x = getattr(self, f'stage{i + 1}')(x)
            if i == self.num_layers - 1:  # last block
                x = self.norm(x)
            out.append(x)
        return out if self.pyramid_reconstruction else x

    def forward_features(self, x):
        x = self.patch_embed(x)  # Bx(56x56)xL
        if self.use_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # get grid-like patches
        x_wo_cls_token = x[:, 1:] if self.use_cls_token else x
        x_wo_cls_token = self.grid_patches(x_wo_cls_token)  # Bx (7x7)x(8x8) xL

        # get visible tokens by random shuffle
        if self.mask_ratio == 0.:
            visible_tokens = x_wo_cls_token
        else:
            num_visible = self.num_visible  # 12 = ⎣49 * (1 - 0.75)⎦
            shuffle = torch.randperm(self.num_patches)  # 49 = 7 * 7
            visible_tokens = x_wo_cls_token[:, shuffle[:num_visible]]    # Bx12x(8x8)xL
        visible_tokens = visible_tokens.flatten(1, 2)  # Bx(12x8x8)xL

        if self.use_cls_token:
            visible_tokens = torch.cat((x[:, [0]], visible_tokens), dim=1)

        encoded_visible_patches = self.blocks(visible_tokens)

        # w/o cls token
        if self.use_cls_token:
            if self.pyramid_reconstruction:
                encoded_visible_patches = [p[:, 1:] for p in encoded_visible_patches]
            else:
                encoded_visible_patches = encoded_visible_patches[:, 1:]

        return encoded_visible_patches, shuffle

    def forward(self, x):
        encoded_visible_patches, shuffle = self.forward_features(x)
        return self.head(encoded_visible_patches), shuffle
