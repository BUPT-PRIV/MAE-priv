import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul

from mae.vision_transformer import VisionTransformer
from mae.pyramid_reconstruction_image_transformer import PriTEncoder
from mae.layers import to_2tuple

__all__ = [
    'vit_small',
    'vit_base',
    'vit_large',
    'vit_conv_small',
    'vit_conv_base',
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class VisionTransformerDecoder(VisionTransformer):
    def __init__(self, mask_ratio=0.75, use_mean_pooling=False, **kwargs):
        super().__init__(**kwargs)

        self.use_mean_pooling = use_mean_pooling
        # Use fixed 2D sin-cos position embedding
        self.pos_embed = self.build_2d_sincos_position_embedding(embed_dim=self.embed_dim)
        self.patch_size = self.patch_embed.patch_size  # 16
        self.num_patches = self.patch_embed.num_patches  # 14*14=196
        self.mask_ratio = mask_ratio
        self.masked_size = int(self.mask_ratio * self.num_patches)  # 147
        self.visible_size = self.num_patches - self.masked_size  # 49

        if use_mean_pooling:
            self.cls_token = None  # no cls token
        self.head = None  # no cls head

        # # weight initialization
        # for name, m in self.named_modules():
        #     if isinstance(m, nn.Linear):
        #         if 'qkv' in name:
        #             # treat the weights of Q, K, V separately
        #             val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
        #             nn.init.uniform_(m.weight, -val, val)
        #         else:
        #             nn.init.xavier_uniform_(m.weight)
        #         nn.init.zeros_(m.bias)

        # if isinstance(self.patch_embed, PatchEmbed):
        #     # xavier_uniform initialization
        #     val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
        #     nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
        #     nn.init.zeros_(self.patch_embed.proj.bias)

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

    def forward(self, x):
        x = self.patch_embed(x)  # BNC = B(HW)C = Bx(14*14)x768

        if not self.use_mean_pooling:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        shuffle = torch.randperm(self.num_patches)
        if not self.use_mean_pooling:
            shuffle = torch.cat([torch.zeros(1, dtype=torch.long), shuffle + 1])  # [0, ...]

        shuffle_token = x[:, shuffle, :]

        if not self.use_mean_pooling:
            visible_token = shuffle_token[:, :self.visible_size + 1, :]  # Bx(14*14*0.25+1)x768 = Bx50x768
            # masked_token = shuffle_token[:, self.visible_size + 1:, :]  # Bx(14*14*0.75-1)x768 = Bx146x768
            shuffle = shuffle[1:] - 1
        else:
            visible_token = shuffle_token[:, :self.visible_size, :]  # Bx(14*14*0.25)x768 = Bx49x768
            # masked_token = shuffle_token[:, self.visible_size:, :]  # Bx(14*14*0.75)x768 = Bx147x768

        encoded_visible_patches = self.blocks(visible_token)
        encoded_visible_patches = self.norm(encoded_visible_patches)

        if not self.use_mean_pooling:
            encoded_visible_patches = encoded_visible_patches[:, 1:, :]

        return encoded_visible_patches, shuffle

    def get_num_layers(self):
        return len(self.blocks)


class ConvStem(nn.Module):
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def vit_small(**kwargs):
    model = VisionTransformerDecoder(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


def vit_base(**kwargs):
    model = VisionTransformerDecoder(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


def vit_large(**kwargs):
    model = VisionTransformerDecoder(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


def vit_conv_small(**kwargs):
    # minus one ViT block
    model = VisionTransformerDecoder(
        patch_size=16, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


def vit_conv_base(**kwargs):
    # minus one ViT block
    model = VisionTransformerDecoder(
        patch_size=16, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


def prit_vit_base(**kwargs):
    model = PriTEncoder(
        patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        strides=(1,), depths=(12,), dims=(768,), **kwargs)
    model.default_cfg = _cfg(num_classes=0)
    return model


def prit_base(**kwargs):
    model = PriTEncoder(
        patch_size=4, embed_dim=96, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        strides=(1, 2, 2, 2), depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), **kwargs)
    model.default_cfg = _cfg(num_classes=0)
    return model
