# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import register_model

from .layers import Block, PatchEmbed
from .utils import build_2d_sincos_position_embedding


def trunc_normal_(tensor, mean=0., std=1.):
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_cls_token=True,
                 mask_ratio=0.75):
        super().__init__()
        self.use_cls_token = use_cls_token

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.mask_ratio = mask_ratio
        self.masked_size = int(self.mask_ratio * self.num_patches)
        self.visible_size = self.num_patches - self.masked_size

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 2D sine-cosine positional embeddings
        pos_embed = build_2d_sincos_position_embedding(*self.patch_embed.patch_shape, embed_dim)
        if use_cls_token:
            pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
            pos_embed = torch.cat([pe_token, pos_embed], dim=1)
        self.register_buffer('pos_embed', pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)
        if use_cls_token:
            trunc_normal_(self.cls_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shuffle = torch.rand(x.shape[0], self.num_patches).argsort(1).to(x.device)
        visible = shuffle[:, :self.visible_size]
        if self.use_cls_token:
            visible = torch.cat([torch.zeros([x.shape[0], 1], dtype=torch.long).to(x.device), visible + 1], dim=1)

        x = self.patch_embed(x)
        if self.use_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        visible_tokens = x.gather(1, visible.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        for blk in self.blocks:
            visible_tokens = blk(visible_tokens)

        # not return cls_token
        if self.use_cls_token:
            visible_tokens = visible_tokens[:, 1:, :]

        visible_tokens = self.norm(visible_tokens)

        return visible_tokens, shuffle


class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None):
        super().__init__()
        self.num_classes = 3 * patch_size ** 2

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, self.num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.head(self.norm(x))  # [B, N, 3*16^2]
        return x


class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 normalized_pixel=False,
                 mask_ratio=0.75,
                 use_cls_token=True,
                 **kwargs,  # avoid the error from create_fn in timm
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            mask_ratio=mask_ratio,
            use_cls_token=use_cls_token,
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
        )

        self.patch_size = patch_size
        self.normalized_pixel = normalized_pixel
        self.num_patches = self.encoder.num_patches
        self.visible_size = self.encoder.visible_size

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        pos_embed = build_2d_sincos_position_embedding(*self.encoder.patch_embed.patch_shape, decoder_embed_dim)
        self.register_buffer('decoder_pos_embed', pos_embed)

        trunc_normal_(self.mask_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mask_token'}  # TODO check
        # return {'encoder.pos_embed', 'encoder.cls_token', 'mask_token', 'decoder_pos_embed'}

    def get_target(self, target):
        B, C, H, W = target.shape
        target = target.view(
            B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size
        )  # Bx3x224x224 --> Bx3x16x14x16x14
        # Bx3x14x16x14x16 --> Bx(14*14)x(3*16*16)
        target = target.permute(0, 2, 4, 1, 3, 5).reshape(B, self.num_patches, -1)
        if self.normalized_pixel:
            target = F.layer_norm(target, target.shape[-1:], eps=1e-6)
        return target

    def forward(self, x, restruct=False):
        encoded_visible_tokens, shuffle = self.encoder(x)  # [B, N_vis, C_e]
        encoded_visible_tokens = self.encoder_to_decoder(encoded_visible_tokens)  # [B, N_vis, C_d]

        shared_mask_tokens = self.mask_token.expand(x.shape[0], self.num_patches - self.visible_size, -1)
        decoder_input = torch.cat([encoded_visible_tokens, shared_mask_tokens], dim=1)  # [B, N_all, C_d]
        decoder_input = decoder_input.gather(
            1, shuffle.argsort(1).unsqueeze(-1).expand(-1, -1, decoder_input.shape[-1]))
        decoder_input = decoder_input + self.decoder_pos_embed

        # decode all tokens
        decoder_output = self.decoder(decoder_input)  # [B, N_all, C_out]

        if restruct:
            return decoder_output, shuffle, self.visible_size

        # target
        with torch.no_grad():
            target = self.get_target(x)

        return self.loss(decoder_output, target, masked_index=shuffle[:, self.visible_size:])

    def loss(self, x, y, masked_index=None):
        if masked_index is not None:
            masked_index = masked_index.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = x.gather(1, masked_index)
            y = y.gather(1, masked_index)
        return F.mse_loss(x, y)


@register_model
def pretrain_mae_tiny_patch16_224(decoder_dim, decoder_depth, decoder_num_heads, **kwargs):
    """5M parameters"""
    if decoder_num_heads is None:
        decoder_num_heads = decoder_dim // 64
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=192,
        encoder_depth=12,
        encoder_num_heads=3,
        decoder_embed_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


@register_model
def pretrain_mae_small_patch16_224(decoder_dim, decoder_depth, decoder_num_heads, **kwargs):
    """22M parameters"""
    if decoder_num_heads is None:
        decoder_num_heads = decoder_dim // 64
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        decoder_embed_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


@register_model
def pretrain_mae_base_patch16_224(decoder_dim, decoder_depth, decoder_num_heads, **kwargs):
    """86M parameters"""
    if decoder_num_heads is None:
        decoder_num_heads = decoder_dim // 64
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


@register_model
def pretrain_mae_large_patch16_224(decoder_dim, decoder_depth, decoder_num_heads, **kwargs):
    """304M parameters"""
    if decoder_num_heads is None:
        decoder_num_heads = decoder_dim // 64
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        decoder_embed_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


@register_model
def pretrain_mae_huge_patch14_224(decoder_dim, decoder_depth, decoder_num_heads, **kwargs):
    """632M parameters"""
    if decoder_num_heads is None:
        decoder_num_heads = decoder_dim // 64
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=14,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        decoder_embed_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


@register_model
def pretrain_mae_giant_patch14_224(decoder_dim, decoder_depth, decoder_num_heads, **kwargs):
    """1.6B parameters"""
    if decoder_num_heads is None:
        decoder_num_heads = decoder_dim // 64
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=14,
        encoder_embed_dim=1664,
        encoder_depth=48,
        encoder_num_heads=16,
        decoder_embed_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model
