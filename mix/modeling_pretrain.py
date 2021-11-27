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

from utils.layers import trunc_normal_ as __call_trunc_normal_
from utils.registry import register_model
from mae.modeling_finetune import Block, _cfg, PatchEmbed
from .mix import Mix_MAE

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'mix_mae_base_patch16_224',
    'mix_mae_large_patch16_224',
]


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_mean_pooling=False,
                 use_learnable_pos_emb=False, mask_ratio=0.75):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.use_mean_pooling = use_mean_pooling

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = num_patches = self.patch_embed.num_patches

        self.mask_ratio = mask_ratio
        self.masked_size = int(self.mask_ratio * self.num_patches)  # 147
        self.visible_size = self.num_patches - self.masked_size  # 49

        if use_mean_pooling:
            self.cls_token = None
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # 2D sine-cosine positional embeddings
            self.pos_embed = self.build_2d_sincos_position_embedding(embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def build_2d_sincos_position_embedding(self, embed_dim=768, temperature=10000., decode=False):
        h, w = self.patch_embed.patch_shape
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
            pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
            pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        pos_embed.requires_grad = False
        return pos_embed

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        x = self.patch_embed(x)
        shuffle = torch.randperm(self.num_patches)

        if not self.use_mean_pooling:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed

        if not self.use_mean_pooling:
            shuffle = torch.cat([torch.zeros(1, dtype=torch.long), shuffle + 1])

        shuffle_token = x[:, shuffle, :]

        if not self.use_mean_pooling:
            visible_token = shuffle_token[:, :self.visible_size + 1, :]  # Bx(14*14*0.25+1)x768 = Bx50x768
            shuffle = shuffle[1:] - 1
        else:
            visible_token = shuffle_token[:, :self.visible_size, :]  # Bx(14*14*0.25)x768 = Bx49x768

        for blk in self.blocks:
            visible_token = blk(visible_token)

        visible_token = self.norm(visible_token)

        # not return cls_token
        if not self.use_mean_pooling:
            visible_token = visible_token[:, 1:, :]

        return visible_token, shuffle


class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, patch_size=16, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,
                 ):
        super().__init__()
        self.num_classes = num_classes = 6 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

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
                 encoder_num_classes=0,
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
                 use_learnable_pos_emb=False,
                 normalized_pixel=False,
                 mask_ratio=0.75,
                 use_mean_pooling=False,
                 mix_mode='batch',
                 mix_alpha=0.5,
                 **kwargs,  # avoid the error from create_fn in timm
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
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
            use_learnable_pos_emb=use_learnable_pos_emb,
            mask_ratio=mask_ratio,
            use_mean_pooling=use_mean_pooling,
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
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
            init_values=init_values)

        self.num_patches = self.encoder.num_patches
        self.patch_size = patch_size
        self.visible_size = self.encoder.visible_size
        self.mix = Mix_MAE(mode=mix_mode, alpha=mix_alpha)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = self.encoder.build_2d_sincos_position_embedding(decoder_embed_dim, decode=True)

        self.normalized_pixel = normalized_pixel

        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def _mse_loss(self, x, y, masked_index=None):
        if masked_index is not None:
            return F.mse_loss(x[:, masked_index, :], y[:, masked_index, :], reduction="mean")
        else:
            return F.mse_loss(x, y, reduction="mean")

    def forward(self, x):
        x, target = self.mix(x)

        B, C, H, W = target.size()

        encoded_visible_patches, shuffle = self.encoder(x)  # [B, N_vis, C_e]
        encoded_visible_patches = self.encoder_to_decoder(encoded_visible_patches)  # [B, N_vis, C_d]

        decoder_input = torch.cat(
            [encoded_visible_patches, self.mask_token.repeat([B, self.num_patches - self.visible_size, 1])], dim=1
        )[:, shuffle.sort()[1], :]
        decoder_input = decoder_input + self.decoder_pos_embed

        # decode (encoded_visible_patches + mask_token)
        decoder_output = self.decoder(decoder_input)  # Bx(14*14)x512

        # target
        target = target.view(
            [B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size]
        )  # Bx3x224x224 --> Bx3x16x14x16x14
        # Bx3x14x16x14x16 --> Bx(14*14)x(16*16*3)
        target = target.permute([0, 2, 4, 3, 5, 1]).reshape(B, self.num_patches, -1, C)

        if self.normalized_pixel:
            mean = target.mean(dim=-2, keepdim=True)
            std = target.var(dim=-2, unbiased=True, keepdim=True).sqrt()
            target = (target - mean) / (std + 1e-6)
        target = target.view(B, self.num_patches, -1)

        return self._mse_loss(decoder_output, target, masked_index=shuffle[self.visible_size:])


@register_model
def mix_mae_small_patch16_224(decoder_dim, decoder_depth, decoder_num_heads, **kwargs):
    if decoder_num_heads is None:
        decoder_num_heads = decoder_dim // 64
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_embed_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def mix_mae_base_patch16_224(decoder_dim, decoder_depth, decoder_num_heads, **kwargs):
    if decoder_num_heads is None:
        decoder_num_heads = decoder_dim // 64
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_embed_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def mix_mae_large_patch16_224(decoder_dim, decoder_depth, decoder_num_heads, **kwargs):
    if decoder_num_heads is None:
        decoder_num_heads = decoder_dim // 64
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_embed_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model