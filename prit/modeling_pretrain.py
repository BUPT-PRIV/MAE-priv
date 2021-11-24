from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple

from utils.layers import trunc_normal_ as __call_trunc_normal_
from utils.registry import register_model

from .layers import Block, PatchEmbed, PatchDownsample, PatchUpsample, Output
from .utils import build_2d_sincos_position_embedding, _cfg


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PriTEncoder(nn.Module):
    """
    PyramidReconstructionImageTransformer Encoder
    """

    def __init__(self,
                 # args for ViT (timm)
                 # w/o `num_classes`, `distilled`, `detph`, `representation_size` and `weight_init`.
                 # default value of `patch_size` and `embed_dim` changed.
                 img_size=224, patch_size=4, in_chans=3, embed_dim=96, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 # more args for ViT (BeiT)
                 qk_scale=None, init_values=0.,
                 # args for PriT
                 strides=(1, 2, 2, 2), depths=(2, 2, 6, 2), dims=(48, 96, 192, 384),
                 mask_ratio=0.75, use_mean_pooling=True, pyramid_reconstruction=False):
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
            mask_ratio (float): mask ratio
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
        self.mask_ratio = mask_ratio
        self.pyramid_reconstruction = pyramid_reconstruction
        self.use_mean_pooling = use_mean_pooling
        self.use_cls_token = use_cls_token = not use_mean_pooling

        self.stride = stride = patch_size * reduce(mul, strides)  # 32 = 4 * 2 * 2 * 2 * 1
        self.out_size = out_size = (img_size[0] // stride, img_size[1] // stride)  # 7 = 224 / 32
        self.num_patches = out_size[0] * out_size[1]  # 49 = 7 * 7
        self.num_visible = int(self.num_patches * (1 - self.mask_ratio))  # 12 = ⎣49 * (1 - 0.75)⎦
        self.num_masked = self.num_patches - self.num_visible  # 37 = 49 - 12

        grid_size = stride // patch_size  # 8 = 32 / 4
        self.split_shape = (out_size[0], grid_size, out_size[1], grid_size)  # (7, 8, 7, 8)

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        grid_h, grid_w = self.patch_embed.grid_size
        assert out_size[0] * grid_size == grid_h and out_size[1] * grid_size == grid_w

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = build_2d_sincos_position_embedding(
            grid_h, grid_w, embed_dim, use_cls_token=use_cls_token)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, sum(depths))]  # stochastic depth decay rule
        for i in range(self.num_layers):
            downsample = i > 0 and (strides[i] == 2 or dims[i - 1] != dims[i])
            self.add_module(f'stage{i + 1}', nn.Sequential(
                PatchDownsample(dims[i - 1], dims[i], self.num_visible, stride=strides[i],
                    norm_layer=norm_layer, with_cls_token=use_cls_token) if downsample else nn.Identity(),
                self._build_blocks(dims[i], num_heads, depths[i],
                    dpr=[dpr.pop() for _ in range(depths[i])], init_values=init_values),
            ))
        self.norm = norm_layer(self.num_features)

        # weight initialization
        self.apply(self._init_weights)
        if use_cls_token:
            trunc_normal_(self.cls_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _build_blocks(self, dim, num_heads, depth, dpr=None, init_values=0.):
        dpr = dpr or ([0.] * depth)
        blocks = [Block(
            dim=dim, num_heads=num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale, drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i],
            norm_layer=self.norm_layer, act_layer=self.act_layer, init_values=init_values)
            for i in range(depth)]
        return nn.Sequential(*blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

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

        # get visible tokens by random shuffle
        shuffle = torch.randperm(self.num_patches)  # 49 = 7 * 7
        visible_tokens = x_wo_cls_token[:, shuffle[:self.num_visible]]    # Bx12x(8x8)xL
        visible_tokens = visible_tokens.flatten(1, 2)  # Bx(12x8x8)xL

        if self.use_cls_token:
            visible_tokens = torch.cat((x[:, [0]], visible_tokens), dim=1)

        # [Bx(12*8*8)xL, Bx(12*4*4)xL, Bx(12*2*2)xL, Bx(12*1*1)xL]
        encoded_visible_patches = self.blocks(visible_tokens)

        # w/o cls token
        if self.use_cls_token:
            encoded_visible_patches = [p[:, 1:] for p in encoded_visible_patches]

        return encoded_visible_patches, shuffle


class PriTDecoder1(nn.Module):
    """
    PyramidReconstructionImageTransformer Decoder
    """

    def __init__(self, encoder: PriTEncoder, decoder_dim=512, decoder_depth=8, decoder_num_heads=8):
        super().__init__()
        stage_idx = encoder.num_layers  # 4

        stride = encoder.patch_size * reduce(mul, encoder.strides[:stage_idx])  # 32 for stage4
        img_size = encoder.img_size  # 224
        out_size = (img_size[0] // stride, img_size[1] // stride)  # 7 for stage4
        num_features = encoder.dims[stage_idx - 1]
        num_heads = decoder_dim // (num_features // encoder.num_heads)
        decoder_num_heads = decoder_num_heads or num_heads
        self.num_layers = encoder.num_layers  # 4
        self.num_patches = encoder.num_patches  # 49
        self.num_visible = encoder.num_visible  # 12
        self.num_masked = encoder.num_masked  # 37
        self.stride = stride

        # build encoder linear projection(s) and mask token(s)
        self.encoder_linear_proj = nn.Linear(num_features, decoder_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # pos_embed
        self.decoder_pos_embed = build_2d_sincos_position_embedding(
            out_size[0], out_size[1], decoder_dim)

        # build decoder
        self.decoder_blocks = encoder._build_blocks(decoder_dim, num_heads, decoder_depth)
        self.decoder_norm = encoder.norm_layer(decoder_dim)
        self.decoder_linear_proj = nn.Linear(decoder_dim, stride ** 2 * 3)

        # weight initialization
        self.apply(self._init_weights)
        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'decoder_pos_embed', 'mask_token'}

    def forward(self, x, shuffle):
        """
            B: batch size
            V: num_visible
            M: num_masked
            P: num_patches, P=V+M
            G: grid_h * grid_w = Gh * Gw
            D: decoder_dim
            S: stride
        """

        # encoder linear projection
        encoded_visible_patches = self.encoder_linear_proj(x)  # B x V x D

        # un-shuffle
        mask_token = self.mask_token.expand(x.size(0), self.num_masked, -1)  # B x M x D
        all_tokens = torch.cat((encoded_visible_patches, mask_token), dim=1)  # B x P x D = B x (V + M) x D
        all_tokens = all_tokens[:, shuffle.argsort()]  # B x P x D

        # with positional embedding
        all_tokens = all_tokens + self.decoder_pos_embed

        # decode all tokens
        masked_inds = shuffle[self.num_visible:]
        decoded_all_tokens = self.decoder_blocks(all_tokens)  # B x P x D
        decoded_masked_tokens = decoded_all_tokens[:, masked_inds] if self.num_masked > 0 else decoded_all_tokens  # B x M x D
        decoded_masked_tokens = self.decoder_norm(decoded_masked_tokens)  # B x M x D
        decoded_masked_tokens = self.decoder_linear_proj(decoded_masked_tokens)  # B x M x 32*32*3

        return decoded_masked_tokens


class PriTDecoder2(nn.Module):
    """
    PyramidReconstructionImageTransformer Decoder
    """

    def __init__(self, encoder: PriTEncoder, decoder_dim=512, decoder_depth=8, decoder_num_heads=8):
        super().__init__()
        stage_idx = 1

        stride = encoder.patch_size * reduce(mul, encoder.strides[:stage_idx])  # 4 for stage1
        img_size = encoder.img_size  # 224
        out_size = (img_size[0] // stride, img_size[1] // stride)  # 56 for stage1
        num_features = encoder.dims[stage_idx - 1]
        num_heads = decoder_dim // (num_features // encoder.num_heads)
        decoder_num_heads = decoder_num_heads or num_heads
        self.num_layers = encoder.num_layers  # 4
        self.num_patches = encoder.num_patches  # 49
        self.num_visible = encoder.num_visible  # 12
        self.num_masked = encoder.num_masked  # 37
        self.stride = stride

        # build encoder linear projection(s) and mask token(s)
        for stage_idx, dim in enumerate(encoder.dims, 1):
            setattr(self, f'encoder_linear_proj{stage_idx}', nn.Linear(dim, decoder_dim))
            setattr(self, f'mask_token{stage_idx}', nn.Parameter(torch.zeros(1, 1, 1, decoder_dim)))

        # build upsampler(s)
        for stage_idx, s in enumerate(encoder.strides[1:], 1):
            setattr(self, f'upsampler{stage_idx}',
                nn.Identity() if s == 1 else PatchUpsample(self.num_patches, stride=s))

        # build and pos embed(s) and output attention(s)
        for stage_idx in range(1, self.num_layers + 1):
            s = encoder.patch_size * reduce(mul, encoder.strides[:stage_idx])
            pos_embed = self._build_pos_embed(img_size[0] // s, img_size[1] // s, decoder_dim,
                grid_size=encoder.strides[stage_idx] if stage_idx < self.num_layers else 1)
            setattr(self, f'pos_embed{stage_idx}', pos_embed)
            setattr(self, f'output{stage_idx}', Output(decoder_dim, num_heads))

        # pos_embed
        self.decoder_pos_embed = self._build_pos_embed(
            out_size[0], out_size[1], decoder_dim, grid_size=encoder.stride // stride)

        # build decoder
        self.decoder_blocks = encoder._build_blocks(decoder_dim, num_heads, decoder_depth)
        self.decoder_norm = encoder.norm_layer(decoder_dim)
        self.decoder_linear_proj = nn.Linear(decoder_dim, stride ** 2 * 3)

        # weight initialization
        self.apply(self._init_weights)
        for i in range(self.num_layers):
            trunc_normal_(getattr(self, f'mask_token{i + 1}'), std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _build_pos_embed(self, h, w, dim, grid_size=1):
        split_shape = (h // grid_size, grid_size, w // grid_size, grid_size)
        decoder_pos_embed = build_2d_sincos_position_embedding(h, w, dim)
        decoder_pos_embed = nn.Parameter(
            self.grid_patches(decoder_pos_embed, split_shape), requires_grad=False)
        return decoder_pos_embed.flatten(1, 2)  # BxNxL

    def grid_patches(self, x, split_shape):
        B, N, L = x.shape
        x = x.reshape(B, *split_shape, L)               # Bx  (7x8x7x8)  xL
        x = x.permute([0, 1, 3, 2, 4, 5])               # Bx   7x7x8x8   xL
        x = x.reshape(B, x.size(1) * x.size(2), -1, L)  # Bx (7x7)x(8x8) xL
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'decoder_pos_embed'} | {f'mask_token{i + 1}' for i in range(self.num_layers)}

    def forward(self, xs, shuffle):
        """
            B: batch size
            V: num_visible
            M: num_masked
            P: num_patches, P=V+M
            G: grid_h * grid_w = Gh * Gw
            D: decoder_dim
            S: stride
        """

        # FPN-like
        out = None
        for i, x in enumerate(xs[::-1]):
            stage_idx = len(xs) - i

            # encode visible patches
            encoder_linear_proj = getattr(self, f'encoder_linear_proj{stage_idx}')
            encoded_visible_patches = encoder_linear_proj(x)  # B x (V*G) x D
            B, VG, L = encoded_visible_patches.shape
            G = VG // self.num_visible
            encoded_visible_patches = encoded_visible_patches.view(B, -1, G, L)  # B x V x G x D

            # un-shuffle
            mask_token = getattr(self, f'mask_token{stage_idx}')  # 1 x 1 x 1 x D
            mask_token = mask_token.expand(x.size(0), self.num_masked, G, -1)  # B x M x G x D
            all_tokens = torch.cat((encoded_visible_patches, mask_token), dim=1)  # B x P x G x D
            all_tokens = all_tokens[:, shuffle.argsort()]  # B x P x G x D
            all_tokens = all_tokens.reshape(B, -1, L)  # B x (P*G) x D

            # upsample
            if out is not None:
                upsampled = getattr(self, f'upsampler{stage_idx}')(out)  # B x N x D
                all_tokens = all_tokens + upsampled

            # pos embedding
            all_tokens = all_tokens + getattr(self, f'pos_embed{stage_idx}')

            out = getattr(self, f'output{stage_idx}')(all_tokens)

        # pos embedding
        out = out + self.decoder_pos_embed

        # decode all tokens
        masked_inds = shuffle[self.num_visible:]
        decoded_all_tokens = self.decoder_blocks(out)  # B x (P*G) x D
        decoded_all_tokens = decoded_all_tokens.view(B, self.num_patches, -1, L)  # B x P x G x D
        decoded_masked_tokens = decoded_all_tokens[:, masked_inds] if self.num_masked > 0 else decoded_all_tokens  # B x M x G x D
        decoded_masked_tokens = decoded_masked_tokens.reshape(B, -1, L)  # B x (M*G) x D
        decoded_masked_tokens = self.decoder_norm(decoded_masked_tokens)  # B x (M*G) x D
        decoded_masked_tokens = self.decoder_linear_proj(decoded_masked_tokens)  # B x (M*G) x S*S*C = B x (37*8*8) x (4*4*3)

        return decoded_masked_tokens


class PriT1(nn.Module):
    """
    Build a PyramidReconstructionImageTransformer (PriT) model with a encoder and encoder
    """

    def __init__(self, encoder, decoder_dim=512, decoder_depth=8, decoder_num_heads=8, normalized_pixel=True):
        super().__init__()
        self.normalized_pixel = normalized_pixel

        # build encoder
        self.encoder: PriTEncoder = encoder()
        self.num_patches = self.encoder.num_patches
        self.num_visible = self.encoder.num_visible

        # build decoder
        self.decoder = PriTDecoder1(self.encoder, decoder_dim=decoder_dim,
            decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads)

    @torch.jit.ignore
    def no_weight_decay(self):
        return (set('encoder.' + name for name in list(self.encoder.no_weight_decay())) |
                set('decoder.' + name for name in list(self.decoder.no_weight_decay())))

    def get_target(self, img, masked_inds=None) -> torch.Tensor:
        B, C, H, W = img.shape
        S = self.decoder.stride
        target = img.view(B, C, H // S, S, W // S, S)  # BxCx7x32x7x32
        target = target.permute([0, 2, 4, 3, 5, 1]).reshape(B, -1, S * S, C)  # Bx49x(32*32)xC

        if masked_inds is not None:
            target = target[:, masked_inds]  # Bx37x(32*32)xC

        # patch normalize
        if self.normalized_pixel:
            mean = target.mean(dim=-2, keepdim=True)
            std = target.var(dim=-2, unbiased=True, keepdim=True).sqrt()
            target = (target - mean) / (std + 1e-6)

        target = target.view(B, -1, S * S * C)  # Bx37x(32*32*C)
        return target

    def forward(self, x):
        """
            B: batch size
            V: num_visible
            M: num_masked
            P: num_patches, P=V+M
            G: grid_h * grid_w = Gh * Gw
            D: decoder_dim
            S: stride
        """

        # encode visible patches
        encoded_visible_patches_multi_stages, shuffle = self.encoder(x)
        encoded_visible_patches = encoded_visible_patches_multi_stages[-1]  # Bx(12*1*1)xL

        # decode
        decoded_masked_tokens = self.decoder(encoded_visible_patches, shuffle)  # Bx(M*G)x(S*S*C)

        # generate target
        masked_target = self.get_target(x, masked_inds=shuffle[self.num_visible:])  # Bx(M*G)x(S*S*C)

        return self.loss(decoded_masked_tokens, masked_target)

    def loss(self, img, target):
        return F.mse_loss(img, target)


class PriT2(nn.Module):
    """
    Build a PyramidReconstructionImageTransformer (PriT) model with a encoder and encoder
    """

    def __init__(self, encoder, decoder_dim=512, decoder_depth=8, decoder_num_heads=8, normalized_pixel=True):
        super().__init__()
        self.normalized_pixel = normalized_pixel

        # build encoder
        self.encoder: PriTEncoder = encoder(pyramid_reconstruction=True)
        self.num_patches = self.encoder.num_patches
        self.num_visible = self.encoder.num_visible

        # build decoder
        self.decoder = PriTDecoder2(self.encoder, decoder_dim=decoder_dim,
            decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads)

    def get_target(self, img, masked_inds=None) -> torch.Tensor:
        B, C, H, W = img.shape
        ES, S, P = self.encoder.stride, self.decoder.stride, self.num_patches
        Ph, Pw = H // ES, W // ES
        Gh = Gw = ES // S
        target = img.view(B, C, Ph, Gh, S, Pw, Gw, S)  # BxCx7x1x32x7x1x32
        target = target.permute([0, 2, 5, 3, 6, 4, 7, 1]).reshape(B, P, -1, C)  # BxPx(Gh*Gw*S*S)xC

        if masked_inds is not None:
            target = target[:, masked_inds]  # BxMx(Gh*Gw*S*S)xC

        # patch normalize
        if self.normalized_pixel:
            mean = target.mean(dim=-2, keepdim=True)
            std = target.std(dim=-2, keepdim=True)
            target = (target - mean) / (std + 1e-6)

        target = target.view(B, -1, S * S * C)  # Bx(M*G)x(S*S*C)
        return target

    def forward(self, x):
        """
            B: batch size
            V: num_visible
            M: num_masked
            P: num_patches, P=V+M
            G: grid_h * grid_w = Gh * Gw
            D: decoder_dim
            S: stride
        """

        # encode visible patches
        # [Bx(V*8*8)xC1, Bx(V*4*4)xC2, Bx(V*2*2)xC3, Bx(V*1*1)xC4]
        encoded_visible_patches_multi_stages, shuffle = self.encoder(x)

        # decode
        decoded_masked_tokens = self.decoder(encoded_visible_patches_multi_stages, shuffle)  # Bx(M*G)x(S*S*C)

        # generate target
        masked_target = self.get_target(x, masked_inds=shuffle[self.num_visible:])  # Bx(M*G)x(S*S*C)

        return self.loss(decoded_masked_tokens, masked_target)

    def loss(self, img, target):
        return F.mse_loss(img, target)


@register_model
def pretrain_prit_mae_small_patch16_224(pretrained=False, **kwargs):
    normalized_pixel=kwargs.pop('normalized_pixel')
    model = PriT1(
        partial(
            PriTEncoder,
            img_size=224,
            patch_size=16,
            embed_dim=384,
            strides=[1],
            depths=[12],
            dims=[384],
            num_heads=6,
            **kwargs,
        ),
        decoder_dim=192,
        decoder_depth=1,
        decoder_num_heads=3,
        normalized_pixel=normalized_pixel)
    model.default_cfg = _cfg()
    return model
