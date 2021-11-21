import math
from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .layers import DropPath, PatchEmbed, Mlp


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


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
        elif stride == 2:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(stride)

        self.reduction = nn.Linear(dim_in, dim_out, bias=False)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        if self.pool is not None:
            x_wo_cls_token = x[:, 1:] if self.with_cls_token else x

            B, VG, L = x_wo_cls_token.shape  # 12, G=8*8, 4x4, 2x2, 1x1
            Gh = Gw = int((VG // self.num_visible) ** 0.5)

            # get grid-like patches
            x_wo_cls_token = x_wo_cls_token.permute(0, 2, 1)  # BxLxVG
            x_wo_cls_token = x_wo_cls_token.reshape(B, -1, Gh, Gw)  # Bx(L*V)xGhxGw

            x_wo_cls_token = self.pool(x_wo_cls_token)  # Bx(LxV)x Gh/2 x Gw/2

            # reshape
            x_wo_cls_token = x_wo_cls_token.view(B, L, -1)  # BxLx(VG/4)
            x_wo_cls_token = x_wo_cls_token.permute(0, 2, 1)  # Bx(VG/4)xL

            x = torch.cat((x[:, [0]], x_wo_cls_token), dim=1) if self.with_cls_token else x_wo_cls_token

        x = self.reduction(x)
        x = self.norm(x)
        return x


class PatchUpsample(nn.Module):
    def __init__(self, num_visible, stride=2):
        super().__init__()

        if stride != 2:
            raise ValueError(stride)

        self.num_visible = num_visible
        self.stride = stride
        self.upsampler = partial(F.interpolate, scale_factor=stride, mode='bilinear', align_corners=False)

    def forward(self, x):
        B, VG, L = x.shape
        Gh = Gw = int((VG // self.num_visible) ** 0.5)

        x = x.permute(0, 2, 1)
        x = x.reshape(B, -1, Gh, Gw)
        x = self.upsampler(x)

        # reshape
        x = x.view(B, L, -1)  # BxLx(Gh*2*Gw*2)
        x = x.permute(0, 2, 1)

        return x


class PriTEncoder(nn.Module):
    """
    PyramidReconstructionImageTransformer Encoder
    """

    def __init__(self,
                 # args for ViT, w/o `num_classes`, `distilled`, `detph` and `representation_size`.
                 # default value of `patch_size` and `embed_dim` changed.
                 img_size=224, patch_size=4, in_chans=3, embed_dim=96, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, weight_init='',
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
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme

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
        self.num_tokens = 0 if use_mean_pooling else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.img_size = img_size = _pair(img_size)

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
            nn.init.trunc_normal_(self.cls_token, std=.02)
        self.pos_embed = self.build_2d_sincos_position_embedding(grid_h, grid_w, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        for i in range(self.num_layers):
            downsample = i > 0 and (strides[i] == 2 or dims[i - 1] != dims[i])
            self.add_module(f'stage{i + 1}', nn.Sequential(
                PatchDownsample(dims[i - 1], dims[i], self.num_visible, stride=strides[i],
                    norm_layer=norm_layer, with_cls_token=use_cls_token) if downsample else nn.Identity(),
                self._build_blocks(dims[i], num_heads, depths[i], dpr=[dpr.pop() for _ in range(depths[i])]),
            ))
        self.norm = norm_layer(self.num_features)

        # weight initialization
        self.init_weights(weight_init)

    def init_weights(self, mode='moco_v3'):
        assert mode in {'moco_v3', 'timm'}
        # weight initialization, MOCO v3
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if mode == 'moco_v3':
                    if 'qkv' in name:
                        # treat the weights of Q, K, V separately
                        val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                        nn.init.uniform_(m.weight, -val, val)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                elif mode == 'timm':
                    nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if mode == 'moco_v3' and isinstance(self.patch_embed, PatchEmbed):
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
        if self.mask_ratio == 0.:
            visible_tokens = x_wo_cls_token
        else:
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
    PyramidReconstructionImageTransformer Dncoder
    """

    def __init__(self, encoder: PriTEncoder, stage_idx, decoder_dim=512, decoder_depth=8):
        super().__init__()
        self.stage_idx = stage_idx

        self.stride = stride = encoder.patch_size * reduce(mul, encoder.strides[:stage_idx])  # 32 for stage4
        img_size = encoder.img_size  # 224
        out_size = (img_size[0] // stride, img_size[1] // stride)  # 7 for stage4
        num_features = encoder.dims[stage_idx - 1]
        num_heads = decoder_dim // (num_features // encoder.num_heads)
        self.num_patches = encoder.num_patches  # 49
        self.num_visible = encoder.num_visible  # 12
        self.num_masked = encoder.num_masked  # 37

        # build encoder linear projection
        self.encoder_linear_proj = nn.Linear(num_features, decoder_dim)

        # build decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = encoder.build_2d_sincos_position_embedding(
            out_size[0], out_size[1], decoder_dim, decode=True)
        self.decoder_blocks = encoder._build_blocks(decoder_dim, num_heads, decoder_depth)
        self.decoder_norm = encoder.norm_layer(decoder_dim)
        self.decoder_linear_proj = nn.Linear(decoder_dim, stride ** 2 * 3)

        # weight initialization, MOCO v3
        for name, m in self.decoder_blocks.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        masked_inds = shuffle[self.num_visible:]
        mask_token = self.mask_token.expand(x.size(0), self.num_masked, -1)  # B x M x D
        all_tokens = torch.cat((encoded_visible_patches, mask_token), dim=1)  # B x P x D = B x (V + M) x D
        all_tokens = all_tokens[:, shuffle.argsort()]  # B x P x D

        # with positional embedding
        all_tokens = all_tokens + self.decoder_pos_embed

        # decode all tokens
        decoded_all_tokens = self.decoder_blocks(all_tokens)  # B x P x D
        decoded_masked_tokens = decoded_all_tokens[:, masked_inds] if self.num_masked > 0 else decoded_all_tokens  # B x M x D
        decoded_masked_tokens = self.decoder_norm(decoded_masked_tokens)  # B x M x D
        decoded_masked_tokens = self.decoder_linear_proj(decoded_masked_tokens)  # B x M x 32*32*3

        return decoded_masked_tokens


class PriTDecoder2(nn.Module):

    def __init__(self, encoder: PriTEncoder, stage_idx, decoder_dim=512, decoder_depth=8):
        super().__init__()
        self.stage_idx = stage_idx

        self.stride = stride = encoder.patch_size * reduce(mul, encoder.strides[:stage_idx])  # 32 for stage4
        img_size = encoder.img_size  # 224
        out_size = (img_size[0] // stride, img_size[1] // stride)  # 7 for stage4
        num_features = encoder.dims[stage_idx - 1]
        num_heads = decoder_dim // (num_features // encoder.num_heads)
        self.num_patches = encoder.num_patches  # 49
        self.num_visible = encoder.num_visible  # 12
        self.num_masked = encoder.num_masked  # 37

        # build encoder linear projection(s)
        self.encoder_linear_projs = nn.ModuleList(nn.Linear(dim, decoder_dim) for dim in encoder.dims)

        # mask token(s)
        for i in range(encoder.num_layers):
            setattr(self, f'mask_token{i + 1}', nn.Parameter(torch.zeros(1, 1, 1, decoder_dim)))

        # pos_embed
        grid_size = encoder.stride // stride
        split_shape = (out_size[0] // grid_size, grid_size, out_size[1] // grid_size, grid_size)
        decoder_pos_embed = encoder.build_2d_sincos_position_embedding(
            out_size[0], out_size[1], decoder_dim, decode=True)
        self.decoder_pos_embed = nn.Parameter(
            encoder.grid_patches(decoder_pos_embed, split_shape), requires_grad=False)

        # build upsampler(s)
        self.upsamplers = nn.ModuleList(
            (nn.Identity() if stride == 1 else PatchUpsample(self.num_patches, stride=stride))
            for stride in encoder.strides
        )

        # build decoder
        self.decoder_blocks = encoder._build_blocks(decoder_dim, num_heads, decoder_depth)
        self.decoder_norm = encoder.norm_layer(decoder_dim)
        self.decoder_linear_proj = nn.Linear(decoder_dim, stride ** 2 * 3)

        # weight initialization, MOCO v3
        for name, m in self.decoder_blocks.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xs, shuffle):
        upsampled = None
        for i, x in enumerate(xs):
            # encode visible patches
            encoded_visible_patches = self.encoder_linear_projs[i](x)  # B x (V*G) x D
            B, VG, L = encoded_visible_patches.shape
            encoded_visible_patches = encoded_visible_patches.reshape(B, self.num_visible, -1, L)  # B x V x G x D
            G = encoded_visible_patches.size(2)  # G = grid_h * grid_w = Gh * Gw

            # un-shuffle
            mask_token = getattr(self, f'mask_token{i + 1}')  # 1 x 1 x 1 x D
            mask_token = mask_token.expand(x.size(0), self.num_masked, G, -1)  # B x M x G x D
            all_tokens = torch.cat((encoded_visible_patches, mask_token), dim=1)  # B x P x G x D
            all_tokens = all_tokens[:, shuffle.argsort()]  # B x P x G x D
            all_tokens = all_tokens.reshape(B, -1, L)  # B x (P*G) x D

            if upsampled is not None:
                all_tokens = all_tokens + upsampled

            # TODO add fpn-like Attention here

            upsampled = self.upsamplers[i](all_tokens)

        all_tokens = upsampled / len(xs)

        # decode all tokens
        masked_inds = shuffle[self.num_visible:]
        decoded_all_tokens = self.decoder_blocks(all_tokens)  # B x (P*G) x D
        decoded_all_tokens = decoded_all_tokens.view(B, self.num_patches, -1, L)  # B x P x G x D
        decoded_masked_tokens = decoded_all_tokens[:, masked_inds] if self.num_masked > 0 else decoded_all_tokens  # B x M x G x D
        decoded_masked_tokens = decoded_masked_tokens.reshape(B, -1, L)  # B x (M*G) x D
        decoded_masked_tokens = self.decoder_norm(decoded_masked_tokens)  # B x (M*G) x D
        decoded_masked_tokens = self.decoder_linear_proj(decoded_masked_tokens)  # B x (M*G) x S*S*C

        return decoded_masked_tokens
