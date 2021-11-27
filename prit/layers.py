from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.layers import DropPath, to_2tuple


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class LocalBlock(Block):
    def __init__(self, num_patches, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_patches = num_patches

    def forward(self, x):
        B, N, L = x.shape
        if self.gamma_1 is None:
            x = x.view(-1, N // self.num_patches, L)  # B*12 x 8*8 x L
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x.view(B, N, L)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x.view(-1, N // self.num_patches, L)  # B*12 x 8*8 x L
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x.view(B, N, L)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
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


class Output(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x
