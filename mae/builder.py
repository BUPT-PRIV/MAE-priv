import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .vision_transformer import Block
from .mix_mae import Mix_MAE

class MAE(nn.Module):
    """
    Build a MAE model with a encoder and encoder
    https://arxiv.org/abs/2111.06377
    """

    def __init__(self, encoder, image_size=224, decoder_dim=512, decoder_depth=8, normalized_pixel=False,
                 mix_mode=None, mix_alpha=0.0):
        super(MAE, self).__init__()

        self.image_size = image_size
        self.normalized_pixel = normalized_pixel  # TODO

        # build encoder
        self.encoder = encoder()
        self.num_patches = self.encoder.num_patches
        self.patch_size = self.encoder.patch_size
        self.visible_size = self.encoder.visible_size
        self.use_mean_pooling = self.encoder.use_mean_pooling

        # build dim_proj
        # Bx(14*14*0.75)x512 = Bx147x512
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.dim_proj = nn.Linear(self.encoder.embed_dim, decoder_dim)
        self.decoder_pos_embed = self.encoder.build_2d_sincos_position_embedding(embed_dim=decoder_dim, decode=True)

        # build decoder
        self.decoder = self._build_decoder(
            decoder_dim=decoder_dim, decoder_head=decoder_dim // 64, decoder_depth=decoder_depth
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim, eps=1e-6)
        self.decoder_linear_proj = nn.Linear(decoder_dim, self.patch_size[0] * self.patch_size[1] * 3)

        # mix mae
        self.mix_up = None
        if mix_mode is not None:
            self.mix_up = Mix_MAE(mode=mix_mode, alpha=mix_alpha)

        # weight initialization
        for name, m in self.decoder.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_decoder(self, decoder_dim=512, decoder_head=8, decoder_depth=8):
        blocks = [
            Block(
                dim=decoder_dim, num_heads=decoder_head, mlp_ratio=4, qkv_bias=True, drop=0.,
                attn_drop=0., drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU)
            for i in range(decoder_depth)
        ]
        return nn.Sequential(*blocks)

    def _mse_loss(self, x, y, masked_index=None):
        if masked_index is not None:
            return F.mse_loss(x[:, masked_index, :], y[:, masked_index, :], reduction="mean")
        else:
            return F.mse_loss(x, y, reduction="mean")

    def forward(self, x):
        if self.mix_up is not None:
            x, target = self.mix_up(x)
        else:
            target = x

        # encode visible token
        B, C, H, W = x.size()  # B, 3, 224, 224
        encoded_visible_patches, shuffle = self.encoder(x)

        # un-shuffle  with positional embedding
        encoded_visible_patches = self.dim_proj(encoded_visible_patches)  # Bx49x768 --> Bx49x512
        decoder_input = torch.cat(
            [encoded_visible_patches, self.mask_token.repeat([B, self.num_patches - self.visible_size, 1])], dim=1
        )[:, shuffle.sort()[1], :]  # Bx49x512 + Bx147x512 --> Bx196x512
        decoder_input = decoder_input + self.decoder_pos_embed

        # decode (encoded_visible_patches + mask_token)
        decoder_output = self.decoder(decoder_input)  # Bx(14*14)x512
        decoder_output = self.decoder_norm(decoder_output)
        decoder_output = self.decoder_linear_proj(decoder_output)  # Bx(14*14)x512 --> Bx(14*14)x(16*16*3)

        # target
        target = target.view(
            [B, C, H // self.patch_size[0], self.patch_size[0], W // self.patch_size[1], self.patch_size[1]]
        )  # Bx3x224x224 --> Bx3x16x14x16x14
        # Bx3x14x16x14x16 --> Bx(14*14)x(16*16*3)
        target = target.permute([0, 2, 4, 3, 5, 1]).reshape(B, self.num_patches, -1, C)
        if self.normalized_pixel:
            mean = target.mean(dim=-2, keepdim=True)
            std = target.var(dim=-2, unbiased=True, keepdim=True).sqrt()
            target = (target - mean) / (std + 1e-6)
        target = target.view(B, self.num_patches, -1)

        return self._mse_loss(decoder_output, target, masked_index=shuffle[self.visible_size:])
