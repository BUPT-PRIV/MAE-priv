import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mix_mae import Mix_MAE
from .pyramid_reconstruction_image_transformer import PriTEncoder, PriTDecoder1, PriTDecoder2
from .vision_transformer import Block


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
        self.num_visible = self.encoder.num_visible

        # build encoder_linear_proj
        # Bx(14*14*0.75)x512 = Bx147x512
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.encoder_linear_proj = nn.Linear(self.encoder.embed_dim, decoder_dim)
        self.decoder_pos_embed = self.encoder.build_2d_sincos_position_embedding(embed_dim=decoder_dim, decode=True)

        # build decoder
        self.decoder_blocks = self._build_decoder(
            decoder_dim=decoder_dim, decoder_head=decoder_dim // 64, decoder_depth=decoder_depth
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim, eps=1e-6)

        self.color_channel = 6 if mix_mode is not None else 3
        self.decoder_linear_proj = nn.Linear(decoder_dim, self.patch_size[0] * self.patch_size[1] * self.color_channel)

        # mix mae
        self.mix_up = None
        if mix_mode is not None:
            self.mix_up = Mix_MAE(mode=mix_mode, alpha=mix_alpha)

        # weight initialization
        for name, m in self.decoder_blocks.named_modules():
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

    def get_target(self, img, masked_inds=None) -> torch.Tensor:
        B, C, H, W = img.shape
        S = self.patch_size[0]
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
        if self.mix_up is not None:
            x, target = self.mix_up(x)
        else:
            target = x

        # encode visible patches
        encoded_visible_patches, shuffle = self.encoder(x)
        encoded_visible_patches = self.encoder_linear_proj(encoded_visible_patches)  # Bx49x768 --> Bx49x512

        # un-shuffle
        mask_token = self.mask_token.repeat(x.size(0), self.num_patches - self.num_visible, 1)
        decoder_input = torch.cat((encoded_visible_patches, mask_token), dim=1)
        decoder_input = decoder_input[:, shuffle.argsort()]  # Bx49x512 + Bx147x512 --> Bx196x512

        # with positional embedding
        decoder_input = decoder_input + self.decoder_pos_embed

        # decode all tokens
        decoder_output = self.decoder_blocks(decoder_input)  # Bx(14*14)x512
        decoder_output = self.decoder_norm(decoder_output)
        decoder_output = self.decoder_linear_proj(decoder_output)  # Bx(14*14)x512 --> Bx(14*14)x(16*16*3)

        # target
        target = self.get_target(target)  # Bx(14*14)x(16*16*3)

        return self.loss(decoder_output, target, masked_inds=shuffle[self.num_visible:])

    def loss(self, x, y, masked_inds=None):
        if masked_inds is not None:
            return F.mse_loss(x[:, masked_inds], y[:, masked_inds], reduction="mean")
        else:
            return F.mse_loss(x, y, reduction="mean")


class PriT1(nn.Module):
    """
    Build a PyramidReconstructionImageTransformer (PriT) model with a encoder and encoder
    """

    def __init__(self, encoder, decoder_dim=512, decoder_depth=8, normalized_pixel=False):
        super().__init__()
        self.normalized_pixel = normalized_pixel

        # build encoder
        self.encoder: PriTEncoder = encoder()
        self.num_patches = self.encoder.num_patches
        self.num_visible = self.encoder.num_visible

        # build decoder
        self.decoder = PriTDecoder1(
            self.encoder, decoder_dim=decoder_dim, decoder_depth=decoder_depth)

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

    def __init__(self, encoder, decoder_dim=512, decoder_depth=8, normalized_pixel=False):
        super().__init__()
        self.normalized_pixel = normalized_pixel

        # build encoder
        self.encoder: PriTEncoder = encoder(pyramid_reconstruction=True)
        self.num_patches = self.encoder.num_patches
        self.num_visible = self.encoder.num_visible

        # build decoder
        self.decoder = PriTDecoder2(
            self.encoder, decoder_dim=decoder_dim, decoder_depth=decoder_depth)

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
