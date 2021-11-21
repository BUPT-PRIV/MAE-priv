import math
from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mix_mae import Mix_MAE
from .pyramid_reconstruction_image_transformer import PriTEncoder, _init_vit_weights
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
        target = img.view(B, C, H // S, S, W // S, S)
        target = target.permute([0, 2, 4, 3, 5, 1]).reshape(B, self.num_patches, -1, C)  # Bx49x(32*32)*C

        if masked_inds is not None:
            target = target[:, masked_inds]  # Bx37x(32*32)*C

        # patch normalize
        if self.normalized_pixel:
            mean = target.mean(dim=-2, keepdim=True)
            std = target.var(dim=-2, unbiased=True, keepdim=True).sqrt()
            target = (target - mean) / (std + 1e-6)

        target = target.flatten(2)  # Bx37x(32*32*C)
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

        masked_inds = shuffle[self.num_visible:]
        return self.loss(decoder_output[:, masked_inds], target[:, masked_inds])

    def loss(self, x, y):
        return F.mse_loss(x, y)


class PriT(nn.Module):
    """
    Build a PyramidReconstructionImageTransformer (PriT) model with a encoder and encoder
    """

    stage_idx = 4

    def __init__(self, encoder, decoder_dim=512, decoder_depth=8,
                 normalized_pixel=False, pyramid_reconstruction=False):
        super().__init__()
        self.normalized_pixel = normalized_pixel
        self.pyramid_reconstruction = pyramid_reconstruction
        if pyramid_reconstruction:
            raise NotImplementedError

        # build encoder
        self.encoder: PriTEncoder = encoder()
        stride = self.encoder.patch_size * reduce(mul, self.encoder.strides[:self.stage_idx])  # 32
        img_size = self.encoder.img_size  # 224
        out_size = (img_size[0] // stride, img_size[1] // stride)  # 7
        num_features = self.encoder.dims[self.stage_idx - 1]
        num_heads = decoder_dim // (num_features // self.encoder.num_heads)
        self.num_patches = self.encoder.num_patches
        self.num_visible = self.encoder.num_visible

        # build encoder linear projection
        self.encoder_linear_proj = nn.Linear(num_features, decoder_dim)

        # build decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, decoder_dim))
        self.decoder_blocks = self.encoder._build_blocks(decoder_dim, num_heads, decoder_depth)
        self.decoder_norm = self.encoder.norm_layer(decoder_dim)
        self.decoder_linear_proj = nn.Linear(decoder_dim, stride ** 2 * 3)

        # pos_embed
        grid_size = self.encoder.stride // stride
        split_shape = (out_size[0] // grid_size, grid_size, out_size[1] // grid_size, grid_size)
        decoder_pos_embed = self.encoder.build_2d_sincos_position_embedding(
            out_size[0], out_size[1], decoder_dim, decode=True)
        self.decoder_pos_embed = nn.Parameter(
            self.encoder.grid_patches(decoder_pos_embed, split_shape), requires_grad=False)

        # weight initialization for decoder, ViT (timm)
        self.init_weights()

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

    def init_weights(self):
        self.apply(_init_vit_weights)

    def get_target(self, img, masked_inds=None) -> torch.Tensor:
        B, C, H, W = img.shape
        S = self.encoder.stride
        target = img.view(B, C, H // S, S, W // S, S)
        target = target.permute([0, 2, 4, 3, 5, 1]).reshape(B, self.num_patches, -1, C)  # Bx49x(32*32)*C

        if masked_inds is not None:
            target = target[:, masked_inds]  # Bx37x(32*32)*C

        # patch normalize
        if self.normalized_pixel:
            mean = target.mean(dim=-2, keepdim=True)
            std = target.std(dim=-2, keepdim=True)
            target = (target - mean) / (std + 1e-6)

        target = target.flatten(2)  # Bx37x(32*32*C)
        return target

    def forward(self, x):
        # encode visible patches
        encoded_visible_patches_multi_stages, shuffle = self.encoder(x)  # Bx12xL
        if not self.pyramid_reconstruction:
            encoded_visible_patches = encoded_visible_patches_multi_stages[-1]
        else:
            raise NotImplementedError

        encoded_visible_patches = self.encoder_linear_proj(encoded_visible_patches)  # Bx12x decoder_dim
        B, VG, L = encoded_visible_patches.shape
        encoded_visible_patches = encoded_visible_patches.reshape(B, self.num_visible, -1, L)  # Bx12xGx decoder_dim
        G = encoded_visible_patches.size(2)  # G = grid_h * grid_w, for stage4 output, G=1

        # un-shuffle
        num_masked = self.num_patches - self.num_visible
        masked_inds = shuffle[self.num_visible:]
        mask_token = self.mask_token.expand(x.size(0), num_masked, G, -1)  # Bx37xGx decoder_dim
        all_tokens = torch.cat((encoded_visible_patches, mask_token), dim=1)  # Bx49xGx decoder_dim
        all_tokens = all_tokens[:, shuffle.argsort()]  # Bx49xGx decoder_dim

        # with positional embedding
        all_tokens = all_tokens + self.decoder_pos_embed
        all_tokens = all_tokens.reshape(B, -1, L)  # Bx(49xG)x decoder_dim

        # decode all tokens
        decoded_all_tokens = self.decoder_blocks(all_tokens)  # Bx49x decoder_dim
        decoded_all_tokens = decoded_all_tokens.view(B, self.num_patches, -1, L)  # Bx49xGx decoder_dim
        decoded_masked_tokens = decoded_all_tokens[:, masked_inds] if num_masked > 0 else decoded_all_tokens
        decoded_masked_tokens = decoded_masked_tokens.reshape(B, -1, L)  # Bx(37xG)xD
        decoded_masked_tokens = self.decoder_norm(decoded_masked_tokens)
        decoded_masked_tokens = self.decoder_linear_proj(decoded_masked_tokens)  # Bx(37xG)x(32*32*3)

        # generate target
        masked_target = self.get_target(x, masked_inds)  # Bx37x(32*32*3)

        return self.loss(decoded_masked_tokens, masked_target)

    def loss(self, img, target):
        return F.mse_loss(img, target)
