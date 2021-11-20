import math
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .vision_transformer import Block
from .pyramid_reconstruction_image_transformer import PriTEncoder, _init_vit_weights


class MAE(nn.Module):
    """
    Build a MAE model with a encoder and encoder
    https://arxiv.org/abs/2111.06377
    """

    def __init__(self, encoder, image_size=224, decoder_dim=512, decoder_depth=8, normalized_pixel=False):
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
        target = x.view(
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


class PriT(nn.Module):
    """
    Build a PyramidReconstructionImageTransformer (PriT) model with a encoder and encoder
    """

    def __init__(self, encoder: PriTEncoder, decoder_dim=512, decoder_depth=8, normalized_pixel=False):
        self.encoder = encoder
        self.normalized_pixel = normalized_pixel

        patch_size = encoder.patch_embed.patch_size
        norm_layer = encoder.norm_layer
        self.stride = encoder.stride
        self.num_patches = encoder.num_patches
        self.num_visible = encoder.visible_num

        # build encoder linear projection
        self.encoder_linear_proj = nn.Linear(encoder.num_features, decoder_dim)

        # build decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = encoder.build_2d_sincos_position_embedding(embed_dim=decoder_dim, decode=True)
        self.decoder_blocks = encoder._build_blocks(decoder_dim, decoder_depth)
        self.decoder_norm = norm_layer(decoder_dim, eps=1e-6)
        self.decoder_linear_proj = nn.Linear(decoder_dim, patch_size[0] * patch_size[1] * 3)

        # weight initialization for decoder, ViT (timm)
        self.init_weights()

        # weight initialization, MOCO v3
        for name, m in self.decoder.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def init_weights(self):
        self.apply(_init_vit_weights)

    def get_target(self, img):
        B, C, H, W = img
        target = img.view([B, C, H // self.stride, self.stride, W // self.stride, self.stride])
        target = target.permute([0, 2, 4, 3, 5, 1]).reshape(B, self.num_patches, -1, C)  # Bx49x(56*56)*C
        return target

    def forward(self, x):
        # encode visible patches
        encoded_visible_patches, shuffle = self.encoder(x)  # Bx12xL
        encoded_visible_patches = self.encoder_linear_proj(encoded_visible_patches)  # Bx12x decoder_dim

        # un-shuffle
        num_masked = self.num_patches - self.num_visible
        masked_inds = shuffle[self.num_visible:]
        mask_token = self.mask_token.repeat([x.size(0), num_masked, 1])
        all_tokens = torch.cat((encoded_visible_patches, mask_token), dim=1)  # Bx12x decoder_dim + Bx37x decoder_dim --> Bx49x decoder_dim
        all_tokens = all_tokens[:, shuffle.argsort()]

        # with positional embedding
        all_tokens = all_tokens + self.decoder_pos_embed

        # decode all tokens
        decoded_all_token = self.decoder_blocks(all_tokens)  # Bx49x decoder_dim
        decoded_masked_token = decoded_all_token[:, masked_inds] if num_masked > 0 else decoded_all_token
        decoded_masked_token = self.decoder_norm(decoded_masked_token)
        decoded_masked_token = self.decoder_linear_proj(decoded_masked_token)  # Bx37x(56*56*3)

        # generate target
        target: torch.Tensor = self.get_target(x)  # Bx49x(56*56)*3
        masked_target = target[:, masked_inds] if num_masked > 0 else target
        # normalize target
        if self.normalized_pixel:
            mean = masked_target.mean(dim=-2, keepdim=True)
            std = masked_target.std(dim=-2, keepdim=True)
            masked_target = (masked_target - mean) / (std + 1e-6)
        masked_target = masked_target.flatten(2)  # Bx49x(56*56*3)

        return self.loss(decoded_masked_token, masked_target)

    def loss(self, img, target):
        return F.mse_loss(img, target)
