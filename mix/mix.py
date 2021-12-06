import torch.nn as nn
import torch

class Mix_MAE(nn.Module):

    def __init__(self, mode, alpha, num=2):
        super(Mix_MAE, self).__init__()
        mode_dict = {
            'base': self.base_mix,
            'decoder': self.decoder_mix,
        }
        self.mode = mode
        self.alpha = alpha
        self.forward = mode_dict[mode]
        # self.mix = nn.Conv2d(3*num, 3, 1)

    @torch.no_grad()
    def base_mix(self, img):
        B = img.shape[0]
        mix_ind = torch.randperm(B)
        mix_img = img.clone()[mix_ind]
        x = self.alpha * img + (1 - self.alpha) * mix_img  # B, 3, 224, 224
        target = torch.cat([img, mix_img], dim=1)  # B, 6, 224, 224
        return x, target

    def decoder_mix(self, img):
        B = img.shape[0]
        mix_ind = torch.randperm(B)
        mix_img = img.clone()[mix_ind]
        # x = self.alpha * img + (1 - self.alpha) * mix_img  # B, 3, 224, 224
        target = torch.cat([img, mix_img], dim=1)  # B, 6, 224, 224
        # x = self.mix(target)
        return target, mix_ind