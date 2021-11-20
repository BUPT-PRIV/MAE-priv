import torch.nn as nn
import torch

class Mix_MAE(nn.Module):

    def __init__(self, mode, alpha):
        super(Mix_MAE, self).__init__()
        mode_dict = {
            'batch': self.batch_mix,
            'pair': self.pair_mix,
        }

        self.mode = mode
        self.alpha = alpha
        self.forward = mode_dict[mode]

    def batch_mix(self, img):
        x = self.alpha * img + (1-self.alpha) * img.flip(0)  # B, 3, 224, 224
        target = torch.cat([img, img.flip(0)], dim=1)  # B, 6, 224, 224
        return x, target

    def pair_mix(self, x):
        assert x.shape[0] % 2 == 0
        num = x.shape[0] // 2
        x0, x1 = x[:num], x[num:]
        x = self.alpha * x0 + (1-self.alpha) * x1
        target = torch.cat([x0, x1], dim=1)
        return x, target
