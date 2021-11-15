import torch
import vits
import mae.builder
from functools import partial


model = mae.builder.MAE(partial(vits.__dict__["vit_base"], mask_ratio=0.75),
                        image_size=224, decoder_dim=512, decoder_depth=8)

print(model)


x = torch.ones([1, 3, 224, 224])

y = model(x)

print(y)




