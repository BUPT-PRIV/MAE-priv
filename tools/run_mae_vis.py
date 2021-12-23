import _init_paths
from mae import modeling_pretrain

import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
import yaml
from PIL import Image
from einops import rearrange

from utils import create_model
from utils.data_constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils.datasets import DataAugmentationForMAE


def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser('Visualization of restruction', add_help=False)

    parser.add_argument('--img_path', type=str, help='input image path')
    parser.add_argument('--save_path', type=str, help='save file path')
    parser.add_argument('--model_path', type=str, help='checkpoint path of model')
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--decoder_dim', default=512, type=int,
                        help='images input size for backbone')
    parser.add_argument('--decoder_depth', default=8, type=int,
                        help='images input size for backbone')
    parser.add_argument('--decoder_num_heads', default=None, type=int,
                        help='images input size for backbone')

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    return parser.parse_args(remaining)


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        normalized_pixel=args.normlize_target,
        use_cls_token=args.use_cls_token,
        mask_ratio=args.mask_ratio,
    )
    return model


def main(args):
    print(args)
    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    with open(args.img_path, 'rb') as f:
        img = Image.open(f)
        img.convert('RGB')
        print("img path:", args.img_path)

    transforms = DataAugmentationForMAE(args)
    img = transforms(img)

    with torch.no_grad():
        img = img[None, :]
        # bool_masked_pos = bool_masked_pos[None, :]
        img = img.to(device, non_blocking=True)
        # bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        outputs, shuffle, visible_size = model(img, restruct=True)
        mask_idxs = shuffle[visible_size:]
        # mask_patches = outputs[0, mask_idxs].reshape([-1, patch_size[0] * patch_size[1], 3]) # 147 16*16 3

        # save original img
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
        ori_img = img * std + mean  # in [0, 1]
        ori_img = ori_img[0]
        img = T.ToPILImage()(ori_img)
        img.save(f"{args.save_path}/ori_img.jpg")

        img_patches = rearrange(ori_img, 'c (h p1) (w p2) -> (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[1])

        # mask img
        mask_patches = img_patches.clone()
        mask_patches[mask_idxs] = 0.0
        mask_img = rearrange(mask_patches, '(h w) (p1 p2) c -> c (h p1) (w p2)', h=args.input_size // patch_size[0],
                             p1=patch_size[0])
        mask_img = T.ToPILImage()(mask_img)
        mask_img.save(f"{args.save_path}/mask_img.jpg")

        img_mean = torch.mean(img_patches, dim=-2, keepdim=True)
        img_var = torch.std(img_patches, dim=-2, keepdim=True)
        re_img = outputs[0].reshape([-1, patch_size[0] * patch_size[1], 3]) * (img_var.sqrt() + 1e-6) + img_mean

        # img_patches[mask_idxs] = mask_img
        re_img = rearrange(re_img, '(h w) (p1 p2) c -> c (h p1) (w p2)', h=args.input_size // patch_size[0],
                           p1=patch_size[0])
        re_img = T.ToPILImage()(re_img)
        re_img.save(f"{args.save_path}/re_img.jpg")


if __name__ == '__main__':
    opts = get_args()
    main(opts)
