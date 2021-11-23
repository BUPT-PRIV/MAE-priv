# --------------------------------------------------------
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from .checkpoint import *
from .cross_entropy import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from .data_constants import *
from .deepspeed_cfg import *
from .dist import *
from .logger import *
from .metrics import accuracy, AverageMeter
from .mixup import Mixup, FastCollateMixup
from .model import unwrap_model, get_state_dict, freeze, unfreeze
from .model_ema import ModelEma, ModelEmaV2
from .native_scaler import *
from .optim_factory import create_optimizer
from .transforms import *
from .registry import register_model, model_entrypoint
from .model_builder import create_model
from .transforms_factory import create_transform