from mixup import Mixup
from cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from model_ema import ModelEma
from lars import LARS
from auto_augment import augment_and_mix_transform, rand_augment_transform
from layer_decay import LayerDecayValueAssigner
from optim_factory import create_optimizer
from log import setup_default_logging