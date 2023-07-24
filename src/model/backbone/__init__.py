from .resnet_fpn import *
from .twins_fpn import *


def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        if config['resolution'] == (8, 2):
            return ResNetFPN_8_2(config['resnetfpn'], config['is_rgb'])
        elif config['resolution'] == (16, 4):
            return ResNetFPN_16_4(config['resnetfpn'], config['is_rgb'])
        elif config['resolution'] == (8, 4, 2):
            return ResNetFPN_8_4_2(config['resnetfpn'], config['is_rgb'])
    elif config['backbone_type'] == 'Twins':
        if config['resolution'] == (8, 4, 2):
            return TwinsFPN_8_4_2(config['resnetfpn'], config['resolution'])
        elif config['resolution'] == (16, 8, 4, 2):
            return TwinsFPN_16_8_4_2(config['resnetfpn'], config['resolution'])
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
