import os.path

from .gvt import *


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def torch_init_model(model, total_dict, key, rank=0):
    if key in total_dict:
        state_dict = total_dict[key]
    else:
        state_dict = total_dict
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict=state_dict, prefix=prefix, local_metadata=local_metadata, strict=True,
                                     missing_keys=missing_keys, unexpected_keys=unexpected_keys, error_msgs=error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    if rank == 0:
        print("missing keys:{}".format(missing_keys))
        print('unexpected keys:{}'.format(unexpected_keys))
        print('error msgs:{}'.format(error_msgs))


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if in_planes != planes:
            self.shortcut = nn.Sequential(conv1x1(in_planes, planes), nn.BatchNorm2d(planes))
        else:
            self.shortcut = None

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        if self.shortcut is None:
            return self.relu(x + y)
        else:
            return self.relu(self.shortcut(x) + y)


class TwinsFPN_8_4_2(nn.Module):
    def __init__(self, config, resolution=(8, 4, 2)):
        super().__init__()
        # Config
        self.resolution = resolution
        model_type = config['model_type']
        if model_type == 'small':
            self.vit = alt_gvt_small_first2_layers()
        elif model_type == 'base':
            self.vit = alt_gvt_base_first2_layers()
        elif model_type == 'large':
            self.vit = alt_gvt_large_first2_layers()
        block_dims = config['block_dims']
        embed_dims = self.vit.embed_dims  # [128, 256]

        if os.path.exists(config['vit_path']):
            state_dict = torch.load(config['vit_path'], map_location='cpu')
            # torch_init_model(self.vit, state_dict, key='none')
            self.vit.load_state_dict(state_dict, strict=False)
        else:
            print(f'WARNING: {config["vit_path"]} is not existed, please ignore it if you are testing.')

        # 1/2 Encoder layers
        self.conv1 = nn.Sequential(nn.Conv2d(3, block_dims[0] // 2, kernel_size=7, stride=2, padding=3, bias=False),
                                   nn.BatchNorm2d(block_dims[0] // 2), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, block_dims[0] // 2, block_dims[0], stride=1)  # 1/2

        # FPN upsample
        self.layer3_outconv = nn.Sequential(conv1x1(embed_dims[1], block_dims[2]), nn.BatchNorm2d(block_dims[2]))

        self.layer2_outconv = nn.Sequential(conv1x1(embed_dims[0], block_dims[2]), nn.BatchNorm2d(block_dims[2]))
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
        )

        self.layer1_outconv = nn.Sequential(conv1x1(block_dims[0], block_dims[1]), nn.BatchNorm2d(block_dims[1]))
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
            nn.BatchNorm2d(block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inp_dim, dim, stride=1):
        layer1 = block(inp_dim, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        return nn.Sequential(*layers)

    def forward(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=x.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=x.device).reshape(1, 3, 1, 1)
        x = (x - mean) / std
        # ResNet Backbone
        x1 = self.layer1(self.conv1(x))  # [1/2]

        # VIT Backbone
        [x2, x3] = self.vit.forward_features(x)  # [1/4, 1/8]

        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)  # [1/4,C=196]

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out + x2_out_2x)

        if self.resolution == (8, 4, 2):
            return [x3_out, x2_out, x1_out]
        else:
            return [x3_out, x1_out]


class TwinsFPN_16_8_4_2(nn.Module):
    def __init__(self, config, resolution=(16, 8, 4, 2)):
        super().__init__()
        # Config
        self.resolution = resolution
        model_type = config['model_type']
        if model_type == 'small':
            self.vit = alt_gvt_small_first3_layers()
        elif model_type == 'base':
            self.vit = alt_gvt_base_first3_layers()
        elif model_type == 'large':
            self.vit = alt_gvt_large_first3_layers()
        block_dims = config['block_dims']
        embed_dims = self.vit.embed_dims  # [96, 192, 384]

        state_dict = torch.load(config['vit_path'], map_location='cpu')
        # torch_init_model(self.vit, state_dict, key='none')
        self.vit.load_state_dict(state_dict, strict=False)

        # 1/2 Encoder layers
        self.conv1 = nn.Sequential(nn.Conv2d(3, block_dims[0] // 2, kernel_size=7, stride=2, padding=3, bias=False),
                                   nn.BatchNorm2d(block_dims[0] // 2), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, block_dims[0] // 2, block_dims[0], stride=1)  # 1/2

        # FPN upsample
        self.layer4_outconv = nn.Sequential(conv1x1(embed_dims[2], block_dims[3]), nn.BatchNorm2d(block_dims[3]))

        self.layer3_outconv = nn.Sequential(conv1x1(embed_dims[1], block_dims[3]), nn.BatchNorm2d(block_dims[3]))
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[3], block_dims[3]),
            nn.BatchNorm2d(block_dims[3]),
            nn.LeakyReLU(),
            conv3x3(block_dims[3], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
        )

        self.layer2_outconv = nn.Sequential(conv1x1(embed_dims[0], block_dims[2]), nn.BatchNorm2d(block_dims[2]))
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
        )

        self.layer1_outconv = nn.Sequential(conv1x1(block_dims[0], block_dims[1]), nn.BatchNorm2d(block_dims[1]))
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
            nn.BatchNorm2d(block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inp_dim, dim, stride=1):
        layer1 = block(inp_dim, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        return nn.Sequential(*layers)

    def forward(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=x.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=x.device).reshape(1, 3, 1, 1)
        x = (x - mean) / std
        # ResNet Backbone
        x1 = self.layer1(self.conv1(x))  # [1/2]

        # VIT Backbone
        [x2, x3, x4] = self.vit.forward_features(x)  # [1/4, 1/8, 1/16]

        # FPN
        x4_out = self.layer4_outconv(x4)  # [1/16]

        x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)
        x3_out = self.layer3_outconv(x3)
        x3_out = self.layer3_outconv2(x3_out + x4_out_2x)  # [1/8]

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)  # [1/4]

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out + x2_out_2x)

        return [x4_out, x3_out, x2_out, x1_out]
