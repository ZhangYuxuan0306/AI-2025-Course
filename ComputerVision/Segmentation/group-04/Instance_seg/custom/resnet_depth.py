import torch
import torch.nn as nn
from mmdet.models.backbones.resnet import ResNet
from mmdet.registry import MODELS

@MODELS.register_module()
class ResNetDepth(ResNet):
    """双分支: RGB+Depth融合"""
    def __init__(self, depth_stem_channels=64, **kwargs):
        super().__init__(**kwargs)

        self.depth_stem = nn.Sequential(
            nn.Conv2d(1, depth_stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(depth_stem_channels),
            nn.ReLU(inplace=True)
        )
        self.depth_match_conv = nn.Conv2d(depth_stem_channels, self.stem_channels, 1) \
            if depth_stem_channels != self.stem_channels else nn.Identity()
        
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2 * self.stem_channels, self.stem_channels, 1, bias=False),
            nn.BatchNorm2d(self.stem_channels)
        )

    def forward(self, x):
        assert x.size(1) == 4, f"Expect 4-ch input, got {x.shape}"
        rgb = x[:, :3, :, :]
        depth = x[:, 3:, :, :]

        rgb = torch.nan_to_num(rgb)
        depth = torch.nan_to_num(depth)

        # RGB stem
        if self.deep_stem:
            rgb_feat = self.stem(rgb)
        else:
            rgb_feat = self.conv1(rgb)
            rgb_feat = self.norm1(rgb_feat)
            rgb_feat = self.relu(rgb_feat)

        # Depth stem + channel match
        depth_feat = self.depth_stem(depth)
        depth_feat = self.depth_match_conv(depth_feat)

        # 融合
        alpha = torch.sigmoid(self.gate_conv(torch.cat([rgb_feat, depth_feat], dim=1)))
        fused_feat = alpha * rgb_feat + (1 - alpha) * depth_feat

        fused_feat = rgb_feat + depth_feat
        fused_feat = self.maxpool(fused_feat)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            fused_feat = res_layer(fused_feat)
            if i in self.out_indices:
                outs.append(fused_feat)
        return tuple(outs)
