"""
reid/osnet_fallback.py

Self-contained, dependency-free OSNet_x1_0 architecture.

This exists ONLY as a fallback for environments where `torchreid` cannot be
installed. It mirrors the Omni-Scale Network design (multi-stream, multiple
receptive fields fused by a channel-gating "Unified Aggregation Gate") used
by the official OSNet_x1_0. Prefer installing torchreid and using the real
model definition (see reid_model.py) so official pretrained checkpoints load
cleanly with matching layer names -- this fallback is a best-effort clone for
when that isn't possible, and pretrained weights should be loaded with
strict=False against it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, k, stride=1, padding=0, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LightConv3x3(nn.Module):
    """Depthwise-separable 3x3 conv, the building block of each OS stream."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.relu(self.bn(x))


class ChannelGate(nn.Module):
    """Unified Aggregation Gate: learns per-channel weights for each scale stream."""

    def __init__(self, in_c, reduction=16):
        super().__init__()
        mid = max(in_c // reduction, 8)
        self.fc1 = nn.Conv2d(in_c, mid, 1)
        self.fc2 = nn.Conv2d(mid, in_c, 1)

    def forward(self, x):
        gate = F.adaptive_avg_pool2d(x, 1)
        gate = F.relu(self.fc1(gate), inplace=True)
        gate = torch.sigmoid(self.fc2(gate))
        return x * gate


class OSBlock(nn.Module):
    """Omni-Scale residual block: fuses streams of depth 1..T via the gate."""

    def __init__(self, in_c, out_c, T=4, bottleneck_reduction=4):
        super().__init__()
        mid_c = out_c // bottleneck_reduction
        self.conv1 = ConvLayer(in_c, mid_c, 1)

        self.streams = nn.ModuleList()
        for t in range(1, T + 1):
            layers = [LightConv3x3(mid_c, mid_c) for _ in range(t)]
            self.streams.append(nn.Sequential(*layers))

        self.gate = ChannelGate(mid_c)
        self.conv3 = ConvLayer(mid_c, out_c, 1, act=False)
        self.downsample = None
        if in_c != out_c:
            self.downsample = ConvLayer(in_c, out_c, 1, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        fused = sum(self.gate(stream(x1)) for stream in self.streams)
        out = self.conv3(fused)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class OSNet(nn.Module):
    def __init__(self, num_classes=1000, channels=(64, 256, 384, 512), feature_dim=512):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.conv1 = ConvLayer(3, c1, 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            OSBlock(c1, c2), OSBlock(c2, c2), nn.AvgPool2d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            OSBlock(c2, c3), OSBlock(c3, c3), nn.AvgPool2d(2, stride=2)
        )
        self.conv4 = nn.Sequential(OSBlock(c3, c4), OSBlock(c4, c4))
        self.conv5 = ConvLayer(c4, feature_dim, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_pool(f).flatten(1)
        if self.training:
            return self.fc(v)
        return v  # inference: return the embedding directly


def osnet_x1_0(num_classes=1000, **kwargs):
    return OSNet(num_classes=num_classes, channels=(64, 256, 384, 512), feature_dim=512)
