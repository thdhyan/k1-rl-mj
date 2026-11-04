"""Frozen ResNet18 (ImageNet weights) for depth images, up to layer3, so the feature map keeps where things are in the image.

Global average pooling would lose the ball's image position, which is the thing the student needs. State dict keys match
torchvision's resnet18, so resnet18-f37072fd.pth loads directly; torchvision itself is not needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_DEPTH_M = 6.0
_MEAN, _STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class _Block(nn.Module):
  def __init__(self, inp: int, out: int, stride: int = 1):
    super().__init__()
    self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(out)
    self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(out)
    self.downsample = None
    if stride != 1 or inp != out:
      self.downsample = nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False), nn.BatchNorm2d(out))

  def forward(self, x):
    skip = x if self.downsample is None else self.downsample(x)
    return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + skip)


class FrozenDepthEncoder(nn.Module):
  """depth (n, h, w) in metres -> feature map (n, 256, h/16, w/16). Weights frozen, batch norm in eval mode."""

  def __init__(self, weights_path: str):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = nn.Sequential(_Block(64, 64), _Block(64, 64))
    self.layer2 = nn.Sequential(_Block(64, 128, 2), _Block(128, 128))
    self.layer3 = nn.Sequential(_Block(128, 256, 2), _Block(256, 256))
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    self.load_state_dict({k: v for k, v in sd.items() if not k.startswith(("layer4", "fc"))}, strict=True)
    self.register_buffer("mean", torch.tensor(_MEAN)[None, :, None, None])
    self.register_buffer("std", torch.tensor(_STD)[None, :, None, None])
    self.requires_grad_(False)
    self.eval()

  def train(self, mode: bool = True):  # stays in eval whatever the caller asks
    return super().train(False)

  @torch.no_grad()
  def forward(self, depth: torch.Tensor) -> torch.Tensor:
    # missing returns (0 or inf) read as far; nearer is brighter
    d = torch.where((depth > 0) & torch.isfinite(depth), depth, torch.full_like(depth, MAX_DEPTH_M))
    x = (1.0 - d.clamp_max(MAX_DEPTH_M) / MAX_DEPTH_M)[:, None].repeat(1, 3, 1, 1)
    x = (x - self.mean) / self.std
    x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 3, 2, 1)
    return self.layer3(self.layer2(self.layer1(x)))
