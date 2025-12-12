"""PyTorch implementation of the AggregationNet for UVQ 1.5.

This module contains the AggregationNetCore and AggregationNet classes,
which are used to aggregate features from different subnets (e.g., content
and distortion) in the UVQ model.

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import defaultdict

import torch
from torch import nn

NUM_CHANNELS_PER_SUBNET = 128
NUM_FILTERS = 256
CONV2D_KERNEL_SIZE = (1, 1)
MAXPOOL2D_KERNEL_SIZE = (16, 16)

DEBUG = False

BN_DEFAULT_EPS = 0.001
BN_DEFAULT_MOMENTUM = 1
DROPOUT_RATE = 0.2


def fix_bn(m):
  classname = m.__class__.__name__
  if classname.find("BatchNorm") != -1:
    m.eval()


class AggregationNetCore(nn.Module):
  """Core network for aggregating features from multiple subnets.

  This module takes features from different subnets (e.g., "content",
  "distortion"), concatenates them, and processes them through a series of
  convolutional, normalization, activation, pooling, and linear layers
  to produce a final aggregated output.

  Attributes:
    num_subnets: The number of subnets whose features are being aggregated.
    subnets: A list of subnet names.
    conv1: First convolutional layer.
    relu1: ReLU activation function.
    tanh: Tanh activation function for the final output scaling.
    dropout1: Dropout layer.
    linear1: Linear layer for the final output.
    maxpool1: Max pooling layer.
    ln1: Layer Normalization layer.
    resizer: Adaptive average pooling to resize input features.
  """

  def __init__(
      self,
      subnets: list[str],
      num_channels_per_subnet=NUM_CHANNELS_PER_SUBNET,
      num_filters=NUM_FILTERS,
      conv2d_kernel_size=CONV2D_KERNEL_SIZE,
      maxpool2d_kernel_size=MAXPOOL2D_KERNEL_SIZE,
      bn_eps=BN_DEFAULT_EPS,
      bn_momentum=BN_DEFAULT_MOMENTUM,
      dropout_rate=DROPOUT_RATE,
  ):
    super(AggregationNetCore, self).__init__()
    self.num_subnets = len(subnets)
    self.subnets = subnets
    self.conv1 = nn.Conv2d(
        self.num_subnets * num_channels_per_subnet,
        num_filters,
        kernel_size=conv2d_kernel_size,
        bias=True,
    )
    self.relu1 = nn.ReLU()
    self.tanh = nn.Tanh()
    self.dropout1 = nn.Dropout(p=dropout_rate)

    self.linear1 = nn.Linear(bias=True, in_features=num_filters, out_features=1)

    self.maxpool1 = nn.MaxPool2d(kernel_size=(4, 4))
    self.ln1 = nn.LayerNorm(normalized_shape=[256, 4, 4], eps=0.001)
    self.resizer = nn.AdaptiveAvgPool2d((4, 4))

  def get_hidden_feat(self, features: dict[str, torch.Tensor]):
    resized_features_list = [
        self.resizer(features[subnet]) for subnet in self.subnets
    ]
    x = (
        resized_features_list[0]
        if len(self.subnets) == 1
        else torch.cat(resized_features_list, dim=1)
    )
    x = self.conv1(x)
    x = self.ln1(x)
    x = self.relu1(x)
    x = self.maxpool1(x)
    x = self.dropout1(x)
    x = nn.Flatten()(x)
    return x

  def forward(self, features: dict[str, torch.Tensor]):
    resized_features_list = [
        self.resizer(features[subnet]) for subnet in self.subnets
    ]
    x = (
        resized_features_list[0]
        if len(self.subnets) == 1
        else torch.cat(resized_features_list, dim=1)
    )
    if DEBUG:
      print("\n--- AggregationNet Input ---")
      print(f"Resized features shape: {x.shape}")
      print(f"Resized features mean: {torch.mean(x).item():.4f}")
      print(f"Resized features std: {torch.std(x).item():.4f}")
    x = self.conv1(x)
    if DEBUG:
      print(
          "--- PyTorch AggregationNet after conv1 ---"
          f" Shape: {x.shape}, Mean: {torch.mean(x).item():.4f}, Std:"
          f" {torch.std(x).item():.4f}"
      )
    x = self.ln1(x)
    if DEBUG:
      print(
          "--- PyTorch AggregationNet after ln1 ---"
          f" Shape: {x.shape}, Mean: {torch.mean(x).item():.4f}, Std:"
          f" {torch.std(x).item():.4f}"
      )
    x = self.relu1(x)
    if DEBUG:
      print(
          "--- PyTorch AggregationNet after relu1 ---"
          f" Shape: {x.shape}, Mean: {torch.mean(x).item():.4f}, Std:"
          f" {torch.std(x).item():.4f}"
      )
    x = self.maxpool1(x)
    if DEBUG:
      print(
          "--- PyTorch AggregationNet after maxpool1 ---"
          f" Shape: {x.shape}, Mean: {torch.mean(x).item():.4f}, Std:"
          f" {torch.std(x).item():.4f}"
      )
    x = self.dropout1(x)
    if DEBUG:
      print(
          "--- PyTorch AggregationNet after dropout1 ---"
          f" Shape: {x.shape}, Mean: {torch.mean(x).item():.4f}, Std:"
          f" {torch.std(x).item():.4f}"
      )
    x = nn.Flatten()(x)
    x = self.linear1(x)
    if DEBUG:
      print(
          "--- PyTorch AggregationNet after linear1 ---"
          f" Shape: {x.shape}, Mean: {torch.mean(x).item():.4f}, Std:"
          f" {torch.std(x).item():.4f}"
      )
    x = self.tanh(x) * 2 + 3  # Scale to [1, 5]
    return x


class AggregationNet(nn.Module):
  """Aggregates content and distortion features using an AggregationNetCore.

  This module wraps the AggregationNetCore to process content and distortion
  features. It handles feature permutation, model loading, and setting
  the model to evaluation mode if required.

  Attributes:
    model: An instance of AggregationNetCore.
  """

  def __init__(
      self,
      model_path=None,
      pretrained=True,
      eval_mode=True,
  ):
    super(AggregationNet, self).__init__()
    self.model = AggregationNetCore(
        ["content", "distortion"],
    )

    if pretrained:
      self.load_state_dict(model_path)
    if eval_mode:
      self.model.eval()
      self.freeze_model(self.model)

  def freeze_model(self, model: nn.Module):
    for p in model.parameters():
      p.requires_grad = False

  def get_hidden_features(
      self,
      content_features,
      distortion_features,
  ):
    feature_results = defaultdict(list)
    r = self.model.get_hidden_feat({
        "content": content_features.permute(0, 3, 1, 2),
        "distortion": distortion_features.permute(0, 3, 1, 2),
    })
    feature_results["uvq_1p5_features"].append(r)
    feature_results = {
        feature: torch.mean(torch.stack(results), dim=0)
        for feature, results in feature_results.items()
    }
    return feature_results

  def forward(
      self,
      content_features,
      distortion_features,
  ):
    feature_results = defaultdict(list)
    r = self.model({
        "content": content_features.permute(0, 3, 1, 2),
        "distortion": distortion_features.permute(0, 3, 1, 2),
    })
    feature_results["uvq_1p5_features"].append(r)
    feature_results = {
        feature: torch.mean(torch.stack(results), dim=0)
        for feature, results in feature_results.items()
    }
    return feature_results

  def load_state_dict(self, model_path) -> nn.Module:
    model = torch.load(
        model_path, weights_only=True, map_location=torch.device("cpu")
    )
    self.model.load_state_dict(model)
    return model
