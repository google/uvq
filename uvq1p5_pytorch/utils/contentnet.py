"""Pytorch implementation of the ContentNet used in UVQ 1.5.

This module contains the ContentNetCore and ContentNet classes.
ContentNetCore is a feature extractor based on an EfficientNet-B0 like
architecture. ContentNet wraps the core model, handles loading pretrained
weights, and provides a forward pass for video inputs.

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

import os

import torch
from torch import nn
import torch.nn.functional as F

from . import custom_nn_layers

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "content_net.pth"
)


def fix_bn(m):
  classname = m.__class__.__name__
  if classname.find("BatchNorm") != -1:
    m.eval()


class ContentNetCore(nn.Module):
  """ContentNet is based on the EfficientNet architecture.

  One can achieve the same network by modifying torchvision's efficientnet_b0
  model. The features layers will be the same, but the final classifier layers
  will need to change. In addition if the intention is to use the baseline
  weights from the tensorflow implementation, changes must be made to support
  the "same" padding used in tensorflow convolution layers. In this
  implementation we have opted to not use the torchivision's efficientNet and
  instead implement the layers from scratch with introduction of a
  Conv2dSamePadding layer.
  """

  def __init__(self, dropout=0.2):
    super().__init__()
    stochastic_depth_prob_step = 0.0
    stochastic_depth_prob = [x * stochastic_depth_prob_step for x in range(16)]
    self.features = nn.Sequential(
        custom_nn_layers.Conv2dNormActivationSamePadding(
            3, 32, kernel_size=3, stride=2, activation_layer=nn.SiLU
        ),
        custom_nn_layers.MBConvSamePadding(
            32, 1, 16, 3, 1, stochastic_depth_prob[0]
        ),
        custom_nn_layers.MBConvSamePadding(
            16, 6, 24, 3, 2, stochastic_depth_prob[1]
        ),
        custom_nn_layers.MBConvSamePadding(
            24, 6, 24, 3, 1, stochastic_depth_prob[2]
        ),
        custom_nn_layers.MBConvSamePadding(
            24, 6, 40, 5, 2, stochastic_depth_prob[3]
        ),
        custom_nn_layers.MBConvSamePadding(
            40, 6, 40, 5, 1, stochastic_depth_prob[4]
        ),
        custom_nn_layers.MBConvSamePadding(
            40, 6, 80, 3, 2, stochastic_depth_prob[5]
        ),
        custom_nn_layers.MBConvSamePadding(
            80, 6, 80, 3, 1, stochastic_depth_prob[6]
        ),
        custom_nn_layers.MBConvSamePadding(
            80, 6, 80, 3, 1, stochastic_depth_prob[7]
        ),
        custom_nn_layers.MBConvSamePadding(
            80, 6, 112, 5, 1, stochastic_depth_prob[8]
        ),
        custom_nn_layers.MBConvSamePadding(
            112, 6, 112, 5, 1, stochastic_depth_prob[9]
        ),
        custom_nn_layers.MBConvSamePadding(
            112, 6, 112, 5, 1, stochastic_depth_prob[10]
        ),
        custom_nn_layers.MBConvSamePadding(
            112, 6, 192, 5, 2, stochastic_depth_prob[11]
        ),
        custom_nn_layers.MBConvSamePadding(
            192, 6, 192, 5, 1, stochastic_depth_prob[12]
        ),
        custom_nn_layers.MBConvSamePadding(
            192, 6, 192, 5, 1, stochastic_depth_prob[13]
        ),
        custom_nn_layers.MBConvSamePadding(
            192, 6, 192, 5, 1, stochastic_depth_prob[14]
        ),
        custom_nn_layers.MBConvSamePadding(
            192, 6, 320, 3, 1, stochastic_depth_prob[15]
        ),
        custom_nn_layers.Conv2dSamePadding(320, 128, kernel_size=2, stride=1),
    )
    self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

  def freeze_backbone(self):
    """Freezes the head of the model, i.e. the avgpool and classifier layers.

    This is useful for finetuning the model on a new dataset without changing
    the head.
    """
    for param in self.features.parameters():
      param.requires_grad = False

  def freeze_head(self):
    """Freezes the head of the model, i.e. the classifier layers.

    This is useful for finetuning the model on a new dataset.
    """
    for param in self.classifier.parameters():
      param.requires_grad = False
    self.features.apply(fix_bn)

  def forward(self, x):
    x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    return self.features(x)


class ContentNet(nn.Module):
  """ContentNet module for extracting features from video frames.

  This module wraps the ContentNetCore and handles loading pretrained weights,
  setting evaluation mode, and providing a forward pass for video inputs.
  """

  def __init__(
      self,
      model_path=MODEL_PATH,
      eval_mode=True,
      pretrained=True,
  ):
    super().__init__()
    self.model = ContentNetCore()
    if pretrained:
      self.load_state_dict(model_path)
    if eval_mode:
      self.model.eval()
    self.features_transpose = (0, 2, 3, 1)

  def load_state_dict(self, model_path) -> torch.nn.Module:
    model = torch.load(
        model_path, weights_only=True, map_location=torch.device("cpu")
    )
    self.model.load_state_dict(model)
    return model

  def predict_and_get_features(self, frame):
    features = self.model(frame)
    return features.permute(*self.features_transpose)

  def forward(self, video):
    """Gets the predicted labels and features for all frames(seconds) in a video.

    Args:
        video (np.ndarray): 5d array of shape (num_seconds, fps, channels=3,
          height=1080, width=1920) with values in [-1, 1] range that will be
          resized to 256x256 before being processed by the model.

    Returns:
        feature (np.ndarray): 4d array of shape (num_seconds, 8, 8,
        channels) of features to be used in aggregation

    Note that even thought the input video can have any fps, computation is
    performed only on the first frame of each second.
    """
    input_video = video[:, 0]
    feature = self.predict_and_get_features(input_video)
    return feature
