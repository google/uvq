"""Pytorch implementation of the DistortionNet for UVQ 1.5.

This module contains the DistortionNet model, which is based on a modified
EfficientNet architecture. It's used to extract features from video frames
for quality assessment.

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

import functools
import os

import torch
from torch import nn

from . import custom_nn_layers

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "checkpoints",
    "converted_models_distortion.pth",
)

default_distortionnet_batchnorm2d = functools.partial(
    nn.BatchNorm2d, eps=0.001, momentum=0
)

# Input video size
VIDEO_HEIGHT = 1080
VIDEO_WIDTH = 1920
VIDEO_CHANNELS = 3

# Input patch size
PATCH_HEIGHT = 360
PATCH_WIDTH = 640
PATCH_DEPTH = 1

# Output feature size
DIM_HEIGHT_FEATURE = 24
DIM_WIDTH_FEATURE = 24
DIM_CHANNEL_FEATURE = 128


def fix_bn(m):
  classname = m.__class__.__name__
  if classname.find("BatchNorm") != -1:
    m.eval()


class DistortionNetCore(nn.Module):
  """DistortionNet is based on the EfficientNet architecture.

  One can achieve the same network by modifying torchvision's efficientnet_b0
  model. The features layers will be the same, but the final avgpool and
  classifier layers will need to change. Instead of AdaptiveAvgPool2D will need
  to use MaxPool, and the classifier will consist of two linear layers. In
  addition if the intention is to use the baseline weights from the tensorflow
  implementation, changes must be made to support the "same" padding used in
  tensorflow convolution layers. In this implementation we have opted to not use
  the torchivision's efficientNet and instead implement the layers from scratch
  with introduction of a Conv2dSamePadding layer.
  """

  def __init__(self, dropout=0.2):
    super().__init__()
    stochastic_depth_prob_step = 0.0125
    stochastic_depth_prob = [x * stochastic_depth_prob_step for x in range(16)]
    self.features = nn.Sequential(
        custom_nn_layers.Conv2dNormActivationSamePadding(
            3,
            32,
            kernel_size=3,
            stride=2,
            activation_layer=nn.SiLU,
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            32,
            1,
            16,
            3,
            1,
            stochastic_depth_prob[0],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            16,
            6,
            24,
            3,
            2,
            stochastic_depth_prob[1],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            24,
            6,
            24,
            3,
            1,
            stochastic_depth_prob[2],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            24,
            6,
            40,
            5,
            2,
            stochastic_depth_prob[3],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            40,
            6,
            40,
            5,
            1,
            stochastic_depth_prob[4],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            40,
            6,
            80,
            3,
            2,
            stochastic_depth_prob[5],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            80,
            6,
            80,
            3,
            1,
            stochastic_depth_prob[6],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            80,
            6,
            80,
            3,
            1,
            stochastic_depth_prob[7],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            80,
            6,
            112,
            5,
            1,
            stochastic_depth_prob[8],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            112,
            6,
            112,
            5,
            1,
            stochastic_depth_prob[9],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            112,
            6,
            112,
            5,
            1,
            stochastic_depth_prob[10],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            112,
            6,
            192,
            5,
            2,
            stochastic_depth_prob[11],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            192,
            6,
            192,
            5,
            1,
            stochastic_depth_prob[12],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            192,
            6,
            192,
            5,
            1,
            stochastic_depth_prob[13],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            192,
            6,
            192,
            5,
            1,
            stochastic_depth_prob[14],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            192,
            6,
            320,
            3,
            1,
            stochastic_depth_prob[15],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.Conv2dSamePadding(
            320, DIM_CHANNEL_FEATURE, kernel_size=2, stride=1, bias=False
        ),
    )
    self.avgpool = nn.Sequential(
        nn.MaxPool2d(kernel_size=(5, 13), stride=1, padding=0),
        custom_nn_layers.PermuteLayerNHWC(),
    )

  def freeze_backbone(self):
    """Freezes the head of the model, i.e. the avgpool and classifier layers.

    This is useful for finetuning the model on a new dataset without changing
    the head.
    """
    for param in self.features.parameters():
      param.requires_grad = False

  def freeze_head(self):
    """Freezes the head of the model, i.e. the avgpool and classifier layers.

    This is useful for finetuning the model on a new dataset without changing
    the head.
    """
    for param in self.avgpool.parameters():
      param.requires_grad = False
    for param in self.classifier.parameters():
      param.requires_grad = False
    self.features.apply(fix_bn)

  def forward(self, x):
    x = self.features(x)
    features = self.avgpool(x)
    return features


class DistortionNet(nn.Module):
  """Pytorch implementation of the DistortionNet for UVQ 1.5.

  This module wraps the DistortionNetCore and handles the video patching
  and feature aggregation. It takes a video as input, splits it into patches,
  passes the patches through the DistortionNetCore, and then reshapes
  the features back into a spatial grid.
  """

  def __init__(
      self,
      model_path=MODEL_PATH,
      eval_mode=True,
      pretrained=True,
      video_height=VIDEO_HEIGHT,
      video_width=VIDEO_WIDTH,
      video_channels=VIDEO_CHANNELS,
      patch_height=PATCH_HEIGHT,
      patch_width=PATCH_WIDTH,
      depth=PATCH_DEPTH,
      feature_channels=DIM_CHANNEL_FEATURE,
      feature_height=DIM_HEIGHT_FEATURE,
      feature_width=DIM_WIDTH_FEATURE,
  ):
    super(DistortionNet, self).__init__()
    self.model = DistortionNetCore()
    if pretrained:
      self.load_state_dict(model_path)
    if eval_mode:
      self.model.eval()
    self.num_patches_x = int(video_width / patch_width)
    self.num_patches_y = int(video_height / patch_height)
    self.feature_channels = feature_channels
    self.feature_height = feature_height
    self.feature_width = feature_width
    self.patch_width = patch_width
    self.patch_height = patch_height
    self.video_width = video_width
    self.video_height = video_height
    self.video_channels = video_channels
    self.depth = depth
    self.patch_feature_height = int(feature_height / self.num_patches_y)
    self.patch_feature_width = int(feature_width / self.num_patches_x)
    self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

  def load_state_dict(self, model_path) -> nn.Module:
    model = torch.load(
        model_path, weights_only=True, map_location=torch.device("cpu")
    )
    self.model.load_state_dict(model)
    return model

  def predict_and_get_features(self, frame):
    features = self.model(frame)
    return features

  def forward(self, video):
    """Gets the predicted labels and features for all frames in a video.

    Args:
        video (np.ndarray): 5d array of shape (num_seconds, fps, channels=3,
          height=1080, width=1920) with values in [-1, 1] range

    Returns:
        feature (np.ndarray): 4d array of shape (num_seconds, 24, 24,
        channels) of features to be used in aggregation
    """
    video = video[:, 0]
    num_sec = video.shape[0]
    c, h, w = video.shape[1], video.shape[2], video.shape[3]

    # Validate patch dimensions
    assert (
        h == self.num_patches_y * self.patch_height
    ), "Height must be divisible by num_patches_y"
    assert (
        w == self.num_patches_x * self.patch_width
    ), "Width must be divisible by num_patches_x"

    # Reshape video into batched patches
    video_reshaped = video.reshape(
        num_sec,
        c,
        self.num_patches_y,
        self.patch_height,
        self.num_patches_x,
        self.patch_width,
    )
    video_reshaped = video_reshaped.permute(0, 2, 4, 1, 3, 5)
    batched_patches = video_reshaped.reshape(
        -1, c, self.patch_height, self.patch_width
    )

    # Batch prediction
    batch_features = self.predict_and_get_features(batched_patches)

    # Reshape features into spatial grid
    feature = batch_features.reshape(
        num_sec,
        self.num_patches_y,
        self.num_patches_x,
        self.patch_feature_height,
        self.patch_feature_width,
        self.feature_channels,
    )
    feature = feature.permute(0, 1, 3, 2, 4, 5)
    feature = feature.reshape(
        num_sec,
        self.num_patches_y * self.patch_feature_height,
        self.num_patches_x * self.patch_feature_width,
        self.feature_channels,
    )
    return feature
