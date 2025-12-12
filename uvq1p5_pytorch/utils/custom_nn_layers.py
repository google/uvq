"""Custom neural network layers used in the UVQ1P5 PyTorch implementation.

This module contains custom layers such as 'same' padding convolutions
(2D and 3D), a custom Interpolate layer, and various Inception-style
blocks adapted for 3D inputs, often emulating TensorFlow's padding behavior.

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

from functools import partial
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation
from torchvision.ops import StochasticDepth
from torchvision.ops.misc import ConvNormActivation


contentnet_default_batchnorm2d = partial(
    nn.BatchNorm2d, eps=0.001, momentum=0.99
)


class Interpolate(nn.Module):
  """Custom layer to imitate the BilinearResize layer from tensorflow
  using the torch.nn.functional.interpolate
  """

  def __init__(
      self, size=None, scale_factor=None, mode="bilinear", align_corners=False
  ):
    super(Interpolate, self).__init__()
    self.size = size
    self.scale_factor = scale_factor
    self.mode = mode
    self.align_corners = align_corners

  def forward(self, x):
    x = F.interpolate(
        x,
        size=self.size,
        scale_factor=self.scale_factor,
        mode=self.mode,
        align_corners=self.align_corners,
    )
    return x


class Conv2dSamePadding(nn.Module):
  """2D Convolution layer with 'same' padding.

  This class emulates the "same" padding behavior of TensorFlow's Conv2D.
  It calculates the necessary padding to ensure that the output tensor
  has a spatial size equal to the input size divided by the stride.
  The padding is applied before the standard `nn.Conv2d` operation.
  """

  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      stride,
      padding=None,
      dilation=1,
      groups=1,
      bias=False,
  ):
    """Custom Conv2D layer to immitate the padding="same" functionality
    of the tensorflow convolutions.
    """
    super(Conv2dSamePadding, self).__init__()
    self.conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )
    self.stride = stride if isinstance(stride, tuple) else (stride, stride)
    self.kernel_size = (
        kernel_size
        if isinstance(kernel_size, tuple)
        else (kernel_size, kernel_size)
    )
    self.groups = groups

  def forward(self, x):
    ih, iw = x.size()[2], x.size()[3]
    kh, kw = self.kernel_size[0], self.kernel_size[1]
    sh, sw = (
        self.stride
        if isinstance(self.stride, tuple)
        else (self.stride, self.stride)
    )

    ph = max((ih - 1) // sh * sh + kh - ih, 0)
    pw = max((iw - 1) // sw * sw + kw - iw, 0)

    pad_top = ph // 2
    pad_bottom = ph - pad_top
    pad_left = pw // 2
    pad_right = pw - pad_left

    x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    return self.conv(x_padded)


class Conv2dNormActivationSamePadding(ConvNormActivation):
  """A Convolution-Normalisation-Activation layer with 'same' padding.

  This class extends `torchvision.ops.misc.ConvNormActivation` to use
  `Conv2dSamePadding` instead of the standard `nn.Conv2d`, providing
  "same" padding behavior similar to TensorFlow.
  """

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Tuple[int, int]] = 3,
      stride: Union[int, Tuple[int, int]] = 1,
      padding: Optional[Union[int, Tuple[int, int], str]] = None,
      groups: int = 1,
      norm_layer: Optional[
          Callable[..., torch.nn.Module]
      ] = contentnet_default_batchnorm2d,
      activation_layer: Optional[
          Callable[..., torch.nn.Module]
      ] = torch.nn.ReLU,
      dilation: Union[int, Tuple[int, int]] = 1,
      inplace: Optional[bool] = True,
      bias: Optional[bool] = None,
  ) -> None:
    super().__init__(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups,
        norm_layer,
        activation_layer,
        dilation,
        inplace,
        bias,
        Conv2dSamePadding,
    )


class MBConvSamePadding(nn.Module):
  """MobileNetV2/EfficientNet style inverted residual block with 'same' padding.

  This block uses `Conv2dNormActivationSamePadding` for all convolutions
  to ensure "same" padding behavior. It includes an expansion convolution,
  a depthwise convolution, a Squeeze-Excitation layer, and a projection
  convolution. A stochastic depth is applied if `use_res_connect` is True.
  """

  def __init__(
      self,
      input_channels: int,
      expand_ratio: int,
      out_channels: int,
      kernel: int,
      stride: int,
      stochastic_depth_prob: float,
      norm_layer: Callable[..., nn.Module] = contentnet_default_batchnorm2d,
      se_layer: Callable[..., nn.Module] = SqueezeExcitation,
  ) -> None:
    super().__init__()

    if not (1 <= stride <= 2):
      raise ValueError("illegal stride value")

    self.use_res_connect = stride == 1 and input_channels == out_channels

    layers: List[nn.Module] = []
    activation_layer = nn.SiLU

    # expand
    expanded_channels = input_channels * expand_ratio
    if expanded_channels != input_channels:
      layers.append(
          Conv2dNormActivationSamePadding(
              input_channels,
              expanded_channels,
              kernel_size=1,
              stride=1,
              norm_layer=norm_layer,
              activation_layer=activation_layer,
          )
      )

    # depthwise
    layers.append(
        Conv2dNormActivationSamePadding(
            expanded_channels,
            expanded_channels,
            kernel_size=kernel,
            stride=stride,
            groups=expanded_channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
    )

    # squeeze and excitation
    squeeze_channels = max(1, input_channels // 4)
    layers.append(
        se_layer(
            expanded_channels,
            squeeze_channels,
            activation=partial(nn.SiLU, inplace=True),
        )
    )

    # project
    layers.append(
        Conv2dNormActivationSamePadding(
            expanded_channels,
            out_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=None,
        )
    )

    self.block = nn.Sequential(*layers)
    self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
    self.out_channels = out_channels

  def forward(self, input):
    result = self.block(input)
    if self.use_res_connect:
      result = self.stochastic_depth(result)
      result += input
    return result


class PermuteLayerNHWC(nn.Module):

  def __init__(self, permutation=(0, 2, 3, 1)):
    super(PermuteLayerNHWC, self).__init__()
    self.dims = permutation

  def forward(self, x):
    x = x.to_dense()
    return x.permute(*self.dims)


class Conv3DSamePadding(nn.Module):
  """3D Convolution layer with 'same' padding.

  This class emulates the "same" padding behavior of TensorFlow's Conv3D,
  where the output size is the input size divided by the stride, by
  calculating and applying appropriate padding before the convolution.
  """

  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      stride=1,
      dilation=1,
      groups=1,
      bias=True,
  ):
    super(Conv3DSamePadding, self).__init__()
    self.conv = nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        0,
        dilation,
        groups,
        bias,
    )
    self.stride = self.conv.stride
    self.kernel_size = self.conv.kernel_size
    self.dilation = self.conv.dilation

  def forward(self, x):
    input_size = x.size()[2:]
    effective_filter_shape = [
        (self.kernel_size[i] - 1) * self.dilation[i] + 1
        for i in range(len(input_size))
    ]
    padding = [
        max(
            (math.ceil(input_size[i] / self.stride[i]) - 1) * self.stride[i]
            + effective_filter_shape[i]
            - input_size[i],
            0,
        )
        for i in range(len(input_size))
    ]
    for i in range(len(input_size)):
      if input_size[i] % self.stride[i] != 0:
        padding[i] += 1
    padding = padding[::-1]
    padding = [(pad // 2, pad - pad // 2) for pad in padding]
    padding = tuple([val for sublist in padding for val in sublist])
    x = nn.functional.pad(x, padding)

    x = self.conv(x)
    return x


class MaxPool3dSame(nn.Module):
  """3D Max Pooling layer with 'same' padding.

  This class emulates the "same" padding behavior for MaxPool3d,
  calculating and applying appropriate padding to the input tensor
  before performing the max pooling operation, such that the output
  size is the input size divided by the stride.
  """

  def __init__(self, kernel_size, stride):
    super(MaxPool3dSame, self).__init__()
    self.kernel_size = kernel_size
    self.stride = stride

  def forward(self, x):
    input_size = x.size()[2:]

    padding = [
        max(
            (math.ceil(input_size[i] / self.stride[i]) - 1) * self.stride[i]
            + self.kernel_size[i]
            - input_size[i],
            0,
        )
        for i in range(len(input_size))
    ]

    for i in range(len(input_size)):
      if input_size[i] % self.stride[i] != 0:
        padding[i] += 1

    padding = padding[::-1]
    padding = [(pad // 2, pad - pad // 2) for pad in padding]
    padding = tuple([val for sublist in padding for val in sublist])

    x = nn.functional.pad(x, padding)

    return nn.functional.max_pool3d(
        x, kernel_size=self.kernel_size, stride=self.stride, padding=0
    )


class InceptionB0B3(nn.Module):
  """A block used in an Inception-like 3D CNN architecture.

  This block consists of an optional initial MaxPool3dSame, followed by a
  1x1x1 convolution, BatchNorm, ReLU, global average pooling, another 1x1x1
  convolution, a sigmoid activation, and finally, channel-wise re-weighting
  of the features after the first ReLU. This structure is reminiscent of
  Squeeze-and-Excitation, applied in a 3D context.
  """

  def __init__(self, conv1_in, conv1_out, pool_shape, include_maxpool=False):
    super(InceptionB0B3, self).__init__()
    self.include_maxpool = include_maxpool
    self.maxpool = (
        MaxPool3dSame(kernel_size=(3, 3, 3), stride=(1, 1, 1))
        if include_maxpool
        else None
    )
    self.conv3d_1 = Conv3DSamePadding(
        conv1_in, conv1_out, kernel_size=1, stride=1, bias=False
    )
    self.batch_norm = nn.BatchNorm3d(conv1_out)
    self.relu = nn.ReLU()
    self.avgpool = nn.AvgPool3d(kernel_size=pool_shape, stride=1)
    self.conv3d_2 = Conv3DSamePadding(
        conv1_out, conv1_out, kernel_size=1, stride=1, bias=False
    )
    self.sigmoid = nn.Sigmoid()
    self.tile_shape = pool_shape

  def forward(self, x):
    if self.include_maxpool:
      x = self.maxpool(x)
    x = self.conv3d_1(x)
    x = self.batch_norm(x)
    relu_out = self.relu(x)
    x = self.avgpool(relu_out)
    x = self.conv3d_2(x)
    x = x.repeat(
        1, 1, self.tile_shape[0], self.tile_shape[1], self.tile_shape[2]
    )
    x = self.sigmoid(x)
    x = x * relu_out
    return x


class InceptionB1B2(nn.Module):
  """A branch block for a 3D Inception-like architecture.

  This block processes the input through a series of 3D convolutions with
  'same' padding. It starts with a 1x1x1 convolution, followed by a
  (1, 3, 3) and a (3, 1, 1) convolution. It also incorporates a channel
  re-weighting mechanism: global average pooling is applied to the features
  after the last ReLU, followed by a 1x1x1 convolution and a sigmoid
  activation. This sigmoid output is then used to re-weight the channels
  of the main path.
  """

  def __init__(self, conv1_in, conv1_out, conv2_out, pool_shape):
    super(InceptionB1B2, self).__init__()
    self.conv3d_1 = Conv3DSamePadding(
        conv1_in, conv1_out, kernel_size=1, stride=1, bias=False
    )
    self.batch_norm1 = nn.BatchNorm3d(conv1_out)
    self.relu = nn.ReLU()
    self.conv3d_2 = Conv3DSamePadding(
        conv1_out, conv2_out, kernel_size=(1, 3, 3), stride=1, bias=False
    )
    self.batch_norm2 = nn.BatchNorm3d(conv2_out)

    self.conv3d_3 = Conv3DSamePadding(
        conv2_out, conv2_out, kernel_size=(3, 1, 1), stride=1, bias=True
    )
    self.avgpool = nn.AvgPool3d(kernel_size=pool_shape, stride=1)
    self.conv3d_4 = Conv3DSamePadding(
        conv2_out, conv2_out, kernel_size=(1, 1, 1), stride=1, bias=False
    )
    self.sigmoid = nn.Sigmoid()
    self.tile_shape = pool_shape

  def forward(self, x):
    x = self.conv3d_1(x)
    x = self.batch_norm1(x)
    x = self.relu(x)
    x = self.conv3d_2(x)
    x = self.batch_norm2(x)
    x = self.relu(x)
    x = self.conv3d_3(x)
    relu_out = self.relu(x)
    x = self.avgpool(relu_out)
    x = self.conv3d_4(x)
    x = x.repeat(
        1, 1, self.tile_shape[0], self.tile_shape[1], self.tile_shape[2]
    )
    x = self.sigmoid(x)
    x = x * relu_out
    return x


class InceptionMixed(nn.Module):
  """A mixed block for a 3D Inception-like architecture.

  This block consists of four parallel branches (b0, b1, b2, b3), each
  processing the input tensor. The outputs of these branches are
  concatenated along the channel dimension.

  - b0 and b3 are instances of `InceptionB0B3`. b3 includes an initial max
    pooling layer.
  - b1 and b2 are instances of `InceptionB1B2`.
  """

  def __init__(
      self,
      conv1_in,
      b0_conv1_out,
      b0_pool_shape,
      b1_conv1_out,
      b1_conv2_out,
      b1_pool_shape,
      b2_conv1_out,
      b2_conv2_out,
      b2_pool_shape,
      b3_conv1_out,
      b3_pool_shape,
  ):
    super(InceptionMixed, self).__init__()
    self.b0 = InceptionB0B3(conv1_in, b0_conv1_out, b0_pool_shape)
    self.b1 = InceptionB1B2(conv1_in, b1_conv1_out, b1_conv2_out, b1_pool_shape)
    self.b2 = InceptionB1B2(conv1_in, b2_conv1_out, b2_conv2_out, b2_pool_shape)
    self.b3 = InceptionB0B3(
        conv1_in, b3_conv1_out, b3_pool_shape, include_maxpool=True
    )

  def forward(self, x):
    b0_out = self.b0(x)
    b1_out = self.b1(x)
    b2_out = self.b2(x)
    b3_out = self.b3(x)
    return torch.cat([b0_out, b1_out, b2_out, b3_out], dim=1)


class InceptionMixedBlock(nn.Module):
  """A large block composed of multiple InceptionMixed modules and pooling layers.

  This block forms a significant part of a 3D CNN architecture,
  processing input features through a series of:
  - An initial 3D convolution, batch normalization, ReLU, and max pooling.
  - An `InceptionB1B2` block.
  - Another max pooling layer.
  - Several `InceptionMixed` blocks (3b and 3c).
  - A max pooling layer.
  - More `InceptionMixed` blocks (4b through 4f).
  - A final max pooling layer.
  - The last two `InceptionMixed` blocks (5b and 5c).

  All convolutional and pooling layers within this block use 'same' padding
  emulating TensorFlow's behavior.
  """

  def __init__(self):
    super(InceptionMixedBlock, self).__init__()
    self.conv3d_1a = Conv3DSamePadding(
        3,
        64,
        kernel_size=(3, 7, 7),
        stride=(2, 2, 2),
        bias=False,
    )
    self.batch_norm_1a = nn.BatchNorm3d(64)
    self.relu = nn.ReLU()
    self.maxpool_2a = MaxPool3dSame(kernel_size=(1, 3, 3), stride=(1, 2, 2))
    self.pre_2b2c = InceptionB1B2(
        64,
        64,
        192,
        (3, 45, 80),
    )
    self.maxpool_3a = MaxPool3dSame(kernel_size=(1, 3, 3), stride=(1, 2, 2))
    self.inception_mixed1_3b = InceptionMixed(
        192,
        64,
        (3, 23, 40),
        96,
        128,
        (3, 23, 40),
        16,
        32,
        (3, 23, 40),
        32,
        (3, 23, 40),
    )
    self.inception_mixed2_3c = InceptionMixed(
        256,
        128,
        (3, 23, 40),
        128,
        192,
        (3, 23, 40),
        32,
        96,
        (3, 23, 40),
        64,
        (3, 23, 40),
    )
    self.maxpool_4a = MaxPool3dSame(kernel_size=(3, 3, 3), stride=(2, 2, 2))
    self.inception_mixed3_4b = InceptionMixed(
        480,
        192,
        (2, 12, 20),
        96,
        208,
        (2, 12, 20),
        16,
        48,
        (2, 12, 20),
        64,
        (2, 12, 20),
    )
    self.inception_mixed4_4c = InceptionMixed(
        512,
        160,
        (2, 12, 20),
        112,
        224,
        (2, 12, 20),
        24,
        64,
        (2, 12, 20),
        64,
        (2, 12, 20),
    )
    self.inception_mixed5_4d = InceptionMixed(
        512,
        128,
        (2, 12, 20),
        128,
        256,
        (2, 12, 20),
        24,
        64,
        (2, 12, 20),
        64,
        (2, 12, 20),
    )
    self.inception_mixed6_4e = InceptionMixed(
        512,
        112,
        (2, 12, 20),
        144,
        288,
        (2, 12, 20),
        32,
        64,
        (2, 12, 20),
        64,
        (2, 12, 20),
    )
    self.inception_mixed7_4f = InceptionMixed(
        528,
        256,
        (2, 12, 20),
        160,
        320,
        (2, 12, 20),
        32,
        128,
        (2, 12, 20),
        128,
        (2, 12, 20),
    )
    self.maxpool_5a = MaxPool3dSame(kernel_size=(2, 2, 2), stride=(2, 2, 2))
    self.inception_mixed8_5b = InceptionMixed(
        832,
        256,
        (1, 6, 10),
        160,
        320,
        (1, 6, 10),
        32,
        128,
        (1, 6, 10),
        128,
        (1, 6, 10),
    )
    self.inception_mixed9_5c = InceptionMixed(
        832,
        384,
        (1, 6, 10),
        192,
        384,
        (1, 6, 10),
        48,
        128,
        (1, 6, 10),
        128,
        (1, 6, 10),
    )

  def forward(self, x):
    x = self.conv3d_1a(x)
    x = self.batch_norm_1a(x)
    x = self.relu(x)
    x = self.maxpool_2a(x)
    x = self.pre_2b2c(x)
    x = self.maxpool_3a(x)
    x = self.inception_mixed1_3b(x)
    x = self.inception_mixed2_3c(x)
    x = self.maxpool_4a(x)
    x = self.inception_mixed3_4b(x)
    x = self.inception_mixed4_4c(x)
    x = self.inception_mixed5_4d(x)
    x = self.inception_mixed6_4e(x)
    x = self.inception_mixed7_4f(x)
    x = self.maxpool_5a(x)
    x = self.inception_mixed8_5b(x)
    x = self.inception_mixed9_5c(x)
    return x
