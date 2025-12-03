"""CompressionNet model and inference helper.

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
import numpy as np
import torch

from torch import nn

from . import custom_nn_layers

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoint/compressionnet_pytorch_statedict.pt")


# Input video size
VIDEO_HEIGHT = 720
VIDEO_WIDTH = 1280
VIDEO_CHANNELS = 3

# Input patch size (video is broken to patches and input to model)
PATCH_HEIGHT = 180
PATCH_WIDTH = 320
PATCH_DEPTH = 5

# Output feature size
DIM_HEIGHT_FEATURE = 16
DIM_WIDTH_FEATURE = 16
DIM_CHANNEL_FEATURE = 100

OUTPUT_LABEL_DIM = 1


class CompressionNet(nn.Module):
    """
    The CompressionNet in architecture is mainly based on InceptionV1
    which is found as googlenet in torchvision. One can modify googlenet to
    achieve at the same network, but some changes would be necessary:
    (1) changing the Conv2d and BatchNorm2d operations into Conv3d
    and BatchNorm3d operation. (2) If one chooses to use the baseline
    weights provided from the TensorFlow implementation, then they need to
    make the necessary changes to support "same" padding offered in Tensorflow
    convolutional layers. (3) Another change would be to use the hooks to output the
    "features" layer's output rather than the final outputs of the network;
    these features are later used by the aggregation network.

    In this implementation, we have opted to implement the layers from scratch,
    rather than modifying the torchvision's googlenet implementation.
    """

    def __init__(self):
        super(CompressionNet, self).__init__()
        self.inception_block1 = custom_nn_layers.InceptionMixedBlock()
        self.final_conv3d = nn.Conv3d(
            1024, 100, kernel_size=(1, 3, 7), stride=1, bias=False
        )
        self.avgpool_3d = nn.AvgPool3d(kernel_size=(1, 4, 4), stride=1)
        self.conv3d_0c = custom_nn_layers.Conv3DSamePadding(
            100, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=True
        )
        self.sigmoid = nn.Sigmoid()
        self.nonlinear1 = nn.Linear(1600, 1600, bias=True)
        self.relu = nn.ReLU()
        self.nonlinear2 = nn.Linear(1600, 1600, bias=True)

    def forward(self, x):
        inception_b1 = self.inception_block1(x)
        inception_v1_conv3d = self.final_conv3d(inception_b1)
        x = self.avgpool_3d(inception_v1_conv3d)
        features = inception_v1_conv3d.squeeze(dim=2)
        x = self.conv3d_0c(x)
        x = torch.mean(x, dim=(0, 1, 2))
        compress_level_orig = self.sigmoid(x)
        reshape_3 = features.permute(0, 2, 3, 1).reshape(features.shape[0], -1)
        non_linear1 = self.nonlinear1(reshape_3)
        non_linear1 = self.relu(non_linear1)
        _ = self.nonlinear2(non_linear1)
        return features, compress_level_orig


class CompressionNetInference:
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
        label_dim=OUTPUT_LABEL_DIM,
    ):
        self.model = CompressionNet()
        if pretrained:
            self.load_state_dict(model_path)
        if eval_mode:
            self.model.eval()
        self.features_transpose = (0, 2, 3, 1)
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
        self.label_dim = label_dim

    def load_state_dict(self, model_path) -> torch.nn.Module:
        model = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(model)
        return model

    def predict(self, patch):
        """
        patch is a 5d numpy array of shape (batch_size, channel, depth, width, height)
        batch_size is assumed to be 1 for now. But should work with larger.
        """
        with torch.no_grad():
            _, label_probs = self.model(torch.Tensor(patch))
        return label_probs.detach().numpy()

    def predict_and_get_features(self, patch) -> tuple[np.ndarray, np.ndarray]:
        """
        patch is a 5d numpy array of shape (batch_size, channel, depth=5, width, height)
        batch_size is assumed to be 1 for now. But should work with larger.
        """
        with torch.no_grad():
            features, label_probs = self.model(torch.Tensor(patch))
        return (
            features.detach().numpy().transpose(*self.features_transpose),
            label_probs.detach().numpy()[0],
        )

    def get_labels_and_features_for_all_frames(
        self,
        video: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Gets the predicted labels and features for all frames in a video.

        Args:
            video (np.ndarray): 5d array of shape (num_seconds, fps, channels=3, height=720, width=1280) with values in [-1, 1] range

        Returns:
            feature (np.ndarray): 4d array of shape (num_seconds, 16, 16, channels=100) of features to be used in aggregation
            label (np.ndarray): 2d array of shape (num_seconds, 4, 4, 1) of predicted compression level for each second*fps frame partitioned into 4x4 patches
        """
        # TODO: allow converting video to a batch of patches and running batch prediction instead of sliding window for loops
        label = np.ndarray(
            (video.shape[0], self.num_patches_y, self.num_patches_x, self.label_dim),
            np.float32,
        )
        feature = np.ndarray(
            (
                video.shape[0],
                self.feature_height,
                self.feature_width,
                self.feature_channels,
            ),
            np.float32,
        )

        video = video.transpose(0, 2, 1, 3, 4)
        patch = np.ndarray(
            (
                1,
                self.video_channels,
                self.depth,
                self.patch_height,
                self.patch_width,
            ),
            np.float32,
        )

        for k in range(video.shape[0]):
            for j in range(self.num_patches_y):
                for i in range(self.num_patches_x):
                    patch[0, :] = video[
                        k,
                        :,
                        :,
                        j * self.patch_height : (j + 1) * self.patch_height,
                        i * self.patch_width : (i + 1) * self.patch_width,
                    ]

                    patch_feature, patch_label = self.predict_and_get_features(patch)
                    feature[
                        k,
                        j
                        * self.patch_feature_height : (j + 1)
                        * self.patch_feature_height,
                        i
                        * self.patch_feature_width : (i + 1)
                        * self.patch_feature_width,
                        :,
                    ] = patch_feature

                    label[k, j, i, :] = patch_label
        return feature, label
