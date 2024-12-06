from functools import partial

import numpy as np
import torch

from torch import nn, Tensor

from .custom_nn_layers import (
    Conv2dNormActivationSamePadding,
    Conv2dSamePadding,
    MBConvSamePadding,
    PermuteLayerNHWC,
)

MODEL_PATH = "checkpoint/distortionnet_pytorch_statedict.pt"

default_distortionnet_batchnorm2d = partial(nn.BatchNorm2d, eps=0.001, momentum=0)

# Input video size
VIDEO_HEIGHT = 720
VIDEO_WIDTH = 1280
VIDEO_CHANNELS = 3

# Input patch size (video is broken to patches and input to model)
PATCH_HEIGHT = 360
PATCH_WIDTH = 640
PATCH_DEPTH = 1

# Output feature size
DIM_HEIGHT_FEATURE = 16
DIM_WIDTH_FEATURE = 16
DIM_CHANNEL_FEATURE = 100

OUTPUT_LABEL_DIM = 26


class DistortionNet(nn.Module):
    """
    DistortionNet is based on the EfficientNet architecture. One can achieve the same
    network by modifying torchvision's efficientnet_b0 model. The features layers will
    be the same, but the final avgpool and classifier layers will need to change. Instead
    of AdaptiveAvgPool2D will need to use MaxPool, and the classifier will consist of two
    linear layers.
    In addition if the intention is to use the baseline weights from the tensorflow implementation,
    changes must be made to support the "same" padding used in tensorflow convolution layers.
    In this implementation we have opted to not use the torchivision's efficientNet and instead
    implement the layers from scratch with intorduction of a Conv2dSamePadding layer.
    """

    def __init__(self, dropout=0.2):
        super().__init__()
        stochastic_depth_prob_step = 0.0125
        stochastic_depth_prob = [x * stochastic_depth_prob_step for x in range(16)]
        self.features = nn.Sequential(
            Conv2dNormActivationSamePadding(
                3,
                32,
                kernel_size=3,
                stride=2,
                activation_layer=nn.SiLU,
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                32,
                1,
                16,
                3,
                1,
                stochastic_depth_prob[0],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                16,
                6,
                24,
                3,
                2,
                stochastic_depth_prob[1],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                24,
                6,
                24,
                3,
                1,
                stochastic_depth_prob[2],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                24,
                6,
                40,
                5,
                2,
                stochastic_depth_prob[3],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                40,
                6,
                40,
                5,
                1,
                stochastic_depth_prob[4],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                40,
                6,
                80,
                3,
                2,
                stochastic_depth_prob[5],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                80,
                6,
                80,
                3,
                1,
                stochastic_depth_prob[6],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                80,
                6,
                80,
                3,
                1,
                stochastic_depth_prob[7],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                80,
                6,
                112,
                5,
                1,
                stochastic_depth_prob[8],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                112,
                6,
                112,
                5,
                1,
                stochastic_depth_prob[9],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                112,
                6,
                112,
                5,
                1,
                stochastic_depth_prob[10],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                112,
                6,
                192,
                5,
                2,
                stochastic_depth_prob[11],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                192,
                6,
                192,
                5,
                1,
                stochastic_depth_prob[12],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                192,
                6,
                192,
                5,
                1,
                stochastic_depth_prob[13],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                192,
                6,
                192,
                5,
                1,
                stochastic_depth_prob[14],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            MBConvSamePadding(
                192,
                6,
                320,
                3,
                1,
                stochastic_depth_prob[15],
                norm_layer=default_distortionnet_batchnorm2d,
            ),
            Conv2dSamePadding(320, 100, kernel_size=(12, 20), stride=1, bias=False),
        )
        self.avgpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(5, 13), stride=1, padding=0),
            PermuteLayerNHWC(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Flatten(),
            nn.Linear(6400, 512, bias=True),
            nn.ReLU6(),
            nn.Linear(512, 26, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        features = self.avgpool(x)
        label_probs = self.classifier(features)
        return features, label_probs


class DistortionNetInference:
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
        self.model = DistortionNet()
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
        self.label_dim = label_dim

    def load_state_dict(self, model_path) -> nn.Module:
        model = torch.load(model_path)
        self.model.load_state_dict(model)
        return model

    def predict(self, frame):
        with torch.no_grad():
            _, label_probs = self.model(Tensor(frame))
        return label_probs.detach().numpy()

    def predict_and_get_features(self, frame) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            features, label_probs = self.model(torch.Tensor(frame))
        return (
            features.detach().numpy(),
            label_probs.detach().numpy(),
        )

    def get_labels_and_features_for_all_frames(
        self,
        video,
    ):
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

        if self.depth == 1:
            # TODO: this is for distortionnet; for compressionnet keep only the "else" part.
            patch = np.ndarray(
                (1, self.video_channels, self.patch_height, self.patch_width),
                np.float32,
            )
        else:
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
                    if self.depth == 1:
                        # TODO: this would work for distortionnet, it's not used here and should be cleaned up; keep only the "else" part
                        patch[0, :] = video[
                            k,
                            0,
                            :,
                            j * self.patch_height : (j + 1) * self.patch_height,
                            i * self.patch_width : (i + 1) * self.patch_width,
                        ]
                    else:
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
