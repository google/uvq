"""ContentNet model and inference helper.

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


from typing import Union
import os
import numpy as np
import pandas as pd
import torch

from torch import nn

from . import custom_nn_layers

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoint/contentnet_pytorch.pt")
LABELS_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoint/contentnet_labels.csv")

# Output feature size
DIM_HEIGHT_FEATURE = 16
DIM_WIDTH_FEATURE = 16
DIM_CHANNEL_FEATURE = 100

# ContentNet specs
DIM_LABEL_CONTENT = 3862


class ContentNet(nn.Module):
    """
    ContentNet is based on the EfficientNet architecture. One can achieve the same
    network by modifying torchvision's efficientnet_b0 model. The features layers will
    be the same, but the final classifier layers will need to change.
    In addition if the intention is to use the baseline weights from the tensorflow implementation,
    changes must be made to support the "same" padding used in tensorflow convolution layers.
    In this implementation we have opted to not use the torchivision's efficientNet and instead
    implement the layers from scratch with introduction of a Conv2dSamePadding layer.
    """

    def __init__(self, num_classes=DIM_LABEL_CONTENT, dropout=0.2):
        super().__init__()
        stochastic_depth_prob_step = 0.0125
        stochastic_depth_prob = [x * stochastic_depth_prob_step for x in range(16)]
        self.features = nn.Sequential(
            custom_nn_layers.Conv2dNormActivationSamePadding(
                3, 32, kernel_size=3, stride=2, activation_layer=nn.SiLU
            ),
            custom_nn_layers.MBConvSamePadding(32, 1, 16, 3, 1, stochastic_depth_prob[0]),
            custom_nn_layers.MBConvSamePadding(16, 6, 24, 3, 2, stochastic_depth_prob[1]),
            custom_nn_layers.MBConvSamePadding(24, 6, 24, 3, 1, stochastic_depth_prob[2]),
            custom_nn_layers.MBConvSamePadding(24, 6, 40, 5, 2, stochastic_depth_prob[3]),
            custom_nn_layers.MBConvSamePadding(40, 6, 40, 5, 1, stochastic_depth_prob[4]),
            custom_nn_layers.MBConvSamePadding(40, 6, 80, 3, 2, stochastic_depth_prob[5]),
            custom_nn_layers.MBConvSamePadding(80, 6, 80, 3, 1, stochastic_depth_prob[6]),
            custom_nn_layers.MBConvSamePadding(80, 6, 80, 3, 1, stochastic_depth_prob[7]),
            custom_nn_layers.MBConvSamePadding(80, 6, 112, 5, 1, stochastic_depth_prob[8]),
            custom_nn_layers.MBConvSamePadding(112, 6, 112, 5, 1, stochastic_depth_prob[9]),
            custom_nn_layers.MBConvSamePadding(112, 6, 112, 5, 1, stochastic_depth_prob[10]),
            custom_nn_layers.MBConvSamePadding(112, 6, 192, 5, 2, stochastic_depth_prob[11]),
            custom_nn_layers.MBConvSamePadding(192, 6, 192, 5, 1, stochastic_depth_prob[12]),
            custom_nn_layers.MBConvSamePadding(192, 6, 192, 5, 1, stochastic_depth_prob[13]),
            custom_nn_layers.MBConvSamePadding(192, 6, 192, 5, 1, stochastic_depth_prob[14]),
            custom_nn_layers.MBConvSamePadding(192, 6, 320, 3, 1, stochastic_depth_prob[15]),
            custom_nn_layers.Interpolate(size=(16, 16), mode="bilinear", align_corners=False),
            custom_nn_layers.Conv2dSamePadding(320, 100, kernel_size=16, stride=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Flatten(),
            nn.Linear(100, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.features(x)
        x = self.avgpool(features)
        label_probs = self.classifier(x)
        return features, label_probs


class ContentNetInference:
    def __init__(
        self, model_path=MODEL_PATH, num_classes=DIM_LABEL_CONTENT, eval_mode=True, pretrained=True
    ):
        self.model = ContentNet(num_classes=num_classes)
        if pretrained:
            self.load_state_dict(model_path)
        if eval_mode:
            self.model.eval()
        self.label_mapping: pd.DataFrame = self.load_labels_df(LABELS_CSV_PATH)
        self.features_transpose = (0, 2, 3, 1)

    def load_state_dict(self, model_path) -> torch.nn.Module:
        model = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(model)
        return model

    def predict(self, frame):
        with torch.no_grad():
            _, label_probs = self.model(torch.Tensor(np.expand_dims(frame, 0)))
        return label_probs.detach().numpy()

    def predict_and_get_features(self, frame) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            features, label_probs = self.model(torch.Tensor(np.expand_dims(frame, 0)))
        return (
            features.detach().numpy().transpose(*self.features_transpose),
            label_probs.detach().numpy()[0],
        )

    def load_labels_df(self, csv_path) -> pd.DataFrame:
        return pd.read_csv(csv_path)

    def label_probabilities_to_text(
        self, label_probs: Union[list, np.ndarray], top_n: int = 1
    ) -> tuple[list, list, list]:
        """
        Converts the label probabilities to text.

        Args:
            label_probs (list or np.ndarray): 1d array of shape (num_classes=3862) of predicted porbabilities for each class
            top_n (int): number of top predictions to return

        Returns:
            predicted (list): list of top_n predicted labels
            probs (list): list of top_n predicted probabilities
            indices (list): list of top_n predicted indices
        """
        top_indices = label_probs.argsort()[: -top_n - 1 : -1]
        probs = label_probs[top_indices]
        predicted = self.label_mapping.merge(
            pd.DataFrame({"prob": probs, "Index": top_indices}), on="Index"
        ).sort_values("prob", ascending=False)
        return (
            predicted["Name"].tolist(),
            predicted["prob"].tolist(),
            predicted["Index"].tolist(),
        )

    def get_labels_and_features_for_all_frames(
        self, video: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Gets the predicted labels and features for all frames(seconds) in a video.

        Args:
            video (np.ndarray): 5d array of shape (num_seconds, fps, channels=3, height=496, width=496) with values in [-1, 1] range

        Returns:
            feature (np.ndarray): 4d array of shape (num_seconds, 16, 16, channels=100) of features to be used in aggregation
            label (np.ndarray): 2d array of shape (num_seconds, num_classes=3862) of predicted porbabilities for each class

        Note that even thought the input video can have any fps, computation is performed only on the first frame of each second.
        """
        label = np.ndarray((video.shape[0], DIM_LABEL_CONTENT), np.float32)
        feature = np.ndarray(
            (
                video.shape[0],
                DIM_HEIGHT_FEATURE,
                DIM_WIDTH_FEATURE,
                DIM_CHANNEL_FEATURE,
            ),
            np.float32,
        )

        for k in range(video.shape[0]):
            frame_features, frame_labels = self.predict_and_get_features(
                video[k, 0, :, :, :]
            )

            feature[k, :, :, :] = frame_features
            label[k, :] = frame_labels
        return feature, label
