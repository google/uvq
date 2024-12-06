import numpy as np
import pandas as pd
import torch

from torch import nn, Tensor

from .custom_nn_layers import (
    Conv2dNormActivationSamePadding,
    Conv2dSamePadding,
    Interpolate,
    MBConvSamePadding,
)
from .video_reader import VideoReader

MODEL_PATH = "checkpoint/contentnet_pytorch.pt"
LABELS_CSV_PATH = "checkpoint/contentnet_labels.csv"

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
    implement the layers from scratch with intorduction of a Conv2dSamePadding layer.
    """

    def __init__(self, num_classes=DIM_LABEL_CONTENT, dropout=0.2):
        super().__init__()
        stochastic_depth_prob_step = 0.0125
        stochastic_depth_prob = [x * stochastic_depth_prob_step for x in range(16)]
        self.features = nn.Sequential(
            Conv2dNormActivationSamePadding(
                3, 32, kernel_size=3, stride=2, activation_layer=nn.SiLU
            ),
            MBConvSamePadding(32, 1, 16, 3, 1, stochastic_depth_prob[0]),
            MBConvSamePadding(16, 6, 24, 3, 2, stochastic_depth_prob[1]),
            MBConvSamePadding(24, 6, 24, 3, 1, stochastic_depth_prob[2]),
            MBConvSamePadding(24, 6, 40, 5, 2, stochastic_depth_prob[3]),
            MBConvSamePadding(40, 6, 40, 5, 1, stochastic_depth_prob[4]),
            MBConvSamePadding(40, 6, 80, 3, 2, stochastic_depth_prob[5]),
            MBConvSamePadding(80, 6, 80, 3, 1, stochastic_depth_prob[6]),
            MBConvSamePadding(80, 6, 80, 3, 1, stochastic_depth_prob[7]),
            MBConvSamePadding(80, 6, 112, 5, 1, stochastic_depth_prob[8]),
            MBConvSamePadding(112, 6, 112, 5, 1, stochastic_depth_prob[9]),
            MBConvSamePadding(112, 6, 112, 5, 1, stochastic_depth_prob[10]),
            MBConvSamePadding(112, 6, 192, 5, 2, stochastic_depth_prob[11]),
            MBConvSamePadding(192, 6, 192, 5, 1, stochastic_depth_prob[12]),
            MBConvSamePadding(192, 6, 192, 5, 1, stochastic_depth_prob[13]),
            MBConvSamePadding(192, 6, 192, 5, 1, stochastic_depth_prob[14]),
            MBConvSamePadding(192, 6, 320, 3, 1, stochastic_depth_prob[15]),
            Interpolate(size=(16, 16), mode="bilinear", align_corners=False),
            Conv2dSamePadding(320, 100, kernel_size=16, stride=1),
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
        self, model_path=MODEL_PATH, num_classes=3862, eval_mode=True, pretrained=True
    ):
        self.model = ContentNet(num_classes=num_classes)
        if pretrained:
            self.load_state_dict(model_path)
        if eval_mode:
            self.model.eval()
        self.label_mapping: pd.DataFrame = self.load_labels_df(LABELS_CSV_PATH)
        self.features_transpose = (0, 2, 3, 1)

    def load_state_dict(self, model_path) -> torch.nn.Module:
        model = torch.load(model_path)
        self.model.load_state_dict(model)
        return model

    def predict(self, frame):
        with torch.no_grad():
            _, label_probs = self.model(Tensor(np.expand_dims(frame, 0)))
        return label_probs.detach().numpy()

    def predict_and_get_features(self, frame) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            features, label_probs = self.model(torch.Tensor(np.expand_dims(frame, 0)))
        return (
            features.detach().numpy().transpose(*self.features_transpose),
            label_probs.detach().numpy()[0],
        )

    def load_labels_df(self, csv_path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        return df

    def label_probabilities_to_text(self, label_probs, top_n=1):
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

    def infer_from_input_video(self, video_filename, video_length, transpose=False):
        video, video_resized = VideoReader.load_video(
            video_filename, video_length, transpose
        )
        video_features, video_labels = self.get_labels_and_features_for_all_frames(
            video_resized
        )
        return video_features, video_labels

    def get_labels_and_features_for_all_frames(self, video):
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
