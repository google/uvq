import typing
import os
from collections import defaultdict

import numpy as np
import torch

from torch import nn

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoint/aggregationnet_models")

NUM_CHANNELS_PER_SUBNET = 100
NUM_FILTERS = 256
CONV2D_KERNEL_SIZE = (1, 1)
MAXPOOL2D_KERNEL_SIZE = (16, 16)

BN_DEFAULT_EPS = 0.001
BN_DEFAULT_MOMENTUM = 1
DROPOUT_RATE = 0.2


class AggregationNet(nn.Module):
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
        super(AggregationNet, self).__init__()
        self.num_subnets = len(subnets)
        self.subnets = subnets
        self.conv1 = nn.Conv2d(
            self.num_subnets * num_channels_per_subnet, num_filters, kernel_size=conv2d_kernel_size, bias=True
        )
        self.bn1 = nn.BatchNorm2d(num_filters, eps=bn_eps, momentum=bn_momentum)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=maxpool2d_kernel_size)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.linear1 = nn.Linear(bias=True, in_features=num_filters, out_features=1)

    def forward(self, features: dict[str, torch.Tensor]):
        x = (
            features[self.subnets[0]]
            if len(self.subnets) == 1
            else torch.cat([features[i] for i in self.subnets], dim=1)
        )
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = nn.Flatten()(x)
        x = self.linear1(x)
        x = torch.mean(x, dim=0)
        return x


class AggregationNetInference:
    def __init__(
        self,
        model_dir=MODEL_DIR,
        pretrained=True,
    ):
        self.models: dict[str, torch.nn.Module] = self.load_models(model_dir)
        if pretrained:
            for _, model in self.models.items():
                model.eval()

    def predict(
        self,
        compression_features: np.ndarray,
        content_features: np.ndarray,
        distortion_features: np.ndarray,
    ) -> dict[str, float]:
        feature_results = defaultdict(list)
        for model_name, features in self.get_model_names_iterator():
            with torch.no_grad():
                r = self.models[model_name](
                    {
                        "compression": torch.Tensor(
                            compression_features.transpose(0, 3, 1, 2)
                        ),
                        "content": torch.Tensor(content_features.transpose(0, 3, 1, 2)),
                        "distortion": torch.Tensor(
                            distortion_features.transpose(0, 3, 1, 2)
                        ),
                    }
                )
            feature_results["_".join(features)].append(r[0].item())
        feature_results = {
            feature: np.mean(results) for feature, results in feature_results.items()
        }
        return feature_results

    def load_models(self, models_dir: str) -> dict[str, torch.nn.Module]:
        models = {}
        for model_name, features in self.get_model_names_iterator():
            model_path = os.path.join(models_dir, model_name + ".pt")
            model = AggregationNet(features)
            state_dict = self.read_state_dict(model_path)
            model.load_state_dict(state_dict)
            models[model_name] = model
        return models

    def read_state_dict(self, model_path: str) -> dict[str, torch.Tensor]:
        state_dict = torch.load(model_path, weights_only=True)
        return state_dict

    def get_model_names_iterator(self) -> typing.Iterator[tuple[str, list[str]]]:
        trainset = "ytugc20s"
        all_trainset_subindex = ["0", "1", "2", "3", "4"]
        all_feature = [
            "compression",
            "content",
            "distortion",
            "compression_content",
            "compression_distortion",
            "content_distortion",
            "compression_content_distortion",
        ]
        aggregation_model = "avgpool"
        for feature in all_feature:
            for trainset_subindex in all_trainset_subindex:
                model_name = "%s_%s_%s_%s" % (
                    trainset,
                    trainset_subindex,
                    aggregation_model,
                    feature,
                )
                current_features = feature.split("_")
                yield model_name, current_features
