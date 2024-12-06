import os
from collections import defaultdict
from typing import Iterator

import numpy as np
import torch

from torch import nn

MODEL_DIR = "checkpoint/aggregationnet_models"


class AggregationNet(nn.Module):
    def __init__(self, subnets: list[str]):
        super(AggregationNet, self).__init__()
        self.num_subnets = len(subnets)
        self.subnets = subnets
        self.conv1 = nn.Conv2d(
            self.num_subnets * 100, 256, kernel_size=(1, 1), bias=True
        )
        self.bn1 = nn.BatchNorm2d(256, eps=0.001, momentum=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(16, 16))
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(bias=True, in_features=256, out_features=1)

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
        eval_mode=True,
        pretrained=True,
    ):
        model_dir = model_dir or MODEL_DIR
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
        state_dict = torch.load(model_path)
        return state_dict

    def get_model_names_iterator(self) -> Iterator[tuple[str, list[str]]]:
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
