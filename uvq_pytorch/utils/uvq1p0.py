"""UVQ1.0 Pytorch model wrapper.

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
import sys

from . import aggregationnet
from . import compressionnet
from . import contentnet
from . import distortionnet

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"
        )
    )
)
import video_reader


class UVQ1p0:
  """Wrapper class for UVQ 1.0 model inference."""

  def __init__(self, eval_mode=True):
    self.contentnet = contentnet.ContentNetInference(eval_mode=eval_mode)
    self.compressionnet = compressionnet.CompressionNetInference(
        eval_mode=eval_mode
    )
    self.distortionnet = distortionnet.DistortionNetInference(
        eval_mode=eval_mode
    )
    self.aggregationnet = aggregationnet.AggregationNetInference(
        pretrained=eval_mode
    )

  def infer(
      self,
      video_filename: str,
      video_length: int,
      transpose: bool = False,
  ) -> dict[str, float]:
    """Runs UVQ 1.0 inference on a video file.

    Args:
        video_filename: Path to the video file.
        video_length: Length of the video in seconds.
        transpose: Whether to transpose the video before processing.

    Returns:
        A dictionary containing the UVQ 1.0 scores.
    """
    video_resized1, video_resized2 = self.load_video(
        video_filename, video_length, transpose
    )
    content_features, _ = (
        self.contentnet.get_labels_and_features_for_all_frames(
            video=video_resized2
        )
    )
    compression_features, _ = (
        self.compressionnet.get_labels_and_features_for_all_frames(
            video=video_resized1,
        )
    )
    distortion_features, _ = (
        self.distortionnet.get_labels_and_features_for_all_frames(
            video=video_resized1,
        )
    )
    results = self.aggregationnet.predict(
        compression_features, content_features, distortion_features
    )
    return results

  def load_video(self, video_filename, video_length, transpose=False):
    video, video_small = video_reader.load_video_1p0(
        video_filename, video_length, transpose
    )
    video = video.transpose(0, 1, 4, 2, 3)
    video_small = video_small.transpose(0, 1, 4, 2, 3)
    return video, video_small
