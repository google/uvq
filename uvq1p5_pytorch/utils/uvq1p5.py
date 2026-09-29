"""Pytorch implementation of the UVQ1p5 model.

This module contains the UVQ1p5 model, including the ContentNet, DistortionNet,
and AggregationNet. It provides functionality to load a video and infer
its quality using the trained UVQ1p5 model.

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
from typing import Any

import torch
import torch.nn as nn
import torchvision

from . import aggregationnet
from . import contentnet
from . import distortionnet

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', 'utils'
        )
    )
)
import video_reader


class UVQ1p5Core(nn.Module):
  """UVQ 1.5 core model."""

  def __init__(
      self,
      content_net,
      distortion_net,
      aggregation_net,
  ):
    super(UVQ1p5Core, self).__init__()
    self.content_net = content_net
    self.distortion_net = distortion_net
    self.aggregation_net = aggregation_net

  def forward(self, video):
    content_features = self.content_net(video)
    distortion_features = self.distortion_net(video)

    pred_dict = self.aggregation_net(content_features, distortion_features)
    quality_pred = pred_dict['uvq_1p5_features']
    return quality_pred


class UVQ1p5(nn.Module):
  """UVQ 1.5 model."""

  def __init__(self, eval_mode=True, pretrained=True):
    super(UVQ1p5, self).__init__()
    model_path_content_net = os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "content_net.pth"
    )
    model_path_distortion_net = os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "distortion_net.pth"
    )
    model_path_aggregation_net = os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "aggregation_net.pth"
    )

    self.content_net = contentnet.ContentNet(
        model_path=model_path_content_net,
        eval_mode=eval_mode,
        pretrained=pretrained,
    )
    self.distortion_net = distortionnet.DistortionNet(
        model_path=model_path_distortion_net,
        eval_mode=eval_mode,
        pretrained=pretrained,
    )
    self.aggregation_net = aggregationnet.AggregationNet(
        model_path=model_path_aggregation_net,
        eval_mode=eval_mode,
        pretrained=pretrained,
    )

    self.uvq1p5_core = UVQ1p5Core(
        self.content_net, self.distortion_net, self.aggregation_net
    )
    if eval_mode:
      self.uvq1p5_core.eval()

  def infer(
      self,
      video_filename: str,
      video_length: int,
      transpose: bool,
      fps: int = 1,
      orig_fps: float | None = None,
      ffmpeg_path: str = "ffmpeg",
      device: str = "cpu",
      batch_size: int | None = None,
  ) -> dict[str, Any]:
    """Runs UVQ 1.5 inference on a video file.

    Args:
      video_filename: Path to the video file.
      video_length: Length of the video in seconds.
      transpose: Whether to transpose the video.
      fps: Frames per second to sample for inference.
      orig_fps: Original frames per second of the video, used for frame index
        calculation.
      ffmpeg_path: Path to ffmpeg executable.
      device: Device to run inference on (e.g., 'cpu' or 'cuda').
      batch_size: Maximum number of sampled frames per inference batch.
        Defaults to 1 on CPU and 24 on other devices.

    Returns:
      A dictionary containing the overall UVQ 1.5 score, per-frame scores,
      and frame indices.
    """
    if batch_size is None:
      batch_size = 1 if device == "cpu" else 24

    # Keep convolution weights in the layout of the decoded RGB input on CPU.
    if device == "cpu":
      self.uvq1p5_core.to(memory_format=torch.channels_last)

    predictions = []
    with torch.inference_mode():
      for video_batch, _ in video_reader.iter_video_batches_1p5(
          video_filename,
          video_length,
          transpose,
          video_fps=fps,
          batch_size=batch_size,
          ffmpeg_path=ffmpeg_path,
      ):
        batch = torch.from_numpy(video_batch).permute(0, 3, 1, 2)
        batch = batch.unsqueeze(1).to(device)
        predictions.append(self.uvq1p5_core(batch))

    prediction = torch.cat(predictions, dim=0)

    video_score = torch.mean(prediction).item()
    frame_scores = prediction.cpu().numpy().flatten().tolist()

    if orig_fps:
      frame_indices = [
          int(round(i * orig_fps / fps)) for i in range(len(frame_scores))
      ]
    else:
      frame_indices = list(range(len(frame_scores)))

    return {
        "uvq1p5_score": video_score,
        "per_frame_scores": frame_scores,
        "frame_indices": frame_indices,
    }

  def load_video(
      self,
      video_filename: str,
      video_length: int,
      transpose: bool = False,
      fps: int = 1,
      ffmpeg_path: str = "ffmpeg",
  ) -> tuple[torch.Tensor, int]:
    """Loads and preprocesses a video for UVQ 1.5 inference.

    Args:
      video_filename: Path to the video file.
      video_length: Length of the video in seconds.
      transpose: Whether to transpose the video.
      fps: Frames per second to sample.

    Returns:
      A tuple containing the loaded video as a torch tensor and the number of
      real frames.
    """
    video, num_real_frames = video_reader.load_video_1p5(
        video_filename,
        video_length,
        transpose,
        video_fps=fps,
        video_height=1080,
        video_width=1920,
        ffmpeg_path=ffmpeg_path,
    )
    video = video.transpose(0, 1, 4, 2, 3)
    return torch.from_numpy(video).float(), num_real_frames
