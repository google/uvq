"""UVQ inference entry point for 1.0 and 1.5 models.

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

import argparse
import math
import os
import json
from typing import Any

from utils.probe import get_nb_frames
from utils.probe import get_r_frame_rate
from utils.probe import get_video_duration

from uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5
from uvq_pytorch.utils.uvq1p0 import UVQ1p0


def main():
  parser = setup_parser()
  args = parser.parse_args()

  video_filename = args.video_filename
  transpose = args.transpose
  output_filepath = args.output

  duration = get_video_duration(video_filename)
  if duration is None:
    print(f"Could not get duration for {video_filename}, skipping.")
    return
  video_length = math.ceil(duration)

  orig_fps = get_r_frame_rate(video_filename)
  if orig_fps is None:
    print(f"Could not get frame rate for {video_filename}, skipping.")
    return

  fps = args.fps
  if fps == -1:
    fps = orig_fps
    nb_frames = get_nb_frames(video_filename)
    if nb_frames is not None and nb_frames > 0:
      video_length = math.ceil(nb_frames / fps)
    else:
      print(
          f"Could not get frame count for {video_filename}, relying on"
          " duration."
      )

  if video_length == 0:
    print(f"Skipping {video_filename} due to 0 length.")
    return

  if args.model_version == "1.5":
    uvq_inference = UVQ1p5()
    if args.device == "cuda":
      uvq_inference.cuda()

    results = uvq_inference.infer(
        video_filename,
        video_length,
        transpose,
        fps=fps,
        orig_fps=orig_fps,
    )
  elif args.model_version == "1.0":
    uvq_inference = UVQ1p0()
    # UVQ1.0 infer doesn't support fps or padding args.
    # It uses its own video reader, which has fixed 5 fps sampling.
    # If fps is passed for 1.0, it will be ignored by 1.0 infer method.
    print("Running UVQ 1.0 inference. FPS argument is ignored (uses 5fps).")
    results = uvq_inference.infer(
        video_filename,
        video_length,
        transpose,
    )
  else:
    print(f"Unknown model version: {args.model_version}")
    return

  print(results)

  if output_filepath != "":
    write_dict_to_file(results, output_filepath)


def write_dict_to_file(d: dict[str, Any], output_filepath: str) -> None:
  """Writes the key-value pairs of a dictionary to a file in JSON format.

  If the directory for the output file does not exist, it will be created.

  Args:
    d: The dictionary to write to the file.
    output_filepath: The path to the output file.
  """
  dirname = os.path.dirname(output_filepath)
  if dirname != "" and not os.path.exists(dirname):
    os.makedirs(dirname)
  with open(output_filepath, "w") as f:
    json.dump(d, f, indent=2)


def setup_parser():
  """Sets up the argument parser for the UVQ inference script."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "video_filename",
      type=str,
      help="Path to the video file",
  )
  parser.add_argument(
      "--model_version",
      type=str,
      default="1.5",
      choices=["1.0", "1.5"],
      help="UVQ model version to use for inference.",
  )
  parser.add_argument(
      "--transpose",
      action="store_true",
      help="If specified, the video will be transposed before processing",
  )
  parser.add_argument(
      "--output",
      type=str,
      help="Path to the output file",
      default="",
      required=False,
  )
  parser.add_argument(
      "--device",
      type=str,
      default="cpu",
      help="Device to run inference on (e.g., 'cpu' or 'cuda').",
  )
  parser.add_argument(
      "--fps",
      type=int,
      default=1,
      help="Frames per second to sample for UVQ1.5. -1 to sample all frames."
      " Ignored for UVQ1.0.",
  )
  return parser


if __name__ == "__main__":
  main()
