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
import tqdm

from utils import probe

from uvq1p5_pytorch.utils import uvq1p5
from uvq_pytorch.utils import uvq1p0


def run_batch_inference(args):
  """Runs inference on a list of videos."""
  if not args.output:
    print("Error: --output must be specified when input is a .txt file list.")
    return

  if args.model_version == "1.5":
    model = uvq1p5.UVQ1p5()
  elif args.model_version == "1.0":
    model = uvq1p0.UVQ1p0()
    print("Running UVQ 1.0 batch inference. FPS argument is ignored (uses 5fps).")
  else:
    return  # Should not happen
  if args.device == "cuda":
    model.cuda()

  try:
    with open(args.input, "r") as f:
      video_paths = [line.strip() for line in f if line.strip()]
  except FileNotFoundError:
    print(f"Error: Input file list not found at {args.input}")
    return

  results_to_write = []
  for video_path in tqdm.tqdm(video_paths, desc="Processing videos"):
    try:
      transpose_flag = False
      dimensions = probe.get_dimensions(video_path, args.ffprobe_path)
      if dimensions and dimensions[0] < dimensions[1]:
        print(
            f"Portrait video {video_path} detected, setting transpose=True for"
            " this video."
        )
        transpose_flag = True

      duration = probe.get_video_duration(video_path, args.ffprobe_path)
      orig_fps = probe.get_r_frame_rate(video_path, args.ffprobe_path)
      nb_frames = probe.get_nb_frames(video_path, args.ffprobe_path)

      video_length = 0
      if duration:
        video_length = math.ceil(duration)
      elif orig_fps and nb_frames:
        video_length = math.ceil(nb_frames / orig_fps)

      if video_length == 0:
        print(f"Could not determine video length for {video_path}, skipping.")
        continue

      fps_to_use = args.fps
      if args.model_version == "1.5" and fps_to_use == -1:
        if orig_fps:
          fps_to_use = orig_fps
        else:
          print(
              f"Cannot determine r_frame_rate for {video_path} with fps=-1,"
              " skipping."
          )
          continue

      if args.model_version == "1.5":
        results = model.infer(
            video_path,
            video_length,
            transpose_flag,
            fps=fps_to_use,
            orig_fps=orig_fps,
            ffmpeg_path=args.ffmpeg_path,
        )
        score = results["uvq1p5_score"]
      elif args.model_version == "1.0":
        results = model.infer(
            video_path,
            video_length,
            transpose_flag,
        )
        score = float(results["compression_content_distortion"])
      
      if args.batch_json_output:
        results["video_name"] = os.path.basename(video_path)
        results_to_write.append(results)
      else:
        results_to_write.append(f"{os.path.basename(video_path)},{score}")
    except Exception as e:
      print(f"Error processing {video_path}: {e}")

  # Write results
  try:
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
      os.makedirs(output_dir)
    
    if args.batch_json_output:
      write_dict_to_file(results_to_write, args.output)
    else:
      with open(args.output, "w") as f_out:
        for line in results_to_write:
          f_out.write(line + "\n")
    print(f"Batch inference complete. Results saved to {args.output}")
  except IOError as e:
    print(f"Error writing to output file {args.output}: {e}")


def run_single_inference(args):
  """Runs inference on a single video."""
  video_filename = args.input
  transpose = args.transpose
  output_filepath = args.output

  duration = probe.get_video_duration(video_filename, args.ffprobe_path)
  if duration is None:
    print(f"Could not get duration for {video_filename}, skipping.")
    return
  video_length = math.ceil(duration)

  orig_fps = probe.get_r_frame_rate(video_filename, args.ffprobe_path)
  if orig_fps is None:
    print(f"Could not get frame rate for {video_filename}, skipping.")
    return

  fps = args.fps
  if fps == -1:
    fps = orig_fps
    nb_frames = probe.get_nb_frames(video_filename, args.ffprobe_path)
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
    uvq_inference = uvq1p5.UVQ1p5()
    if args.device == "cuda":
      uvq_inference.cuda()

    results = uvq_inference.infer(
        video_filename,
        video_length,
        transpose,
        fps=fps,
        orig_fps=orig_fps,
        ffmpeg_path=args.ffmpeg_path,
    )
  elif args.model_version == "1.0":
    uvq_inference = uvq1p0.UVQ1p0()
    # UVQ1.0 infer doesn't support fps or padding args.
    # It uses its own video reader, which has fixed 5 fps sampling.
    # If fps is passed for 1.0, it will be ignored by 1.0 infer method.
    print("Running UVQ 1.0 inference. FPS argument is ignored (uses 5fps).")
    results = uvq_inference.infer(
        video_filename,
        video_length,
        transpose,
    )
    results = {k: float(v) for k, v in results.items()}
  else:
    print(f"Unknown model version: {args.model_version}")
    return

  results["video_name"] = video_filename
  if args.output_all_stats:
    print(json.dumps(results))
  else:
    if args.model_version == "1.5":
      print(results["uvq1p5_score"])
    elif args.model_version == "1.0":
      print(results["compression_content_distortion"])

  if output_filepath != "":
    write_dict_to_file(results, output_filepath)


def main():
  parser = setup_parser()
  args = parser.parse_args()

  if args.input.endswith(".txt"):
    run_batch_inference(args)
  else:
    run_single_inference(args)


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
      "input",
      type=str,
      help=(
          "Path to a single video file or a .txt file with one video path per"
          " line."
      ),
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
      help="Path to the output file. Required if input is a .txt list.",
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
  parser.add_argument(
      "--batch_json_output",
      action="store_true",
      help="If specified, outputs batch results in JSON format including per " \
      "frame scores instead of just overall mean score.",
  )
  parser.add_argument(
      "--output_all_stats",
      action="store_true",
      help="If specified, print all stats in JSON format to stdout.",
  )
  parser.add_argument(
      "--ffmpeg_path",
      type=str,
      default="ffmpeg",
      help="Path to ffmpeg executable.",
  )
  parser.add_argument(
      "--ffprobe_path",
      type=str,
      default="ffprobe",
      help="Path to ffprobe executable.",
  )
  return parser


if __name__ == "__main__":
  main()
