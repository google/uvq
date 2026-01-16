"""probe utils using ffprobe.

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

import math
import subprocess


def get_dimensions(
    video_path, ffprobe_path="ffprobe"
) -> tuple[int, int] | None:
  """Get video width and height using ffprobe."""
  cmd = [
      ffprobe_path,
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=width,height",
      "-of",
      "csv=s=x:p=0",
      video_path,
  ]
  try:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
    )
    output = result.stdout.strip()
    if "x" in output:
      width, height = map(int, output.split("x"))
      return width, height
    return None
  except Exception as e:
    print(f"Error getting dimensions for {video_path}: {e}")
    return None


def get_nb_frames(video_path, ffprobe_path="ffprobe") -> int | None:
  """Get video nb_frames using ffprobe."""
  cmd = [
      ffprobe_path,
      "-v",
      "error",
      "-count_frames",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=nb_read_frames",
      "-of",
      "default=noprint_wrappers=1:nokey=1",
      video_path,
  ]
  try:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
    )
    output = result.stdout.strip()
    if not output or output == "N/A":
      return None
    return int(output)
  except Exception as e:
    print(f"Error getting nb_frames for {video_path}: {e}")
    return None


def get_r_frame_rate(video_path, ffprobe_path="ffprobe") -> int | None:
  """Get video r_frame_rate using ffprobe."""
  cmd = [
      ffprobe_path,
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=r_frame_rate",
      "-of",
      "default=noprint_wrappers=1:nokey=1",
      video_path,
  ]
  try:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
    )
    output = result.stdout.strip()
    if not output:
      print(f"Could not get r_frame_rate for {video_path}")
      return None
    if "/" in output:
      num, den = output.split("/")
      if int(den) == 0:
        return None
      return int(math.ceil(float(num) / float(den)))
    else:
      return int(math.ceil(float(output)))
  except Exception as e:
    print(f"Error getting r_frame_rate for {video_path}: {e}")
    return None


def get_video_duration(video_path, ffprobe_path="ffprobe") -> float | None:
  """Get video duration in seconds using ffprobe."""
  cmd = [
      ffprobe_path,
      "-v",
      "error",
      "-show_entries",
      "format=duration",
      "-of",
      "default=noprint_wrappers=1:nokey=1",
      video_path,
  ]
  try:
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    duration = float(result.stdout)
    return duration
  except Exception as e:
    print(f"Error getting duration for {video_path}: {e}")
    return None
