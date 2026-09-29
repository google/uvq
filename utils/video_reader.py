"""utility to resize and load the resized video.

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

import logging
import os
import subprocess
import tempfile
from collections.abc import Iterator

import numpy as np


def _extend_array(rgb: bytearray, total_len: int) -> bytearray:
  """Extends the byte array (or truncates) to be total_len."""
  missing = total_len - len(rgb)
  if missing < 0:
    rgb = rgb[0:total_len]
  else:
    rgb.extend(bytearray(missing))
  return rgb


def load_video_1p0(
    filepath: str,
    video_length: int,
    transpose: bool = False,
    video_fps: int = 5,
    ffmpeg_path: str = "ffmpeg",
) -> tuple[np.ndarray, np.ndarray]:
  """Load input video for UVQ 1.0.

  Args:
    filepath: Path to the video file.
    video_length: Length of the video in seconds.
    transpose: Whether to transpose the video.
    video_fps: Frames per second to sample for inference.

  Returns:
    A tuple containing the loaded video and resized video as numpy arrays.
  """
  video_height = 720
  video_width = 1280
  video_channel = 3
  input_height_content = 496
  input_width_content = 496
  # Rotate video if requested
  if transpose:
    transpose_param = "transpose=1,"
  else:
    transpose_param = ""

  # Sample at constant frame rate, and save as RGB24 (RGBRGB...)
  fd, temp_filename = tempfile.mkstemp()
  fd_small, temp_filename_small = tempfile.mkstemp()
  filter_complex = (
      f"[0:v]{transpose_param}scale=w={video_width}:h={video_height}:"
      f"flags=bicubic:force_original_aspect_ratio=1,"
      f"pad={video_width}:{video_height}:(ow-iw)/2:(oh-ih)/2,"
      f"format=rgb24,split=2[out1][tmp],"
      f"[tmp]scale={input_width_content}:{input_height_content}:flags=bilinear[out2]"
  )
  cmd = (
      f"{ffmpeg_path}  -i {filepath} -filter_complex \"{filter_complex}\""
      f" -map [out1] -r {video_fps} -f rawvideo -pix_fmt rgb24 -y {temp_filename}"
      f" -map [out2] -r {video_fps} -f rawvideo -pix_fmt rgb24 -y"
      f" {temp_filename_small}"
  )

  try:
    logging.info("Run with cmd:% s\n", cmd)
    subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError as error:
    logging.fatal(
        "Run with cmd: %s \n terminated with return code %s\n%s",
        cmd,
        str(error.returncode),
        error.output,
    )
    raise error

  # For video, the entire video is divided into 1s chunks in 5 fps
  with (
      open(temp_filename, "rb") as rgb_file,
      open(temp_filename_small, "rb") as rgb_file_small,
  ):
    single_frame_size = video_width * video_height * video_channel
    full_decode_size = video_length * video_fps * single_frame_size
    rgb_file.seek(0, 2)
    rgb_file_size = rgb_file.tell()
    rgb_file.seek(0)
    assert rgb_file_size >= single_frame_size, (
        f"Decoding failed to output a single frame: {rgb_file_size} <"
        f" {single_frame_size}"
    )
    if rgb_file_size < full_decode_size:
      logging.warning(
          "Decoding may be truncated: %d bytes (%d frames) < %d bytes (%d"
          " frames), or video length (%ds) may be too incorrect",
          rgb_file_size,
          rgb_file_size / single_frame_size,
          full_decode_size,
          full_decode_size / single_frame_size,
          video_length,
      )

    rgb = _extend_array(bytearray(rgb_file.read()), full_decode_size)
    rgb_small = _extend_array(
        bytearray(rgb_file_small.read()),
        video_length
        * video_fps
        * input_width_content
        * input_height_content
        * video_channel,
    )
    video = (
        np.reshape(
            np.frombuffer(rgb, "uint8"),
            (video_length, int(video_fps), video_height, video_width, 3),
        )
        / 255.0
        - 0.5
    ) * 2
    video_resized = (
        np.reshape(
            np.frombuffer(rgb_small, "uint8"),
            (
                video_length,
                int(video_fps),
                input_height_content,
                input_width_content,
                3,
            ),
        )
        / 255.0
        - 0.5
    ) * 2

  # Delete temp files
  os.close(fd)
  os.remove(temp_filename)
  os.close(fd_small)
  os.remove(temp_filename_small)
  logging.info("Load %s done successfully.", filepath)
  return video, video_resized


def _video_1p5_command(
    filepath: str,
    transpose: bool,
    video_fps: int,
    video_height: int,
    video_width: int,
    ffmpeg_path: str,
) -> list[str]:
  """Build the FFmpeg command used by the UVQ 1.5 reader."""
  transpose_filter = "transpose=2," if transpose else ""
  video_filter = (
      f"fps={video_fps}:start_time=0:round=up,{transpose_filter}"
      f"scale=w={video_width}:h={video_height}:flags=bicubic,format=rgb24"
  )
  return [
      ffmpeg_path,
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      filepath,
      "-vf",
      video_filter,
      "-fps_mode",
      "passthrough",
      "-f",
      "rawvideo",
      "-pix_fmt",
      "rgb24",
      "pipe:1",
  ]


def iter_video_batches_1p5(
    filepath: str,
    video_length: int,
    transpose: bool = False,
    video_fps: int = 1,
    video_height: int = 1080,
    video_width: int = 1920,
    batch_size: int = 24,
    ffmpeg_path: str = "ffmpeg",
) -> Iterator[tuple[np.ndarray, int]]:
  """Decode normalized RGB frames for UVQ 1.5 in batches.

  A truncated decode is padded with black frames up to
  ``video_length * video_fps`` frames.

  Args:
    filepath: Path to the video file.
    video_length: Length of the video in seconds.
    transpose: Whether to transpose the video.
    video_fps: Frames per second to sample for inference.
    video_height: Height of the video to resize to.
    video_width: Width of the video to resize to.
    batch_size: Maximum number of frames per yielded batch.
    ffmpeg_path: Path to the FFmpeg executable.

  Yields:
    A tuple containing a float32 array of shape
    (frames, video_height, video_width, 3) with values in [-1, 1], and the
    number of real decoded frames in that batch (0 for padding batches).

  Raises:
    ValueError: If batch_size is less than 1.
    RuntimeError: If FFmpeg returns an incomplete RGB frame.
    subprocess.CalledProcessError: If FFmpeg exits with an error.
    RuntimeError: If FFmpeg does not decode a single frame.
  """
  if batch_size < 1:
    raise ValueError("batch_size must be at least 1")

  video_channels = 3
  single_frame_size = video_width * video_height * video_channels
  expected_frames = video_length * video_fps
  command = _video_1p5_command(
      filepath,
      transpose,
      video_fps,
      video_height,
      video_width,
      ffmpeg_path,
  )
  logging.info("Run with cmd: %s\n", subprocess.list2cmdline(command))

  num_real_frames = 0
  emitted_frames = 0
  with tempfile.TemporaryFile() as error_file:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=error_file,
    )
    assert process.stdout is not None
    completed_read = False

    try:
      while True:
        rgb = process.stdout.read(batch_size * single_frame_size)
        if not rgb:
          break
        if len(rgb) % single_frame_size != 0:
          raise RuntimeError(
              "FFmpeg returned an incomplete RGB frame: "
              f"{len(rgb)} bytes is not divisible by {single_frame_size}"
          )

        decoded_frames = len(rgb) // single_frame_size
        num_real_frames += decoded_frames
        frames_to_emit = min(decoded_frames, expected_frames - emitted_frames)
        if frames_to_emit <= 0:
          continue

        rgb_array = np.frombuffer(
            rgb, dtype=np.uint8, count=frames_to_emit * single_frame_size
        )
        video = rgb_array.reshape(
            frames_to_emit,
            video_height,
            video_width,
            video_channels,
        ).astype(np.float32)
        video *= 2.0 / 255.0
        video -= 1.0
        emitted_frames += frames_to_emit
        yield video, frames_to_emit
      completed_read = True
      return_code = process.wait()
    finally:
      process.stdout.close()
      if process.poll() is None:
        process.terminate()
        process.wait()

    if completed_read and return_code != 0:
      error_file.seek(0)
      error_output = error_file.read().decode(errors="replace")
      raise subprocess.CalledProcessError(
          return_code, command, stderr=error_output
      )

  if num_real_frames < 1:
    raise RuntimeError(
        f"Decoding failed to output a single frame for '{filepath}'"
    )
  if num_real_frames < expected_frames:
    logging.warning(
        "Decoding may be truncated: %d frames < %d frames, or video length"
        " (%ds) may be too incorrect",
        num_real_frames,
        expected_frames,
        video_length,
    )

  while emitted_frames < expected_frames:
    padding_frames = min(batch_size, expected_frames - emitted_frames)
    padding = np.full(
        (padding_frames, video_height, video_width, video_channels),
        -1.0,
        dtype=np.float32,
    )
    emitted_frames += padding_frames
    yield padding, 0


def load_video_1p5(
    filepath: str,
    video_length: int,
    transpose: bool = False,
    video_fps: int = 1,
    video_height: int = 1080,
    video_width: int = 1920,
    ffmpeg_path: str = "ffmpeg",
) -> tuple[np.ndarray, int]:
  """Load input video for UVQ 1.5.

  Args:
    filepath: Path to the video file.
    video_length: Length of the video in seconds.
    transpose: Whether to transpose the video.
    video_fps: Frames per second to sample for inference.
    video_height: Height of the video to resize to.
    video_width: Width of the video to resize to.
    ffmpeg_path: Path to the FFmpeg executable.

  Returns:
    A tuple containing the loaded video as a numpy array and the number of
    real frames.
  """
  batches = []
  num_real_frames = 0
  for batch, real_frames in iter_video_batches_1p5(
      filepath,
      video_length,
      transpose,
      video_fps,
      video_height,
      video_width,
      ffmpeg_path=ffmpeg_path,
  ):
    batches.append(batch)
    num_real_frames += real_frames

  video = np.concatenate(batches).reshape(
      video_length, video_fps, video_height, video_width, 3
  )
  logging.info("Load %s done successfully.", filepath)
  return video, num_real_frames
