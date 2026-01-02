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


def yield_video_1p5_chunks(
    filepath: str,
    video_length: int,
    transpose: bool = False,
    video_fps: int = 1,
    video_height: int = 1080,
    video_width: int = 1920,
    ffmpeg_path: str = "ffmpeg",
    chunk_size_frames: int = 16,
):
  """Yields chunks of the video as numpy arrays.

  Args:
    filepath: Path to the video file.
    video_length: Length of the video in seconds.
    transpose: Whether to transpose the video.
    video_fps: Frames per second to sample for inference.
    video_height: Height of the video to resize to.
    video_width: Width of the video to resize to.
    chunk_size_frames: Number of frames to yield per chunk.
  
  Yields:
    A tuple containing:
      - A chunk of the loaded video as a numpy array (batch, 1, h, w, c).
      - The number of real frames in the entire video (only available once determined).
  """
  video_channel = 3
  # Rotate video if requested
  if transpose:
    transpose_param = "transpose=2,"  # rotate 90 degrees counterclockwise
  else:
    transpose_param = ""

  # Sample at constant frame rate, and save as RGB24 (RGBRGB...)
  fd, temp_filename = tempfile.mkstemp()
  cmd = (
      f"{ffmpeg_path} -i {filepath} -vf"
      f" {transpose_param}scale=w={video_width}:h={video_height}:flags=bicubic,format=rgb24"
      f" -r {video_fps} -f rawvideo -pix_fmt rgb24 -y {temp_filename}"
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
  try:
    with open(temp_filename, "rb") as rgb_file:
      single_frame_size = video_width * video_height * video_channel
      full_decode_size = video_length * video_fps * single_frame_size
      rgb_file.seek(0, 2)
      rgb_file_size = rgb_file.tell()
      rgb_file.seek(0)
      num_real_frames = rgb_file_size // single_frame_size
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

      chunk_size_bytes = chunk_size_frames * single_frame_size
      
      # Read and yield chunks
      read_frames = 0
      while read_frames < num_real_frames:
        chunk_bytes = rgb_file.read(chunk_size_bytes)
        if not chunk_bytes:
          break
          
        # Handle partial chunks (e.g. end of file)
        # We read len(chunk_bytes). We simply divide by single_frame_size.
        # If there are leftovers < single_frame_size (partial frame), we ignore them.
        current_chunk_frames = len(chunk_bytes) // single_frame_size
        
        if current_chunk_frames == 0:
           break

        # Truncate to valid frames bytes
        valid_bytes = current_chunk_frames * single_frame_size
        if len(chunk_bytes) > valid_bytes:
            logging.warning("Read partial frame at end of file, truncating.")
            chunk_bytes = chunk_bytes[:valid_bytes]

        if current_chunk_frames == 0:
           break

        video_chunk = (
            np.reshape(
                np.frombuffer(chunk_bytes, "uint8"),
                (current_chunk_frames, 1, video_height, video_width, 3),
            ).astype(np.float32)
            / 255.0
            - 0.5
        ) * 2
        
        yield video_chunk, num_real_frames
        read_frames += current_chunk_frames

  finally:
    # Delete temp files
    os.close(fd)
    if os.path.exists(temp_filename):
      os.remove(temp_filename)
    logging.info("Load %s done successfully.", filepath)


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
  
  Note: This loads the entire video into memory. Use yield_video_1p5_chunks for large videos.
  
  Args:
    filepath: Path to the video file.
    video_length: Length of the video in seconds.
    transpose: Whether to transpose the video.
    video_fps: Frames per second to sample for inference.
    video_height: Height of the video to resize to.
    video_width: Width of the video to resize to.

  Returns:
    A tuple containing the loaded video as a numpy array and the number of
    real frames.
  """
  chunks = []
  num_real_frames = 0
  for chunk, n_frames in yield_video_1p5_chunks(
      filepath, video_length, transpose, video_fps, video_height, video_width, ffmpeg_path
  ):
    chunks.append(chunk)
    num_real_frames = n_frames
  
  if not chunks:
      return np.array([]), 0

  # Reconstruct the full video array.
  # Shape will be (TotalFrames, 1, H, W, 3).
  video = np.concatenate(chunks, axis=0) 
      
  return video, num_real_frames
