"""Tests for the bounded UVQ 1.5 video reader."""

import io
import unittest
from unittest import mock

import numpy as np

from utils import video_reader


class _FakeProcess:
  """Minimal subprocess.Popen stand-in for raw-video reader tests."""

  def __init__(self, output: bytes, return_code: int = 0):
    self.stdout = io.BytesIO(output)
    self._configured_return_code = return_code
    self.returncode = None

  def poll(self):
    return self.returncode

  def terminate(self):
    self.returncode = -15

  def wait(self):
    if self.returncode is None:
      self.returncode = self._configured_return_code
    return self.returncode


class VideoReaderTest(unittest.TestCase):

  def test_ffmpeg_samples_before_scaling_without_shell_command(self):
    path = "video path with spaces;not-a-command.mp4"
    command = video_reader._video_1p5_command(
        path,
        transpose=True,
        video_fps=1,
        video_height=1080,
        video_width=1920,
        ffmpeg_path="ffmpeg",
    )

    self.assertIn(path, command)
    video_filter = command[command.index("-vf") + 1]
    self.assertLess(video_filter.index("fps="), video_filter.index("scale="))
    self.assertLess(
        video_filter.index("fps="), video_filter.index("transpose=")
    )
    self.assertIn("start_time=0:round=up", video_filter)

  @mock.patch("utils.video_reader.subprocess.Popen")
  def test_batches_are_float32_and_truncated_input_is_padded(self, popen):
    first_frame = bytes([0, 64, 128, 192, 224, 255])
    popen.return_value = _FakeProcess(first_frame)

    batches = list(
        video_reader.iter_video_batches_1p5(
            "input.mp4",
            video_length=2,
            video_fps=1,
            video_height=1,
            video_width=2,
            batch_size=1,
        )
    )

    self.assertEqual([real_frames for _, real_frames in batches], [1, 0])
    self.assertEqual(batches[0][0].dtype, np.float32)
    np.testing.assert_allclose(
        batches[0][0].reshape(-1),
        np.frombuffer(first_frame, dtype=np.uint8).astype(np.float32)
        * (2.0 / 255.0)
        - 1.0,
    )
    np.testing.assert_array_equal(batches[1][0], -1.0)

  @mock.patch("utils.video_reader.subprocess.Popen")
  def test_incomplete_rgb_frame_is_rejected(self, popen):
    popen.return_value = _FakeProcess(b"short")

    with self.assertRaisesRegex(RuntimeError, "incomplete RGB frame"):
      list(
          video_reader.iter_video_batches_1p5(
              "input.mp4",
              video_length=1,
              video_height=1,
              video_width=2,
          )
      )


if __name__ == "__main__":
  unittest.main()
