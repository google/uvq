"""Utility functions of the UVQ model.

Copyright 2022 Google LLC

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


import csv
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow.compat.v1 import gfile, logging, saved_model
from tensorflow.compat.v1 import ConfigProto, Session, Graph


# Explicitly set input size
VIDEO_HEIGHT = 720
VIDEO_WIDTH = 1280
VIDEO_FPS = 5
VIDEO_CHANNEL = 3

# Output feature size
DIM_HEIGHT_FEATURE = 16
DIM_WIDTH_FEATURE = 16
DIM_CHANNEL_FEATURE = 100

# ContentNet specs
INPUT_HEIGHT_CONTENT = 496
INPUT_WIDTH_CONTENT = 496
INPUT_CHANNEL_CONTENT = 3
DIM_LABEL_CONTENT = 3862


def load_video(filepath, video_length, transpose=False):
  """Load input video."""
  # Rotate video if requested
  if transpose:
    transpose_param = 'transpose=1,'
  else:
    transpose_param = ''

  # Sample at constant frame rate, and save as RGB24 (RGBRGB...)
  fd, temp_filename = tempfile.mkstemp()
  cmd = (
      'ffmpeg  -i %s'
      ' -vf "%sscale=w=%d:h=%d:force_original_aspect_ratio=1,'
      'pad=%d:%d:(ow-iw)/2:(oh-ih)/2"'
      ' -r %d -f rawvideo -pix_fmt rgb24'
      ' -y %s'
  ) % (filepath, transpose_param, VIDEO_WIDTH, VIDEO_HEIGHT,
       VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS,
       temp_filename)
  try:
    logging.info('Run with cmd:% s\n', cmd)
    subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError as error:
    logging.error('Run with cmd: %s \n terminated with return code %s\n%s',
                  cmd, str(error.returncode), error.output)

  # For video, the entire video is divided into 1s chunks in 5 fps
  video = np.ndarray((video_length, VIDEO_FPS, VIDEO_HEIGHT, VIDEO_WIDTH,
                      VIDEO_CHANNEL), np.float32)
  video_resized = np.ndarray((video_length, VIDEO_FPS, INPUT_HEIGHT_CONTENT,
                              INPUT_WIDTH_CONTENT, INPUT_CHANNEL_CONTENT),
                             np.float32)

  with gfile.Open(temp_filename, 'rb') as rgb_file:
    rgb = bytearray(rgb_file.read())
    frame_size = VIDEO_WIDTH * VIDEO_HEIGHT * VIDEO_CHANNEL
    total_len = video_length * VIDEO_FPS * frame_size
    missing = total_len - len(rgb)
    if missing < 0:
      rgb = rgb[0 : total_len]
    else:
      rgb.extend(bytearray(missing))

    for i in range(video_length):
      for j in range(VIDEO_FPS):
        index = i * VIDEO_FPS + j
        frame = bytes(rgb[index*frame_size : (index+1)*frame_size])
        im = Image.frombytes('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), frame)
        video[i, j, :] = (np.resize(im.getdata(),
                                    (VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_CHANNEL))
                          / 255.0 - 0.5) * 2
        im_resized = im.resize((INPUT_HEIGHT_CONTENT, INPUT_WIDTH_CONTENT),
                               resample=Image.BILINEAR)

        video_resized[
            i, j, :] = (np.resize(im_resized.getdata(),
                                  (INPUT_HEIGHT_CONTENT, INPUT_WIDTH_CONTENT,
                                   INPUT_CHANNEL_CONTENT)) / 255.0 - 0.5) * 2

  # Delete temp files
  os.close(fd)
  os.remove(temp_filename)
  logging.info('Load %s done successfully.', filepath)
  return video, video_resized


def generate_content_feature(video, model_path, input_node, output_nodes):
  """Extract features from ContentNet."""
  with Session(graph=Graph(), config=ConfigProto(
      allow_soft_placement=True, log_device_placement=False)) as sess:
    saved_model.loader.load(
        sess, [saved_model.tag_constants.SERVING], model_path)

    label = np.ndarray((video.shape[0], DIM_LABEL_CONTENT), np.float32)
    feature = np.ndarray((video.shape[0], DIM_HEIGHT_FEATURE,
                          DIM_WIDTH_FEATURE, DIM_CHANNEL_FEATURE), np.float32)

    patch = np.ndarray((1, INPUT_HEIGHT_CONTENT, INPUT_WIDTH_CONTENT,
                        INPUT_CHANNEL_CONTENT), np.float32)

    for k in range(video.shape[0]):
      patch[0, :] = video[k, 0, :, :, :]
      patch_feature, patch_label = sess.run(output_nodes,
                                            feed_dict={input_node: patch})

      feature[k, :, :, :] = patch_feature
      label[k, :] = patch_label
    return feature, label


def generate_subnet_feature(video, model_path, input_width, input_height,
                            input_fps, feature_width, feature_height,
                            feature_channel, label_dim, input_node,
                            output_nodes):
  """Extract features from CompresionNet or DistortionNet."""
  with Session(graph=Graph(), config=ConfigProto(
      allow_soft_placement=True, log_device_placement=False)) as sess:
    saved_model.loader.load(
        sess, [saved_model.tag_constants.SERVING], model_path)

    num_h = int(VIDEO_HEIGHT / input_height)
    num_w = int(VIDEO_WIDTH / input_width)

    label = np.ndarray((video.shape[0], num_h, num_w, label_dim), np.float32)
    feature = np.ndarray((video.shape[0], num_h * feature_height,
                          num_w * feature_width, feature_channel), np.float32)

    if input_fps == 1:
      patch = np.ndarray((1, input_height, input_width,
                          video.shape[-1]), np.float32)
    else:
      patch = np.ndarray((1, input_fps, input_height, input_width,
                          video.shape[-1]), np.float32)

    for k in range(video.shape[0]):
      for j in range(num_h):
        for i in range(num_w):
          if input_fps == 1:
            patch[0, :] = video[k, 0, j * input_height:(j + 1) * input_height,
                                i * input_width:(i + 1) * input_width, :]
          else:
            patch[0, :] = video[k, :, j * input_height:(j + 1) * input_height,
                                i * input_width:(i + 1) * input_width, :]

          patch_feature, patch_label = sess.run(output_nodes,
                                                feed_dict={input_node: patch})

          feature[k, j * feature_height:(j + 1) * feature_height, i *
                  feature_width:(i + 1) * feature_width, :] = patch_feature

          label[k, j, i, :] = patch_label

    return feature, label


def generate_features(video_id, video_length, filepath, model_dir, output_dir,
                      transpose=False):
  """Generate features from input video."""
  video, video_resized = load_video(filepath, video_length, transpose)

  feature_compression, label_compression = generate_subnet_feature(
      video, '%s/compressionnet_baseline' % model_dir,
      320, 180, 5,  # input height, weight, fps,
      4, 4, 100, 1,  # feature map height, width, channels, and label_size
      'input_orig:0',
      ['feature_layer_orig:0', 'compress_level_orig:0'])

  feature_content, label_content = generate_content_feature(
      video_resized, '%s/contentnet_baseline' % model_dir,
      'map/TensorArrayV2Stack/TensorListStack:0',
      ['final_conv2d/Conv2D:0', 'class_confidence:0'])

  feature_distortion, label_distortion = generate_subnet_feature(
      video, '%s/distortionnet_baseline' % model_dir,
      640, 360, 1,  # input height, weight, fps,
      8, 8, 100, 26,  # feature map height, width, channels, and label_size
      'input_images:0',
      ['feature_map:0', 'dist_type_prediction/dist_type_predictions:0'])

  # Save features
  fd, temp = tempfile.mkstemp()

  feature_compression.astype('float32').tofile(temp)
  out_feature = '%s/%s_feature_compression.binary' % (output_dir, video_id)
  gfile.Copy(temp, out_feature, overwrite=True)

  feature_content.astype('float32').tofile(temp)
  out_feature = '%s/%s_feature_content.binary' % (output_dir, video_id)
  gfile.Copy(temp, out_feature, overwrite=True)

  feature_distortion.astype('float32').tofile(temp)
  out_feature = '%s/%s_feature_distortion.binary' % (output_dir, video_id)
  gfile.Copy(temp, out_feature, overwrite=True)

  # Feature labels
  np.savetxt(temp, label_compression.reshape(label_compression.shape[0], -1),
             fmt='%0.3f', delimiter=',')
  out_feature = '%s/%s_label_compression.csv' % (output_dir, video_id)
  gfile.Copy(temp, out_feature, overwrite=True)

  np.savetxt(temp, label_content.reshape(label_content.shape[0], -1),
             fmt='%0.3f', delimiter=',')
  out_feature = '%s/%s_label_content.csv' % (output_dir, video_id)
  gfile.Copy(temp, out_feature, overwrite=True)

  np.savetxt(temp, label_distortion.reshape(label_distortion.shape[0], -1),
             fmt='%0.3f', delimiter=',')
  out_feature = '%s/%s_label_distortion.csv' % (output_dir, video_id)
  gfile.Copy(temp, out_feature, overwrite=True)

  os.close(fd)
  os.remove(temp)


def load_features(video_id, dim_time, feature_dir):
  """Load pre-generated features."""
  input_compression_feature = '%s/%s_feature_compression.binary' % (
      feature_dir, video_id)
  with gfile.Open(input_compression_feature, 'rb') as input_file:
    s = input_file.read()
    with Session() as sess:
      feature_1d = tf.io.decode_raw(s, out_type=tf.float32)

      feature = tf.reshape(feature_1d,
                           [1, dim_time, DIM_HEIGHT_FEATURE,
                            DIM_WIDTH_FEATURE, DIM_CHANNEL_FEATURE])
      feature_compression = sess.run(feature)

  input_content_feature = '%s/%s_feature_content.binary' % (
      feature_dir, video_id)
  with gfile.Open(input_content_feature, 'rb') as input_file:
    s = input_file.read()
    with Session() as sess:
      feature_1d = tf.io.decode_raw(s, out_type=tf.float32)

      feature = tf.reshape(feature_1d,
                           [1, dim_time, DIM_HEIGHT_FEATURE,
                            DIM_WIDTH_FEATURE, DIM_CHANNEL_FEATURE])
      feature_content = sess.run(feature)

  input_distortion_feature = '%s/%s_feature_distortion.binary' % (
      feature_dir, video_id)
  with gfile.Open(input_distortion_feature, 'rb') as input_file:
    s = input_file.read()
    with Session() as sess:
      feature_1d = tf.io.decode_raw(s, out_type=tf.float32)

      feature = tf.reshape(feature_1d,
                           [1, dim_time, DIM_HEIGHT_FEATURE,
                            DIM_WIDTH_FEATURE, DIM_CHANNEL_FEATURE])
      feature_distortion = sess.run(feature)

  return feature_compression, feature_content, feature_distortion


def prediction(video_id, video_length, model_dir, feature_dir, output_dir):
  """Predict quality (MOS)."""
  trainset = 'ytugc20s'
  all_trainset_subindex = ['0', '1', '2', '3', '4']

  all_feature = ['compression',
                 'content',
                 'distortion',
                 'compression_content',
                 'compression_distortion',
                 'content_distortion',
                 'compression_content_distortion',
                 ]
  aggregation_model = 'avgpool'
  all_outputs = []
  for feature in all_feature:
    aggregated_mos = 0
    for trainset_subindex in all_trainset_subindex:
      model_name = '%s_%s_%s_%s' % (
          trainset, trainset_subindex, aggregation_model, feature)

      with Session(graph=Graph(), config=ConfigProto(
          allow_soft_placement=True, log_device_placement=False)) as sess:
        saved_model.loader.load(
            sess, [saved_model.tag_constants.SERVING],
            '%s/aggregationnet_baseline/%s' % (
                model_dir, model_name))

        [feature_compression, feature_content, feature_distortion
         ] = load_features(video_id, video_length, feature_dir)
        feature_compression = feature_compression[
            :, 0:video_length, :, :, :]
        feature_content = feature_content[
            :, 0:video_length, :, :, :]
        feature_distortion = feature_distortion[
            :, 0:video_length, :, :, :]

        pred_mos = sess.run(
            'Model/mos:0',
            feed_dict={'feature_compression:0': feature_compression,
                       'feature_content:0': feature_content,
                       'feature_distortion:0': feature_distortion,
                       })
        pred_mos = pred_mos[0][0]
        aggregated_mos += pred_mos

    aggregated_mos /= len(all_trainset_subindex)
    all_outputs.append([video_id, feature, aggregated_mos])

    fd, temp = tempfile.mkstemp()
    with gfile.Open(temp, 'w') as f:
      writer = csv.writer(f)
      writer.writerows(all_outputs)
    out_file = '%s/%s_uvq.csv' % (output_dir, video_id)
    gfile.Copy(temp, out_file, overwrite=True)
    os.close(fd)
    os.remove(temp)
