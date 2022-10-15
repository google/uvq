"""Run inference with the UVQ model.

Example usage:

python3 uvq_main.py \
  --input_files="Gaming_1080P-0ce6,20,Gaming_1080P-0ce6_orig.mp4" \
  --output_dir=results \
  --model_dir=models \
  --transpose=False


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

import tensorflow as tf

import uvq_utils as utils
from tensorflow.compat.v1 import flags, gfile


FLAGS = tf.compat.v1.flags.FLAGS

# Following parameters are required
flags.DEFINE_string('input_files', '', 'configuration of input files.')
flags.DEFINE_string('output_dir', '', 'Directory to save results.')
flags.DEFINE_string('model_dir', 'models', 'Directory to UVQ models.')
flags.DEFINE_bool('transpose', False, 'Whether to tranpose the input video.')


def main(_):
  # Input must be in format: video_id,video_length,file_path
  video_id, video_length, filepath = FLAGS.input_files.split(',')
  video_length = int(video_length)

  output_dir = '%s/%s' % (FLAGS.output_dir, video_id)
  feature_dir = '%s/features' % output_dir
  if not gfile.IsDirectory(feature_dir):
    gfile.MakeDirs(feature_dir)

  utils.generate_features(video_id, video_length, filepath, FLAGS.model_dir,
                          feature_dir, FLAGS.transpose)

  utils.prediction(video_id, video_length, FLAGS.model_dir, feature_dir,
                   output_dir)


if __name__ == '__main__':
  tf.compat.v1.app.run()
