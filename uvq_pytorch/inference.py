"""UVQ (pytorch) inference entry point.

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

# run using: python3 uvq_pytorch/inference.py <path_to_video> <video_length_in_seconds> [--output <path_to_output_file>] [--transpose]
import argparse
import os
import sys

from utils import aggregationnet, compressionnet, contentnet, distortionnet

# append the path to the uvq/utils folder to the system path so that we can import video_reader
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
    )
)
import video_reader


# Output feature size
DIM_HEIGHT_FEATURE = 16
DIM_WIDTH_FEATURE = 16
DIM_CHANNEL_FEATURE = 100


class UVQInference:
    def infer(
        self, video_filename: str, video_length: int, transpose: bool = False
    ) -> dict[str, float]:
        """
        Args:
            video_filename: Path to the video file
            video_length: Length of the video in frames
            transpose: whether to transpose the video before processing
        Returns:
            A dictionary containing the UVQ scores for each category
        """

        self.contentnet = contentnet.ContentNetInference()
        self.compressionnet = compressionnet.CompressionNetInference()
        self.distotionnet = distortionnet.DistortionNetInference()
        self.aggregationnet = aggregationnet.AggregationNetInference()

        video_resized1, video_resized2 = self.load_video(
            video_filename, video_length, transpose
        )
        content_features, content_labels = (
            self.contentnet.get_labels_and_features_for_all_frames(video=video_resized2)
        )
        compression_features, compression_labels = (
            self.compressionnet.get_labels_and_features_for_all_frames(
                video=video_resized1,
            )
        )
        distortion_features, distortion_labels = (
            self.distotionnet.get_labels_and_features_for_all_frames(
                video=video_resized1,
            )
        )
        results = self.aggregationnet.predict(
            compression_features, content_features, distortion_features
        )
        print(results)
        return results

    def load_video(self, video_filename, video_length, transpose=False):
        video, video_small = video_reader.VideoReader.load_video(
            video_filename, video_length, transpose
        )
        video = video.transpose(0, 1, 4, 2, 3)
        video_small = video_small.transpose(0, 1, 4, 2, 3)
        return video, video_small


def main():
    parser = setup_parser()
    args = parser.parse_args()
    video_filename = args.video_filename
    video_length = args.video_length
    transpose = args.transpose
    output_filepath = args.output

    uvq_inference = UVQInference()
    results: dict[str, float] = uvq_inference.infer(
        video_filename, video_length, transpose
    )

    if output_filepath != "":
        write_dict_to_file(results, output_filepath)


def write_dict_to_file(d: dict, output_filepath: str) -> None:
    dirname = os.path.dirname(output_filepath)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(output_filepath, "w") as f:
        for key, value in d.items():
            f.write(f"{key}: {value}\n")


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video_filename",
        type=str,
        help="Path to the video file",
    )
    parser.add_argument(
        "video_length",
        type=int,
        help="Length of the video in seconds",
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
    return parser


if __name__ == "__main__":
    main()
