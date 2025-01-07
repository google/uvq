import argparse
import os

import numpy as np
import torch

from utils.aggregationnet import AggregationNetInference
from utils.compressionnet import CompressionNetInference
from utils.contentnet import ContentNetInference
from utils.distortionnet import DistortionNetInference
from utils.video_reader import VideoReader

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

        self.contentnet = ContentNetInference()
        self.compressionnet = CompressionNetInference()
        self.distotionnet = DistortionNetInference()
        self.aggregationnet = AggregationNetInference()

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
        return VideoReader.load_video(video_filename, video_length, transpose)


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
    if not os.path.exists(dirname):
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
        help="Length of the video in frames",
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
