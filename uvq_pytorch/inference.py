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
    def infer(self, video_filename, video_length, transpose=False):

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
        video_resized1, video_resized2 = VideoReader.load_video(
            video_filename, video_length, transpose
        )
        return video_resized1, video_resized2


def main():
    import sys

    video_filename = sys.argv[1]
    video_length = int(sys.argv[2])
    transpose = False
    uvq_inference = UVQInference()
    results: dict[str, float] = uvq_inference.infer(
        video_filename, video_length, transpose
    )


if __name__ == "__main__":
    main()
