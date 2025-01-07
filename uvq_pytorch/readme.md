To test the pytorch implementation, `cd` into the `uvq_pytorch` directory.
Download an example video. E.g. [Gaming_1080P-0ce6_orig.mp4](https://storage.googleapis.com/ugc-dataset/vp9_compressed_videos/Gaming_1080P-0ce6_orig.mp4)
Then run:
```
python inference.py Gaming_1080P-0ce6_orig.mp4 20
```
first argument being the name of the file and the second one the length of the video.

This will print out a dictionary containing UVQ scores based on different combinations of CompressionNet, DistortionNet, and ContentNet.
