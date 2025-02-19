To test the pytorch implementation, `cd` into the `uvq_pytorch` directory.
Download an example video. E.g. [Gaming_1080P-0ce6_orig.mp4](https://storage.googleapis.com/ugc-dataset/vp9_compressed_videos/Gaming_1080P-0ce6_orig.mp4)
Then run:
```
python inference.py Gaming_1080P-0ce6_orig.mp4 20
```
first argument being the name of the file and the second one the length of the video (in seconds).
This will print out a dictionary containing UVQ scores based on different combinations of CompressionNet, DistortionNet, and ContentNet.

Two additional arguments are accepted: `--transpose` specifies that the video should be transposed before processing. If a `--output` argument is provided, a file with that name will be created and the output results will be written to it as well.
