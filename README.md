# UVQ: Universal Video Quality Model 

This repository contains checkpointed models of Google's Universal Video Quality (UVQ) model.
UVQ is a no-reference perceptual video quality assessment model that is designed to work
well on user-generated content, where there is no pristine reference.

News: UVQ 1.5 is released with better generalizability, robustness, and efficiency !

Read this blog post for an overview of UVQ:

"[UVQ: Measuring YouTube's Perceptual Video Quality](https://ai.googleblog.com/2022/08/uvq-measuring-youtubes-perceptual-video.html)", Google AI Blog 2022

More details are available in our paper:

Yilin Wang, Junjie Ke, Hossein Talebi, Joong Gon Yim, Neil Birkbeck, Balu Adsumilli, Peyman Milanfar, Feng Yang, "[Rich features for perceptual quality assessment of UGC videos](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Rich_Features_for_Perceptual_Quality_Assessment_of_UGC_Videos_CVPR_2021_paper.html)", CVPR 2021.

The corresponding data from the paper is available for download from: [YouTube UGC Dataset](https://media.withyoutube.com)

## Running the code

### Dependencies

You must have [FFmpeg](http://www.ffmpeg.org/) installed and available on your path.

The models and code require Python 3 (tested with 3.13.7) and [PyTorch](https://pytorch.org/).

With virtualenv, you can install the requirements to a virtual environment:
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Predict Quality

You can grab some examples videos from the [YouTube UGC Dataset](https://media.withyoutube.com). For example, you can get Gaming_1080P-0ce6_orig.mp4 using curl:

```
curl -o Gaming_1080P-0ce6_orig.mp4 https://storage.googleapis.com/ugc-dataset/vp9_compressed_videos/Gaming_1080P-0ce6_orig.mp4
```

You can then run inference using `uvq_inference.py`. Use the `--model_version`
flag to select between UVQ 1.0 (`1.0`) and UVQ 1.5 (`1.5`).

**UVQ 1.5 (Default)**

```bash
python uvq_inference.py Gaming_1080P-0ce6_orig.mp4 --model_version 1.5
```

This will output a dictionary containing the UVQ 1.5 scores:
```
{'uvq1p5_score': 3.880362033843994, 'per_frame_scores': [4.021927833557129, 4.013788223266602, 4.110747814178467, 4.142043113708496, 4.1536993980407715, 4.147506237030029, 4.149798393249512, 4.149064064025879, 4.149083137512207, 4.133814811706543, 3.5636682510375977, 3.8045108318328857, 3.630220413208008, 3.6495614051818848, 3.6260201930999756, 3.6136975288391113, 3.5050578117370605, 3.7031033039093018, 3.676196575164795, 3.663726806640625], 'frame_indices': [...]}
```

**UVQ 1.0**

```bash
python uvq_inference.py Gaming_1080P-0ce6_orig.mp4 --model_version 1.0
```

This will output a dictionary containing the UVQ 1.0 scores:
```
{'compression': 3.927565574645996, 'content': 3.948335313796997, 'distortion': 4.26719913482666, 'compression_content': 3.953589344024658, 'compression_distortion': 4.061836576461792, 'content_distortion': 4.070189666748047, 'compression_content_distortion': 4.060612201690674}
```

We provide multiple predicted scores, using different combinations of UVQ features.
`compression_content_distortion` (combining three features) is our default score for Mean Opinion Score (MOS) prediction for UVQ 1.0.

#### Optional Arguments

*   `--transpose`: Transpose the video before processing (e.g., for portrait videos).
*   `--output OUTPUT`: Path to save the output scores to a file. Scores will be saved in JSON format.
*   `--device DEVICE`: Device to run inference on (e.g., `cpu` or `cuda`).
*   `--fps FPS`: (UVQ 1.5 only) Frames per second to sample. Default is 1. Use -1 to sample all frames.


## Performance

With default `--fps 1` sampling, UVQ 1.5 can run faster than real-time on multi-core CPUs.
CPU inference speed was measured on a virtual machine with an AMD EPYC 7B13 processor, using `Gaming_1080P-0ce6_orig.mp4` (20 seconds duration, 1080p resolution), sampling 1 frame per second (20 frames total).

Example command:
```bash
time taskset -c 0-3 python uvq_inference.py Gaming_1080P-0ce6_orig.mp4 --output pred.json
```

The wall-clock time varies by the number of cores assigned:

*   8 cores (`taskset -c 0-7`): ~13.8 seconds
*   4 cores (`taskset -c 0-3`): ~17.9 seconds
*   2 cores (`taskset -c 0-1`): ~26.5 seconds
*   1 core (`taskset -c 0-0`): ~43.6 seconds

Your runtime may vary based on CPU architecture, clock speed, and system load.

## Contributors

[//]: contributor-faces

<a href="https://github.com/yilinwang01"><img src="https://avatars.githubusercontent.com/u/30224449?v=4" title="yilinwang01" width="80" height="80"></a>
<a href="https://github.com/nbirkbeck"><img src="https://avatars.githubusercontent.com/u/6225937?v=4" title="nbirkbeck" width="80" height="80"></a>
<a href="https://github.com/megabalu"><img src="https://avatars.githubusercontent.com/u/99928166?v=4" title="megabalu" width="80" height="80"></a>
<a href="https://github.com/yaohongwu-g"><img src="https://avatars.githubusercontent.com/u/188632463?v=4" title="yaohongwu-g" width="80" height="80"></a>

[//]: contributor-faces

