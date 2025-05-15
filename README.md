# UVQ: Universal Video Quality Model 

This repository contains checkpointed models of Google's Universal Video Quality (UVQ) model.
UVQ is a no-reference perceptual video quality assessment model that is designed to work
well on user-generated content, where there is no pristine reference.

Read this blog post for an overview of UVQ:

"[UVQ: Measuring YouTube's Perceptual Video Quality](https://ai.googleblog.com/2022/08/uvq-measuring-youtubes-perceptual-video.html)", Google AI Blog 2022

More details are available in our paper:

Yilin Wang, Junjie Ke, Hossein Talebi, Joong Gon Yim, Neil Birkbeck, Balu Adsumilli, Peyman Milanfar, Feng Yang, "[Rich features for perceptual quality assessment of UGC videos](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Rich_Features_for_Perceptual_Quality_Assessment_of_UGC_Videos_CVPR_2021_paper.html)", CVPR 2021.

The corresponding data from the paper is available for download from: [YouTube UGC Dataset](https://media.withyoutube.com)

## Running the code

### Dependencies

You must have [FFmpeg](http://www.ffmpeg.org/) installed and available on your path.

The models and code require Python 3.6 (or greater) and [Tensorflow](https://www.tensorflow.org/install).

With virtualenv, you can install the requirements to a virtual environment:
```
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Predict Quality

You can grab some examples videos from the [YouTube UGC Dataset](https://media.withyoutube.com). For example, you can get Gaming_1080P-0ce6_orig.mp4 using curl:

```
curl -o Gaming_1080P-0ce6_orig.mp4 https://storage.googleapis.com/ugc-dataset/vp9_compressed_videos/Gaming_1080P-0ce6_orig.mp4
```

You can then run the example:

```bash
mkdir -p results
python3 uvq_main.py --input_files="Gaming_1080P-0ce6_orig,20,Gaming_1080P-0ce6_orig.mp4" --output_dir results --model_dir models
```

#### Input file formatting
The input files format is a line with the following fields:

`id,video_length,filepath`

#### Results

The `output_dir` will contain a csv file with the results for each model. For example,
```bash
cat results/Gaming_1080P-0ce6_orig/Gaming_1080P-0ce6_orig_uvq.csv
```
Gives:
```bash
Gaming_1080P-0ce6,compression,3.927867603302002
Gaming_1080P-0ce6,content,3.945391607284546
Gaming_1080P-0ce6,distortion,4.267196607589722
Gaming_1080P-0ce6,compression_content,3.9505696296691895
Gaming_1080P-0ce6,compression_distortion,4.062019920349121
Gaming_1080P-0ce6,content_distortion,4.067790699005127
Gaming_1080P-0ce6,compression_content_distortion,4.058663845062256
```

We provide multiple predcited scores, using different combinations of UVQ features.
`compression_content_distortion` (combining three features) is our default score for Mean Opinion Score (MOS) prediction.

The output features folder includes UVQ labels and raw features:
```bash
Gaming_1080P-0ce6_orig_feature_compression.binary
Gaming_1080P-0ce6_orig_feature_content.binary
Gaming_1080P-0ce6_orig_feature_distortion.binary
Gaming_1080P-0ce6_orig_label_compression.csv
Gaming_1080P-0ce6_orig_label_content.csv
Gaming_1080P-0ce6_orig_label_distortion.csv
```
UVQ labels (.csv, each row corresponding to 1s chunk):<br />
compression: 16 compression levels per row, corresponding to 4x4 subregions of the entire frame.<br />
distortion: 26 distortion types defined in [KADID-10k](http://database.mmsp-kn.de/kadid-10k-database.html) for 2x2 subregions. The first element is the undefined type. <br /> 
content: 3862 content labels defined in [YouTube-8M](https://research.google.com/youtube8m/).<br />

UVQ raw features (in binary):<br />
25600 float numbers per 1s chunk.


## Contributors

[//]: contributor-faces

<a href="https://github.com/yilinwang01"><img src="https://avatars.githubusercontent.com/u/30224449?v=4" title="yilinwang01" width="80" height="80"></a>
<a href="https://github.com/nbirkbeck"><img src="https://avatars.githubusercontent.com/u/6225937?v=4" title="nbirkbeck" width="80" height="80"></a>
<a href="https://github.com/megabalu"><img src="https://avatars.githubusercontent.com/u/99928166?v=4" title="megabalu" width="80" height="80"></a>
<a href="https://github.com/yaohongwu-g"><img src="https://avatars.githubusercontent.com/u/188632463?v=4" title="yaohongwu-g" width="80" height="80"></a>

[//]: contributor-faces

