# Simultaneous semantic segmentation and outlier detection

Code to reproduce the results from
<div class="highlight highlight-html"><pre>
<b><a href=https://arxiv.org/abs/1908.01098>Simultaneous Semantic Segmentation and Outlier Detection in Presence of Domain Shift</a>
<a href=https://github.com/pb-brainiac>Petra Bevandić</a>, <a href=https://ivankreso.github.io/>Ivan Krešo</a>, <a href=https://github.com/orsic>Marin Oršić</a>, <a href=http://www.zemris.fer.hr/~ssegvic/index_en.html>Siniša Šegvić</a></b>
GCPR, 2019.
</pre></div>

## Run code
### Requirements
```
pip install -r requirements.txt
```

### Prepare data
Download <a href=https://github.com/fyu/lsun>lsun</a> val and <a href=http://host.robots.ox.ac.uk/pascal/VOC/voc2007/>Pascal 2007</a> train and test sets. If you use LInux, you can do this by running the download script:
```bash
cd data
./download.sh
```
If you are on Windows, use the script as a guide for organizing the data.

Create a <a href=http://www.wilddash.cc/accounts/login>Wilddash account</a> and download:
* wd_val_01.zip

Either download and extract to `data/` or create a symbolic link `data/wd_val_01`

### Prepare libs:
If you are no Linux, you can build the necessary libs by running the build script:
```bash
cd libs
./build.sh
```
If you are on Windows, use the script as a guide.

### Run inference
The inference script performs validation of segmentation and outlier detection. Semantic segmentation is validated by measuring mean intersection over union (mIoU) on the WildDash validation dataset.

Outlier detection is validated by measuring average precision (AP) in two ways. To validate negative image detection, we measure AP between WildDash val and `AP-iters` random subsets of LSUN val. We validate outlier patch detection on WildDash validation images with randomly pasted Pascal animals (we generate `AP-iters` sets of WildDash validation images with outlier patches).

We provide two models trained on Vistas and ImageNet datesets. The base model only has the segmentation head, and was trained to 
output uniform distribution on outlier pixels. The two head model contains a segmentation head and a separate outlier detection head.

If `save-outputs` is set to 1, the script saves the outputs of outlier detection and semantic segmentation, as well as the combined output into `./outputs/save_name/{confidence|segmentation|seg_with_conf}`. The `reshape-size` parameter represents the size of the smaller side of the input image.

```
python inference.py --model models/base.py --params params/base_oe.pt --save-outputs=0 --reshape-size=540 --verbose=1 --AP-iters 5
python inference.py --model models/two_head.py --params params/two_head.pt --save-outputs=1 --save-name="two_head" --reshape-size=540 --verbose=0 --AP-iters 5
```
