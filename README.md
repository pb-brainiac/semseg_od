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
Download <a href=https://github.com/fyu/lsun>lsun</a> val and <a href=http://host.robots.ox.ac.uk/pascal/VOC/voc2007/>Pascal 2007</a> train and test sets. 
```bash
cd data
./download.sh
```

Create a <a href=http://www.wilddash.cc/accounts/login>Wilddash account</a> and download:
* wd_val_01.zip

Either download and extract to `data/` or create a symbolic link `data/wd_val_01`

### Prepare libs:
```bash
cd libs
./build.sh
```

### Run inference
```
python inference.py --model models/base.py --params params/base_oe.pt --save-outputs=0 --reshape-size=540 --verbose=1 --AP-iters 5
python inference.py --model models/two_head.py --params params/two_head.pt --save-outputs=1 --save-name="two_head" --reshape-size=540 --verbose=0 --AP-iters 5
```
