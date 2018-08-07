# Panoptic segmentation project

Author: Kevin Dethoor

Small project to perform a panoptic segementation on [COCO data set](http://cocodataset.org/).
It relies on the famous DeepLab for semantic segmentation and MaskRCNN for detection. 

# Project structure

* The `Inference` notebook contains the lines of code to infer both stuffs and detections on COCO.
* The `StuffInstanceToPanoptic2chPng` notebook shows how stuff images and detections are merged to make a 2-channel PNG image.
* The `stuffInstanceToPanoptic2chPng` does the same as the notebook of the same name, but it will process a whole folder rather than just one image. It uses `panoptic_2ch_conversion_config.json` as a config file
* the `panopticutils`folder uses

# Notes

Symlinks to your `coco` folder and to your `pycocotools` folder will be required for the code to work properly. This therefore assumes that you have built `pycocotools` - it's in the submodule if you haven't downloaded it yet.

# Current status
At the moment, the whole pipeline is unfortunately incomplete. COCO val set has been converted to 2-channel PNG images but
* some images from the panoptic data set seem to be absent from either the detection or the stuff data sets - they were skipped
* since the panopticapi assumes all images are properly provided, it isn't possible to run the code unless it is modified
* some labels seem to be incorrectly converted, and this needs to be checked.

Once these problems are resolved, one should be able to evaluate the approach.

# Some commands

To convert a detection and a segmentation mask into a 2-channel PNG:
```python stuffIntanceToPanoptic2chPng.py --config_path "panoptic_2ch_conversion_config.json"```

To convert from the 2 channel format to the COCO format:
```python panopticapi/format_converter.py --source_folder data/out/val2017/p2ch/ --images_json_file ../coco/annotations/panoptic_val2017.json --categories_json_file panopticapi/panoptic_coco_categories.json --segmentations_folder data/out/val2017/segm --predictions_json_file data/out/val2017/segm.json```

# Sources

* [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)<br>
L. C. Chen, G. Papandreou, I. Kokkinos et al.,<br>
In arXiv, 2016.

* [DeepLab with Pytorch](https://github.com/kazuto1011/deeplab-pytorch)<br>
K. Nakashima,<br>
On GitHub, 2018.

* [Detectron 2018](https://github.com/facebookresearch/detectron)<br>
R. Girshick, I. Radosavovic, G. Gkioxari, P. Dollár and K. He,<br>
On GitHub, 2018.

* [A Pytorch implementation of Detectron](https://github.com/roytseng-tw/Detectron.pytorch#supported-network-modules)<br>
R. Tseng,<br>
On GitHub, 2018.

* [Mask RCNN](https://arxiv.org/abs/1703.06870)<br>
A. Kirillov, K. He, R. Girshick, C. Rother, P. Dollár,<br>
In arXiv, 2018.

* [Panoptic segmentation](https://arxiv.org/abs/1801.00868)<br>
K. He, G. Gkioxari, P. Dollár, R. Girshick,<br>
In qrXiv, 2018.