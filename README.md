# object_detection_in_remote_sensing_images
## Introduction
Object detection in remote sensing images has been widely used in many fields, such as forest and ocean monitoring, military target discrimination and so on. With the development of science and technology, people hope to process massive remote sensing image data and obtain useful information automatically. However, object detection in remote sensing images is always a challenging task due to complex background, variable illumination conditions, large image size, wide range of object scales and angle orientations, dense arrangement and overlapping occlusion between objects. To solve the above problems, we propose an object detection framework based on convolutional neural networks and attention mechanism, with the main work as follows:

Firstly, we analyze the research background and its significance, and then deeply discuss the research status concerning object detection in remote sensing images using deep learning methods. Also, we introduce the basic structure and corresponding mathematical principles of Cascade R-CNN and Swin Transformer, and analyze the advantages of each module.

Secondly, DOTA devkit is used to segment training set and validation set in DOTA-v1.0 and merge detection results, whereas mmdetection is used to build a basic detector framework. In this paper, ResNet-50, ResNet-101, Swin-T, Swin-S and Swin-B backbone networks are utilized to train and test the model in both 1x and 3x training schedules, and the performance of different detectors is compared to verify the advantages of attention mechanism in object detection. The mAP is improved by about 5.1 percentage points from the ResNet-50_1x baseline.

Finally, various data augmentation strategies are used to improve the detection performance of the model. The strategies include image flipping, multi-scale random cropping, shift scale rotation, brightness and contrast transformation, color transformation, multi-scale testing and so on. In this paper, the effectiveness of various data augmentation strategies is illustrated through ablation experiments. Data augmentation strategies have improved mAP by 6.0 percentage points from the Swin-T_3x baseline to 77.7, achieving good detection performance.

## Usage
### Set up Environment
```
wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
conda create -n py37 python=3.7
conda activate py37
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
conda install pytorch=1.7.0 torchvision cudatoolkit=10.2 -c pytorch
```
### mmdetection
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html
pip install -r requirements/build.txt
pip install -v -e .
```
### DOTA_devkit
```
conda install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```
### Train
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup ./tools/dist_train.sh ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_max_ms_train_3x_coco_dota.py 8 > nohup.log 2>&1 &
```
### Test
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_max_ms_train_3x_coco_dota.py "./work_dirs/cascade_rcnn_swin-t-p4-w7_fpn_max_ms_train_3x_coco_dota/epoch_35.pth" 8 --out "/home/ZhangYumeng/Workspace/DOTA_devkit-master/DOTA_devkit-master/outcome_max_35.pkl" --eval bbox
```
### Merge
```
python HBB_pkl_reader.py --pkl_name outcome_max_35.pkl 
```
## Results
Evaluation before test @34th epoch:
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.463
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.709
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.521
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.330
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.488
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.544
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.552
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.567
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.567
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.404
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.611
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.654
```

After merge:
```
map: 0.7772415280063075
classaps:  [96.15830979 81.61095892 51.94498409 67.58662255 70.36897231 84.70424018
 91.13452151 95.60817726 67.86447364 86.79844594 65.46443074 66.80557472
 85.52208765 73.23524553 81.05524719]
```

## Reference
Some of the source code is modified from the DOTA_devkit and mmdetection codebase.

DOTA devkit：https://github.com/CAPTAIN-WHU/DOTA_devkit

mmdetection：https://github.com/open-mmlab/mmdetection

Swin Transformer：https://github.com/SwinTransformer/Swin-Transformer-Object-Detection

-----

Copyright (c) 2022 Yumeng Zhang. All rights reserved.

Date modified: 2022/06/02
